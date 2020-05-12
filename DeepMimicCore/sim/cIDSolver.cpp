#include "cIDSolver.hpp"
#include "SimCharacter.h"
#include "../Extras/InverseDynamics/btMultiBodyTreeCreator.hpp"
#include "sim/CtPDController.h"
#include <util/BulletUtil.h>
#include <iostream>

cIDSolver::cIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, eIDSolverType type)
{
	assert(sim_char != nullptr);
	assert(world != nullptr);
	mSimChar = sim_char;
	mCharController = std::dynamic_pointer_cast<cCtPDController>(sim_char->GetController()).get();
	assert(mCharController!=nullptr && mCharController->IsValid());
	mMultibody = sim_char->GetMultiBody().get();
	mWorldScale = sim_char->GetWorld()->GetScale();
	mWorld = world;
	mFloatingBase = !(mMultibody->hasFixedBase());
	mDof = mMultibody->getNumDofs();
	if (mFloatingBase == true)
	{
		mDof += 6;
	}

	mNumLinks = mMultibody->getNumLinks() + 1;

	// init ID model
	btInverseDynamicsBullet3::btMultiBodyTreeCreator id_creator;
	if (-1 == id_creator.createFromBtMultiBody(mMultibody, false))
	{
		b3Error("error creating tree\n");
	}
	else
	{
		mInverseModel = btInverseDynamicsBullet3::CreateMultiBodyTree(id_creator);
		btInverseDynamicsBullet3::vec3 gravity(cBulletUtil::tVectorTobtVector(gGravity));

		mInverseModel->setGravityInWorldFrame(gravity * mWorldScale);
	}

	// init id mapping
	for (int inverse_id = 0; inverse_id < mNumLinks; inverse_id++)
	{
		int world_id;
		if (0 == inverse_id)
		{
			btMultiBodyLinkCollider * base_collider = mMultibody->getBaseCollider();
			if (nullptr == base_collider)
			{
				continue;
			}
			else
			{
				world_id = base_collider->getWorldArrayIndex();
			}
		}
		else
		{
			world_id = mMultibody->getLinkCollider(inverse_id - 1)->getWorldArrayIndex();
		}
		mWorldId2InverseId[world_id] = inverse_id;
		mInverseId2WorldId[inverse_id] = world_id;
	}

	omega_buffer = new btVector3[mNumLinks];
	vel_buffer = new btVector3[mNumLinks];

    mType = type;
}

cIDSolver::~cIDSolver()
{
	delete [] omega_buffer;
	delete [] vel_buffer;
}

eIDSolverType cIDSolver::GetType()
{
    return mType;
}

void cIDSolver::RecordMultibodyInfo(std::vector<tMatrix>& local_to_world_rot, std::vector<tVector>& link_pos_world) const
{
	// std::cout <<"begin info \n";
	assert(local_to_world_rot.size() == mNumLinks);
	assert(link_pos_world.size() == mNumLinks);

	
	for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
	{
		// set up rot & pos
		if (0 == ID_link_id)
		{
			local_to_world_rot[0] = cMathUtil::RotMat(cBulletUtil::btQuaternionTotQuaternion(mMultibody->getWorldToBaseRot().inverse()));
			link_pos_world[0] = cBulletUtil::btVectorTotVector1(mMultibody->getBasePos());
		}
		else
		{
			int multibody_link_id = ID_link_id - 1;
			auto & cur_trans = mMultibody->getLinkCollider(multibody_link_id)->getWorldTransform();
			local_to_world_rot[ID_link_id] = cBulletUtil::btMatrixTotMatrix1(cur_trans.getBasis());
			link_pos_world[ID_link_id] = cBulletUtil::btVectorTotVector1(cur_trans.getOrigin());
		}
	}
}


void cIDSolver::RecordMultibodyInfo(std::vector<tMatrix>& local_to_world_rot, std::vector<tVector>& link_pos_world,\
 std::vector<tVector> & link_omega_world, std::vector<tVector> & link_vel_world) const
{
	// std::cout <<"begin info \n";
	local_to_world_rot.resize(mNumLinks);
	link_pos_world.resize(mNumLinks);
	link_omega_world.resize(mNumLinks);
	link_vel_world.resize(mNumLinks);

	// std::cout <<" links num = " << mNumLinks << std::endl;
			
	mMultibody->compTreeLinkVelocities(omega_buffer, vel_buffer);
	for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
	{
		// set up rot & pos
		if (0 == ID_link_id)
		{
			local_to_world_rot[0] = cMathUtil::RotMat(cBulletUtil::btQuaternionTotQuaternion(mMultibody->getWorldToBaseRot().inverse()));
			link_pos_world[0] = cBulletUtil::btVectorTotVector1(mMultibody->getBasePos());
			link_vel_world[0] = cBulletUtil::btVectorTotVector0(mMultibody->getBaseVel());
			link_omega_world[0]= cBulletUtil::btVectorTotVector0(mMultibody->getBaseOmega());
		}
		else
		{
			int multibody_link_id = ID_link_id - 1;
			auto & cur_trans = mMultibody->getLinkCollider(multibody_link_id)->getWorldTransform();
			local_to_world_rot[ID_link_id] = cBulletUtil::btMatrixTotMatrix1(cur_trans.getBasis());
			link_pos_world[ID_link_id] = cBulletUtil::btVectorTotVector1(cur_trans.getOrigin());
			link_vel_world[ID_link_id] = cBulletUtil::btVectorTotVector0(quatRotate(cur_trans.getRotation(), vel_buffer[ID_link_id]));
			link_omega_world[ID_link_id] = cBulletUtil::btVectorTotVector0(quatRotate(cur_trans.getRotation(), omega_buffer[ID_link_id]));
		}
		// std::cout <<"link " << ID_link_id <<" pos  = " << link_pos_world[ID_link_id] .transpose() << std::endl;
	}
}

void cIDSolver::RecordGeneralizedInfo(tVectorXd & q, tVectorXd & q_dot) const
{
	q.resize(mDof), q_dot.resize(mDof);
	q.setZero(), q_dot.setZero();
	// root
	if (mFloatingBase == true)
	{
		// q = [rot, pos], rot = euler angle in XYZ rot
		tQuaternion world_to_base = cBulletUtil::btQuaternionTotQuaternion(mMultibody->getWorldToBaseRot());
		tVector euler_angle_rot = cMathUtil::QuaternionToEulerAngles(world_to_base, eRotationOrder::XYZ);
		//std::cout << "[debug] RecordGenInfo: world to base mat = \n" << world_to_base.toRotationMatrix() << std::endl;
		//std::cout << "[debug] RecordGenInfo: world to base euler angle = \n" << euler_angle_rot.transpose() << std::endl;

		for (int i = 0; i < 3; i++) q(i) = euler_angle_rot[i];
		tVector pos = cBulletUtil::btIDVectorTotVector0(mMultibody->getBasePos());
		for (int i = 3; i < 6; i++)q(i) = pos[i - 3];

		// q_dot = [w, v]
		tVector omega = cBulletUtil::btVectorTotVector0(mMultibody->getBaseOmega());
		tVector vel = cBulletUtil::btVectorTotVector0(mMultibody->getBaseVel());
		for (int i = 0; i < 3; i++)q_dot(i) = omega[i];
		for (int i = 3; i < 6; i++)q_dot(i) = vel[i - 3];
	}

	// other joints
	for (int link_id = 0, cnt = 0; link_id < mMultibody->getNumLinks(); link_id++)
	{
		auto & cur_link = mMultibody->getLink(link_id);
		int offset = cur_link.m_dofOffset;
		if (mFloatingBase == true) offset += 6;
		switch (cur_link.m_jointType)
		{
            case btMultibodyLink::eFeatherstoneJointType::eRevolute:
            {
                q(offset) = mMultibody->getJointPos(link_id);
                q_dot(offset) = mMultibody->getJointVel(link_id);
                break;
            }
            case btMultibodyLink::eFeatherstoneJointType::eSpherical:
            {
                // q: euler angle
                // q_dot: omega
                btScalar * bt_joint_pos = mMultibody->getJointPosMultiDof(link_id);
                btScalar * bt_joint_vel = mMultibody->getJointVelMultiDof(link_id);

                // local to parent frame��quaternion
                tQuaternion joint_pos_q(bt_joint_pos[3], bt_joint_pos[0], bt_joint_pos[1], bt_joint_pos[2]);
                tVector joint_pos_euler = cMathUtil::QuaternionToEulerAngles(joint_pos_q, eRotationOrder::ZYX);
                tVector joint_vel = tVector(bt_joint_vel[0], bt_joint_vel[1], bt_joint_vel[2], 0);

                for (int i = 0; i < 3; i++)
                {
                    q(offset + i) = joint_pos_euler[i];	// in parent frame

                    q_dot(offset + i) = joint_vel[i];	// in parent frame
                }
                break;
            }
            case btMultibodyLink::eFeatherstoneJointType::eFixed:
            {
                break;
            }
            default:
            {
                std::cout << "[error] cIDSolver::RecordGeneralizedInfo: unsupported joint type " << cur_link.m_jointType << std::endl;
                exit(1);
            }
        }
	}
}

void cIDSolver::SetGeneralizedPos(const tVectorXd & q)
{
	assert(q.size() == mDof);
	if (mFloatingBase == true)
	{
		tVector root_pos, euler_angle_rot;
		for(int i=0; i<3; i++) euler_angle_rot[i] = q[i];
		for(int i=3; i<6; i++) root_pos[i-3] = q[i];
		tQuaternion world_to_base_rot = cMathUtil::EulerAnglesToQuaternion(euler_angle_rot, eRotationOrder::XYZ);
		mMultibody->setBasePos(cBulletUtil::tVectorTobtVector(root_pos));
		mMultibody->setWorldToBaseRot(cBulletUtil::tQuaternionTobtQuaternion(world_to_base_rot));
	}

	// other joints besides root
	for (int link_id = 0, cnt = 0; link_id < mMultibody->getNumLinks(); link_id++)
	{
		auto & cur_link = mMultibody->getLink(link_id);
		int offset = cur_link.m_dofOffset;
		if (mFloatingBase == true) offset += 6;
		// std::cout <<"link id = " << link_id <<" type = " << cur_link.m_jointType << std::endl;
		switch (cur_link.m_jointType)
		{
			case btMultibodyLink::eFixed:
			{
				break;
			}
			case btMultibodyLink::eRevolute:
			{
				mMultibody->setJointPos(link_id, q[offset]);
				// q(offset) = mMultibody->getJointPos(link_id);
				break;
			}
			case btMultibodyLink::eSpherical:
			{
				// std::cout << 1<< std::endl;
				tVector joint_pos_euler;
				joint_pos_euler.segment(0, 3) = q.segment(offset, 3);
				// std::cout << 2<< std::endl;
				tQuaternion rot_qua = cMathUtil::EulerAnglesToQuaternion(joint_pos_euler, eRotationOrder::ZYX);
				// std::cout << 3<< std::endl;
				btScalar bt_joint_pos[4] = {rot_qua.x(), rot_qua.y(), rot_qua.z(), rot_qua.w()};
				// std::cout << 4<< std::endl;
				mMultibody->setJointPosMultiDof(link_id, bt_joint_pos);
				break;
			}
			default:
			{
				std::cout <<"[error] cIDSolver::SetGeneralizedInfo unsupported type = " << cur_link.m_jointType << std::endl;
				exit(1);
			}
		}
	}
	btAlignedObjectArray<btVector3> mVecBuffer0;
	btAlignedObjectArray<btQuaternion> mRotBuffer;
	mMultibody->updateCollisionObjectWorldTransforms(mRotBuffer, mVecBuffer0);
	// tVectorXd q_test, q_dot_test;
	// cIDSolver::RecordGeneralizedInfo(q_test, q_dot_test);
	// std::cout << "given q = " << q.transpose() << std::endl;
	// std::cout << "current after set q = " << q_test.transpose() << std::endl;
	// std::cout << "error = " << (q - q_test).norm() << std::endl; 
	// std::cout <<"over cIDSolver::SetGeneralizedInfo done\n";
	// exit(1);
}

void cIDSolver::SetGeneralizedVel(const tVectorXd & q)
{
	assert(q.size() == mDof);
	if (mFloatingBase == true)
	{
		// tVector omega = cBulletUtil::btVectorTotVector0(mMultibody->getBaseOmega());
		// tVector vel = cBulletUtil::btVectorTotVector0(mMultibody->getBaseVel());
		// for (int i = 0; i < 3; i++)q_dot(i) = omega[i];
		// for (int i = 3; i < 6; i++)q_dot(i) = vel[i - 3];

		tVector omega = tVector::Zero(), vel = tVector::Zero();
		for(int i=0; i<3; i++) omega[i]= q(i), vel[i] = q(i + 3);
		mMultibody->setBaseOmega(cBulletUtil::tVectorTobtVector(omega));
		mMultibody->setBaseVel(cBulletUtil::tVectorTobtVector(vel));
		// tVector root_pos, euler_angle_rot;
		// for(int i=0; i<3; i++) euler_angle_rot[i] = q[i];
		// for(int i=3; i<6; i++) root_pos[i-3] = q[i];
		// tQuaternion world_to_base_rot = cMathUtil::EulerAnglesToQuaternion(euler_angle_rot, eRotationOrder::XYZ);
		// mMultibody->setBasePos(cBulletUtil::tVectorTobtVector(root_pos));
		// mMultibody->setWorldToBaseRot(cBulletUtil::tQuaternionTobtQuaternion(world_to_base_rot));
	}

	// other joints besides root
	for (int link_id = 0, cnt = 0; link_id < mMultibody->getNumLinks(); link_id++)
	{
		auto & cur_link = mMultibody->getLink(link_id);
		int offset = cur_link.m_dofOffset;
		if (mFloatingBase == true) offset += 6;
		// std::cout <<"link id = " << link_id <<" type = " << cur_link.m_jointType << std::endl;
		switch (cur_link.m_jointType)
		{
			case btMultibodyLink::eFixed:
			{
				break;
			}
			case btMultibodyLink::eRevolute:
			{
				mMultibody->setJointVel(link_id, q[offset]);
				// mMultibody->setJointPos(link_id, q[offset]);
				// q(offset) = mMultibody->getJointPos(link_id);
				break;
			}
			case btMultibodyLink::eSpherical:
			{
				// // std::cout << 1<< std::endl;
				// tVector joint_pos_euler;
				// joint_pos_euler.segment(0, 3) = q.segment(offset, 3);
				// // std::cout << 2<< std::endl;
				// tQuaternion rot_qua = cMathUtil::EulerAnglesToQuaternion(joint_pos_euler, eRotationOrder::ZYX);
				// // std::cout << 3<< std::endl;
				// btScalar bt_joint_pos[4] = {rot_qua.x(), rot_qua.y(), rot_qua.z(), rot_qua.w()};
				// // std::cout << 4<< std::endl;
				// mMultibody->setJointPosMultiDof(link_id, bt_joint_pos);
				btScalar bt_joint_vel[] = {q[offset], q[offset+1], q[offset+2], 0};
				mMultibody->setJointVelMultiDof(link_id, bt_joint_vel);
				break;
			}
			default:
			{
				std::cout <<"[error] cIDSolver::SetGeneralizedInfo unsupported type = " << cur_link.m_jointType << std::endl;
				exit(1);
			}
		}
	}
	// tVectorXd q_test, q_dot_test;
	// cIDSolver::RecordGeneralizedInfo(q_test, q_dot_test);
	// std::cout << "given q_dot = " << q.transpose() << std::endl;
	// std::cout << "current after set q = " << q_dot_test.transpose() << std::endl;
	// std::cout << "error = " << (q - q_dot_test).norm() << std::endl; 
	// std::cout <<"over cIDSolver::SetGeneralizedVel done\n";
	// exit(1);
}

void cIDSolver::RecordJointForces(std::vector<tVector> & mJointForces) const
{
	mJointForces.resize(mMultibody->getNumLinks());
    for (int i = 0; i < mMultibody->getNumLinks(); i++)
	{
		const btMultibodyLink& cur_link = mMultibody->getLink(i);
		switch (cur_link.m_jointType)
		{
			case btMultibodyLink::eSpherical:
			{
				// in joint frame
				btScalar * local_torque_bt = mMultibody->getJointTorqueMultiDof(i);
				tVector local_torque = tVector(local_torque_bt[0], local_torque_bt[1], local_torque_bt[2], 0);
				mJointForces[i] = local_torque;
				break;
			}
			case btMultibodyLink::eRevolute:
			{
				btVector3 local_torque_bt = mMultibody->getJointTorque(i) * mMultibody->getLink(i).getAxisTop(0);// rotation axis in link frame, sometimes not identity
				tVector local_torque = cBulletUtil::btVectorTotVector0(local_torque_bt);
				mJointForces[i] = local_torque;
				//std::cout << "-[debug] cIDSolver::RecordJointForces get joint "<< i << " raw force = " << mMultibody->getJointTorque(i) << std::endl;
				//std::cout << "-[debug] cIDSolver::RecordJointForces get joint "<< i << " mul force = " << local_torque.transpose() << std::endl;
				//std::cout << "-[debug] cIDSolver::RecordJointForces get joint "<< i << " axis = " << cBulletUtil::btVectorTotVector0(mMultibody->getLink(i).getAxisTop(0)).transpose() << std::endl;
				break;
			}
			case btMultibodyLink::eFixed:
			{
				mJointForces[i].setZero();
				break;
			}
			default:
			{
				std::cout << "[error] cIDSolver::RecordJointForces: Unsupported joint type " << cur_link.m_jointType << std::endl;
				break;
			}
		}
	}
}

/**
 * \brief					Record current action of the character
 * \param					the ref of action which will be revised then
*/
void cIDSolver::RecordAction(tVectorXd & action) const
{
	assert(mCharController != nullptr);
	action = mCharController->GetCurAction();
	// std::cout << "void cIDSolver::RecordAction get pd target: " << pd_target.transpose() << std::endl;
	// exit(1);
}

/**
 * \brief					Record current pd target pose of the character
 * \param					the ref of pd target in this time
*/
void cIDSolver::RecordPDTarget(tVectorXd & pd_target) const	
{
	// ball joints are in quaternions in pd target 
	assert(mCharController != nullptr);
	pd_target = mCharController->GetCurPDTargetPose();
}

void cIDSolver::RecordContactForces(std::vector<tForceInfo> &mContactForces, double mCurTimestep, std::map<int, int> &mWorldId2InverseId) const
{
    mContactForces.clear();
	int num_contact_manifolds = mWorld->getDispatcher()->getNumManifolds();
	for (int i = 0; i < num_contact_manifolds; i++)
	{
		const btPersistentManifold * manifold = mWorld->getDispatcher()->getInternalManifoldPointer()[i];

		const int body0_id = manifold->getBody0()->getWorldArrayIndex(),
			body1_id = manifold->getBody1()->getWorldArrayIndex();

		int num_contact_pts = manifold->getNumContacts();
		tForceInfo cur_contact_info;
		for (int j = 0; j < num_contact_pts; j++)
		{
			tForceInfo contact_info;
			const btManifoldPoint & pt = manifold->getContactPoint(j);
			btScalar linear_friction_force1 = pt.m_appliedImpulseLateral1 / mCurTimestep;
			btScalar linear_friction_force2 = pt.m_appliedImpulseLateral2 / mCurTimestep;
			tVector lateral_friction_dir1 = cBulletUtil::btVectorTotVector0(pt.m_lateralFrictionDir1),
				lateral_friction_dir2 = cBulletUtil::btVectorTotVector0(pt.m_lateralFrictionDir2);
			double impulse = pt.m_appliedImpulse;
			tVector normal = cBulletUtil::btVectorTotVector0(pt.m_normalWorldOnB);
			tVector force0 = impulse / mCurTimestep * normal;
			tVector friction = linear_friction_force1 * lateral_friction_dir1 + linear_friction_force2 * lateral_friction_dir2;
			tVector pos, force;

			// verify collision
			bool collision_valid = false;
			if (mWorldId2InverseId.end() != mWorldId2InverseId.find(body0_id))
			{
				contact_info.mId = mWorldId2InverseId[body0_id];
				pos = cBulletUtil::btVectorTotVector1(pt.getPositionWorldOnA());
				force = (force0 + friction);

				// add contact forces for body 0
				contact_info.mPos = pos;
				contact_info.mForce = force;
				mContactForces.push_back(contact_info);

				collision_valid = true;
			}
			if (mWorldId2InverseId.end() != mWorldId2InverseId.find(body1_id))
			{
				contact_info.mId = mWorldId2InverseId[body1_id];
				pos = cBulletUtil::btVectorTotVector1(pt.getPositionWorldOnB());
				force = -(force0 + friction);

				// add contact forces for body 1
				contact_info.mPos = pos;
				contact_info.mForce = force;
				mContactForces.push_back(contact_info);
				collision_valid = true;
			}

			if (false == collision_valid)
			{
				// ignore other collisions
				std::cout << "[warn] cIDSolver::GetContactForces: ignore collision forces!\n";
				continue;
			}
		}
	}
}

void cIDSolver::ApplyContactForcesToID(const std::vector<tForceInfo> &mContactForces, const std::vector<tVector> & mLinkPos, const std::vector<tMatrix> & mLinkRot) const
{
	assert(mLinkRot.size() == mNumLinks);
	assert(mLinkPos.size() == mNumLinks);
	tVector base_force = tVector::Zero(), base_torque = tVector::Zero();
	for (auto & cur_force : mContactForces)
	{
		int ID_link_id = cur_force.mId;
		int multibody_id = ID_link_id - 1;
		tVector link_pos_world = mLinkPos[ID_link_id];
		tMatrix local_to_world = mLinkRot[ID_link_id];
		tVector joint_pos_world;
		if (0 == ID_link_id)
		{
			joint_pos_world = link_pos_world;
		}
		else
		{
			joint_pos_world = local_to_world * cBulletUtil::btVectorTotVector0(-mMultibody->getLink(multibody_id).m_dVector) + link_pos_world;
		}
		tVector force_arm_world = cur_force.mPos - joint_pos_world;
		// 1. get contact force in world frame -> local frame
		tVector contact_force_world = cur_force.mForce;
		tVector contact_force_local = local_to_world.transpose() * contact_force_world;
		// 2. get calculate contact torque in world frame -> local frame
		tVector contact_torque_world = force_arm_world.cross3(contact_force_world);
		tVector contact_torque_local = local_to_world.transpose() * contact_torque_world;

		//std::cout << "[debug] ApplyContactForcesToID: for link " << ID_link_id << ", contact force = " << contact_force_world.transpose() << ", force arm = " << force_arm_world.transpose() << std::endl;
		//std::cout << "\tfinal add force = " << contact_force_local.transpose() << ", final add torque = " << contact_torque_local.transpose() << std::endl;

		mInverseModel->addUserForce(ID_link_id, cBulletUtil::tVectorTobtVector(contact_force_local));
		mInverseModel->addUserMoment(ID_link_id, cBulletUtil::tVectorTobtVector(contact_torque_local));
		base_force += contact_force_world;
	}
	//std::cout << "[debug] cIDSolver::ApplyContactForces: base constrained force world = " << base_force.transpose() << std::endl;
	//std::cout << "[debug] simulation base constrained force world = " << cBulletUtil::btVectorTotVector0(static_cast<cCollisionWorld *> (mWorld)->base_cons_force).transpose() << std::endl;
}


void cIDSolver::SolveIDSingleStep(std::vector<tVector> & solved_joint_forces,
	const std::vector<tForceInfo> & contact_forces,
	const std::vector<tVector> & link_pos, 
	const std::vector<tMatrix> &link_rot, 
	const tVectorXd & mBuffer_q,
	const tVectorXd & mBuffer_u,
	const tVectorXd & mBuffer_u_dot,
	int frame_id,
	const std::vector<tVector> &external_forces,
	const std::vector<tVector> &external_torques) const
{

// #define DEBUG_STEP
#ifdef DEBUG_STEP
	std::ofstream fout("test3.txt", std::ios::app);
	fout <<"----------------------------\n";
	fout <<"solve id for frame " << frame_id << std::endl;
#endif
	// it should have only mNumLinks-1 elements.
	// std::cout << 1 << std::endl;
	solved_joint_forces.resize(mNumLinks-1);
	// std::cout << 2 << std::endl;
	for(auto & x : solved_joint_forces) x.setZero();
#ifdef DEBUG_STEP
	// std::cout << 3 << std::endl;
	fout <<"link pos : ";

	for(auto & x : link_pos) fout << x.transpose() <<" ";
	fout <<"\nlink rot : " ;
	for(auto & x : link_rot) fout << x.transpose() << std::endl;
	fout <<"\n external forces and torques : ";
	for(int idx = 0; idx < mNumLinks; idx++) fout << external_forces[idx].transpose() <<" " << external_torques[idx].transpose() <<" ";
	fout << std::endl;
#endif
	ApplyContactForcesToID(contact_forces, link_pos, link_rot);
	// std::cout << 4 << std::endl;
	ApplyExternalForcesToID(link_pos, link_rot, external_forces, external_torques);
	// std::cout << 5 << std::endl;

	// solve ID: these joint forces are in joint frame
	// but the reference
	btInverseDynamicsBullet3::vecx solve_joint_force_bt(mDof);

#ifdef DEBUG_STEP
	fout <<"q " << frame_id - 1 <<" " << mBuffer_q.transpose() << std::endl;
	fout <<"u " << frame_id - 1 <<" " << mBuffer_u.transpose() << std::endl;
	fout <<"u dot " << frame_id - 1 <<" " << mBuffer_u_dot.transpose() << std::endl;
#endif
	// std::cout << 6 << std::endl;
	mInverseModel->calculateInverseDynamics(
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_q),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u_dot),
		&solve_joint_force_bt);
	// std::cout << 7 << std::endl;
	// convert the solve "solve_joint_force_bt" into individual joint forces for each joint.
	Eigen::VectorXd solved_joint_force_full_concated = cBulletUtil::btIDArrayToEigenArray(solve_joint_force_bt);
#ifdef DEBUG_STEP
	fout <<"result = " << solved_joint_force_full_concated.transpose() << std::endl;
#endif

	std::vector<double> true_joint_force(0);
	for (int i = 0; i < mMultibody->getNumLinks(); i++)
	{
		auto & cur_link = mMultibody->getLink(i);
		int offset = cur_link.m_dofOffset;
		if (mFloatingBase == true) offset += 6;
		const int cnt = cur_link.m_dofCount;
		switch (cur_link.m_jointType)
		{
			case btMultibodyLink::eFeatherstoneJointType::eRevolute:
			{
				assert(cnt == 1);
				/*
					solved_joint_force : in joint frame
					mJointForces[i]: in link frame.

					transform solved joint force to link frame
				*/
				double value = solved_joint_force_full_concated[offset];
				tVector axis_in_body_frame = cBulletUtil::btVectorTotVector0(mMultibody->getLink(i).getAxisTop(0));
				tQuaternion body_to_this = cMathUtil::EulerToQuaternion(cKinTree::GetBodyAttachTheta(mSimChar->GetBodyDefs(), i), eRotationOrder::XYZ);//
				tQuaternion this_to_body = body_to_this.inverse();
				tVector axis_in_joint_frame = cMathUtil::QuatRotVec(body_to_this, axis_in_body_frame);
				tVector torque_in_joint_frame = value * axis_in_joint_frame;
				tVector torque_in_link_frame = cMathUtil::QuatRotVec(this_to_body, torque_in_joint_frame);
				solved_joint_forces[i] = torque_in_link_frame;
				break;
			}
			case btMultibodyLink::eFeatherstoneJointType::eSpherical:
			{
				assert(cnt == 3);
				solved_joint_forces[i] = tVector(solved_joint_force_full_concated[offset], solved_joint_force_full_concated[offset + 1], solved_joint_force_full_concated[offset + 2], 0);
				break;
			}
			case btMultibodyLink::eFeatherstoneJointType::eFixed:
			{
				break;
			}
			default:
			{
				std::cout << "[error] cIDSolver::SolveID: unsupported joint type " << cur_link.m_jointType;
				exit(1);
				break;
			}
		}
	}
}


void cIDSolver::ApplyExternalForcesToID(const std::vector<tVector> & link_poses,
	const std::vector<tMatrix> & link_rot,
	const std::vector<tVector> & ext_forces,
	const std::vector<tVector> & ext_torques) const
{
	// attention: mInverseModel MUST be set up well before it is called.
	assert(link_poses.size() == mNumLinks);
	assert(link_rot.size() == mNumLinks);
	assert(ext_forces.size() == mNumLinks);
	assert(ext_torques.size() == mNumLinks);
	//add external user force
	for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
	{
		btInverseDynamicsBullet3::vec3 bt_com;
		btInverseDynamicsBullet3::mat33 bt_rot;
		// get current COM of this body "ID_link_id" relative to the COM of whole system.
		mInverseModel->getBodyCoM(ID_link_id, &bt_com);		// world com, ����ʵ������Joint pos
		mInverseModel->getBodyTransform(ID_link_id, &bt_rot); // body to world
		tVector link_pos_world = link_poses[ID_link_id];
		tMatrix local_to_world_rot = link_rot[ID_link_id];

		// calculate global applied position.
		tVector force_arm_world = tVector::Zero();
		if (ID_link_id == 0)
		{
			// for root link, the COM and joint pos are coincidenced
			force_arm_world = tVector::Zero();
		}
		else
		{
			int multibody_id = ID_link_id - 1;
			tVector joint_pos_world;
			auto & link = mMultibody->getLink(multibody_id);
			joint_pos_world = link_pos_world + local_to_world_rot * cBulletUtil::btVectorTotVector0(-link.m_dVector);
			force_arm_world = link_pos_world - joint_pos_world;
		}

		// convert forces & torques to local frame(joint frame aka fixed frame)
		tVector force_world = ext_forces[ID_link_id];
		tVector force_local = local_to_world_rot.transpose() * force_world;

		tVector torque_world = ext_torques[ID_link_id];
		tVector torque_local = local_to_world_rot.transpose() * torque_world;

		tVector force_torque_world = force_arm_world.cross3(force_world);
		tVector force_torque_local = local_to_world_rot.transpose() * force_torque_world;

		// set up external force, external torque to Inverse Model.
		mInverseModel->addUserForce(ID_link_id, cBulletUtil::tVectorTobtVector(force_local));
		mInverseModel->addUserMoment(ID_link_id, cBulletUtil::tVectorTobtVector(force_torque_local));
		mInverseModel->addUserMoment(ID_link_id, cBulletUtil::tVectorTobtVector(torque_local));

		//if (ID_link_id == 0) continue;
		//std::cout << "for link " << ID_link_id << std::endl;
		//std::cout << "local to world rot ID = \n" << local_to_world_rot << std::endl;
		//std::cout << "local to world rot multi = \n" << cBulletUtil::btMatrixTotMatrix1(mMultibody->getLinkCollider(ID_link_id - 1)->getWorldTransform().getBasis()) << std::endl;;
		//std::cout << "user force world = " << force_world.transpose() << std::endl;
		//std::cout << "user force local = " << force_local.transpose() << std::endl;
		//std::cout << "user force torque world = " << force_torque_world.transpose() << std::endl;
		//std::cout << "user force torque local = " << force_torque_local.transpose() << std::endl;
		//std::cout << "user torque world = " << torque_world.transpose() << std::endl;
		//std::cout << "user torque local = " << torque_local.transpose() << std::endl;
	}
}

tVectorXd cIDSolver::CalcGeneralizedVel(const tVectorXd & q_before, const tVectorXd & q_after, double timestep) const
{
	tVectorXd q_dot = tVectorXd::Zero(mDof);

	assert(q_before.size() == mDof);
	assert(q_after.size() == mDof);

	int dof_offset = 0;
	for (int i = 0; i < mNumLinks; i++)
	{
		if (0 == i)
		{
			// for floating root joint
			if (true == mFloatingBase)
			{
				tVector pos_before = tVector(q_before[3], q_before[4], q_before[5], 1),
					pos_after = tVector(q_after[3], q_after[4], q_after[5], 1);
				tVector vel_after = (pos_after - pos_before) / timestep;

				// calculate angular velocity, then put it into q_dot
				tVector euler_angle_before = tVector(q_before[0], q_before[1], q_before[2], 0),
					euler_angle_after = tVector(q_after[0], q_after[1], q_after[2], 0);
				tQuaternion quater_before = cMathUtil::EulerAnglesToQuaternion(euler_angle_before, eRotationOrder::XYZ).inverse(),
					quater_after = cMathUtil::EulerAnglesToQuaternion(euler_angle_after, eRotationOrder::XYZ).inverse();

				tVector omega_after = cMathUtil::CalcAngularVelocity(quater_before, quater_after, timestep);
				// std::cout <<"omega after = " << omega_after.transpose() << std::endl;
				// std::cout <<"quater before = " << quater_before.coeffs().transpose() << std::endl;
				// std::cout <<"quater after = " << quater_after.coeffs().transpose() << std::endl;
				// exit(1);
				q_dot.segment(0, 3) = omega_after.segment(0, 3);
				q_dot.segment(3, 3) = vel_after.segment(0, 3);
				dof_offset += 6;
				continue;
			}
		}
		else // other joints
		{
			int multibody_id = i - 1;
			auto & link = mMultibody->getLink(multibody_id);
			switch (link.m_jointType)
			{
			case btMultibodyLink::eFeatherstoneJointType::eSpherical:
			{
				// calculate angular velocity, then put it into q_dot
				tVector euler_angle_before = tVector::Zero(), euler_angle_after = tVector::Zero();
				euler_angle_before.segment(0, 3) = q_before.segment(dof_offset, 3);
				euler_angle_after.segment(0, 3) = q_after.segment(dof_offset, 3);

				tQuaternion quater_before = cMathUtil::EulerAnglesToQuaternion(euler_angle_before, eRotationOrder::ZYX),
					quater_after = cMathUtil::EulerAnglesToQuaternion(euler_angle_after, eRotationOrder::ZYX);

				// convert the ang vel from parent frame to local frame, then write in
				tVector ang_vel_parent = cMathUtil::CalcAngularVelocity(quater_before, quater_after, timestep);
				tVector ang_vel_local = cMathUtil::RotMat(quater_before).transpose() * ang_vel_parent;
				q_dot.segment(dof_offset, 3) = ang_vel_local.segment(0, 3);
				dof_offset += 3;
				break;
			}
			case btMultibodyLink::eFeatherstoneJointType::eRevolute:
			{
				q_dot(dof_offset) = (q_after(dof_offset) - q_before(dof_offset)) / timestep;
				dof_offset += 1;

				break;
			}
			default:
				break;
			}
		}
	}
	return q_dot;
}

/*
	@Function: CalcMomentum
		Given the current state of multibody including pos, ori, vel, omega,
		this function can calculate linear momentum and angular momentum of this character
*/
void cIDSolver::CalcMomentum(const std::vector<tVector> & mLinkPos, 
    const std::vector<tMatrix> & mLinkRot,
    const std::vector<tVector> & mLinkVel, const std::vector<tVector> & mLinkOmega,
    tVector & mLinearMomentum, tVector & mAngMomentum)const
{
    assert(mLinkPos.size() == mNumLinks);
    assert(mLinkRot.size() == mNumLinks);
    assert(mLinkVel.size() == mNumLinks);
    assert(mLinkVel.size() == mNumLinks);
    mLinearMomentum.setZero();
    mAngMomentum.setZero();
    
    // 1. init all values
    std::vector<double> mass(mNumLinks);
    double total_mass = mSimChar->CalcTotalMass();
    std::vector<tVector> inertia(mNumLinks);
    tVector COM = tVector::Zero();
    for(int i=0; i<mNumLinks; i++)
    {
        if(0 == i)
        {
            mass[i] = mMultibody->getBaseMass();
            inertia[i] = cBulletUtil::btVectorTotVector0(mMultibody->getBaseInertia());
        }
        else
        {
            mass[i] = mMultibody->getLinkMass(i-1);
            inertia[i] = cBulletUtil::btVectorTotVector0(mMultibody->getLinkInertia(i-1));
        }
        COM += mLinkPos[i] * mass[i] / total_mass;
    }

    // 2. calculate linear momentum and angular momentum
    for(int i=0; i<mNumLinks; i++)
    {
        // calculate linear momentum = mv
        mLinearMomentum += mass[i] * mLinkVel[i];

        // calculate angular momentum = m * (x - COM) + R * Ibody * R.T * w
        mAngMomentum += mass[i] * (mLinkPos[i] - COM) + 
            mLinkRot[i] * inertia[i].asDiagonal() * mLinkRot[i].transpose() * mLinkOmega[i];
    }
    // std::cout << "linear mome = " << mLinearMomentum.transpose() << std::endl;

}

/*
    @Function: CalcDiscreteVelAndOmega
		Given mLinkPosCur, mLinkRotCur, mLinkPosNext and mLinkRotNext, this function will try to calculate
		the linear velocity and angular vel of each link COM in mLinkDiscreteVel and mLinkDiscreteOmega respectly
*/
void cIDSolver::CalcDiscreteVelAndOmega(
    const std::vector<tVector> & mLinkPosCur, 
    const std::vector<tMatrix> & mLinkRotCur,
    const std::vector<tVector> & mLinkPosNext, 
    const std::vector<tMatrix> & mLinkRotNext,
	double timestep,
    std::vector<tVector> & mLinkDiscreteVel,
    std::vector<tVector> & mLinkDiscreteOmega 
)const
{
	mLinkDiscreteVel.resize(mNumLinks);
	mLinkDiscreteOmega.resize(mNumLinks);
	assert(timestep > 1e-6);
	for(int i=0; i< mNumLinks; i++)
	{
		// discrete vel = (p_next - p_cur) / timestep
		// discrete omega = (rot_next - rot_cur) / timestep, note that this '-' is quaternion minus
		mLinkDiscreteVel[i] = (mLinkPosNext[i] - mLinkPosCur[i]) / timestep;
		mLinkDiscreteOmega[i] = cMathUtil::CalcQuaternionVel(
			cMathUtil::RotMatToQuaternion(mLinkRotCur[i]),
			cMathUtil::RotMatToQuaternion(mLinkRotNext[i]),
			timestep
		);
	}

}