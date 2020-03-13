#include "cIDSolver.hpp"
#include "SimCharacter.h"
#include "../Extras/InverseDynamics/btMultiBodyTreeCreator.hpp"
#include <util/BulletUtil.h>
#include <iostream>

cIDSolver::cIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, eIDSolverType type)
{
	mSimChar = sim_char;
	mMultibody = sim_char->GetMultiBody().get();
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
	if (-1 == id_creator.createFromBtMultiBody(mMultibody, true))
	{
		b3Error("error creating tree\n");
	}
	else
	{
		mInverseModel = btInverseDynamicsBullet3::CreateMultiBodyTree(id_creator);
		btInverseDynamicsBullet3::vec3 gravity(cBulletUtil::tVectorTobtVector(gGravity));

		mInverseModel->setGravityInWorldFrame(gravity);
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

    mType = type;
}

eIDSolverType cIDSolver::GetType()
{
    return mType;
}

void cIDSolver::RecordMultibodyInfo(std::vector<tMatrix>& local_to_world_rot, std::vector<tVector>& link_pos_world) const
{
	local_to_world_rot.resize(mNumLinks);
	link_pos_world.resize(mNumLinks);

	for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
	{
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
	const std::vector<tVector> link_pos, 
	const std::vector<tMatrix> link_rot, 
	const tVectorXd * mBuffer_q,
	const tVectorXd * mBuffer_u,
	const tVectorXd * mBuffer_u_dot,
	int frame_id,
	const std::vector<tVector> &external_forces,
	const std::vector<tVector> &external_torques) const
{

	// std::ofstream fout("test3.txt", std::ios::app);
	// fout <<"----------------------------\n";
	// fout <<"solve id for frame " << frame_id << std::endl;
	// it should have only mNumLinks-1 elements.
	solved_joint_forces.resize(mNumLinks-1);
	for(auto & x : solved_joint_forces) x.setZero();
	// fout <<"link pos : ";
	// for(auto & x : link_pos) fout << x.transpose() <<" ";
	// fout <<"\nlink rot : " ;
	// for(auto & x : link_rot) fout << x.transpose() <<" ";
	// fout <<"\n external forces and torques : ";
	// for(int idx = 0; idx < mNumLinks; idx++) fout << external_forces[idx].transpose() <<" " << external_torques[idx].transpose() <<" ";
	ApplyContactForcesToID(contact_forces, link_pos, link_rot);
	ApplyExternalForcesToID(link_pos, link_rot, external_forces, external_torques);

	// solve ID: these joint forces are in joint frame
	// but the reference
	btInverseDynamicsBullet3::vecx solve_joint_force_bt(mDof);

	// fout <<"q " << frame_id - 1 <<" " << mBuffer_q[frame_id - 1].transpose() << std::endl;
	// fout <<"u " << frame_id - 1 <<" " << mBuffer_u[frame_id - 1].transpose() << std::endl;
	// fout <<"u dor " << frame_id - 1 <<" " << mBuffer_u_dot[frame_id - 1].transpose() << std::endl;
	mInverseModel->calculateInverseDynamics(
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_q[frame_id - 1]),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u[frame_id - 1]),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u_dot[frame_id - 1]),
		&solve_joint_force_bt);
	
	// convert the solve "solve_joint_force_bt" into individual joint forces for each joint.
	Eigen::VectorXd solved_joint_force_full_concated = cBulletUtil::btIDArrayToEigenArray(solve_joint_force_bt);
	// fout <<"result = " << solved_joint_force_full_concated.transpose() << std::endl;
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
// //#define VERBOSE 
// 	{
// 		double threshold = 1e-8;
// 		double sum_error = 0;
// #ifdef VERBOSE
// 		std::cout << "solve joint torque diff = ";
// #endif
// 		Eigen::VectorXd solved_joint_force_full_concated = cBulletUtil::btIDArrayToEigenArray(solve_joint_force_bt);
// 		std::vector<double> true_joint_force(0);
// 		double err = 0;
// 		// output torque
// 		if (mFloatingBase == true)
// 		{
// 			double val = 0;
// 			for (int i = 0; i < 6; i++)
// 			{
// 				val = solved_joint_force_full_concated[i];
// 				if (std::fabs(val) < threshold)  val = 0;
// #ifdef VERBOSE
// 				std::cout << val << " ";
// #endif
// 				err += val * val;
// 			}
// #ifdef VERBOSE
// 			std::cout << "| ";
// #endif
// 		}

// 		for (int i = 0; i < mMultibody->getNumLinks(); i++)
// 		{
// 			auto & cur_link = mMultibody->getLink(i);
// 			int offset = cur_link.m_dofOffset;
// 			if (mFloatingBase == true) offset += 6;
// 			const int cnt = cur_link.m_dofCount;
// 			switch (cur_link.m_jointType)
// 			{
// 			case btMultibodyLink::eFeatherstoneJointType::eRevolute:
// 			{
// 				assert(cnt == 1);
// 				/*
// 					solved_joint_force : in joint frame
// 					mJointForces[i]: in link frame.

// 					transform solved joint force to link frame

// 				*/
// 				double value = solved_joint_force_full_concated[offset];
// 				tVector axis_in_body_frame = cBulletUtil::btVectorTotVector0(mMultibody->getLink(i).getAxisTop(0));
// 				tQuaternion body_to_this = cMathUtil::EulerToQuaternion(cKinTree::GetBodyAttachTheta(mSimChar->GetBodyDefs(), i), eRotationOrder::XYZ);//
// 				tQuaternion this_to_body = body_to_this.inverse();
// 				tVector axis_in_joint_frame = cMathUtil::QuatRotVec(body_to_this, axis_in_body_frame);
// 				tVector torque_in_joint_frame = value * axis_in_joint_frame;
// 				tVector torque_in_link_frame = cMathUtil::QuatRotVec(this_to_body, torque_in_joint_frame);
// 				solved_joint_forces[i] = torque_in_link_frame;

// 				for (int id = 0; id < 3; id++)
// 				{
// 					double diff_val = torque_in_link_frame[id] - exp_joint_forces[i][id];
// 					if (std::fabs(diff_val) < threshold) diff_val = 0;
// 					true_joint_force.push_back(torque_in_link_frame[id]);
// #ifdef VERBOSE
// 					std::cout << diff_val << " ";
// #endif
// 					err += diff_val * diff_val;
// 				}
// #ifdef VERBOSE
// 				std::cout << " | ";
// #endif
// 				break;
// 			}
// 			case btMultibodyLink::eFeatherstoneJointType::eSpherical:
// 			{
// 				assert(cnt == 3);

// 				tVector torque = tVector(solved_joint_force_full_concated[offset], solved_joint_force_full_concated[offset + 1], solved_joint_force_full_concated[offset + 2], 0);

// 				solved_joint_forces[i] = torque;

// 				double val = 0;
// 				for (int dof_id = 0; dof_id < cnt; dof_id++)
// 				{
// 					val = torque[dof_id] - exp_joint_forces[i][dof_id];
// 					if (std::fabs(val) < threshold) val = 0;
// #ifdef VERBOSE
// 					std::cout << val << " ";
// #endif
// 					true_joint_force.push_back(exp_joint_forces[i][dof_id]);
// 					err += val * val;
// 				}
// #ifdef VERBOSE
// 				std::cout << " | ";
// #endif
// 				break;
// 			}
// 			case btMultibodyLink::eFeatherstoneJointType::eFixed:
// 			{
// 				break;
// 			}
// 			default:
// 			{
// 				std::cout << "[error] cIDSolver::SolveID: unsupported joint type " << cur_link.m_jointType;
// 				exit(1);
// 				break;
// 			}
// 			}
// 		}
// #ifdef VERBOSE
// 		std::cout << std::endl;
// #endif
// 		if (err > threshold)
// 		{
// 			std::cout << "true joint force = ";
// 			for (int i = 0; i < true_joint_force.size(); i++)
// 			{
// 				double val = true_joint_force[i];
// 				if (std::abs(val) < threshold) val = 0;
// 				std::cout << val << " ";
// 			}
// 			std::cout << std::endl;

// 			std::cout << "solved joint force = ";
// 			for (int i = 0; i < solved_joint_force_full_concated.size(); i++)
// 			{
// 				double val = solved_joint_force_full_concated[i];
// 				if (std::abs(val) < threshold) val = 0;
// 				std::cout << val << " ";
// 			}

// 			std::cout << std::endl;

// 			std::cout << "[error] ID solved wrong\n";
// 			//exit(1);
// 		}
// 	}
}


void cIDSolver::ApplyExternalForcesToID(const std::vector<tVector> & link_poses,
	const std::vector<tMatrix> & link_rot,
	const std::vector<tVector> & ext_forces,
	const std::vector<tVector> & ext_torques) const
{
	// attension: mInverseModel MUST be set up well before it is called.

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

tVectorXd cIDSolver::CalculateGeneralizedVel(const tVectorXd & q_before, const tVectorXd & q_after, double timestep) const
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