#include <iostream>
#include <fstream>
#include "BulletDynamics/Featherstone/btMultiBody.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h"
#include "BulletDynamics/Featherstone/btMultiBodyMLCPConstraintSolver.h"
#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyLink.h"
#include "BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h"
#include "BulletDynamics/Featherstone/btMultiBodyJointMotor.h"
#include "BulletDynamics/Featherstone/btMultiBodyPoint2Point.h"
#include "BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h"
#include "BulletDynamics/Featherstone/btMultiBodySliderConstraint.h"
#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"

#include "../Extras/InverseDynamics/btMultiBodyTreeCreator.hpp"

#include "SimInverseDynamics.h"
#include <BulletDynamics/Featherstone/cCollisionWorld.h>
#include <util/BulletUtil.h>
#include "SimCharacter.h"

cIDSolver::cIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world)
{
	mSimChar = sim_char;
	mMultibody = sim_char->GetMultiBody().get();
	mWorld = world;
	// init ID struct
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

	// init vars
	mEnableExternalForce = false;
	mEnableExternalTorque = false;
	mEnableSolveID = true;
	mFloatingBase = !(mMultibody->hasFixedBase());
	mDof = mMultibody->getNumDofs();
	if (mFloatingBase == true)
	{
		mDof += 6;
	}

	mNumLinks = mMultibody->getNumLinks() + 1;

	// clear buffer
	for (auto & x : mBuffer_q) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u_dot) x.resize(mDof), x.setZero();

	// set up map
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

	// init other vars
	mEnableExternalForce = true;
	mEnableExternalTorque = true;
	mEnableSolveID = true;
	mFloatingBase = !(mMultibody->hasFixedBase());
	mDof = mMultibody->getNumDofs();
	if (mFloatingBase == true)
	{
		mDof += 6;
	}

	mNumLinks = mMultibody->getNumLinks() + 1;

	mFrameId = 0;
	mSolvingMode = eSolvingMode::POS;

	ClearID();
}

void cIDSolver::SetTimestep(double deltaTime)
{
	mCurTimestep = deltaTime;
}

void cIDSolver::ClearID()
{

	if (mMultibody == nullptr || mInverseModel == nullptr)
	{
		std::cout << "[error] mcIDSolver::ClearID: illegal input" << std::endl;
		exit(1);
	}

	if (mFrameId + 1 >= MAX_FRAME_NUM)
	{
		std::cout << "[error] cIDSolver::ClearID buffer filled \n";
		exit(1);
	}


	mContactForces.clear();
	mJointForces.resize(mNumLinks - 1);
	for (auto & x : mJointForces) x.setZero();
	mSolvedJointForces.resize(mNumLinks - 1);
	for (auto & x : mSolvedJointForces) x.setZero();
	mExternalForces.resize(mNumLinks);
	for (auto & x : mExternalForces) x.setZero();
	mExternalTorques.resize(mNumLinks);
	for (auto & x : mExternalTorques) x.setZero();
	mLinkRot.resize(mNumLinks);
	for (auto & x : mLinkRot) x.setIdentity();
	mLinkPos.resize(mNumLinks);
	for (auto & x : mLinkPos) x.setZero();



	// clear external force
	mInverseModel->clearAllUserForcesAndMoments();
}

void cIDSolver::PreSim()
{
	//std::cout << "--------------------------------\n";
	ClearID();

	// record info
	AddExternalForces();

	// record info
	RecordJointForces();
	RecordGeneralizedInfo(mBuffer_q[mFrameId], mBuffer_u[mFrameId]);
	RecordMultibodyInfo(mLinkRot, mLinkPos);
}

void cIDSolver::PostSim()
{
	mFrameId++;

	RecordGeneralizedInfo(mBuffer_q[mFrameId], mBuffer_u[mFrameId]);

	// record contact forces
	RecordContactForces();


	if (mSolvingMode == eSolvingMode::VEL)
	{
		if (mFrameId >= 1)
			mBuffer_u_dot[mFrameId - 1] = (mBuffer_u[mFrameId] - mBuffer_u[mFrameId - 1]) / mCurTimestep;
	}
	else if (mSolvingMode == eSolvingMode::POS)
	{
		if (mFrameId >= 2)
		{
			tVectorXd old_vel_after = mBuffer_u[mFrameId];
			tVectorXd old_vel_before = mBuffer_u[mFrameId - 1];
			tVectorXd old_accel = (old_vel_after - old_vel_before) / mCurTimestep;
			mBuffer_u[mFrameId - 1] = CalculateGeneralizedVel(mBuffer_q[mFrameId - 2], mBuffer_q[mFrameId - 1], mCurTimestep);
			mBuffer_u[mFrameId] = CalculateGeneralizedVel(mBuffer_q[mFrameId - 1], mBuffer_q[mFrameId], mCurTimestep);
			mBuffer_u_dot[mFrameId - 1] = (mBuffer_u[mFrameId] - mBuffer_u[mFrameId - 1]) / mCurTimestep;

			double threshold = 1e-6;
			{
				tVectorXd diff = old_vel_after - mBuffer_u[mFrameId];
				if (diff.norm() > threshold)
				{
					std::cout << "truth vel = " << old_vel_after.transpose() << std::endl;
					std::cout << "calculated vel = " << mBuffer_u[mFrameId].transpose() << std::endl;
					std::cout << "calculate vel after error = " << (diff).transpose() << " | " << (old_vel_after - mBuffer_u[mFrameId]).norm() << std::endl;
					//exit(1);
				}

				// check vel
				diff = old_vel_before - mBuffer_u[mFrameId - 1];
				if (diff.norm() > threshold)
				{
					std::cout << "truth vel = " << old_vel_after.transpose() << std::endl;
					std::cout << "calculated vel = " << mBuffer_u[mFrameId].transpose() << std::endl;
					std::cout << "calculate vel before error = " << (diff).transpose() << " | " << (old_vel_after - mBuffer_u[mFrameId]).norm() << std::endl;
					//exit(1);
				}

				// check accel
				diff = mBuffer_u_dot[mFrameId - 1] - old_accel;
				if (diff.norm() > threshold)
				{
					std::cout << "truth accel =  " << old_accel.transpose() << std::endl;
					std::cout << "calc accel =  " << mBuffer_u_dot[mFrameId - 1].transpose() << std::endl;
					std::cout << "solved error = " << diff.transpose() << std::endl;
					//exit(1);
				}
			}


		}
	}

	SolveID();
}

void cIDSolver::AddJointForces()
{
	std::cout << "[error] cIDSolver::AddJointForces: this function should not be called in this context\n";
	exit(1);

	// pre setting
	int kp[3], kd[3];
	double torque_limit = 20;
	kp[0] = 100, kd[0] = 10;
	kp[1] = 10, kd[1] = 1;
	kp[2] = 100, kd[2] = 10;
	mMultibody->clearForcesAndTorques();
	for (auto &x : mJointForces) x.setZero();


	assert(mJointForces.size() == mMultibody->getNumLinks());
	for (int i = 0; i < mMultibody->getNumLinks(); i++)
	{
		// 1. pd control force
		const btMultibodyLink& cur_link = mMultibody->getLink(i);
		btTransform cur_trans = mMultibody->getLinkCollider(i)->getWorldTransform();
		btQuaternion local_to_world_rot = cur_trans.getRotation();

		switch (cur_link.m_jointType)
		{
		case btMultibodyLink::eFeatherstoneJointType::eRevolute:
		{
			// all joint pos, joint vel are local, not global
			double val = (0 - mMultibody->getJointPos(i)) * kp[0] + (0 - mMultibody->getJointVel(i)) * kd[0];
			tVector local_torque = cBulletUtil::btVectorTotVector0(mMultibody->getLink(i).getAxisTop(0) * val);
			if (local_torque.norm() > torque_limit)
			{
				//std::cout << "[warn] joint " << i << " exceed torque lim = " << torque_limit << std::endl;
				double scale = std::min(local_torque.norm(), torque_limit);
				local_torque = scale * local_torque.normalized();
			}

			mJointForces[i] = local_torque;
			mMultibody->addJointTorque(i, val);
			break;
		}
		case btMultibodyLink::eFeatherstoneJointType::eSpherical:
		{
			// get joint pos & vel for link i
			btScalar * joint_pos_bt = mMultibody->getJointPosMultiDof(i);
			btScalar * joint_vel_bt = mMultibody->getJointVelMultiDof(i);
			tVector joint_pos = tVector::Zero();
			tVector joint_vel = tVector::Zero();
			const int cnt = cur_link.m_dofCount;
			assert(cnt == 3);

			// set up joint pos, from quaternion to euler angles
			tQuaternion cur_rot(joint_pos_bt[3], joint_pos_bt[0], joint_pos_bt[1], joint_pos_bt[2]);
			joint_pos = cMathUtil::QuaternionToEulerAngles(cur_rot, eRotationOrder::ZYX);

			// set up joint vel
			for (int dof_id = 0; dof_id < 3; dof_id++)
			{
				joint_vel[dof_id] = joint_vel_bt[dof_id];
			}

			tVector local_torque = tVector::Zero();
			for (int dof_id = 0; dof_id < 3; dof_id++)
			{
				double value = kp[dof_id] * (0 - joint_pos[dof_id]) + kd[dof_id] * (0 - joint_vel[dof_id]);
				local_torque[dof_id] = value;
			}
			if (local_torque.norm() > torque_limit)
			{
				//std::cout << "[warn] joint " << i << " exceed torque lim = " << torque_limit << std::endl;
				double scale = std::min(local_torque.norm(), torque_limit);
				local_torque = scale * local_torque.normalized();
			}
			//std::cout << " joint " << i << " torque = " << local_torque.transpose() << std::endl;
			//local_torque = tVector(0, 0, 0, 0);
			//local_torque = tVector(0, 0, 0, 0);

			mJointForces[i] = local_torque;
			btScalar torque_bt[3] = { static_cast<btScalar>(local_torque[0]),\
				static_cast<btScalar>(local_torque[1]),
				static_cast<btScalar>(local_torque[2]) };
			mMultibody->addJointTorqueMultiDof(i, torque_bt);
			//std::cout << "link " << i << " joint world torque =" << world_torque.transpose() << std::endl;
			break;
		}
		case btMultibodyLink::eFeatherstoneJointType::eFixed:
		{
			// �޷�ʩ����
			//����
			break;
		}
		default:
		{
			std::cout << "[error] cIDSolver::AddJointForces: Unsupported joint type " << cur_link.m_jointType << std::endl;
			exit(1);
			break;
		}
		}

	}
}

void cIDSolver::AddExternalForces()
{
	// add random force
	if (true == mEnableExternalForce)
	{
		tVector external_force = tVector::Zero();
		for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
		{
			int multibody_link_id = ID_link_id - 1;
			if (ID_link_id == 0)
			{
				// add for root
				external_force = tVector::Random() * 10;
				mMultibody->addBaseForce(cBulletUtil::tVectorTobtVector(external_force));
			}
			else
			{
				external_force = tVector::Random() * 10;
				// add external force in world frame
				mMultibody->addLinkForce(multibody_link_id, cBulletUtil::tVectorTobtVector(external_force));
			}
			mExternalForces[ID_link_id] = external_force;
		}
	}

	// add random torque
	if (true == mEnableExternalTorque)
	{
		tVector external_torque = tVector::Zero();
		for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
		{
			int multibody_link_id = ID_link_id - 1;
			if (0 == ID_link_id)
			{
				external_torque = tVector::Random() * 10;
				mMultibody->addBaseTorque(cBulletUtil::tVectorTobtVector(external_torque));
			}
			else
			{
				external_torque = tVector::Random() * 10;
				mMultibody->addLinkTorque(multibody_link_id, cBulletUtil::tVectorTobtVector(external_torque));
			}
			mExternalTorques[ID_link_id] = external_torque;
		}
	}
}

void cIDSolver::RecordContactForces()
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

			// �����ײ����ͬ�����壬��ʱ����߼��������⡣
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

void cIDSolver::RecordJointForces()
{
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

void cIDSolver::RecordMultibodyInfo(std::vector<tMatrix>& local_to_world_rot, std::vector<tVector>& link_pos_world) const
{
	assert(local_to_world_rot.size() == mNumLinks);
	assert(link_pos_world.size() == mNumLinks);

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
void cIDSolver::RecordGeneralizedInfo(tVectorXd & _q, tVectorXd & _q_dot) const
{
	_q.resize(mDof), _q.setZero();
	_q_dot.resize(mDof), _q_dot.setZero();
	btInverseDynamicsBullet3::vecx q;
	btInverseDynamicsBullet3::vecx q_dot;
	RecordGeneralizedInfo(q, q_dot);
	for (int i = 0; i < mDof; i++)
	{
		_q[i] = q(i);
		_q_dot[i] = q_dot(i);
	}
}
void cIDSolver::RecordGeneralizedInfo(btInverseDynamicsBullet3::vecx & q, btInverseDynamicsBullet3::vecx & q_dot) const
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



void cIDSolver::SolveID()
{
	if (mEnableSolveID == false) return;
	if ((mSolvingMode == eSolvingMode::VEL && mFrameId < 1)
		||
		(mSolvingMode == eSolvingMode::POS && mFrameId < 2))
	{
		return;
	}

	ApplyContactForcesToID();
	ApplyExternalForcesToID();

	// solve ID: these joint forces are in joint frame
	// but the reference
	btInverseDynamicsBullet3::vecx solve_joint_force_bt(mDof);

	mInverseModel->calculateInverseDynamics(
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_q[mFrameId - 1]),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u[mFrameId - 1]),
		cBulletUtil::EigenArrayTobtIDArray(mBuffer_u_dot[mFrameId - 1]),
		&solve_joint_force_bt);

//#define VERBOSE 
	{
		double threshold = 1e-8;
		double sum_error = 0;
#ifdef VERBOSE
		std::cout << "solve joint torque diff = ";
#endif
		Eigen::VectorXd solved_joint_force = cBulletUtil::btIDArrayToEigenArray(solve_joint_force_bt);
		std::vector<double> true_joint_force(0);
		double err = 0;
		// output torque
		if (mFloatingBase == true)
		{
			double val = 0;
			for (int i = 0; i < 6; i++)
			{
				val = solved_joint_force[i];
				if (std::fabs(val) < threshold)  val = 0;
#ifdef VERBOSE
				std::cout << val << " ";
#endif
				err += val * val;
			}
#ifdef VERBOSE
			std::cout << "| ";
#endif
		}

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
				double value = solved_joint_force[offset];
				tVector axis_in_body_frame = cBulletUtil::btVectorTotVector0(mMultibody->getLink(i).getAxisTop(0));
				tQuaternion body_to_this = cMathUtil::EulerToQuaternion(cKinTree::GetBodyAttachTheta(mSimChar->GetBodyDefs(), i), eRotationOrder::XYZ);//
				tQuaternion this_to_body = body_to_this.inverse();
				tVector axis_in_joint_frame = cMathUtil::QuatRotVec(body_to_this, axis_in_body_frame);
				tVector torque_in_joint_frame = value * axis_in_joint_frame;
				tVector torque_in_link_frame = cMathUtil::QuatRotVec(this_to_body, torque_in_joint_frame);
				mSolvedJointForces[i] = torque_in_link_frame;

				for (int id = 0; id < 3; id++)
				{
					double diff_val = torque_in_link_frame[id] - mJointForces[i][id];
					if (std::fabs(diff_val) < threshold) diff_val = 0;
					true_joint_force.push_back(torque_in_link_frame[id]);
#ifdef VERBOSE
					std::cout << diff_val << " ";
#endif
					err += diff_val * diff_val;
				}
#ifdef VERBOSE
				std::cout << " | ";
#endif
				break;
			}
			case btMultibodyLink::eFeatherstoneJointType::eSpherical:
			{
				assert(cnt == 3);

				tVector torque = tVector(solved_joint_force[offset], solved_joint_force[offset + 1], solved_joint_force[offset + 2], 0);

				mSolvedJointForces[i] = torque;

				double val = 0;
				for (int dof_id = 0; dof_id < cnt; dof_id++)
				{
					val = torque[dof_id] - mJointForces[i][dof_id];
					if (std::fabs(val) < threshold) val = 0;
#ifdef VERBOSE
					std::cout << val << " ";
#endif
					true_joint_force.push_back(mJointForces[i][dof_id]);
					err += val * val;
				}
#ifdef VERBOSE
				std::cout << " | ";
#endif
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
#ifdef VERBOSE
		std::cout << std::endl;
#endif
		if (err > threshold)
		{
			std::cout << "true joint force = ";
			for (int i = 0; i < true_joint_force.size(); i++)
			{
				double val = true_joint_force[i];
				if (std::abs(val) < threshold) val = 0;
				std::cout << val << " ";
			}
			std::cout << std::endl;

			std::cout << "solved joint force = ";
			for (int i = 0; i < solved_joint_force.size(); i++)
			{
				double val = solved_joint_force[i];
				if (std::abs(val) < threshold) val = 0;
				std::cout << val << " ";
			}

			std::cout << std::endl;

			std::cout << "[error] ID solved wrong\n";
			//exit(1);
		}
	}

	// compare result


}

void cIDSolver::ApplyContactForcesToID()
{
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
		// 2. ����
		tVector contact_force_world = cur_force.mForce;
		tVector contact_force_local = local_to_world.transpose() * contact_force_world;
		// 1. ������
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

void cIDSolver::ApplyExternalForcesToID()
{
	//add external user force
	for (int ID_link_id = 0; ID_link_id < mNumLinks; ID_link_id++)
	{
		btInverseDynamicsBullet3::vec3 bt_com;
		btInverseDynamicsBullet3::mat33 bt_rot;
		mInverseModel->getBodyCoM(ID_link_id, &bt_com);		// world com, ����ʵ������Joint pos
		mInverseModel->getBodyTransform(ID_link_id, &bt_rot); // body to world
		tVector link_pos_world = mLinkPos[ID_link_id];
		tMatrix local_to_world_rot = mLinkRot[ID_link_id];

		// ��multibody�л�ȡ
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

		tVector force_world = mExternalForces[ID_link_id];
		tVector force_local = local_to_world_rot.transpose() * force_world;

		tVector torque_world = mExternalTorques[ID_link_id];
		tVector torque_local = local_to_world_rot.transpose() * torque_world;

		tVector force_torque_world = force_arm_world.cross3(force_world);
		tVector force_torque_local = local_to_world_rot.transpose() * force_torque_world;


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

				tVector omega_after = cMathUtil::CalcAngularVelocity(quater_before, quater_after, mCurTimestep);
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
				tVector ang_vel_parent = cMathUtil::CalcAngularVelocity(quater_before, quater_after, mCurTimestep);
				tVector ang_vel_local = cMathUtil::RotMat(quater_before).transpose() * ang_vel_parent;
				q_dot.segment(dof_offset, 3) = ang_vel_local.segment(0, 3);
				dof_offset += 3;
				break;
			}
			case btMultibodyLink::eFeatherstoneJointType::eRevolute:
			{
				q_dot(dof_offset) = (q_after(dof_offset) - q_before(dof_offset)) / mCurTimestep;
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

void cIDSolver::Reset()
{
	// clear buffer
	for (auto & x : mBuffer_q) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u_dot) x.resize(mDof), x.setZero();

	mFrameId = 0;

	ClearID();
}