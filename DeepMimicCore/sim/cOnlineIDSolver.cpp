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

#include "cOnlineIDSolver.hpp"
#include <BulletDynamics/Featherstone/cCollisionWorld.h>
#include <util/BulletUtil.h>
#include "SimCharacter.h"

cOnlineIDSolver::cOnlineIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world):cIDSolver(sim_char, world, eIDSolverType::Online)
{
	// clear buffer
	for (auto & x : mBuffer_q) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u_dot) x.resize(mDof), x.setZero();

	// init other vars
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

	mFrameId = 0;
	mSolvingMode = eSolvingMode::POS;

	ClearID();
}

void cOnlineIDSolver::SetTimestep(double deltaTime)
{
	mCurTimestep = deltaTime;
}

void cOnlineIDSolver::ClearID()
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

void cOnlineIDSolver::PreSim()
{
	//std::cout << "--------------------------------\n";
	ClearID();

	// record info
	AddExternalForces();

	// record info
	RecordJointForces(mJointForces);
	RecordGeneralizedInfo(mBuffer_q[mFrameId], mBuffer_u[mFrameId]);
	RecordMultibodyInfo(mLinkRot, mLinkPos);
	// std::cout <<"online sovler presim record " << mFrameId << std::endl;
	// std::ofstream fout("test1.txt", std::ios::app);
	// fout <<"presim frame id = " << mFrameId;
	// fout << "\n truth joint forces: ";
	// for(auto & x : mJointForces) fout << x.transpose() <<" ";
	// fout << "\n buffer q : ";
	// fout << mBuffer_q[mFrameId].transpose() <<" ";
	// fout << "\n buffer u : ";
	// fout << mBuffer_u[mFrameId].transpose() <<" ";
	// fout << std::endl;
	
}

void cOnlineIDSolver::PostSim()
{
	mFrameId++;

	RecordGeneralizedInfo(mBuffer_q[mFrameId], mBuffer_u[mFrameId]);
	// record contact forces
	RecordContactForces(mContactForces, mCurTimestep, mWorldId2InverseId);


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
            
			// std::ofstream fout("test1.txt", std::ios::app);
            // fout <<"offline buffer u dot calc: \n";
            // fout <<"buffer u " << mFrameId -1 <<" = " << mBuffer_u[mFrameId - 1].transpose()<< std::endl;
            // fout <<"buffer u " << mFrameId <<" = " << mBuffer_u[mFrameId].transpose()<< std::endl;
            // fout <<"buffer u dot " << mFrameId-1 <<" = " << mBuffer_u_dot[mFrameId - 1].transpose()<< std::endl;

			double threshold = 1e-5;
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
					std::cout << "differential error = " << diff.transpose() << std::endl;
					//exit(1);
				}
			}
		}
	}

	// std::cout <<"online sovler post record " << mFrameId << std::endl;
	// std::ofstream fout("test1.txt", std::ios::app);
	// fout <<"post sim frame id = " << mFrameId;
	// fout << "\n contact forces: ";
	// for(auto & x : mContactForces) fout << x.mForce.transpose() <<" ";
	// fout << "\n buffer q : ";
	// fout << mBuffer_q[mFrameId].transpose() <<" ";
	// fout << "\n buffer u : ";
	// fout << mBuffer_u[mFrameId].transpose() <<" ";
	// fout << std::endl;
	SolveIDSingleStep(mSolvedJointForces, mContactForces, mLinkPos, mLinkRot, mBuffer_q[mFrameId-1], mBuffer_u[mFrameId-1], mBuffer_u_dot[mFrameId-1], mFrameId, mExternalForces, mExternalTorques);

	// std::cout <<"online sovler ID record " << mFrameId << std::endl;
	// fout <<"ID frame id = " << mFrameId;
	// fout << "\n solved forces: ";
	// for(auto & x : mSolvedJointForces) fout << x.transpose() <<" ";
	// fout << "\n buffer u_dot : ";
	// fout << mBuffer_u_dot[mFrameId - 1].transpose() << " ";
	// fout <<"\n link pos : ";
	// for(auto & x : mLinkPos) fout << x.transpose() <<" ";
	// fout <<"\n link rot : ";
	// for(auto & x : mLinkRot) fout << x.transpose() <<" ";
	// fout << std::endl;
}

void cOnlineIDSolver::AddJointForces()
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

void cOnlineIDSolver::AddExternalForces()
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

void cOnlineIDSolver::Reset()
{
	// clear buffer
	for (auto & x : mBuffer_q) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u) x.resize(mDof), x.setZero();
	for (auto & x : mBuffer_u_dot) x.resize(mDof), x.setZero();

	mFrameId = 0;

	ClearID();
}


void cOnlineIDSolver::SolveIDSingleStep(std::vector<tVector> & solved_joint_forces,
		const std::vector<tForceInfo> & contact_forces,
		const std::vector<tVector> & link_pos, 
		const std::vector<tMatrix> & link_rot, 
		const tVectorXd & buf_q,
		const tVectorXd & buf_u,
		const tVectorXd & buf_u_dot,
		int frame_id,
		const std::vector<tVector> &external_forces,
		const std::vector<tVector> &external_torques) const
{
	if (mEnableSolveID == false) return;
	if ((mSolvingMode == eSolvingMode::VEL && mFrameId < 1)
		||
		(mSolvingMode == eSolvingMode::POS && mFrameId < 2))
	{
		return;
	}

	// solve for this step
	cIDSolver::SolveIDSingleStep(
		solved_joint_forces,
		contact_forces, 
		link_pos, 
		link_rot, 
		buf_q, 
		buf_u, 
		buf_u_dot, 
		frame_id, 
		external_forces, 
		external_torques);

	// check the result(solved_joint_forces) which should be the same as the ground truth(truth_joint_forces)
	assert(solved_joint_forces.size() == mJointForces.size());
	double err = 0;
	for(int id = 0; id< solved_joint_forces.size(); id++)
	{
		if(cMathUtil::IsSame(solved_joint_forces[id], mJointForces[id], 1e-5) == false)
		{
			err += (solved_joint_forces[id]- mJointForces[id]).norm();
			std::cout <<"[error] online ID solved error: for joint " << id <<" diff : solved = " << solved_joint_forces[id].transpose() <<", truth = " << mJointForces[id].transpose() << std::endl;
		}
	}
	if(err < 1e-5) std::cout << "[log] frame " << frame_id <<" online ID solved accurately\n";
}