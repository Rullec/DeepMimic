#include "anim/Motion.h"
#include "sim/SimInverseDynamics.h"
#include <queue>
#include <iostream>
using namespace std;
using namespace Eigen;

const std::string gIDLogPath_f = "./logs/controller_logs/ID_compute_force.log";
const std::string gIDLogPath_t = "./logs/controller_logs/ID_compute_torque.log";
const std::string gIDLogPath_linkinfo = "./logs/controller_logs/ID_link_info.log";

auto JudgeSameVec = [](VectorXd a, VectorXd b, double eps = 1e-5)
{
	return (a - b).norm() < eps;
};

auto JudgeSame = [](double a, double b, double eps = 1e-5)
{
	return abs(a - b) < eps;
};

// lambda definitions
auto CHECK_INPUT = [](const Eigen::VectorXd & v, int size)->bool
{
	if (v.size() != size) return false;
	else return true;
};

auto ADD_NEW_ITEM = [](Eigen::MatrixXd & mat, const Eigen::VectorXd & vec)->bool
{
	if (mat.cols() != vec.size())
	{
		std::cout << "[error] ADD_NEW_ITEM FAILED" << std::endl;
		return false;
	}

	mat.conservativeResize(mat.rows() + 1, Eigen::NoChange);
	mat.row(mat.rows() - 1) = vec;
	//std::cout << "[debug] inner pose = " << mat.row(mat.rows() - 1);
	return true;
};

cInverseDynamicsInfo::cInverseDynamicsInfo(std::shared_ptr<cSimCharacter> & sim_char)
{
	mNumOfFrames = -1;
	mPoseSize = -1, mStateSize = -1, mActionSize = -1, mContact_infoSize = -1;
	mState.resize(0, 0);
	mPose.resize(0, 0);
	mAction.resize(0, 0);
	mContact_info.resize(0, 0);

	mNumLinks = sim_char->GetNumJoints();
	mSimChar = sim_char;
	mIDStatus = eIDStatus::INVALID;
	mLinkInfo = (std::shared_ptr<cInverseDynamicsInfo::tLinkInfo>)(new cInverseDynamicsInfo::tLinkInfo());

	BuildTopoInfo();

	// init temp & final result
	mReactionForce.resize(mNumLinks);
	mReactionForce_pof.resize(mNumLinks);
	mReactionTorque.resize(mNumLinks);

	for (int i = 0; i < mNumLinks; i++)
	{
		mReactionForce[i].setZero();
		// get joint position
		auto & cur_link = mSimChar->GetJoint(i).GetChild();
		mReactionForce_pof[i] = cur_link->GetRotationCenter();
	}
	
}

void cInverseDynamicsInfo::AddNewFrame(const Eigen::VectorXd & state_, const Eigen::VectorXd & pose_, const Eigen::VectorXd & action_, const Eigen::VectorXd & contact_info_)
{
	/*
	Function: cInverseDynamicsInfo::AddNewFrame

	This function will append a new frame info into the storaged MatrixXd array

*/

	if (mNumOfFrames == -1)
	{
		mNumOfFrames = 0;
		mStateSize = state_.size(), mPoseSize = pose_.size(), mActionSize = action_.size(), mContact_infoSize = contact_info_.size();
		mState.resize(mNumOfFrames, mStateSize);
		mPose.resize(mNumOfFrames, mPoseSize);
		mAction.resize(mNumOfFrames, mActionSize);
		mContact_info.resize(mNumOfFrames, mContact_infoSize);
	}

	if (!CHECK_INPUT(state_, mStateSize)
		|| !CHECK_INPUT(pose_, mPoseSize)
		|| !(CHECK_INPUT(action_, mActionSize) || CHECK_INPUT(action_, 0))	// the size of action can be 0 or mActionSize
		|| !CHECK_INPUT(contact_info_, mContact_infoSize))
	{
		std::cout << "[error] AddNewFrame in InverseDynamicInfo input does not correspond" << std::endl;
		std::cout << "[debug] state size = " << state_.size() << " != " << mStateSize << std::endl;
		std::cout << "[debug] pose size = " << pose_.size() << " != " << mPoseSize << std::endl;
		std::cout << "[debug] action size = " << action_.size() << " != " << mActionSize << std::endl;
		std::cout << "[debug] contact_info size = " << contact_info_.size() << " != " << mContact_infoSize << std::endl;
		exit(1);
	}
	else
	{
		mNumOfFrames++;
		bool succ = true;
		succ &= ADD_NEW_ITEM(mState, state_);
		succ &= ADD_NEW_ITEM(mPose, pose_);
		if (action_.size() != 0) succ &= ADD_NEW_ITEM(mAction, action_);	// in the last frame of the trajectory, the state of action is zero.
		succ &= ADD_NEW_ITEM(mContact_info, contact_info_);
		if (!succ) {
			std::cout << "[error] AddNewFrame in InverseDynamicInfo failed" << std::endl;
			exit(1);
		}
	}

	mIDStatus = eIDStatus::PREPARED;
}

int cInverseDynamicsInfo::GetNumOfFrames() const 
{
	return mNumOfFrames;
}

Eigen::VectorXd cInverseDynamicsInfo::GetPose(double time)
{
	double cur_time = 0;
	int target_frame_id = 0;
	for (int i = 0; i < mNumOfFrames - 1; i++)
	{
		double duration = mPose(i, cMotion::eFrameParams::eFrameTime);
		if (cur_time <= time && time <= cur_time + duration)
		{
			target_frame_id = i;
			break;
		}
		cur_time += duration;
	}
	return mPose.row(target_frame_id).segment(1, mPoseSize - 1);
}

Eigen::VectorXd cInverseDynamicsInfo::GetTorque(double time)
{
	double cur_time = 0;
	int target_frame_id = 0;
	for (int i = 0; i < mNumOfFrames - 2; i++)
	{
		double duration = mPose(i, cMotion::eFrameParams::eFrameTime);
		if (cur_time <= time && time <= cur_time + duration)
		{
			target_frame_id = i;
			break;
		}
		cur_time += duration;
	}
	return mTorque.row(target_frame_id);
}

double cInverseDynamicsInfo::GetMaxTime() const 
{
	double total_time = 0;
	for (int i = 0; i < mNumOfFrames; i++)
	{
		double duration = mPose(i, cMotion::eFrameParams::eFrameTime);
		total_time += duration;
	}
	return total_time;
}

tVector cInverseDynamicsInfo::GetLinkPos(int frame, int body_id) const 
{
	if (frame >= mLinkInfo->mLinkPos.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetmLinkPos(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	tVector pos = tVector::Zero(); 
	pos.segment(0, 3) = mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3).transpose();
	return pos;
}

tVector cInverseDynamicsInfo::GetLinkVel(int frame, int body_id) const 
{
	if (frame >= mLinkInfo->mLinkVel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetmLinkVel(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	tVector vel = tVector::Zero();
	vel.segment(0, 3) = mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3).transpose();
	return vel;
}

tVector cInverseDynamicsInfo::GetLinkAccel(int frame, int body_id)const 
{
	if (frame >= mLinkInfo->mLinkAccel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAccel(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}

	tVector accel = tVector::Zero();
	accel.segment(0, 3) = mLinkInfo->mLinkAccel.block(frame, body_id * 3, 1, 3).transpose();
	return accel;
}

tVector cInverseDynamicsInfo::GetLinkRotation(int frame, int body_id) const // get Quaternion coeff 4*1 [x, y, z, w]
{
	// local to world

	if (frame < 0 || frame >= mLinkInfo->mLinkRot.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkRotation: illegal access" << std::endl;
		exit(1);
	}

	// [x, y, z, w] quaternion
	return mLinkInfo->mLinkRot.block(frame, body_id * 4, 1, 4).transpose();
}

tMatrix cInverseDynamicsInfo::GetLinkTrans(int frame, int body_id) const 
{
	// in this frame, for this link body_id, return the 4x4 transformation matrix local to world
	tMatrix local_to_world_mat = GetLinkRotationMat(frame, body_id);
	tVector link_pos = GetLinkPos(frame, body_id);
	return cMathUtil::TranslateMat(link_pos) * local_to_world_mat;
}

tMatrix cInverseDynamicsInfo::GetLinkRotationMat(int frame, int body_id) const
{
	tVector quat_coeff = GetLinkRotation(frame, body_id);
	tQuaternion local_to_world = tQuaternion(quat_coeff[3], quat_coeff[0], quat_coeff[1], quat_coeff[2]);
	tMatrix local_to_world_mat = cMathUtil::RotateMat(local_to_world);
	return local_to_world_mat;
}

tVector cInverseDynamicsInfo::GetLinkAngularVel(int frame, int body_id)	const // get link angular vel 4*1 [wx, wy, wz, 0]
{
	if (frame < 0 || frame >= mLinkInfo->mLinkAngularVel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAngularVel: illegal access" << std::endl;
		exit(1);
	}

	// [omega_ax, omega_ay, omega_az, 0] angular velocity
	return mLinkInfo->mLinkAngularVel.block(frame, body_id * 4, 1, 4).transpose();
}

tVector cInverseDynamicsInfo::GetLinkAngularAccel(int frame, int body_id)	const // get link angular accel 4*1 [ax, ay, az, 0]
{
	if (frame < 0 || frame >= mLinkInfo->mLinkAngularAccel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAngularAccel: illegal access" << std::endl;
		exit(1);
	}
	// [omega_ax, omega_ay, omega_az, 0] angular velocity
	return mLinkInfo->mLinkAngularAccel.block(frame, body_id * 4, 1, 4).transpose();
}

void cInverseDynamicsInfo::GetLinkContactInfo(int frame, int link, tEigenArr<tVector> & force, tEigenArr<tVector> & point_of_force) const
{
	// check & clear input
	if (frame >= mNumOfFrames)
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkContactInfo: invalid frame input " << frame << std::endl;
	}
	force.clear();
	point_of_force.clear();

	VectorXd contact_info = mContact_info.row(frame);
	for (int i = 0; i < contact_info.size(); i += 7)
	{
		if (int(contact_info[i]) == link)
		{
			tVector cur_pof = tVector::Zero(), cur_force = tVector::Zero();
			cur_pof.segment(0, 3) = contact_info.segment(i + 1, 3);
			cur_force.segment(0, 3) = contact_info.segment(i + 4, 3);
			force.push_back(cur_force);
			point_of_force.push_back(cur_pof);
		}
	}

}

void cInverseDynamicsInfo::SolveInverseDynamics()
{
	/*
	@Function: cInverseDynamicsInfo::SolveInverseDynamics
	@params: void

	As it is, given the character poses info & contact info, this function will computes the actions accordly.
	It can be divided into 2 parts:

	1. compute dynamic link info(pos, vel, accel, angular displacement, angular vel, angular accel)
	2. calculate joint torques using a
	3. convert the torque to PD target by individual PD params
*/

	// clear log path
	ofstream f_clear;
	f_clear.open(gIDLogPath_f);
	f_clear << "";
	f_clear.close();
	f_clear.open(gIDLogPath_t);
	f_clear << "";
	f_clear.close();

	if (mIDStatus != eIDStatus::PREPARED)
	{
		std::cout << "[warn] You do not need to / can not solve Inverse Dynamics, flag = " << mIDStatus << std::endl;
		return;
	}

	if (mSimChar == nullptr)
	{

	}
	;

	// 1. compute dynamic link info
	ComputeLinkInfo();

	// 2. calculate joint toruqe for each frames &
	ComputeJointDynamics();

	// 3. convert the torque to PD target
	ComputePDTarget();


}

void cInverseDynamicsInfo::ComputeLinkInfo()
{
	/*
		@Function: void cInverseDynamicsInfo::ComputeLinkInfo
		@params: void
			Accoroding to the motion info in mState & mPos, this function will compute the linear
		velocity & acceleration for the COM of each link in every frame.
			We supposed that the intergrator empoloies explicit Euler method.
	*/
	// check input 
	if (mNumOfFrames == -1 || mSimChar == nullptr || mLinkInfo == nullptr)
	{
		std::cout << "[error] cInverseDynamicsInfo::ComputeLinkInfo input illegal" << std::endl;
		exit(1);
	}

	// 1. save the status of sim_char. It wil be resumed in the end of this function

	Eigen::VectorXd pose_before = mSimChar->GetPose();

	// 2. get the linear position of each link in each frame
	// 2.1 resize
	int num_bodies = mSimChar->GetNumBodyParts();

	{
		// timesteps
		mLinkInfo->mTimesteps.resize(mNumOfFrames);

		// linear infos
		mLinkInfo->mLinkPos.resize(mNumOfFrames, num_bodies * 3);	// mLinkPos.size == (total num of frames, NumOfBodies * 3);
		mLinkInfo->mLinkVel.resize(mNumOfFrames - 1, num_bodies * 3);
		mLinkInfo->mLinkAccel.resize(mNumOfFrames - 2, num_bodies * 3);

		// ultimate angular infos 
		mLinkInfo->mLinkRot.resize(mNumOfFrames, num_bodies * 4);		// quaternion coeff [x, y, z, w]
		mLinkInfo->mLinkAngularVel.resize(mNumOfFrames - 1, num_bodies * 4);	// axis angle [x,y,z,theta]
		mLinkInfo->mLinkAngularAccel.resize(mNumOfFrames - 2, num_bodies * 4);	// the same axis angle
	}


	for (int i = 0; i < mNumOfFrames; i++)
	{
		Eigen::VectorXd cur_pose = mPose.row(i).segment(1, mPoseSize - 1);
		double cur_timestep = mPose.row(i)[cMotion::eFrameTime];

		mLinkInfo->mTimesteps[i] = cur_timestep;
		mSimChar->SetPose(cur_pose);

		// get & set position info
		for (int body_id = 0; body_id < mSimChar->GetNumBodyParts(); body_id++)
		{
			// 2.2 0 order info computing 
			ComputeLinkInfo0(i, body_id);

			// 2.3 1st order info computing(velocity)
			ComputeLinkInfo1(i - 1, body_id);

			// 2.4 2nd order info computing (acceleration)
			ComputeLinkInfo2(i - 2, body_id);
		}
	}

	PrintLinkInfo();

	// 5. check the storage
	{
		// check velocity
		for (int frame = 0; frame < GetNumOfFrames() - 1; frame++)
		{
			double timestep = mLinkInfo->mTimesteps[frame];
			for (int body = 0; body < this->mSimChar->GetNumBodyParts(); body++)
			{
				tVector cur_vel = GetLinkVel(frame, body);
				tVector cur_pos = GetLinkPos(frame, body), next_pos = GetLinkPos(frame + 1, body);
				tVector predicted_move = cur_vel.cwiseProduct(tVector::Ones() * timestep);
				tVector true_move = next_pos - cur_pos;
				if (JudgeSameVec(predicted_move, true_move) == false)
				{
					printf("[error] judge same vec failed in frame %d body %d\n", frame, body);
					std::cout << "[debug] predicated_move = " << predicted_move.transpose() << std::endl;
					std::cout << "[debug] true_move = " << true_move.transpose() << std::endl;
				}
			}
		}

		// check accel
		for (int frame = 0; frame < GetNumOfFrames() - 2; frame++)
		{
			double timestep = mLinkInfo->mTimesteps[frame];
			for (int body = 0; body < this->mSimChar->GetNumBodyParts(); body++)
			{
				tVector cur_accel = GetLinkAccel(frame, body);
				//std::cout << cur_accel.transpose() << std::endl;
				tVector cur_vel = GetLinkVel(frame, body), next_vel = GetLinkVel(frame + 1, body);
				//std::cout << cur_vel.transpose() <<" " << next_vel.transpose() << std::endl;
				tVector predicted_move = cur_accel.cwiseProduct(tVector::Ones() * timestep);
				//std::cout << timestep << std::endl;
				tVector true_move = next_vel - cur_vel;
				//std::cout << true_move.transpose() << std::endl;
				if (false == JudgeSameVec(true_move, predicted_move))
				{
					printf("[error] judge same accel failed in frame %d body %d\n", frame, body);
					std::cout << "[debug] predicated_move = " << predicted_move.transpose() << std::endl;
					std::cout << "[debug] true_move = " << true_move.transpose() << std::endl;

				}
				//std::cout << predicted_timestep.transpose() << std::endl;
			}
		}
	}

	// 6. resume the status
	mSimChar->SetPose(pose_before);
}

void cInverseDynamicsInfo::ComputeLinkInfo0(int frame, int body_id)
{
	/*
	@Function: cIvnerseDynamicsInfo::ComputeLinkInfo0
	@params: frame Type int
	@params: body_id Type int
	*/

	if (frame < 0) return;

	// linear terms
	{
		tVector cur_pos = mSimChar->GetBodyPartPos(body_id);
		mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3) = cur_pos.segment(0, 3).transpose();
		tVector get_pos = GetLinkPos(frame, body_id);
		if (JudgeSameVec(get_pos, cur_pos) == false)
		{
			std::cout << "[error] mLinkPos set error" << cur_pos.transpose() << " " << get_pos.transpose() << std::endl;
			exit(1);
		}
	}
	
	// ultimate angular term
	{
		auto & cur_part = mSimChar->GetBodyPart(body_id);
		tQuaternion quater = cur_part->GetRotation();
		tVector coeff = quater.coeffs().transpose();
		mLinkInfo->mLinkRot.block(frame, body_id * 4, 1, 4) = coeff.transpose();

		// verify
		tVector get_coef = GetLinkRotation(frame ,body_id);
		if (false == JudgeSameVec(coeff, get_coef))
		{
			std::cout << "[error] ComputeLinkInfo0: set ultimate angular term error: " \
				<< coeff.transpose() << " " << get_coef.transpose() << std::endl;
			exit(1);
		}
	}
}

void cInverseDynamicsInfo::ComputeLinkInfo1(int frame, int body_id)
{
	/*
	@Function: cInverseDynamicsInfo::ComputeLinkInfo1
	@params: the same as above

	This function computes first order info for links, including
		linear vel and angular vel
*/

	if (frame < 0) return;
	double timestep = mLinkInfo->mTimesteps[frame];
	// linear terms
	{
		tVector vel = (GetLinkPos(frame + 1, body_id) - GetLinkPos(frame, body_id)) / timestep;
		mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3) = vel.segment(0, 3).transpose();
		tVector get_vel = GetLinkVel(frame, body_id);
		if (JudgeSameVec(get_vel, vel) == false)
		{
			std::cout << "[error] mLinkVel set error: " << vel.transpose() << " " << get_vel.transpose() << std::endl;
			exit(1);
		}
	}

	// ultimate angular velocity
	{
		tVector q1_coef = GetLinkRotation(frame, body_id), q2_coef = GetLinkRotation(frame + 1, body_id);
		tQuaternion q1 = tQuaternion(q1_coef[3], q1_coef[0], q1_coef[1], q1_coef[2]),
					q2 = tQuaternion(q2_coef[3], q2_coef[0], q2_coef[1], q2_coef[2]);
		tVector omega = cMathUtil::CalcQuaternionVel(q1, q2, timestep);

		// normalized axis angle, format [w_ax, w_ay, w_az, 0]
		mLinkInfo->mLinkAngularVel.block(frame, body_id * 4, 1, 4) = omega.transpose();

		tVector get_vel = GetLinkAngularVel(frame, body_id);

		if (JudgeSameVec(get_vel, omega) == false)
		{
			std::cout << "[error] ComputeLinkInfo1: ultimate angular velocity error = " << get_vel.transpose()\
				<< " " << omega.transpose() << std::endl;
			exit(1);
		}
		/*
			The following code implement this formula:
				w = 0.5 *  dot(q) * q^*
			it works well only when the angular displacement is SMALL, because only in this case 
			can dot(q) be approximated to q_{t+1} - q_t
		*/
		
		// test : w = 0.5 * (q2 - q1) / timestep * conj(q1)
		//btQuaternion q1_bt = btQuaternion(q1.x(), q1.y(), q1.z(), q1.w());
		//btQuaternion q2_bt = btQuaternion(q2.x(), q2.y(), q2.z(), q2.w());
		//tQuaternion q2_conj = q2.conjugate();
		//btQuaternion q2_conj_bt = btQuaternion(q2_conj.x(), q2_conj.y(), q2_conj.z(), q2_conj.w());
		//q1_bt = q1_bt.normalize();
		//q2_bt = q2_bt.normalize();
		////0.5 * (q2_bt - q1_bt) / timestep;
		//btQuaternion omega_bt_quater = (q2_bt - q1_bt) * 2 / timestep * q2_conj_bt;
		//btVector axis = omega_bt_quater.getAxis();
		//tVector omega_bt = tVector(axis.getX(), axis.getY(), axis.getZ(), 0);
		//double mag_bt = omega_bt.norm();
		//omega_bt /= mag_bt;
		//omega_bt[3] = mag_bt;
		//
		//std::cout << omega_bt.transpose() << std::endl;
		//std::cout << omega.transpose() << std::endl;

	}
}

void cInverseDynamicsInfo::ComputeLinkInfo2(int frame, int body_id)
{
	/*
	@Function: cInverseDynamicsInfo::ComputeLinkInfo2
	@params: the same as above

	This function computes second order link motion terms
		linear accel and angular accel

	*/

	if (frame < 0) return;
	double timestep = mLinkInfo->mTimesteps[frame];
	// linear terms
	{
		tVector accel = (GetLinkVel(frame + 1, body_id) - GetLinkVel(frame, body_id)) / timestep;
		mLinkInfo->mLinkAccel.block(frame, body_id * 3, 1, 3) = accel.segment(0, 3).transpose();
		tVector get_accel = GetLinkAccel(frame, body_id);
		if (JudgeSameVec(get_accel, accel) == false)
		{
			std::cout << "[error] mLinkAccel set error " << accel.transpose() << " " << get_accel.transpose() << std::endl;
			exit(1);
		}
	}

	// ultimate angular terms, computed from axis-angle to accel
	{
		
		tVector q1_vel_a = GetLinkAngularVel(frame, body_id), q2_vel_a = GetLinkAngularVel(frame + 1, body_id);

		tVector q1_accel = (q2_vel_a - q1_vel_a) / timestep;
		double mag_accel = q1_accel.norm();

		mLinkInfo->mLinkAngularAccel.block(frame, body_id * 4, 1, 4) = q1_accel.transpose();

		tVector get_accel = GetLinkAngularAccel(frame, body_id);
		if (JudgeSameVec(get_accel, q1_accel) == false)
		{
			std::cout << "[error] ComputeLinkInfo2 error in ultimate accel: " << get_accel.transpose() \
				<< " " << q1_accel.transpose() << std::endl;
			exit(1);
		}
	}
}

void cInverseDynamicsInfo::ComputeJointDynamics()
{
	/*
	@Function: cInverseDynamicsInfo::ComputeJointDynamics const
	@params: torque Type VectorXd, the computation goal

	This function implement the classic ID algorithm "Recursive Inverse Dynamic Method"
	both in C.K. Liu: "A Quick Tutorial on Multibody Dynamics" and in Featherstone's book(2008)
	2 parts:
	1. construct the links' topology tree, which has a root and all of the links who has no successors are leaves
	2. for each frame, calculates the torque & force between links
	*/

	if (mSimChar == nullptr)
	{
		std::cout << "[error] void cInverseDynamicsInfo::ComputeJointDynamics(VectorXd & torque) empty model" << std::endl;
		exit(1);
	}

	// 2. compute forces for each frame
	//for (int frame_id = 0; frame_id < 1; frame_id++)
	mTorque.resize(mNumOfFrames - 2, 3 * mNumLinks);
	for (int frame_id = 0; frame_id < mNumOfFrames - 2; frame_id++)
	{
		ComputeJointDynamicsFrame(frame_id);

		VectorXd cur_torque(3 * mNumLinks);
		for (int j = 0; j < mNumLinks; j++)
		{
			cur_torque.segment(3 * j, 3) = mReactionTorque[j];
		}
		mTorque.row(frame_id) = cur_torque.transpose();
		
	}

}

void cInverseDynamicsInfo::ComputeJointDynamicsFrame(const int frame_id)
{
	// 1. check input 
	if (nullptr == mSimChar  || frame_id >= mNumOfFrames)
	{
		std::cout << "[error] void cInverseDynamicsInfo::ComputeSingleJointTorque: input invalid " << std::endl;
		exit(1);
	}
	int num_links = mSimChar->GetNumBodyParts();

	// 2. iteration for each joint
	for (int index = 0; index < num_links; index++)
	{
		int link_id = mVisitSeq[index];
		//std::cout << "[debug] begin to compute the force for link " << link_id << " " << mSimChar->GetBodyName(link_id) << std::endl;
		ComputeJointDynamicsFrameForce(frame_id, link_id);
		//std::cout << "[debug] begin to compute the torque for link " << link_id << " " << mSimChar->GetBodyName(link_id) << std::endl;
		ComputeJointDynamicsFrameTorque(frame_id, link_id);
	}

}

void cInverseDynamicsInfo::ComputeJointDynamicsFrameForce(const int frame_id, const int link_id)
{
	// TODO: 验证force和torque的逻辑正确性
	// (不必须，可以直接设置torque)然后开始把torque转化为PD target
	// 把得到的torque送入simulation中，看结果。

	ofstream fout(gIDLogPath_f, ios::app);
	fout << "----------------frame " << frame_id << "----------------" << std::endl;
	fout << "----------------begin to calculate link " << link_id << " " << mSimChar->GetBodyName(link_id) << std::endl;

	// 3. computes the force between joints in this frame
	auto & link = mSimChar->GetJoint(link_id).GetChild();
	string link_name = mSimChar->GetBodyName(link_id);
	double link_mass = link->GetMass();
	fout << "mass = " << link_mass << std::endl;
	
	
	tVector link_pos = GetLinkPos(frame_id, link_id);
	fout << "link pos = " << link_pos.transpose() << std::endl;
	tMatrix local_to_world_rot = GetLinkRotationMat(frame_id, link_id);
	tVector local_to_world_coef = GetLinkRotation(frame_id, link_id);
	fout << "link rot quater [x, y, z, w] = " << local_to_world_coef.transpose() << std::endl;
	tMatrix local_to_world_trans = cMathUtil::TranslateMat(link_pos) * local_to_world_rot;
	fout << "link local to world trans = \n" << local_to_world_trans << std::endl;
	tMatrix world_to_local_rot = cMathUtil::InvRigidMat(local_to_world_rot);
	tMatrix world_to_local_trans = cMathUtil::InvRigidMat(local_to_world_trans);
	fout << "link world to local trans = \n" << world_to_local_trans << std::endl;

	// 3.1 computes the COM accel of this link in local frame
	tVector link_accel_world = tVector::Zero();
	link_accel_world = GetLinkAccel(frame_id, link_id);	// in world frame
	fout << "link accel world = " << link_accel_world.transpose() << std::endl;;
	tVector link_accel_local = world_to_local_rot * link_accel_world;
	fout << "link accel local = " << link_accel_local.transpose() << std::endl;;

	// 3.2 tackle the contact_force and reaction force coming from its child joint(link)
	// contact forces
	tEigenArr<tVector> contact_force_lst, contact_point_of_force_lst;	// contact forces in world frame
	GetLinkContactInfo(frame_id, link_id, contact_force_lst, contact_point_of_force_lst);

	// convert contact forces and point of forces to local frame
	tVector link_contact_force_total = tVector::Zero();
	for (int i = 0; i < contact_force_lst.size(); i++)
	{
		// force: free vector, no fixed start point
		tVector cur_contact_force = tVector(
			contact_force_lst[i][0],
			contact_force_lst[i][1],
			contact_force_lst[i][2],
			1);
		// point of force: fixed vector, a point in this space
		tVector cur_contact_pof = tVector(
			contact_point_of_force_lst[i][0],
			contact_point_of_force_lst[i][1],
			contact_point_of_force_lst[i][2],
			1);
		
		fout << "contact force " << i << " global = " << cur_contact_force.transpose() \
			<< "pof = " << cur_contact_pof.transpose() <<std::endl;

		// 这里需要验证
		cur_contact_force = world_to_local_rot * cur_contact_force;
		cur_contact_pof = world_to_local_trans * cur_contact_pof;
		fout << "contact force " << i << " local = " << cur_contact_force.transpose() \
			<< "pof = " << cur_contact_pof.transpose() << std::endl;

		tVector result_force = cur_contact_force;
		link_contact_force_total += result_force;
	}

	// computes the composition of child reaction forces
	// children joints 所给的力，拿到的是在child坐标系下的作用点和力方向，需要做坐标系变换
	const VectorXd child_id_lst = mJointChildren[link_id];
	tVector child_force_total = tVector::Zero();
	for (int i = 0; i < child_id_lst.size(); i++)
	{
		int child_id = child_id_lst[i];

		// get reaction force & pof in world frame
		tVector cur_reaction_force = -mReactionForce[child_id],
			point_of_reaction_force = mReactionForce_pof[child_id];
		fout << "child " << child_id << " reaction force global = " << cur_reaction_force.transpose() << std::endl;
		cur_reaction_force = world_to_local_rot * cur_reaction_force;
		fout << "child " << child_id << " reaction force local = " << cur_reaction_force.transpose() << std::endl;
		child_force_total += cur_reaction_force;
	}

	/*
		3.3 computes the force given by its parent
		f_{parent(link_id)} = mass * accel_local - contact_force_set + f_{child(link_id)}_set

		ATTENTION: we define "f_{any_link}" as the forces applied to any_link which coming from its parent,
			and in this status, the symbol of "f_{any_link}" is positive.
	*/
	tVector f_parent_local = tVector::Zero();
	f_parent_local = link_mass * link_accel_local - link_contact_force_total - child_force_total;
	fout << "force_local = " << f_parent_local.transpose() << std::endl;
	tVector f_parent_global = local_to_world_rot * f_parent_local;
	fout << "force_glocal = " << f_parent_global.transpose() << std::endl;

	tVector f_parent_pof_local = link->GetRotationCenter();
	fout << "force_local_pof = " << f_parent_pof_local.transpose() << std::endl;
	tVector f_parent_pof_global = local_to_world_trans * f_parent_pof_local;
	fout << "force_global_pof = " << f_parent_pof_global.transpose() << std::endl;
	
	// 两个global数存入buffer中
	mReactionForce[link_id] = f_parent_global;
	mReactionForce_pof[link_id] = f_parent_pof_global;
}

void cInverseDynamicsInfo::ComputeJointDynamicsFrameTorque(const int frame_id, const int link_id)
{
	// check input
	if (frame_id < 0 || frame_id >= mNumOfFrames)
	{
		std::cout << "[error] frame id illegel " << std::endl;
		exit(1);
	}

	/*
		I_{ck} * \dot{w} + \dot{w} x (I_{ck} * w) = \tau - c_k x fk - \sum{R * \tau_child + (d-c)x(R_i * f_i)}
	*/
	ofstream fout(gIDLogPath_t, ios::app);
	fout << "----------------frame " << frame_id << "----------------" << std::endl;
	fout << "----------------begin to calculate link " << link_id << " " << mSimChar->GetBodyName(link_id) << std::endl;

	auto & cur_link = mSimChar->GetJoint(link_id).GetChild();
	//std::cout << "[debug] link " << link_id << " inertia = " << cur_link->GetInertia().transpose() << std::endl;
	/*
		\tau = I_{ck} * \dot{w}^l + {w_k}^l x I_{ck} {w_k}^l + c_k x f_k + sum(...)
	*/

	tVector final_torque = tVector::Zero();

	// 1. get inertia
	tVector inertia = cur_link->GetInertia();
	tMatrix inertia_tensor = tMatrix::Identity();
	inertia_tensor(0, 0) = inertia[0];
	inertia_tensor(1, 1) = inertia[1];
	inertia_tensor(2, 2) = inertia[2];
	fout << "inertia = " << inertia.transpose() << std::endl;

	// 2. get angular accel, convert it to local
	tVector link_ang_accel = GetLinkAngularAccel(frame_id, link_id);
	tVector link_ang_vel = GetLinkAngularVel(frame_id, link_id);
	tVector link_pos = GetLinkPos(frame_id, link_id);	// COM world position
	tMatrix local_to_world_rot = GetLinkRotationMat(frame_id, link_id);
	tMatrix local_to_world_trans = cMathUtil::TranslateMat(link_pos) * local_to_world_rot;
	tMatrix world_to_local_rot = cMathUtil::InvRigidMat(local_to_world_rot);
	tMatrix world_to_local_trans = cMathUtil::InvRigidMat(local_to_world_trans);
	fout << "inertia tensor = \n " << inertia_tensor << std::endl;
	fout << "link ang accel = " << link_ang_accel.transpose() << std::endl;
	fout << "link ang vel = " << link_ang_vel.transpose() << std::endl;
	fout << "inertia_tensor * link_ang_vel = " << (inertia_tensor * link_ang_vel).transpose() << std::endl;
	tVector LHS = inertia_tensor * link_ang_accel + link_ang_vel.cross3(inertia_tensor * link_ang_vel);
	fout << "LHS = " << LHS.transpose() << std::endl;
	final_torque += LHS;

	// 3. get the reaction force given by his parent
	tVector link_reaction_force = mReactionForce[link_id],		// in world frame
		link_reaction_force_pof = mReactionForce_pof[link_id];	// in world frame
	fout << "parent reaction force_world = " << link_reaction_force.transpose() << std::endl;
	fout << "parent reaction force pof_world = " << link_reaction_force_pof.transpose() << std::endl;
	link_reaction_force = world_to_local_rot * link_reaction_force;
	link_reaction_force_pof = world_to_local_trans * link_reaction_force_pof;
	fout << "parent reaction force_local = " << link_reaction_force.transpose() << std::endl;
	fout << "parent reaction force pof_local = " << link_reaction_force_pof.transpose() << std::endl;

	tVector link_reaction_torque = link_reaction_force_pof.cross3(link_reaction_force);
	fout << "link reaction torque = " << link_reaction_torque.transpose() << std::endl;
	final_torque -= link_reaction_torque;

	// 4. get the reaction force & torque accordly given by his children
	VectorXd total_children = mJointChildren[link_id];
	for (int i = 0; i < total_children.size(); i++)
	{
		int cur_child = total_children[i];
		fout << "for child " << cur_child << " ";
		tVector cur_force = -mReactionForce[cur_child];			// world frame
		tVector cur_force_pof = mReactionForce_pof[cur_child];	// world frame

		// world frame to this frame
		cur_force = world_to_local_rot * cur_force;
		cur_force_pof = world_to_local_trans * cur_force_pof;
		fout << "reaction force local = " << cur_force.transpose();
		fout << ", reaction force pof local = " << cur_force_pof.transpose();
		tVector cur_torque = cur_force_pof.cross3(cur_force);	// calculate the torque caused by these force
		fout << " torque = " << cur_torque.transpose() << std::endl;
		final_torque -= cur_torque;
	}

	// 5. get the reaction torque given by his children
	for (int i = 0; i < total_children.size(); i++)
	{
		int cur_child = total_children[i];
		tVector cur_torque = -mReactionTorque[cur_child];	// in world frame
		cur_torque = world_to_local_rot * cur_torque;
		fout << "for child " << cur_child << ", reaction torque local = " << cur_torque.transpose() << std::endl;
		final_torque -= cur_torque;
	}

	// 6. torque given by external forces
	tVectorArr contact_force, contact_pof;
	GetLinkContactInfo(frame_id, link_id, contact_force, contact_pof);
	for (int i = 0; i < contact_force.size(); i++)
	{
		tVector cur_contact_force = contact_force[i],
			cur_contact_force_pof = contact_pof[i];
		
		// computes torque
		cur_contact_force = world_to_local_rot * cur_contact_force;
		cur_contact_force_pof = world_to_local_trans * cur_contact_force_pof;
		final_torque -= cur_contact_force_pof.cross3(cur_contact_force);
	}

	tVector final_torque_world = local_to_world_rot * final_torque;
	mReactionTorque[link_id] = final_torque_world;
	fout << "final torque local = " << final_torque.transpose() << std::endl;
	fout << "final torque world = " << final_torque_world.transpose() << std::endl;
}

void cInverseDynamicsInfo::ComputePDTarget() const
{

}

void cInverseDynamicsInfo::PrintLinkInfo()
{
	/*
	Function: cInverseDynamicsInfo::PrintLinkInfo
	@params:

	Output the dynamic link info, from 0 order to 2nd order, including:
		linear: pos, vel, accel
		angular: displacement, vel, accel
	to log file.
	*/

	if (mSimChar == nullptr) return;

	ofstream fout(gIDLogPath_linkinfo);

	int num_bodies = mSimChar->GetNumBodyParts();
	for (int frame = 0; frame < 1; frame++)
	{
		double timestep = mLinkInfo->mTimesteps[frame];
		fout << "----------------frame " << frame << "----------------" << std::endl;
		fout << "timestep = " << timestep << std::endl;
		for (int body_id = 0; body_id < num_bodies; body_id++)
		{
			fout << "--------body " << body_id <<" " << mSimChar->GetBodyName(body_id) << " info--------" << std::endl;
			// 1. output 0 order info 
			{
				tVector link_pos = GetLinkPos(frame, body_id);
				tMatrix link_rot = GetLinkRotationMat(frame, body_id);
				tVector link_rot_quater = GetLinkRotation(frame, body_id);
				tMatrix link_trans = GetLinkTrans(frame, body_id);

				fout << "link_pos = " << link_pos.transpose() << std::endl;
				fout << "link_rotmat = \n " << link_rot << std::endl;
				fout << "link_rot_quater = " << link_rot_quater.transpose() << std::endl;
				fout << "link_trans = \n " << link_trans << std::endl;
			}


			// 2. 1 order info output
			if (frame < mNumOfFrames - 1)
			{
				tVector link_linear_vel = GetLinkVel(frame, body_id);
				tVector link_ang_vel = GetLinkAngularVel(frame, body_id);
				fout << "link_linear_vel = " << link_linear_vel.transpose() << std::endl;
				fout << "link_ang_vel = " << link_ang_vel.transpose() << std::endl;
			}

			// 3. 2 order info output
			if (frame < mNumOfFrames - 2)
			{
				tVector link_linear_accel = GetLinkAccel(frame, body_id);
				tVector link_ang_accel = GetLinkAngularAccel(frame, body_id);
				fout << "link_linear_accel = " << link_linear_accel.transpose() << std::endl;
				fout << "link_angular_accel = " << link_ang_accel.transpose() << std::endl;
			}


			//// 四元数1 + 角位移得到的四元数move = 四元数2, 验证成功
			//{
			//	tVector angular_dist;
			//	angular_dist.setZero();
			//	angular_dist.segment(0, 3) = (q_vel_euler * timestep);
			//	tQuaternion predicted_q_move = cMathUtil::EulerToQuaternion(angular_dist);

			//	tQuaternion predicted_q = predicted_q_move * q1;
			//	fout << "[debug] predicted time 1 quaternion by euler angle = " << predicted_q.coeffs().transpose() << std::endl;

			//}

			//// 角位移 + 四元数得到的角位移 = 角位移2，
			//{
			//	tQuaternion q_move = q2 * q1.conjugate();
			//	tVector euler_move = cMathUtil::QuaternionToEuler(q_move).segment(0, 3);
			//	tVector predicted_q = euler_move + q1_euler;
			//	fout << "[debug] predicted time 1 euler angle by quaternion = " << predicted_q.transpose() << std::endl;
			//}


			////fout << "[debug] time 0 quaternion vel = " << q_vel.coeffs().transpose() \
			////	<< " (" << 2 *cMathUtil::QuaternionToEuler(q_vel).transpose() \
			////	<< ", euler angles = " << q_vel_euler.transpose() << std::endl;
			//
			//fout << "[debug] time 0 + vel * timestep = " << (q1_euler + q_vel_euler * timestep).transpose() << std::endl;
			//fout << "[debug] time 1 = " << q2_euler.transpose() << std::endl;
			fout << "--------[debug] body " << body_id << " end----" << std::endl;
		}
	}
}

void cInverseDynamicsInfo::BuildTopoInfo()
{
	// computes the topological info about the skeleton
	int cur = 0;

	// 1. mJointChildren
	mJointChildren.resize(mNumLinks);
	for (int i = 0; i < mNumLinks; i++) mSimChar->GetChildJoint(i, mJointChildren[i]);

	// 2. mVisitSeq
	Eigen::VectorXd visit_seq;
	visit_seq = VectorXd::Zero(mNumLinks);
	bool * is_visited = new bool[mNumLinks];
	memset(is_visited, 0, sizeof(bool) * mNumLinks);
	is_visited[int(visit_seq[cur])] = true;
	visit_seq[cur++] = mSimChar->GetRootID();

	// from root joint, bfs the whole tree. 
	// why not begin the visit from leaf nodes? because it will be wrong.A link may have many children.
	VectorXd child_joint;
	while (true)
	{
		int seq_size = visit_seq.size();
		for (int i = 0; i < seq_size; i++)
		{
			int cur_joint = visit_seq[i];
			child_joint = mJointChildren[cur_joint];

			for (int j = 0; j < child_joint.size(); j++)
			{
				int cur_child_joint = child_joint[j];
				if (is_visited[cur_child_joint] == false)
				{
					visit_seq[cur++] = cur_child_joint;
					is_visited[cur_child_joint] = true;
				}
			}
		}
		if (seq_size == visit_seq.size()) break;
	}

	// reverse
	mVisitSeq = visit_seq.reverse();
	for (int i = 0; i < mVisitSeq.size(); i++)
	{
		int joint_id = mVisitSeq[i];
		std::cout << "[debug] visit " << joint_id << " " << mSimChar->GetBodyName(joint_id) << std::endl;
	}
}