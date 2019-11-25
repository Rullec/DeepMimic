#include "anim/Motion.h"
#include "sim/SimInverseDynamics.h"

#include <iostream>
using namespace std;
using namespace Eigen;

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

	mSimChar = sim_char;
	mLinkInfo = (std::shared_ptr<cInverseDynamicsInfo::tLinkCOMInfo>)(new cInverseDynamicsInfo::tLinkCOMInfo());
}

/*
	Function: cInverseDynamicsInfo::AddNewFrame
		
	This function will append a new frame info into the storaged MatrixXd array

*/
void cInverseDynamicsInfo::AddNewFrame(const Eigen::VectorXd & state_, const Eigen::VectorXd & pose_, const Eigen::VectorXd & action_, const Eigen::VectorXd & contact_info_)
{
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
}

int cInverseDynamicsInfo::GetNumOfFrames()
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

double cInverseDynamicsInfo::GetMaxTime()
{
	double total_time = 0;
	for (int i = 0; i < mNumOfFrames; i++)
	{
		double duration = mPose(i, cMotion::eFrameParams::eFrameTime);
		total_time += duration;
	}
	return total_time;
}
Eigen::Vector3d cInverseDynamicsInfo::GetLinkPos(int frame, int body_id)
{
	if (frame >= mLinkInfo->mLinkPos.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetmLinkPos(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	Eigen::Vector3d pos = mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3).transpose();
	return pos;
}

Eigen::Vector3d cInverseDynamicsInfo::GetLinkVel(int frame, int body_id)
{
	if (frame >= mLinkInfo->mLinkVel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetmLinkVel(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	Eigen::Vector3d vel = mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3).transpose();
	return vel;
}

Eigen::Vector3d cInverseDynamicsInfo::GetLinkAccel(int frame, int body_id)
{
	if (frame >= mLinkInfo->mLinkAccel.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAccel(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}

	Eigen::Vector3d accel = mLinkInfo->mLinkAccel.block(frame, body_id * 3, 1, 3).transpose();
	return accel;
}

/*
	@Function: void cInverseDynamicsInfo

		Accoroding to the motion info in mState & mPos, this function will compute the linear 
	velocity & acceleration for the COM of each link in every frame.
		We supposed that the intergrator empoloies explicit Euler method.
*/
void cInverseDynamicsInfo::ComputeLinkInfo()
{
	auto JudgeSameVec = [](VectorXd a, VectorXd b, double eps = 1e-5)
	{
		return (a - b).norm() < eps;
	};

	auto JudgeSame = [](double a, double b, double eps = 1e-5)
	{
		return abs(a - b) < eps;
	};

	// check input 
	if (mNumOfFrames == -1 || mSimChar == nullptr || mLinkInfo == nullptr)
	{
		std::cout << "[error] cInverseDynamicsInfo::ComputeLinkInfo input illegal" << std::endl;
		exit(1);
	}

	// 1. save the status of sim_char. It wil be resumed in the end of this function
	Eigen::VectorXd pose_before = mSimChar->GetPose();

	// 2. get the linear position of each link in each frame
	/*
		mLinkInfo::mLinkPos = get_link_pose_from_SimCharacter;
		mLinkInfo::mTimesteps = mPos[i]
	*/

	// 2.1 resize
	int total_dims = mSimChar->GetNumBodyParts() * 3;
	mLinkInfo->mTimesteps.resize(mNumOfFrames);
	mLinkInfo->mLinkPos.resize(mNumOfFrames, total_dims);	// mLinkPos.size == (total num of frames, NumOfBodies * 3);
	mLinkInfo->mLinkPos.setZero();

	// 2.2 compute & set 0 order info(pose & timestep)
	for (int i = 0; i < mNumOfFrames; i++)
	{
		Eigen::VectorXd cur_pose = mPose.row(i).segment(1, mPoseSize - 1);
		double cur_timestep = mPose.row(i)[cMotion::eFrameTime];

		mLinkInfo->mTimesteps[i] = cur_timestep;
		mSimChar->SetPose(cur_pose);
		
		// get &set position info
		for (int body_id = 0; body_id < mSimChar->GetNumBodyParts(); body_id++)
		{
			Eigen::Vector3d cur_pos = mSimChar->GetBodyPartPos(body_id).segment(0, 3).transpose();
			//std::cout << "[debug] frame " << i << " body " << body_id << " pos = " << cur_pos.transpose() << std::endl;
			mLinkInfo->mLinkPos.block(i, body_id * 3, 1, 3) = cur_pos.transpose();
			//std::cout << "[debug] get link pos frame " << i << ", root pos = " << GetLinkPos(i, body_id).transpose() << std::endl;
			Vector3d get_pos = GetLinkPos(i, body_id);
			if (JudgeSameVec(get_pos, cur_pos) == false)
			{
				std::cout << "[error] mLinkPos set error" << cur_pos.transpose() <<" " << get_pos.transpose() << std::endl;
				exit(1);
			}
		}
		
		//std::cout << "[debug] frame " << i << ", total pos = " << mLinkInfo->mLinkPos.row(i) << std::endl;
		
		//exit(1);
	}
	//exit(1);

	// 3. compute the vel
	mLinkInfo->mLinkVel.resize(mNumOfFrames-1, total_dims);
	for (int i = 0; i < mNumOfFrames - 1; i++)
	{
		for (int body_id = 0; body_id < mSimChar->GetNumBodyParts(); body_id++)
		{
			Vector3d vel = (mLinkInfo->mLinkPos.block(i + 1, body_id * 3, 1, 3) - mLinkInfo->mLinkPos.block(i, body_id * 3, 1, 3)).transpose() / mLinkInfo->mTimesteps[i];
			mLinkInfo->mLinkVel.block(i, body_id * 3, 1, 3) = vel.transpose();
			Vector3d get_vel = GetLinkVel(i, body_id);
			if (JudgeSameVec(get_vel, vel) == false)
			{
				std::cout << "[error] mLinkVel set error: " << vel.transpose() << " " << get_vel.transpose() <<std::endl;
				exit(1);
			}
		}
	}

	// 4. compute the acceleration
	mLinkInfo->mLinkAccel.resize(mNumOfFrames - 2, total_dims);
	for (int i = 0; i < mNumOfFrames - 2; i++)
	{
		for (int body_id = 0; body_id < mSimChar->GetNumBodyParts(); body_id++)
		{
			Vector3d accel = (mLinkInfo->mLinkVel.block(i + 1, body_id * 3, 1, 3) -mLinkInfo->mLinkVel.block(i, body_id * 3, 1, 3)).transpose() / mLinkInfo->mTimesteps[i];
			mLinkInfo->mLinkAccel.block(i, body_id * 3, 1, 3) = accel.transpose();
			Vector3d get_accel = GetLinkAccel(i, body_id);
			if (JudgeSameVec(get_accel, accel) == false)
			{
				std::cout << "[error] mLinkAccel set error " << accel.transpose() <<" " << get_accel.transpose() << std::endl;
				exit(1);
			}
		}
	}

	auto &DIVIDE = [](Vector3d a, Vector3d b)->Vector3d
	{
		return Vector3d(a[0] / b[0], a[1] / b[1], a[2] / b[2]);
	};

	// 5. check the storage
	{
		// check velocity
		for (int frame = 0; frame < GetNumOfFrames() - 1; frame++)
		{
			double timestep = mLinkInfo->mTimesteps[frame];
			for (int body = 0; body < this->mSimChar->GetNumBodyParts(); body++)
			{
				Vector3d cur_vel = GetLinkVel(frame, body);
				Vector3d cur_pos = GetLinkPos(frame, body), next_pos = GetLinkPos(frame+1, body);
				Vector3d predicted_move = cur_vel.cwiseProduct(Vector3d::Ones() * timestep);
				Vector3d true_move = next_pos - cur_pos;
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
				Vector3d cur_accel = GetLinkAccel(frame, body);
				//std::cout << cur_accel.transpose() << std::endl;
				Vector3d cur_vel = GetLinkVel(frame, body), next_vel = GetLinkVel(frame + 1, body);
				//std::cout << cur_vel.transpose() <<" " << next_vel.transpose() << std::endl;
				Vector3d predicted_move = cur_accel.cwiseProduct(Vector3d::Ones() * timestep);
				//std::cout << timestep << std::endl;
				Vector3d true_move = next_vel - cur_vel;
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