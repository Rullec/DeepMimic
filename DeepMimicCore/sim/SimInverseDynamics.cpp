#include "anim/Motion.h"
#include "sim/SimInverseDynamics.h"

#include <iostream>
using namespace std;
using namespace Eigen;

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

tQuaternion cInverseDynamicsInfo::GetLinkJointAngle_quaternion(int frame, int body_id)
{
	const auto & target = mLinkInfo->mAngularQuaternionInfo.mLinkJointAngle;
	if (frame >= target.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkJointAngle_quaternion(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}

	tVector q_vec = target.block(frame, body_id * 4, 1, 4).transpose();
	return cMathUtil::CoefVectorToQuaternion(q_vec);
}

tQuaternion cInverseDynamicsInfo::GetLinkAngularVel_quaternion(int frame, int body_id)
{
	const auto & target = mLinkInfo->mAngularQuaternionInfo.mLinkAngularVel;
	if (frame >= target.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAngularVel_quaternion(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	tVector q_vec = target.block(frame, body_id * 4, 1, 4).transpose();
	return cMathUtil::CoefVectorToQuaternion(q_vec);
}

tQuaternion cInverseDynamicsInfo::GetLinkAngularAccel_quaternion(int frame, int body_id)
{
	const auto & target = mLinkInfo->mAngularQuaternionInfo.mLinkAngularAccel;
	if (frame >= target.rows())
	{
		std::cout << "[error] cInverseDynamicsInfo::GetLinkAngularAccel_quaternion(int frame, int body_id) illegal access" << std::endl;
		exit(1);
	}
	tVector q_vec = target.block(frame, body_id * 4, 1, 4).transpose();
	return cMathUtil::CoefVectorToQuaternion(q_vec);
}

Eigen::Vector3d cInverseDynamicsInfo::GetLinkJointAngle_euler(int frame, int body_id)
{
	Vector3d euler = mLinkInfo->mAngularEulerInfo.mLinkJointAngle.block(frame, body_id * 3, 1, 3).transpose();
	return euler;
}

Eigen::Vector3d cInverseDynamicsInfo::GetLinkAngularVel_euler(int frame, int body_id)
{
	Vector3d euler = mLinkInfo->mAngularEulerInfo.mLinkAngularVel.block(frame, body_id * 3, 1, 3).transpose();
	return euler;
}

Eigen::Vector3d cInverseDynamicsInfo::GetLinkAngularAccel_euler(int frame, int body_id)
{
	Vector3d euler = mLinkInfo->mAngularEulerInfo.mLinkAngularAccel.block(frame, body_id * 3, 1, 3).transpose();
	return euler;
}

/*
	@Function: void cInverseDynamicsInfo

		Accoroding to the motion info in mState & mPos, this function will compute the linear 
	velocity & acceleration for the COM of each link in every frame.
		We supposed that the intergrator empoloies explicit Euler method.
*/
void cInverseDynamicsInfo::ComputeLinkInfo()
{
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

		// angular infos - quaternion format
		mLinkInfo->mAngularQuaternionInfo.mLinkJointAngle.resize(mNumOfFrames, num_bodies * 4);
		mLinkInfo->mAngularQuaternionInfo.mLinkAngularVel.resize(mNumOfFrames - 1, num_bodies * 4);
		mLinkInfo->mAngularQuaternionInfo.mLinkAngularAccel.resize(mNumOfFrames - 2, num_bodies * 4);

		// angular infos - euler angles format
		mLinkInfo->mAngularEulerInfo.mLinkJointAngle.resize(mNumOfFrames, num_bodies * 3);
		mLinkInfo->mAngularEulerInfo.mLinkAngularVel.resize(mNumOfFrames - 1, num_bodies * 3);
		mLinkInfo->mAngularEulerInfo.mLinkAngularAccel.resize(mNumOfFrames - 2, num_bodies * 3);
		
	}
	

	for (int i = 0; i < mNumOfFrames; i++)
	{
		Eigen::VectorXd cur_pose = mPose.row(i).segment(1, mPoseSize - 1);
		double cur_timestep = mPose.row(i)[cMotion::eFrameTime];

		mLinkInfo->mTimesteps[i] = cur_timestep;
		mSimChar->SetPose(cur_pose);

		if (i == 2)
			i = 2;
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

	// TODO: output the result of angular velocity and visualization
	ofstream fout("quaternion.log");
	for (int frame = 0; frame < 1; frame++)
	{
		for (int body_id = 0; body_id < num_bodies; body_id++)
		{
			// 1. get rotation and angular velocity
			tQuaternion q1 = GetLinkJointAngle_quaternion(frame, body_id),
					q2 = GetLinkJointAngle_quaternion(frame + 1, body_id),
					q_vel = GetLinkAngularVel_quaternion(frame, body_id);
			
			Vector3d q1_euler = GetLinkJointAngle_euler(frame, body_id),
				q2_euler = GetLinkJointAngle_euler(frame + 1, body_id),
				q_vel_euler = GetLinkAngularVel_euler(frame, body_id);

			double timestep = mLinkInfo->mTimesteps[frame];
			fout << "--------[debug] body " << body_id << "--------" << std::endl;
			fout << "[debug] timestep = " << timestep << std::endl;
			
			fout << "[debug] time 0 quaternion = " << q1.coeffs().transpose()\
				<<" ("<< cMathUtil::QuaternionToEuler(q1).transpose() \
				<< "), euler angles = " << q1_euler.transpose() << std::endl;
			
			fout << "[debug] time 1 quaternion = " << q2.coeffs().transpose() \
				<<" (" << cMathUtil::QuaternionToEuler(q2).transpose() \
				<< ", euler angles = " << q2_euler.transpose() << std::endl;

			// 四元数1 + 角位移得到的四元数move = 四元数2, 验证成功
			{
				tVector angular_dist;
				angular_dist.setZero();
				angular_dist.segment(0, 3) = (q_vel_euler * timestep);
				tQuaternion predicted_q_move = cMathUtil::EulerToQuaternion(angular_dist);

				tQuaternion predicted_q = predicted_q_move * q1;
				fout << "[debug] predicted time 1 quaternion by euler angle = " << predicted_q.coeffs().transpose() << std::endl;

			}

			// 角位移 + 四元数得到的角位移 = 角位移2，
			{
				tQuaternion q_move = q2 * q1.conjugate();
				Vector3d euler_move = cMathUtil::QuaternionToEuler(q_move).segment(0, 3);
				Vector3d predicted_q = euler_move + q1_euler;
				fout << "[debug] predicted time 1 euler angle by quaternion = " << predicted_q.transpose() << std::endl;
			}
			fout << "[debug] time 0 quaternion vel = " << q_vel.coeffs().transpose() \
				<< " (" << 2 *cMathUtil::QuaternionToEuler(q_vel).transpose() \
				<< ", euler angles = " << q_vel_euler.transpose() << std::endl;
			
			fout << "[debug] time 0 + vel * timestep = " << (q1_euler + q_vel_euler * timestep).transpose() << std::endl;
			fout << "[debug] time 1 = " << q2_euler.transpose() << std::endl;
			fout << "--------[debug] body " << body_id << " end----" << std::endl;
		}
	}
	exit(1);
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
				Vector3d cur_pos = GetLinkPos(frame, body), next_pos = GetLinkPos(frame + 1, body);
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

void cInverseDynamicsInfo::ComputeLinkInfo0(int frame, int body_id)
{
	if (frame < 0) return;

	// linear terms
	Eigen::Vector3d cur_pos = mSimChar->GetBodyPartPos(body_id).segment(0, 3).transpose();
	mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3) = cur_pos.transpose();
	Vector3d get_pos = GetLinkPos(frame, body_id);
	if (JudgeSameVec(get_pos, cur_pos) == false)
	{
		std::cout << "[error] mLinkPos set error" << cur_pos.transpose() << " " << get_pos.transpose() << std::endl;
		exit(1);
	}

	// angular terms - quaternion parts
	{
		auto &cur_part = mSimChar->GetBodyPart(body_id);
		tQuaternion quater = cur_part->GetRotation();
		tVector coef = quater.coeffs();// x, y, z, w
		mLinkInfo->mAngularQuaternionInfo.mLinkJointAngle.block(frame, body_id * 4, 1, 4) = coef.transpose();
		tVector get_joint_angle = GetLinkJointAngle_quaternion(frame, body_id).coeffs();
		if (JudgeSameVec(get_joint_angle, coef) == false)
		{
			std::cout << "[error] mLinkJointAngle set error: ideal = " << coef.transpose() << ", get = " << get_joint_angle.transpose() << std::endl;
			exit(1);
		}
	}
	
	// angular terms - euler angles parts
	{
		auto &cur_part = mSimChar->GetBodyPart(body_id);
		tQuaternion quater = cur_part->GetRotation();
		tVector euler = cMathUtil::QuaternionToEuler(quater);
		Vector3d coef = euler.segment(0, 3);
		mLinkInfo->mAngularEulerInfo.mLinkJointAngle.block(frame, body_id * 3, 1, 3) = coef.transpose();
		Vector3d get_joint_angle = GetLinkJointAngle_euler(frame, body_id);
		if (JudgeSameVec(get_joint_angle, coef) == false)
		{
			std::cout << "[error] mLinkJointAngle set error: ideal = " << coef.transpose() << ", get = " << get_joint_angle.transpose() << std::endl;
			exit(1);
		}
	}
}

void cInverseDynamicsInfo::ComputeLinkInfo1(int frame, int body_id)
{
	if (frame < 0) return;

	// linear terms
	Vector3d vel = (mLinkInfo->mLinkPos.block(frame + 1, body_id * 3, 1, 3) - mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3)).transpose() / mLinkInfo->mTimesteps[frame];
	mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3) = vel.transpose();
	Vector3d get_vel = GetLinkVel(frame, body_id);
	if (JudgeSameVec(get_vel, vel) == false)
	{
		std::cout << "[error] mLinkVel set error: " << vel.transpose() << " " << get_vel.transpose() << std::endl;
		exit(1);
	}

	// angular terms - quaternion parts
	{
		tQuaternion q1 = GetLinkJointAngle_quaternion(frame, body_id), q2 = GetLinkJointAngle_quaternion(frame + 1, body_id);

		double timestep = mLinkInfo->mTimesteps[frame];
		tVector q_vel_axisangle = cMathUtil::CalcQuaternionVel(q1, q2, timestep);	// 返回轴角[theta * ax, theta * ay, theta *az, 0];
		double q_vel_magnitude = q_vel_axisangle.norm();
		tVector q_vel_axis = q_vel_axisangle / q_vel_magnitude;
		tQuaternion q_vel = cMathUtil::AxisAngleToQuaternion(q_vel_axis, q_vel_magnitude);

		mLinkInfo->mAngularQuaternionInfo.mLinkAngularVel.block(frame, body_id * 4, 1, 4)
			= q_vel.coeffs().transpose();

		// check get result 
		tVector coef = q_vel.coeffs();
		tVector get_ang_vel = GetLinkAngularVel_quaternion(frame, body_id).coeffs();
		if (JudgeSameVec(get_ang_vel, coef) == false)
		{
			std::cout << "[error] mLinkAngularVel set error: ideal = " << coef.transpose() << ", get = " << get_ang_vel.transpose() << std::endl;
			exit(1);
		}
	}
	
	// angular terms - euler angle parts
	{
		double timestep = mLinkInfo->mTimesteps[frame];
		Vector3d q1 = GetLinkJointAngle_euler(frame, body_id), q2 = GetLinkJointAngle_euler(frame + 1, body_id);
		Vector3d q_vel = (q2 - q1) / timestep;
		mLinkInfo->mAngularEulerInfo.mLinkAngularVel.block(frame, body_id * 3, 1, 3) = q_vel.transpose();

		Vector3d get_ang_vel = GetLinkAngularVel_euler(frame, body_id);
		if (JudgeSameVec(get_ang_vel, q_vel) == false)
		{
			std::cout << "[error] mLinkAngularVel set error: ideal = " << q_vel.transpose() << ", get = " << get_ang_vel.transpose() << std::endl;
			exit(1);
		}
	}
}

void cInverseDynamicsInfo::ComputeLinkInfo2(int frame, int body_id)
{
	if (frame < 0) return;

	// linear terms
	Vector3d accel = (mLinkInfo->mLinkVel.block(frame + 1, body_id * 3, 1, 3) - mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3)).transpose() / mLinkInfo->mTimesteps[frame];
	mLinkInfo->mLinkAccel.block(frame, body_id * 3, 1, 3) = accel.transpose();
	Vector3d get_accel = GetLinkAccel(frame, body_id);
	if (JudgeSameVec(get_accel, accel) == false)
	{
		std::cout << "[error] mLinkAccel set error " << accel.transpose() << " " << get_accel.transpose() << std::endl;
		exit(1);
	}

	// angular terms - quaternion part
	{
		tQuaternion q1 = GetLinkAngularVel_quaternion(frame, body_id), q2 = GetLinkAngularVel_quaternion(frame + 1, body_id);

		double timestep = mLinkInfo->mTimesteps[frame];
		tVector q_accel_axisangle = cMathUtil::CalcQuaternionVel(q1, q2, timestep);	// 返回轴角
		double q_accel_magnitude = q_accel_axisangle.norm();
		tVector q_accel_axis = q_accel_axisangle / q_accel_axisangle.norm();
		tQuaternion q_accel = cMathUtil::AxisAngleToQuaternion(q_accel_axis, q_accel_magnitude);

		mLinkInfo->mAngularQuaternionInfo.mLinkAngularAccel.block(frame, body_id * 4, 1, 4)
			= q_accel.coeffs().transpose();


		// check get result 
		tVector coef = q_accel.coeffs();
		tVector get_ang_accel = GetLinkAngularAccel_quaternion(frame, body_id).coeffs();
		if (JudgeSameVec(get_ang_accel, coef) == false)
		{
			std::cout << "[error] mLinkAngularAccel set error: ideal = " << coef.transpose() << ", get = " << get_ang_accel.transpose() << std::endl;
			exit(1);
		}
	}

	// angular terms - euler part
	{
		double timestep = mLinkInfo->mTimesteps[frame];
		Vector3d q_vel1 = GetLinkAngularVel_euler(frame, body_id), q_vel2 = GetLinkAngularVel_euler(frame + 1, body_id);
		Vector3d q_accel = (q_vel2 - q_vel1) / timestep;
		mLinkInfo->mAngularEulerInfo.mLinkAngularAccel.block(frame, body_id * 3, 1, 3) = q_accel.transpose();

		Vector3d get_accel = GetLinkAngularAccel_euler(frame, body_id);
		if (JudgeSameVec(q_accel, get_accel) == false)
		{
			std::cout << "[error] mLinkAngularAccel euler set error: ideal = " << q_accel.transpose() << ", get = " << get_accel.transpose() << std::endl;
			exit(1);
		}
	}
}