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

tVector cInverseDynamicsInfo::GetLinkRotation(int frame, int body_id)	// get Quaternion coeff 4*1 [x, y, z, w]
{
	// [x, y, z, w] quaternion
	return mLinkInfo->mLinkRot.block(frame, body_id * 4, 1, 4).transpose();
}

tVector cInverseDynamicsInfo::GetLinkAngularVel(int frame, int body_id)	// get link angular vel 4*1 [wx, wy, wz, 0]
{
	// [ax, ay, az, dtheta/dt] axis-angle angular velocity
	return mLinkInfo->mLinkAngularVel.block(frame, body_id * 4, 1, 4).transpose();
}

tVector cInverseDynamicsInfo::GetLinkAngularAccel(int frame, int body_id)	// get link angular accel 4*1 [ax, ay, az, 0]
{
	// [ax, ay, az, dtheta^2/dt^2] axis-angle angular velocity
	return mLinkInfo->mLinkAngularAccel.block(frame, body_id * 4, 1, 4).transpose();
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

	// TODO: output the result of angular velocity and visualization
	ofstream fout("quaternion.log");
	for (int frame = 0; frame < mNumOfFrames - 1; frame++)
	{
		for (int body_id = 0; body_id < num_bodies; body_id++)
		{
			// 1. get rotation and angular velocity
			tVector q1_quater_ultimate = GetLinkRotation(frame, body_id),
					q2_quater_ultimate = GetLinkRotation(frame + 1, body_id);

			double timestep = mLinkInfo->mTimesteps[frame];
			fout << "--------[debug] body " << body_id << "--------" << std::endl;
			fout << "[debug] timestep = " << timestep << std::endl;
			
			// 0 order info output
			{
				tVector coef = GetLinkRotation(frame, body_id);
				tQuaternion q1_quater_ultimate = tQuaternion(coef[3], coef[0], coef[1], coef[2]);
				tVector q1_axis_angle;
				q1_axis_angle.setZero();
				cMathUtil::QuaternionToAxisAngle(q1_quater_ultimate, q1_axis_angle, q1_axis_angle[3]);
				fout << "[debug] time 0 rotation quaterinion = " << coef.transpose()\
					<< ", axis angle = " << q1_axis_angle.transpose() << std::endl;
			}
			
			// 1 order info output
			{
				tVector q1_vel_ultimate = GetLinkAngularVel(frame, body_id);	
				fout << "[debug] time 0 angular velocity axis-angle = " << q1_vel_ultimate.transpose() << std::endl;

			}
			
			// 2 order info output
			{
				tVector q2_accel_ultimate = GetLinkAngularAccel(frame, body_id);
				q2_accel_ultimate *= q2_accel_ultimate[3];
				q2_accel_ultimate[3] = 0;
				fout << "[debug] time 0 angular accel ultimate = " << q2_accel_ultimate.segment(0, 3).transpose() << std::endl;
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
			//	Vector3d euler_move = cMathUtil::QuaternionToEuler(q_move).segment(0, 3);
			//	Vector3d predicted_q = euler_move + q1_euler;
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
	{
		Eigen::Vector3d cur_pos = mSimChar->GetBodyPartPos(body_id).segment(0, 3).transpose();
		mLinkInfo->mLinkPos.block(frame, body_id * 3, 1, 3) = cur_pos.transpose();
		Vector3d get_pos = GetLinkPos(frame, body_id);
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
	if (frame < 0) return;
	double timestep = mLinkInfo->mTimesteps[frame];
	// linear terms
	{
		Vector3d vel = (GetLinkPos(frame + 1, body_id) - GetLinkPos(frame, body_id)) / timestep;
		mLinkInfo->mLinkVel.block(frame, body_id * 3, 1, 3) = vel.transpose();
		Vector3d get_vel = GetLinkVel(frame, body_id);
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

		// normalize
		double mag = omega.norm();
		omega /= mag;
		omega[3] = mag;

		// normalized axis angle, format [ax, ay, az, theta]
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
		//btVector3 axis = omega_bt_quater.getAxis();
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
	if (frame < 0) return;
	double timestep = mLinkInfo->mTimesteps[frame];
	// linear terms
	{
		Vector3d accel = (GetLinkVel(frame + 1, body_id) - GetLinkVel(frame, body_id)) / timestep;
		mLinkInfo->mLinkAccel.block(frame, body_id * 3, 1, 3) = accel.transpose();
		Vector3d get_accel = GetLinkAccel(frame, body_id);
		if (JudgeSameVec(get_accel, accel) == false)
		{
			std::cout << "[error] mLinkAccel set error " << accel.transpose() << " " << get_accel.transpose() << std::endl;
			exit(1);
		}
	}

	// ultimate angular terms, computed from axis-angle to accel
	{
		
		tVector q1_vel_a = GetLinkAngularVel(frame, body_id), q2_vel_a = GetLinkAngularVel(frame + 1, body_id);
		double mag_1 = q1_vel_a[3], mag_2 = q2_vel_a[3];
		q1_vel_a *= mag_1, q2_vel_a *= mag_2;
		q1_vel_a[3] = 0, q2_vel_a[3] = 0;

		tVector q1_accel = (q2_vel_a - q1_vel_a) / timestep;
		double mag_accel = q1_accel.norm();
		mag_accel /= mag_accel;
		q1_accel[3] = mag_accel;

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