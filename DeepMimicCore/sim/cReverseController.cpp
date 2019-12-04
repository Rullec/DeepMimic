#include "cReverseController.h"
#include <iostream>
#include <fstream>
#include <windows.h>

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);
void removeRow(tVectorXd& vec, unsigned int rowToRemove);

cReverseController::cReverseController(cSimCharacter * sim_char)
{
	mChar = sim_char;
	std::ofstream fout("logs/controller_logs/pd_target_debug.log");
	fout << std::endl;
	I.resize(mChar->GetNumDof(), mChar->GetNumDof());
	I.setIdentity();
	mEnableSolving = false;
}

void cReverseController::CalcPDTarget(const tVectorXd & input_torque, const tVectorXd &input_pose, const tVectorXd & input_cur_vel, tVectorXd & output_pd_target)
{
	// 1. calculate pose_next = input_pose + input_cur_vel
	tVectorXd pose_next;
	{
		tVectorXd pose_vel_quaternion;
		cKinTree::VelToPoseDiff(mChar->GetJointMat(), input_pose, input_cur_vel, pose_vel_quaternion);
		pose_next = input_pose + mTimestep * pose_vel_quaternion;
		cKinTree::PostProcessPose(mChar->GetJointMat(), pose_next);
		//std::cout << "input pose = " << input_pose.transpose() << std::endl;
		//std::cout << "next pose = " << pose_next.transpose() << std::endl;
	}

	// 2. calculate pos diff = PD_target - pose_next
	tVectorXd pose_diff;
	// get all joints' pose diff from torque, except root joint.
	// there is no active for ce on root joint, so resolve the root joint's pose_diff is impossible
	CalcPoseDiffFromTorque(input_torque, input_pose, input_cur_vel, pose_diff);
	//std::cout << "pose diff = " << pose_diff.transpose() << std::endl;
	// 3. integrate the pos diff then get the PD target, PD target = pose_next + diff
	{
		int num_joints = mChar->GetNumJoints();
		int num_dof = mChar->GetNumDof();
		const Eigen::MatrixXd & joint_mat = mChar->GetJointMat();
		assert(pose_diff.size() == num_dof);
		assert(pose_next.size() == num_dof);
		
		// clear all.
		output_pd_target.resize(num_dof);
		output_pd_target.setZero();

		for (int id = 1; id < num_joints; id++)
		{
			// for each joint except root, calculate pose_next = pose_cur + pose_diff
			int param_offset = cKinTree::GetParamOffset(joint_mat, id);
			int param_size = cKinTree::GetParamSize(joint_mat, id);

			cKinTree::eJointType type = cKinTree::GetJointType(joint_mat, id);
			switch (type)
			{
			case cKinTree::eJointType::eJointTypeSpherical:
			{
				tQuaternion q1 = cMathUtil::VecToQuat(pose_next.segment(param_offset, param_size));
				tQuaternion q_diff = cMathUtil::AxisAngleToQuaternion(pose_diff.segment(param_offset, param_size));
				tQuaternion q2 = q1 * q_diff;
				output_pd_target.segment(param_offset, param_size) = cMathUtil::QuatToVec(q2);
				break;
			}
			default:
				output_pd_target.segment(param_offset, param_size) = pose_next.segment(param_offset, param_size) + pose_diff.segment(param_offset, param_size);
				break;
			}
		}
	}
}

/*
	@Function: CalcPoseDiffFromTorque
		this function will calculate the "pose_err" in CalcControlForces by the control torque
	@params: input_torque, the torque we applied to this character, as the output of CalcControlForces
	@params: input_pose, the pose when we try to calculate torque in CalcControlForces
	@params: input_vel, the pose velocity we get the same as above
	@params: output_pos_diff, the pose diff between PD target and the reference pose, NOT THE INPUT POSE!
		ATTENTION FOR THE FINAL OUTPUT

*/
void cReverseController::CalcPoseDiffFromTorque(const tVectorXd & input_torque, const tVectorXd & input_pose, const tVectorXd & input_vel, tVectorXd & output_pos_diff)
{
	if (false == mEnableSolving)
	{
		std::cout << "[error] cReverseController::CalcPDTarget didn't prepare well, can not solve PD\n";
		exit(1);
	}

	// \tau = kp * (q_target - q_ref) + kd * (-timestep * q_accel)
	// 1. calculate q_ref = q_cur + timestep * q_vel
	const Eigen::VectorXd& cur_pose = input_pose;	// get current character's pose, quaternions for each joint
	const Eigen::VectorXd& cur_vel = input_vel;	// get char's vel: vel = d(pose)/ dt, the differential quaternions for each joints

	tVectorXd pose_ref;
	{
		Eigen::VectorXd quadternion_dot;
		const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
		// use pose & vel to calculate "pose_inc"
		cKinTree::VelToPoseDiff(joint_mat, cur_pose, cur_vel, quadternion_dot); // pose_inc = dqdt
		pose_ref = cur_pose + mTimestep * quadternion_dot;	// pose_cur + timestep * dqdt = pose_next (predicted)
		cKinTree::PostProcessPose(joint_mat, pose_ref);
	}

	// 2. calulte A and b

	M_s_inv = (M + mTimestep * Kd_dense).ldlt().solve(I);
	A = M_s_inv * Kp_mat;
	b = -M_s_inv * (Kd_mat * cur_vel + C);

	// 3. calculate E and f
	E = Kp_dense - mTimestep * Kd_mat * A;
	f = input_torque + Kd_mat * (cur_vel + mTimestep * b);

	// 4. arrange the matrix and solve final target
	// it can be more efficient if we delete some "zero" dofs, for example the first 7 nums  and each 4th number for spherial joints
	output_pos_diff = E.ldlt().solve(f);

//#define OUTPUT_LOG
#ifdef OUTPUT_LOG
	std::cout << "verbose log atterntion\n";
	std::ofstream fout("logs/controller_logs/pd_target_debug.log", std::ios::app);
	fout << "-----------------------------\n";
	fout << "Kp = \n" << Kp_mat.toDenseMatrix() << std::endl;
	fout << "Kd = \n" << Kd_mat.toDenseMatrix() << std::endl;
	fout << "cur pose = " << cur_pose.transpose() << std::endl;
	fout << "next pose = " << pose_ref.transpose() << std::endl;
	fout << "A = \n" << A << std::endl;
	fout << "b = " << b.transpose() << std::endl;
	fout << "E = \n" << E << std::endl;
	fout << "f = " << f.transpose() << std::endl;
	fout << "E_sub = \n" << E_sub << std::endl;
	fout << "f_sub = " << f_sub.transpose() << std::endl;
	fout << "Err = " << output_pd_target.transpose() << std::endl;
#endif
	
	mEnableSolving = false;
}

void cReverseController::CalcAction(const tVectorXd & input_torque, const tVector & input_cur_pose, const tVectorXd & input_cur_vel, tVector & output_action)
{
	tVectorXd pd_target;
	CalcPDTarget(input_torque, input_cur_pose, input_cur_vel, pd_target);

}

void cReverseController::SetParams(double timestep_, const Eigen::MatrixXd &M_, const Eigen::MatrixXd &C_, const Eigen::VectorXd & kp_, const Eigen::VectorXd & kd_)
{
	mTimestep = timestep_;
	M = M_;
	C = C_;
	
	Kp_mat = kp_.asDiagonal();
	Kp_dense = Kp_mat.toDenseMatrix();
	Kd_mat = kd_.asDiagonal();
	Kd_dense = Kd_mat.toDenseMatrix();
	mEnableSolving = true;
}


void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();

	if (rowToRemove < numRows)
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

	matrix.conservativeResize(numRows, numCols);
}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols() - 1;

	if (colToRemove < numCols)
		matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

	matrix.conservativeResize(numRows, numCols);
}

void removeRow(tVectorXd& vec, unsigned int rowToRemove)
{
	unsigned int numRows = vec.size() - 1;
	unsigned int numCols = 1;

	if (rowToRemove < numRows)
		vec.block(rowToRemove, 0, numRows - rowToRemove, numCols) = vec.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

	vec.conservativeResize(numRows, numCols);
}
