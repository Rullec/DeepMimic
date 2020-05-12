#include "ImpPDController.h"
#include <iostream>

#include "sim/SimCharacter.h"
#include "sim/RBDUtil.h"

#include "util/cTimeUtil.hpp"
#include "util/FileUtil.h"
#include "cReverseController.h"

#include <fstream>

cImpPDController::cImpPDController()
{
	mExternRBDModel = true;
	mEnableSolvePDTargetTest = false;

#if defined(IMP_PD_CTRL_PROFILER)
	mPerfSolveTime = 0;
	mPerfTotalTime = 0;
	mPerfSolveCount = 0;
	mPerfTotalCount = 0;
#endif // IMP_PD_CTRL_PROFILER
}

cImpPDController::~cImpPDController()
{
}

void cImpPDController::Init(cSimCharacter* character, const Eigen::MatrixXd& pd_params, const tVector& gravity)
{
	std::shared_ptr<cRBDModel> model = BuildRBDModel(*character, gravity);
	Init(character, model, pd_params, gravity);
	mExternRBDModel = false;

	// init reverse controller
	mPDTargetSolver = std::make_shared<cReverseController>(character);
}

void cImpPDController::Init(cSimCharacter* character, const std::shared_ptr<cRBDModel>& model, const Eigen::MatrixXd& pd_params, const tVector& gravity)
{
	cExpPDController::Init(character, pd_params);
	mGravity = gravity;
	mRBDModel = model;
	InitGains();
}

void cImpPDController::Clear()
{
	cExpPDController::Clear();
	mExternRBDModel = true;
	mRBDModel.reset();
}

void cImpPDController::UpdateControlForce(double time_step, Eigen::VectorXd& out_tau)
{
	cController::Update(time_step);

#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_BEG(Update_Ctrl_Force)
#endif

	if (time_step > 0)
	{
		if (!mExternRBDModel)
		{
			// 一般都是要更新这个RBDModel的
			UpdateRBDModel();
		}

		// 计算控制力
		CalcControlForces(time_step, out_tau);
	}

#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_END(Update_Ctrl_Force, mPerfTotalTime, mPerfTotalCount)
#endif

#if defined(IMP_PD_CTRL_PROFILER)
	printf("Solve Time: %.5f\n", mPerfSolveTime);
	printf("Total Time: %.5f\n", mPerfTotalTime);
#endif
}

void cImpPDController::SetKp(int joint_id, double kp)
{
	cExpPDController::SetKp(joint_id, kp);

	int param_offset = mChar->GetParamOffset(joint_id);
	int param_size = mChar->GetParamSize(joint_id);

	auto curr_kp = mKp.segment(param_offset, param_size);
	curr_kp.setOnes();
	curr_kp *= kp;
}

void cImpPDController::SetKd(int joint_id, double kd)
{
	cExpPDController::SetKd(joint_id, kd);

	int param_offset = mChar->GetParamOffset(joint_id);
	int param_size = mChar->GetParamSize(joint_id);

	auto curr_kd = mKd.segment(param_offset, param_size);
	curr_kd.setOnes();
	curr_kd *= kd;
}

void cImpPDController::InitGains()
{
	int num_dof = GetNumDof();
	mKp = Eigen::VectorXd::Zero(num_dof);
	mKd = Eigen::VectorXd::Zero(num_dof);

	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);
		if (pd_ctrl.IsValid())
		{
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);

			double kp = pd_ctrl.GetKp();
			double kd = pd_ctrl.GetKd();

			mKp.segment(param_offset, param_size) = Eigen::VectorXd::Ones(param_size) * kp;
			mKd.segment(param_offset, param_size) = Eigen::VectorXd::Ones(param_size) * kd;
		}
	}
}

std::shared_ptr<cRBDModel> cImpPDController::BuildRBDModel(const cSimCharacter& character, const tVector& gravity) const
{
	/*
		这个函数主要为character建立RBDModel，这个模型是用来计算质量矩阵、转动惯量之类的东西的;
	 */
	std::shared_ptr<cRBDModel> model = std::shared_ptr<cRBDModel>(new cRBDModel());
	// 计算指标的初始化需要JointMat和BodyDefs的共同作用
	model->Init(character.GetJointMat(), character.GetBodyDefs(), gravity);
	return model;
}

void cImpPDController::UpdateRBDModel()
{
	// RBDModel竟然还需要更新
	const Eigen::VectorXd& pose = mChar->GetPose();
	const Eigen::VectorXd& vel = mChar->GetVel();
	mRBDModel->Update(pose, vel);
}

void cImpPDController::CalcControlForces(double time_step, Eigen::VectorXd & out_tau)
{
	// #define OUTPUT_LOG_CONTROL_FORCE

#ifdef OUTPUT_LOG_CONTROL_FORCE
	std::cout << "verbose log here\n";
	std::ofstream fout("logs/controller_logs/control_force.log", std::ios::app);
#endif // OUTPUT_LOG_CONTROL_FORCE

	double timestep = time_step;

	const Eigen::VectorXd& pose = mChar->GetPose();	// get current character's pose, quaternions for each joint
	const Eigen::VectorXd& vel = mChar->GetVel();	// get char's vel: vel = d(pose)/ dt, the differential quaternions for each joints

	Eigen::VectorXd tar_pose;
	Eigen::VectorXd tar_vel;
	BuildTargetPose(tar_pose);
	BuildTargetVel(tar_vel);


	// construct mKp & mKd (kp & kd coeffs) to a diagonal mat
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat = mKp.asDiagonal();
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_mat = mKd.asDiagonal();
	Eigen::VectorXd mKp_v = mKp, mKd_v = mKd;
	
	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);

		// if this joint is invalid, clear it.
		// NO FORCE can be applied to these joints
		if (!pd_ctrl.IsValid() || !pd_ctrl.IsActive())
		{
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Kp_mat.diagonal().segment(param_offset, param_size).setZero();
			Kd_mat.diagonal().segment(param_offset, param_size).setZero();
			mKp_v.segment(param_offset, param_size).setZero();
			mKd_v.segment(param_offset, param_size).setZero();
		}
	}

	// Why we need mass matrix
	Eigen::MatrixXd M = mRBDModel->GetMassMat();
	//std::cout << "mass = " << M << std::endl;

#ifdef OUTPUT_LOG_CONTROL_FORCE
	{
		fout << "----------------------------\n";
		fout << "pose = " << pose.transpose() << " | size = " << pose.size() << std::endl;
		fout << "vel = " << vel.transpose() << " | size = " << vel.size() << std::endl;
		fout << "tar pose = " << tar_pose.transpose() << " | size = " << tar_pose.size() << std::endl;
		fout << "tar vel = " << tar_vel.transpose() << " | size = " << tar_vel.size() << std::endl;
		fout << "Kp = \n" << Kp_mat.toDenseMatrix() << std::endl;
		fout << "Kd = \n" << Kd_mat.toDenseMatrix() << std::endl;
		fout << "raw M = \n" << M << std::endl;
	}
#endif

	// similar to Carolios foce?
	const Eigen::VectorXd& C = mRBDModel->GetBiasForce();
	//std::cout << "bias force = " << C.transpose() << std::endl;
	M.diagonal() += timestep * mKd;

	{
#ifdef OUTPUT_LOG_CONTROL_FORCE
	
		fout << "C = \n " << C << std::endl;

#endif
		//std::cout << "after add mKd, M = " << M.transpose() << std::endl;

		// pose_inc = dqdt = q' = 0.5 * w * q, differential of quaternions (pose)
		Eigen::VectorXd pose_inc;
		const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
		// use pose & vel to calculate "pose_inc"
		cKinTree::VelToPoseDiff(joint_mat, pose, vel, pose_inc); // pose_inc = dqdt

#ifdef OUTPUT_LOG_CONTROL_FORCE
		{
			fout << "vel to pose diff = " << pose_inc.transpose() << std::endl;
		}
#endif

		pose_inc = pose + timestep * pose_inc;	// pose_cur + timestep * dqdt = pose_next (predicted)
		// std::cout <<"cImpPDController::CalcControlForces pose next inc = " << pose_inc.transpose() << std::endl;
		cKinTree::PostProcessPose(joint_mat, pose_inc);	// normalize, quaternions make sense
		Eigen::VectorXd pose_err;

		// pose_err: for spherical joints, in local frame
		cKinTree::CalcVel(joint_mat, pose_inc, tar_pose, 1, pose_err);
		Eigen::VectorXd vel_err = tar_vel - vel;
		Eigen::VectorXd acc = Kp_mat * pose_err + Kd_mat * vel_err - C;

#ifdef OUTPUT_LOG_CONTROL_FORCE
		{
			fout << "next pose = " << pose_inc.transpose() << std::endl;
			fout << "pose err = " << pose_err.transpose() << std::endl;
			fout << "vel err = " << vel_err.transpose() << std::endl;
			fout << "Q = " << acc.transpose() << std::endl;
		}
#endif

#if defined(IMP_PD_CTRL_PROFILER)
		TIMER_RECORD_BEG(Solve)
#endif

			acc = M.ldlt().solve(acc);
#if defined(IMP_PD_CTRL_PROFILER)
		TIMER_RECORD_END(Solve, mPerfSolveTime, mPerfSolveCount)
#endif

	// final formular: tau(torque) = kp * pose_err + kd (val - t*acc)
	out_tau += Kp_mat * pose_err + Kd_mat * (vel_err - timestep * acc);

#ifdef OUTPUT_LOG_CONTROL_FORCE
		{
			fout << "acc = " << acc.transpose() << std::endl;
			fout << "tau = " << out_tau.transpose() << std::endl;
		}
#endif
		// begin to solve PD target
		if(true == mEnableSolvePDTargetTest)
		{
			cTimeUtil::Begin("solve PD");
			// check velocity
			{
				tVectorXd target_vel;
				for (int i = 0; i < GetNumJoints(); i++)
				{
					const cPDController & pd_ctrl = GetPDCtrl(i);

					if (true == pd_ctrl.IsValid())
					{
						pd_ctrl.GetTargetVel(target_vel);
						assert(target_vel.norm() < 1e-10);
					}
				}
			}

			// begin to solve pd target
			tVectorXd solved_pd_target;
			this->SolvePDTargetByTorque(timestep, pose, vel, out_tau, solved_pd_target);
			//std::cout << "truth pose = " << tar_pose.transpose() << std::endl;
			//std::cout << "solved pose = " << wait_pd_target.transpose() << std::endl;
			tVectorXd diff = (tar_pose - solved_pd_target);
			for (int i = 0; i < diff.size(); i++)
			{
				if (std::abs(diff[i]) < 1e-10 || std::abs((tar_pose[i] + solved_pd_target[i])) < 1e-7) diff[i] = 0;
			}
			if (diff.segment(7, diff.size() - 7).norm() > 1e-8)
			{
				std::cout << "[warning] cImpPDController::CalcControlForces: truth target pose = " << tar_pose.transpose() << std::endl;
				std::cout << "[warning] cImpPDController::CalcControlForces: solve target pose = " << solved_pd_target.transpose() << std::endl;
				std::cout << "\n[warning] cImpPDController::CalcControlForces: solve pd target diff = " << diff.transpose() << std::endl;
				// exit(1);
			}
			std::cout <<"[log] cImpPDController solve PD Target accurately\n";
			// std::cout <<"[log] cImpPDController solved PD target = " << solved_pd_target.transpose() << std::endl; 
			// std::cout <<"[log] cImpPDController truth PD target = " << solved_pd_target.transpose() << std::endl; 
			cTimeUtil::End("solve PD");
		}
	}
	// exit(1);
}

void cImpPDController::BuildTargetPose(tVectorXd & out_pose) const
{
	out_pose = tVectorXd::Zero(GetNumDof());

	Eigen::VectorXd cur_pose;
	for (int i = 0; i < GetNumJoints(); i++)
	{
		const cPDController & pd_ctrl = GetPDCtrl(i);

		if (true == pd_ctrl.IsValid())
		{
			pd_ctrl.GetTargetTheta(cur_pose);
			int param_offset = mChar->GetParamOffset(i);
			int param_size = mChar->GetParamSize(i);
			out_pose.segment(param_offset, param_size) = cur_pose;
		}
	}
}

void cImpPDController::BuildTargetVel(tVectorXd & out_vel) const
{
	out_vel = tVectorXd::Zero(GetNumDof());

	Eigen::VectorXd cur_vel;
	for (int i = 0; i < GetNumJoints(); i++)
	{
		const cPDController & pd_ctrl = GetPDCtrl(i);

		if (true == pd_ctrl.IsValid())
		{
			pd_ctrl.GetTargetVel(cur_vel);
			int param_offset = mChar->GetParamOffset(i);
			int param_size = mChar->GetParamSize(i);
			out_vel.segment(param_offset, param_size) = cur_vel;
		}
	}
}

void cImpPDController::SetEnableSolvePDTargetTest(bool enable)
{
	std::cout <<"[log] cImpPDController::SetEnableSolvePDTargetTest = " << enable << std::endl;
	mEnableSolvePDTargetTest = enable;
}


/**
 * \brief SolvePDTargetByTorque
 * \param timestep for current frame
 * \param char_pose, the pose of character in this time
 * \param char_vel, the pose vel of character
 * \param torque, torque in each joints
 * \param PDTarget, solved PDTarget result
*/
void cImpPDController::SolvePDTargetByTorque(double timestep, const tVectorXd & char_pose,
		const tVectorXd & char_vel,const tVectorXd & torque, tVectorXd & PDTarget)
{
	Eigen::VectorXd mKp_v = mKp, mKd_v = mKd;
	
	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);

		// if this joint is invalid, clear it.
		// NO FORCE can be applied to these joints
		if (!pd_ctrl.IsValid() || !pd_ctrl.IsActive())
		{
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			mKp_v.segment(param_offset, param_size).setZero();
			mKd_v.segment(param_offset, param_size).setZero();
		}
	}
	mChar->SetPose(char_pose);
	mChar->SetVel(char_vel)	;
	UpdateRBDModel();

	mPDTargetSolver->SetParams(timestep, mRBDModel->GetMassMat(), mRBDModel->GetBiasForce(), mKp_v, mKd_v);
	//std::cout << "input pose = " << pose.transpose() << std::endl;
	//std::cout << "next pose = " << pose_inc.transpose() << std::endl;
	//std::cout << "pose diff = " << pose_err.transpose() << std::endl;
	mPDTargetSolver->CalcPDTarget(torque, char_pose, char_vel, PDTarget);
}