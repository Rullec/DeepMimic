﻿#include "ImpPDController.h"
#include <iostream>

#include "sim/SimCharacter.h"
#include "sim/RBDUtil.h"

#include "util/FileUtil.h"

cImpPDController::cImpPDController()
{
	mExternRBDModel = true;

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
	// 初始化的时候就要建立RBDModel...
	// 给定角色和重力之后就可以建立了, 这个Init只是一个wrapper
	std::shared_ptr<cRBDModel> model = BuildRBDModel(*character, gravity);
	Init(character, model, pd_params, gravity);
	mExternRBDModel = false;
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

void cImpPDController::CalcControlForces(double time_step, Eigen::VectorXd& out_tau)
{
	/*
		感觉很关键，究竟是如何计算控制力的?
	 */
	double t = time_step;

	const Eigen::VectorXd& pose = mChar->GetPose();
	const Eigen::VectorXd& vel = mChar->GetVel();
	Eigen::VectorXd tar_pose;
	Eigen::VectorXd tar_vel;
	BuildTargetPose(tar_pose);	// 在每个joint对应的controler中把目标的position　拿到，放进tar_pose里面
	BuildTargetVel(tar_vel);	// velocity拿到，放到tar_vel里

	// 把mKp 和 mKd两个向量(因为每个joint a pd controller)变成对角阵
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat = mKp.asDiagonal();
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_mat = mKd.asDiagonal();

	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);
		if (!pd_ctrl.IsValid() || !pd_ctrl.IsActive())
		{
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			Kp_mat.diagonal().segment(param_offset, param_size).setZero();
			Kd_mat.diagonal().segment(param_offset, param_size).setZero();
		}
	}

	// 这个里面也是需要动力学的，simulation怎么可能没有动力学呢?
	// 计算torque的时候，所有的参数都是从这个RBDModel中拿到的, 包括:
	/*
		质量阵
		疑似科氏力的"BiasForce"

	 */
	Eigen::MatrixXd M = mRBDModel->GetMassMat();
	const Eigen::VectorXd& C = mRBDModel->GetBiasForce();
	M.diagonal() += t * mKd;

	Eigen::VectorXd pose_inc;
	const Eigen::MatrixXd& joint_mat = mChar->GetJointMat();
	cKinTree::VelToPoseDiff(joint_mat, pose, vel, pose_inc);

	pose_inc = pose + t * pose_inc;
	cKinTree::PostProcessPose(joint_mat, pose_inc);
	// 这里的pos是纯粹的四元数了(因为后处理中总是在做归一化)

	Eigen::VectorXd pose_err;
	cKinTree::CalcVel(joint_mat, pose_inc, tar_pose, 1, pose_err);
	Eigen::VectorXd vel_err = tar_vel - vel;

	// 求加速度acc为什么要减去C?说不清楚
	// 但究竟bias force是什么，还是说不清
	Eigen::VectorXd acc = Kp_mat * pose_err + Kd_mat * vel_err - C;
	
#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_BEG(Solve)
#endif

	//int root_size = cKinTree::gRootDim;
	//int num_act_dofs = static_cast<int>(acc.size()) - root_size;
	//auto M_act = M.block(root_size, root_size, num_act_dofs, num_act_dofs);
	//auto acc_act = acc.segment(root_size, num_act_dofs);
	//acc_act = M_act.ldlt().solve(acc_act);
	
	acc = M.ldlt().solve(acc);

#if defined(IMP_PD_CTRL_PROFILER)
	TIMER_RECORD_END(Solve, mPerfSolveTime, mPerfSolveCount)
#endif
	
	// 然后就可以计算出tau了，最后的公式 tau(torque) = kp * pose_err + kd (val - t*acc)
	out_tau += Kp_mat * pose_err + Kd_mat * (vel_err - t * acc);
}

void cImpPDController::BuildTargetPose(Eigen::VectorXd& out_pose) const
{
	// 传入的out_pose就是外面的"target pose"在这个函数里需要赋值好
	out_pose = Eigen::VectorXd::Zero(GetNumDof());

	//const auto& joint_mat = mChar->GetJointMat();
	//tVector root_pos = mChar->GetRootPos();
	//tQuaternion root_rot = mChar->GetRootRotation();
	//cKinTree::SetRootPos(joint_mat, root_pos, out_pose);
	//cKinTree::SetRootRot(joint_mat, root_rot, out_pose);

	for (int j = 0; j < GetNumJoints(); ++j)
	{
		// 每个关节上都有一个pd controller，设置的参数都是不一样的
		// 拿到以后，把pd controller里面存储的理想目标拿出来
		// 存到out_pose &里面，就完成功能了
		const cPDController& pd_ctrl = GetPDCtrl(j);
		if (pd_ctrl.IsValid())
		{
			Eigen::VectorXd curr_pose;
			pd_ctrl.GetTargetTheta(curr_pose);
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			out_pose.segment(param_offset, param_size) = curr_pose;
		}
	}
}

void cImpPDController::BuildTargetVel(Eigen::VectorXd& out_vel) const
{
	out_vel = Eigen::VectorXd::Zero(GetNumDof());

	//const auto& joint_mat = mChar->GetJointMat();
	//tVector root_vel = mChar->GetRootVel();
	//tVector root_ang_vel = mChar->GetRootAngVel();
	//cKinTree::SetRootVel(joint_mat, root_vel, out_vel);
	//cKinTree::SetRootAngVel(joint_mat, root_ang_vel, out_vel);
	
	for (int j = 0; j < GetNumJoints(); ++j)
	{
		const cPDController& pd_ctrl = GetPDCtrl(j);
		if (pd_ctrl.IsValid())
		{
			Eigen::VectorXd curr_vel;
			pd_ctrl.GetTargetVel(curr_vel);
			int param_offset = mChar->GetParamOffset(j);
			int param_size = mChar->GetParamSize(j);
			out_vel.segment(param_offset, param_size) = curr_vel;
		}
	}
}