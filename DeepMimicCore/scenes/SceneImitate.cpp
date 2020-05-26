﻿#include "SceneImitate.h"
#include "sim/RBDUtil.h"
#include "sim/CtController.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

// std::string gRewardInfopath;
cSceneImitate::RewardParams::RewardParams()
{
	pose_w = 0;
	vel_w = 0;
	end_eff_w = 0;
	root_w = 0;
	com_w = 0;

	// scale params
	pose_scale = 0;
	vel_scale = 0;
	end_eff_scale = 0;
	root_scale = 0;
	com_scale = 0;
	err_scale = 0;

	// root sub reward weight (under the jurisdiction of root_w)
	root_pos_w = 0;
	root_rot_w = 0;
	root_vel_w = 0;
	root_angle_vel_w = 0;
}

void cSceneImitate::DiffLogOutput(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	using namespace Eigen;
	if (mAngleDiffDir.size() == 0 || mAngleDiffDir[mAngleDiffDir.size() - 1] != '/')
	{
		std::cout << "the dir = " << mAngleDiffDir << ", invalid" << std::endl;
	}
	
	const Eigen::VectorXd& pose0 = sim_char.GetPose(), vel0 = sim_char.GetVel(), pose1 = kin_char.GetPose(), vel1 = kin_char.GetVel();
	const auto& joint_mat = sim_char.GetJointMat();
	const auto& body_defs = sim_char.GetBodyDefs();
	const int num_joints = sim_char.GetNumJoints();
	ofstream fout;
	for (int i = 0; i < num_joints; i++)
	{
		const int offset = cKinTree::GetParamOffset(joint_mat, i);
		const int size = cKinTree::GetParamSize(joint_mat, i);
		VectorXd cur_pose = pose0.segment(offset, size);
		VectorXd motion_pose = pose1.segment(offset, size);
		VectorXd cur_vel = vel0.segment(offset, size);
		VectorXd motion_vel = vel1.segment(offset, size);
		string filename = this->mAngleDiffDir + std::to_string(i) + ".txt";
		fout.open(filename, std::ios_base::app);
		if (fout.fail() == true)
		{
			std::cout << "[angle diff log] open " << filename << " failed" << std::endl;
			abort();
		}

		// record it
		fout << "time " << this->GetTime() << ", joint " << i << ", cur pose = " << cur_pose.transpose() << ", motion pose = " << motion_pose.transpose() << std::endl;
		fout << "time " << this->GetTime() << ", joint " << i << ", cur vel = " << cur_vel.transpose() << ", motion vel = " << motion_vel.transpose() << std::endl;
		//std::cout << "joint " << i << " cur_pose = " << cur_pose.transpose() << ", motion pose = " << motion_pose.transpose() << std::endl;

		fout.close();
	}
	
}

double cSceneImitate::CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	// print
//	 std::cout << "compute reward, angle diff = " << mEnableAngleDiffLog <<", dir = " << mAngleDiffDir <<std::endl;
	if (mEnableAngleDiffLog == true)	DiffLogOutput(sim_char, kin_char);

	// get current run time from kin_char
	//double time = kin_char.GetTime();
	//std::cout << "time = " << time << std::endl;
	
	// reward共计5项，pose, vel, end_effector, root, com
	// 五项权重: 
	double pose_w = RewParams.pose_w;
	double vel_w = RewParams.vel_w;
	double end_eff_w = RewParams.end_eff_w;
	double root_w = RewParams.root_w;
	double com_w = RewParams.com_w;
	// char log[200] = {};
	// sprintf(log, "pose_w=%f, vel_w=%f, end_eff_w=%f, root_w=%f, com_w=%f",
	// 		pose_w, vel_w, end_eff_w, root_w, com_w);
	// std::cout << log << std::endl;

	// normalize
	double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
	pose_w /= total_w;
	vel_w /= total_w;
	end_eff_w /= total_w;
	root_w /= total_w;
	com_w /= total_w;

	// 又有一个scale
	const double pose_scale = RewParams.pose_scale;
	const double vel_scale = RewParams.vel_scale;
	const double end_eff_scale = RewParams.end_eff_scale;
	const double root_scale = RewParams.root_scale;
	const double com_scale = RewParams.com_scale;
	const double err_scale = RewParams.err_scale;	// an uniform adjustment
	// memset(log, 0, 200*sizeof(char));
	// sprintf(log, "pose_scale=%f, vel_scale=%f, end_eff_scale=%f, root_scale=%f, com_scale=%f, err_scale=%f",
	// 		pose_scale, vel_scale, end_eff_scale, root_scale, com_scale,
	// 		err_scale);
	// std::cout << log << std::endl;
	

	const auto& joint_mat = sim_char.GetJointMat();
	const auto& body_defs = sim_char.GetBodyDefs();
	double reward = 0;

	// sim_char: simulation character
	// kin_char: the representation of motion data
	const Eigen::VectorXd& pose0 = sim_char.GetPose();
	const Eigen::VectorXd& vel0 = sim_char.GetVel();
	const Eigen::VectorXd& pose1 = kin_char.GetPose();
	const Eigen::VectorXd& vel1 = kin_char.GetVel();	// get the vel of kin char
	
	//Eigen::VectorXd contact_info;
	//contact_info.resize(0);
	//SolveInverseDynamic(sim_char.GetID(), pose0, pose1, vel0, vel1, contact_info);
	tMatrix origin_trans = sim_char.BuildOriginTrans();	// convert all points in sim char world frame into sim char root local frame
	tMatrix kin_origin_trans = kin_char.BuildOriginTrans();	// convert all points in kin char world frame into kinchar root local frame

	tVector com0_world = sim_char.CalcCOM();			// calculate current sim char COM in world frame	
	tVector com_vel0_world = sim_char.CalcCOMVel();		// calculate current sim char COM vel in world frame
	tVector com1_world;
	tVector com_vel1_world;
	cRBDUtil::CalcCoM(joint_mat, body_defs, pose1, vel1, com1_world, com_vel1_world);	// calculate kin char COM and COM vel in world frame

	
	int root_id = sim_char.GetRootID();
	tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
	tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
	tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
	tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
	tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
	tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
	tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
	tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

	// std::ofstream fout(gRewardInfopath, std::ios::app);
	// fout << "---------------\n";
	// fout <<"pose1 = " << pose1.transpose() << std::endl;
	// fout <<"root_pos1 = " << root_pos1.transpose() << std::endl;
	double pose_err = 0;
	double vel_err = 0;
	double end_eff_err = 0;
	double root_err = 0;
	double com_err = 0;
	double heading_err = 0;

	int num_end_effs = 0;
	int num_joints = sim_char.GetNumJoints();
	assert(num_joints == mJointWeights.size());

	double root_rot_w = mJointWeights[root_id];
	// 计算root朝向错误
	pose_err += root_rot_w * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);	// pow(diff_rot_rot_theta, 2)
	vel_err += root_rot_w * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);
//    std::cout << "======================================\n";
	std::vector<double> joint_angle_err, joint_vel_err;
	for (int j = root_id + 1; j < num_joints; ++j)
	{
		// ROOT is not included in this part of code.
		double w = mJointWeights[j];
		double curr_pose_err = cKinTree::CalcPoseErr(joint_mat, j, pose0, pose1);	// calculate the joint angle diff 
		double curr_vel_err = cKinTree::CalcVelErr(joint_mat, j, vel0, vel1);		// calculate the joint vel diff
		joint_angle_err.push_back(curr_pose_err);
		joint_vel_err.push_back(curr_vel_err);

		// add joint angle diff and joint vel diff to pose_err and vel_err
		pose_err += w * curr_pose_err;
		vel_err += w * curr_vel_err;

		// if end effector
		bool is_end_eff = sim_char.IsEndEffector(j);
		if (is_end_eff)
		{
			// calculate the ref end effector pos(pos0) and true end effector pos(pos1) in both world frame
			// according to mPose and mkinchar.getPose()
			tVector pos0 = sim_char.CalcJointPos(j);
			tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, j);

			// ground_h0: get the height of ground. It is zero here.
			// ground_h1: get the origin's y component. It should be zero as well in this case
			double ground_h0 = mGround->SampleHeight(pos0); 
			double ground_h1 = kin_char.GetOriginPos()[1];

			// calculate the relative position of end effector for sim char with respect to root position, then minus ground height
			// calculate the relative position of end effector for kin char with respect to root position, then minus ground height
			tVector pos_rel0 = pos0 - root_pos0;
			tVector pos_rel1 = pos1 - root_pos1;
			pos_rel0[1] = pos0[1] - ground_h0;
			pos_rel1[1] = pos1[1] - ground_h1;

			// represented them in both root joint local frame: 
			pos_rel0 = origin_trans * pos_rel0;
			pos_rel1 = kin_origin_trans * pos_rel1;

			// calculate the end effector diff
			double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
			end_eff_err += curr_end_err;
			++num_end_effs;

			/* summary:
				This portion of code, calculate the ideal relative position of end effector with respect to ideal root pos for sim char
				and its corrosponding vector for kin char.
				Then compare them as an error.
			*/
		}
	}
//    std::cout << "======================================\n";

	if (num_end_effs > 0)
	{
		end_eff_err /= num_end_effs;
	}

	// sim char，应该是simulation的角色(实际)
	// kin char，应该是从motion中读进来的角色(理想)
	double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos());
	double root_ground_h1 = kin_char.GetOriginPos()[1];
	root_pos0[1] -= root_ground_h0;
	root_pos1[1] -= root_ground_h1;
	double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
	// fout <<"root_pos 0 " << root_pos0.transpose() <<"\n ";
	// fout <<"kin char time " << kin_char.GetTime() <<"\n";
	// fout <<"kin char mOrigin " << kin_char.GetOriginPos().transpose() <<"\n";
	// fout <<"root_pos 1 " << root_pos1.transpose() <<"\n";
	double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
	root_rot_err *= root_rot_err;

	double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
	double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();

	// root位置误差, 旋转误差(0.1), 速度误差(1e-2)，角速度误差(1e-3)，合称为root_err
	root_err = RewParams.root_pos_w * root_pos_err
			+ RewParams.root_rot_w * root_rot_err
			+ RewParams.root_vel_w * root_vel_err
			+ RewParams.root_angle_vel_w * root_ang_vel_err;
	com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();

//	 memset(log, 0, 200*sizeof(char));
//	 sprintf(log, "root_pos_w=%f, root_rot_w=%f, root_vel_w=%f, root_angle_vel_w=%f",
//	 		RewParams.root_pos_w, RewParams.root_rot_w , RewParams.root_vel_w,
//	 		RewParams.root_angle_vel_w);
//	 std::cout << log << std::endl;

	double pose_reward = exp(-err_scale * pose_scale * pose_err);	// 各个joint的朝向 (实际 - 理想)^2
	double vel_reward = exp(-err_scale * vel_scale * vel_err);		// joints的速度(实际 - 理想)^2
	double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);	// end_effector位置误差^2
	double root_reward = exp(-err_scale * root_scale * root_err);	// root joint的(位置误差+0.1线速度误差+0.01朝向误差+0.001角速度)^2
	double com_reward = exp(-err_scale * com_scale * com_err);		// 0.1 * (质心速度误差)^2
//    std::cout << "===================================\n";
//    std::cout << "pose_reward:    " << pose_reward << "\n"
//              << "vel_reward:     " << vel_reward << "\n"
//              << "end_eff_reward: " << end_eff_reward << "\n"
//              << "root_reward:    " << root_reward << "\n"
//              << "com_reward:     " << com_reward << "\n";
//    std::cout << "===================================\n";


    /*
        double pose_w = 0.5;
        double vel_w = 0.05;
        double end_eff_w = 0.15;
        double root_w = 0.2;
        double com_w = 0.1;
    */
	reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward
		+ root_w * root_reward + com_w * com_reward;
	// fout << "pose reward = " << pose_reward << " ";
	// fout << "vel reward = " << vel_reward << " ";
	// fout << "end eff reward = " << end_eff_reward << " ";
	/*
			root_pos_err
			+ RewParams.root_rot_w * root_rot_err
			+ RewParams.root_vel_w * root_vel_err
			+ RewParams.root_angle_vel_w * root_ang_vel_err;
	*/
	// fout << "root pos err = " << root_pos_err << " ";
	// fout << "root rot err = " << root_rot_err << " ";
	// fout << "root vel err = " << root_vel_err << " ";
	// fout << "root ang vel err = " << root_ang_vel_err << " ";
	// fout << "root reward = " << root_reward << " ";

	// fout << "com reward = " << com_reward << " ";
	// fout << "final reward = " << reward << " ";
	// fout << std::endl;
	return reward;
}

cSceneImitate::cSceneImitate()
{
	mEnableRandRotReset = false;
	mSyncCharRootPos = true;
	mSyncCharRootRot = false;
	mMotionFile = "";
	mAngleDiffDir = "";
	mRewardFile = "";
	mEnableAngleDiffLog = false;
	mEnableRootRotFail = false;
	mHoldEndFrame = 0;

	// gRewardInfopath = "reward_info_sample.txt";
	// cFileUtil::ClearFile(gRewardInfopath);
}

cSceneImitate::~cSceneImitate()
{
}

void cSceneImitate::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	cRLSceneSimChar::ParseArgs(parser);
	parser->ParseString("motion_file", mMotionFile);
	parser->ParseBool("enable_rand_rot_reset", mEnableRandRotReset);
	parser->ParseBool("sync_char_root_pos", mSyncCharRootPos);
	parser->ParseBool("sync_char_root_rot", mSyncCharRootRot);
	parser->ParseBool("enable_root_rot_fail", mEnableRootRotFail);
	parser->ParseDouble("hold_end_frame", mHoldEndFrame);

	// print angle diff log
	parser->ParseBool("enable_angle_diff_log", mEnableAngleDiffLog);
	parser->ParseString("angle_diff_dir", mAngleDiffDir);

	// read reward weight file
	parser->ParseString("reward_file", mRewardFile);

}

void cSceneImitate::Init()
{
	mKinChar.reset();
	BuildKinChar();

	cRLSceneSimChar::Init();
	InitRewardWeights();
}

double cSceneImitate::CalcReward(int agent_id) const
{
	// 计算reward在这里
	const cSimCharacter* sim_char = GetAgentChar(agent_id);
	bool fallen = HasFallen(*sim_char);

	double r = 0;
	int max_id = 0;
	if (!fallen)
	{
		r = CalcRewardImitate(*sim_char, *mKinChar);
	}
	return r;
}

const std::shared_ptr<cKinCharacter>& cSceneImitate::GetKinChar() const
{
	return mKinChar;
}

void cSceneImitate::EnableRandRotReset(bool enable)
{
	mEnableRandRotReset = enable;
}

bool cSceneImitate::EnabledRandRotReset() const
{
	bool enable = mEnableRandRotReset;
	return enable;
}

cSceneImitate::eTerminate cSceneImitate::CheckTerminate(int agent_id) const
{
	// 这里也有一个检查终结...看起来更有可行性。
	// std::cout <<"cSceneImitate::eTerminate cSceneImitate::CheckTerminate(int agent_id) const" <<std::endl;
	eTerminate terminated = cRLSceneSimChar::CheckTerminate(agent_id);
	if (terminated == eTerminateNull)
	{
		// 如果上面的terminated属于暂时无法判断的话。
		// 就检查Motion是否over.如果over就定为失败停止，否则还是无法判断。
		bool end_motion = false;
		const auto& kin_char = GetKinChar();
		const cMotion& motion = kin_char->GetMotion();

		if (motion.GetLoop() == cMotion::eLoopNone)
		{
			// 不循环的话，就是超时
			double dur = motion.GetDuration();
			double kin_time = kin_char->GetTime();
			end_motion = kin_time > dur + mHoldEndFrame;
		}
		else
		{
			// motion是否停止
			end_motion = kin_char->IsMotionOver();
		}

		if(true == end_motion)
		{
			std::cout <<"[end] motion end!"<<std::endl;
		}
		// 否则就是按照上面来
		terminated = (end_motion) ? eTerminateFail : terminated;
	}
	else
	{
		// std::cout << "[end] character fall down" <<std::endl;
	}
	
	return terminated;
}

std::string cSceneImitate::GetName() const
{
	return "Imitate";
}

void cSceneImitate::SyncKinCharNewCycleInverseDynamic(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const
{
	SyncKinCharNewCycle(sim_char, out_kin_char);
}

void cSceneImitate::ResolveCharGroundIntersectInverseDynamic()
{
	ResolveCharGroundIntersect();
}

bool cSceneImitate::BuildCharacters()
{
	bool succ = cRLSceneSimChar::BuildCharacters();
	if (EnableSyncChar())
	{
		SyncCharacters();
	}
	return succ;
}

void cSceneImitate::CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const
{
	int num_joints = character->GetNumJoints();
	out_weights = Eigen::VectorXd::Ones(num_joints);
	for (int j = 0; j < num_joints; ++j)
	{
		double curr_w = character->GetJointDiffWeight(j);
		assert(std::isnan(curr_w) == false);
		out_weights[j] = curr_w;
	}

	double sum = out_weights.lpNorm<1>();
	assert(sum > 1e-6);
	out_weights /= sum;
}

bool cSceneImitate::BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl)
{
	bool succ = cSceneSimChar::BuildController(ctrl_params, out_ctrl);
	if (succ)
	{
		auto ct_ctrl = dynamic_cast<cCtController*>(out_ctrl.get());
		if (ct_ctrl != nullptr)
		{
			const auto& kin_char = GetKinChar();
			double cycle_dur = kin_char->GetMotionDuration();
			ct_ctrl->SetCyclePeriod(cycle_dur);
		}
	}
	return succ;
}

void cSceneImitate::BuildKinChar()
{
	bool succ = BuildKinCharacter(0, mKinChar);
	if (!succ)
	{
		printf("Failed to build kin character\n");
		assert(false);
	}
}

bool cSceneImitate::BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char) const
{
	auto kin_char = std::shared_ptr<cKinCharacter>(new cKinCharacter());
	const cSimCharacter::tParams& sim_char_params = mCharParams[0];
	cKinCharacter::tParams kin_char_params;

	kin_char_params.mID = id;
	kin_char_params.mCharFile = sim_char_params.mCharFile;
	kin_char_params.mOrigin = sim_char_params.mInitPos;
	kin_char_params.mLoadDrawShapes = false;
	kin_char_params.mMotionFile = mMotionFile;

	bool succ = kin_char->Init(kin_char_params);
	if (succ)
	{
		out_char = kin_char;
	}
	return succ;
}

void cSceneImitate::UpdateCharacters(double timestep)
{
	UpdateKinChar(timestep);
	cRLSceneSimChar::UpdateCharacters(timestep);
}

void cSceneImitate::UpdateKinChar(double timestep)
{
	const auto& kin_char = GetKinChar();
	double prev_phase = kin_char->GetPhase();
	
	kin_char->Update(timestep);

	// auto before_pose = kin_char->GetPose();

	double curr_phase = kin_char->GetPhase();
	// 如果之前阶段比当前阶段大，代表进入新循环了
	if (curr_phase < prev_phase)
	{
		const auto& sim_char = GetCharacter();
		SyncKinCharNewCycle(*sim_char, *kin_char);
	}

	// if(std::fabs(kin_char->GetPose()[0]) > 1e-7)
	// {
	// 	std::cout << "time = " << kin_char->GetTime() <<" frame " << int(kin_char->GetTime() / timestep)\
	// 		<<" kinchar pose[0] > 1e-7\n";
	// 	std::cout << "before pose = " << before_pose.transpose() << std::endl;
	// 	std::cout <<"after pose = " << kin_char->GetPose().transpose() << std::endl;
	// 	exit(1);
	// }
}

void cSceneImitate::ResetCharacters()
{
	cRLSceneSimChar::ResetCharacters();

	ResetKinChar();
	if (EnableSyncChar())
	{
		SyncCharacters();
	}
}

/*
	Reset Kinchar
	1. set pose and vel to pose0 and vel0
	2. set origin rotation is identity
	3. set origin position is the same as simchar init pos
	4. set current time is random time
	5. calculating current pose and vel according to current time
	6. 
*/
void cSceneImitate::ResetKinChar()
{
	// cFileUtil::ClearFile(gRewardInfopath);
	// std::cout <<"-----------------reset begin-------------------\n";
	// 1. get a random time
	double rand_time = CalcRandKinResetTime();
	// double rand_time = 0.00186;

	// 2. get simchar init params (usually 0, 0, 0)
	const cSimCharacter::tParams& char_params = mCharParams[0];
	const auto& kin_char = GetKinChar();

	// 3. kinchar reset. mPose = mPose0, mVel = mVel0
	kin_char->Reset();	// write mPose = mPose0, mVel = mVel0

	// 4. set origin rot and origin pos to init params
	// ATTENTION: after this "SetOriginPos", the kin char's status was set back to the init status. 
	// mOrigin = 0, 0, 0 (init value), and mPose = mPose0 = character pose in time 0.
	kin_char->SetOriginRot(tQuaternion::Identity());
	// auto before_set_origin_pose = kin_char->GetPose();
	// std::cout <<"[debug] before set origin pose = " << before_set_origin_pose.transpose() << std::endl;
	kin_char->SetOriginPos(char_params.mInitPos);
	// if(kin_char->GetOriginPos().norm() > 1e-10)
	// {
	// 	std::cout <<"[error] cSceneImitate::ResetKinChar origin pos = " << kin_char->GetOriginPos().transpose() << std::endl;
	// 	exit(0);
	// }

	// auto after_set_origin_pose = kin_char->GetPose();
	// std::cout <<"[debug] after set origin pose = " << after_set_origin_pose.transpose() << std::endl;
	// auto diff = (after_set_origin_pose - init_pose0_debug);
	// std::cout <<"[debug] diff = " << diff.transpose() << std::endl;
	// std::cout <<"diff norm = " << diff.norm() << std::endl;
	// if(diff.norm() > 1e-10)
	// {
	// 	exit(0);
	// }
	
	// std::cout <<"[debug] before set time mOrigin = " << kin_char->GetOriginPos().transpose() << std::endl;
	// std::cout <<"[debug] set rand time = " << rand_time << std::endl;
	// 5. just set time varible. do not update or change any value
	kin_char->SetTime(rand_time);
	// 6. extract a ref motion from mMotion, here record the root pos as "root_ref_pos" and root rotation "root_ref_rot"
	// then we calculate root_final_pos = mOriginRot * root_ref_pos + mOrigin
	// root_final_rot = mOriginRot * root_ref_root
	// From then on, the mPose's root pos and root rot will be overwritten
	kin_char->Pose(rand_time);
	// std::cout <<"[debug] after Pose(rand_time) pose = " << kin_char->GetPose().transpose() << std::endl;
	// std::cout <<"[debug] after set time mOrigin = " << kin_char->GetOriginPos().transpose() << std::endl;
	// std::cout <<"[debug] after set time pose = " << kin_char->GetPose().transpose() << std::endl;

	if (EnabledRandRotReset())
	{
		std::cout <<"it will break ID! disabled!\n";
		exit(1);
		double rand_theta = mRand.RandDouble(-M_PI, M_PI);
		kin_char->RotateOrigin(cMathUtil::EulerToQuaternion(tVector(0, rand_theta, 0, 0), eRotationOrder::XYZ));
	}

	// std::cout <<"-----------------reset end-------------------\n";
}

void cSceneImitate::SyncCharacters()
{	const auto& kin_char = GetKinChar();
	const Eigen::VectorXd& pose = kin_char->GetPose();
	const Eigen::VectorXd& vel = kin_char->GetVel();
	
	const auto & sim_char = GetCharacter();

	// std::cout <<"------------begin sync char\n";
	// std::cout << "pose = " << sim_char->GetPose().transpose() << std::endl;
	// std::cout << "root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;

	sim_char->SetPose(pose);
	sim_char->SetVel(vel);

	const auto& ctrl = sim_char->GetController();
	auto ct_ctrl = dynamic_cast<cCtController*>(ctrl.get());
	if (ct_ctrl != nullptr)
	{
		double kin_time = GetKinTime();
		ct_ctrl->SetInitTime(kin_time);
	}
	// std::cout <<"------------end sync char\n";
	// std::cout << "pose = " << sim_char->GetPose().transpose() << std::endl;
	// std::cout << "root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;
	// exit(1);
}

bool cSceneImitate::EnableSyncChar() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->HasMotion();
}

void cSceneImitate::InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char)
{
	// nothing to see here
}

void cSceneImitate::InitRewardWeights()
{
	// read weight from file
	ifstream fin(mRewardFile);
	if(!fin)
	{
		std::cout <<"[cSceneImitate] open reward file " << mRewardFile <<" failed" << std::endl;
		abort();
	}
	Json::Reader reader;
	Json::Value root;
	bool succ = reader.parse(fin, root);
	if (!succ)
	{
		std::cout <<"[cSceneImitate] Failed to parse json " << mRewardFile << std::endl;
	}
	SetRewardParams(root);

	// read joint error weight in pose_err calculation from skeleton file("DiffWeight")
	InitJointWeights();
}

void cSceneImitate::SetRewardParams(Json::Value & root)
{
	Json::Value reward_weight_terms=root["reward_weight_terms"],
		scale_terms=root["scale_terms"], 
		root_sub_terms=root["root_sub_terms"];

	RewParams.pose_w = reward_weight_terms["pose_w"].asDouble();
	RewParams.vel_w = reward_weight_terms["vel_w"].asDouble();
	RewParams.end_eff_w = reward_weight_terms["end_eff_w"].asDouble();
	RewParams.root_w = reward_weight_terms["root_w"].asDouble();
	RewParams.com_w = reward_weight_terms["com_w"].asDouble();

	RewParams.pose_scale = scale_terms["pose_scale"].asDouble();
	RewParams.vel_scale = scale_terms["vel_scale"].asDouble();
	RewParams.end_eff_scale = scale_terms["end_eff_scale"].asDouble();
	RewParams.root_scale = scale_terms["root_scale"].asDouble();
	RewParams.com_scale = scale_terms["com_scale"].asDouble();
	RewParams.err_scale = scale_terms["err_scale"].asDouble();

	RewParams.root_pos_w = root_sub_terms["root_pos_w"].asDouble();
	RewParams.root_rot_w = root_sub_terms["root_rot_w"].asDouble();
	RewParams.root_vel_w = root_sub_terms["root_vel_w"].asDouble();
	RewParams.root_angle_vel_w = root_sub_terms["root_angle_vel_w"].asDouble();

}

void cSceneImitate::InitJointWeights()
{
	CalcJointWeights(GetCharacter(), mJointWeights);
}
// To avoid Char Ground Intersect, You must shift the whole character 
void cSceneImitate::ResolveCharGroundIntersect()
{
	cRLSceneSimChar::ResolveCharGroundIntersect();

	// after solving the intersect, we need to update kinchar value again.
	if (EnableSyncChar())
	{
		SyncKinCharRoot();
	}
}

void cSceneImitate::ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const
{
	cRLSceneSimChar::ResolveCharGroundIntersect(out_char);
}

void cSceneImitate::SyncKinCharRoot()
{
	const auto& sim_char = GetCharacter();
	tVector sim_root_pos = sim_char->GetRootPos();
	double sim_heading = sim_char->CalcHeading();

	const auto& kin_char = GetKinChar();
	double kin_heading = kin_char->CalcHeading();

	tQuaternion drot = tQuaternion::Identity();
	if (mSyncCharRootRot)
	{
		drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
	}

	kin_char->RotateRoot(drot);
	kin_char->SetRootPos(sim_root_pos);
}

// When the simulation step into a new cycle of given motion, we need to 
void cSceneImitate::SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const
{
	// Do we sync the rotation of rot? It is an option which is avaliable in config arg_file
	// It is usually closed, because the orientation of the character are important to its motion in most cases.
	// We always want to keep the character running in the same direction or at least, do not have a big bias
	if (mSyncCharRootRot)
	{
		double sim_heading = sim_char.CalcHeading();
		double kin_heading = out_kin_char.CalcHeading();
		// rotate the kin_char heading the same as the simulated one
		tQuaternion drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
		out_kin_char.RotateRoot(drot);
	}

	// Do we sync the position of root? It is in arg_file as well
	// It is usually opened. The given motion can have a shift in root pos between the first & the end frame of it.
	if (mSyncCharRootPos)
	{
		tVector sim_root_pos = sim_char.GetRootPos();
		tVector kin_root_pos = out_kin_char.GetRootPos();
		kin_root_pos[0] = sim_root_pos[0];
		kin_root_pos[2] = sim_root_pos[2];

		tVector origin = out_kin_char.GetOriginPos();
		double dh = kin_root_pos[1] - origin[1];
		double ground_h = mGround->SampleHeight(kin_root_pos);
		kin_root_pos[1] = ground_h + dh;

		out_kin_char.SetRootPos(kin_root_pos);
	}
}

double cSceneImitate::GetKinTime() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->GetTime();
}

bool cSceneImitate::CheckKinNewCycle(double timestep) const
{
	bool new_cycle = false;
	const auto& kin_char = GetKinChar();
	if (kin_char->GetMotion().EnableLoop())
	{
		double cycle_dur = kin_char->GetMotionDuration();
		double time = GetKinTime();
		new_cycle = cMathUtil::CheckNextInterval(timestep, time, cycle_dur);
	}
	return new_cycle;
}


bool cSceneImitate::HasFallen(const cSimCharacter& sim_char) const
{
	bool fallen = cRLSceneSimChar::HasFallen(sim_char);
	if (mEnableRootRotFail)
	{
		fallen |= CheckRootRotFail(sim_char);
	}

	return fallen;
}

bool cSceneImitate::CheckRootRotFail(const cSimCharacter& sim_char) const
{
	const auto& kin_char = GetKinChar();
	bool fail = CheckRootRotFail(sim_char, *kin_char);
	return fail;
}

bool cSceneImitate::CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const
{
	const double threshold = 0.5 * M_PI;

	tQuaternion sim_rot = sim_char.GetRootRotation();
	tQuaternion kin_rot = kin_char.GetRootRotation();
	double rot_diff = cMathUtil::QuatDiffTheta(sim_rot, kin_rot);
	return rot_diff > threshold;
}

double cSceneImitate::CalcRandKinResetTime()
{
	const auto& kin_char = GetKinChar();
	double dur = kin_char->GetMotionDuration();
	double rand_time = cMathUtil::RandDouble(0, dur);
	return rand_time;
}
