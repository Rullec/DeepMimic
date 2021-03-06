﻿#include "SceneSimChar.h"

#include <memory>
#include <ctime>
#include "sim/SimBox.h"
#include "sim/GroundPlane.h"
#include "sim/GroundBuilder.h"
#include "sim/DeepMimicCharController.h"
#include "sim/BuildIDSolver.hpp"
#include "sim/cOnlineIDSolver.hpp"
#include "sim/cOfflineIDSolver.hpp"
#include "util/FileUtil.h"
#include <iostream>
#include <fstream>
using namespace std;

const int gDefaultCharID = 0;
const double gCharViewDistPad = 1;
const double cSceneSimChar::gGroundSpawnOffset = -1; // some padding to prevent parts of character from getting spawned inside obstacles

const size_t gInitGroundUpdateCount = std::numeric_limits<size_t>::max();

cSceneSimChar::tObjEntry::tObjEntry()
{
	mObj = nullptr;
	mEndTime = std::numeric_limits<double>::infinity();
	mColor = tVector(0.5, 0.5, 0.5, 1);
	mPersist = false;
}

bool cSceneSimChar::tObjEntry::IsValid() const
{
	return mObj != nullptr;
}

cSceneSimChar::tJointEntry::tJointEntry()
{
	mJoint = nullptr;
}

bool cSceneSimChar::tJointEntry::IsValid() const
{
	return mJoint != nullptr;
}

cSceneSimChar::tPerturbParams::tPerturbParams()
{
	mEnableRandPerturbs = false;
	mTimer = 0;
	mTimeMin = std::numeric_limits<double>::infinity();
	mTimeMax = std::numeric_limits<double>::infinity();
	mNextTime = 0;
	mMinPerturb = 50;
	mMaxPerturb = 100;
	mMinDuration = 0.1;
	mMaxDuration = 0.5;
}

cSceneSimChar::cSceneSimChar()
{
	mEnableContactFall = true;
	mEnableRandCharPlacement = true;
	mEnableTorqueRecord = false;
	mTorqueRecordFile = "";
	mEnableJointTorqueControl = true;
	//mIDInfo.clear();

	mWorldParams.mNumSubsteps = 1;
	mWorldParams.mScale = 1;
	mWorldParams.mGravity = gGravity;
}

cSceneSimChar::~cSceneSimChar()
{
	Clear();
}

void cSceneSimChar::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	cScene::ParseArgs(parser);

	bool succ = true;

	parser->ParseBool("enable_char_contact_fall", mEnableContactFall);	// 打开角色接触掉落, 这东西默认是开启的。
	parser->ParseBool("enable_rand_char_placement", mEnableRandCharPlacement);
	parser->ParseBool("enable_torque_record", mEnableTorqueRecord);
	parser->ParseString("torque_record_file", mTorqueRecordFile);
	parser->ParseBool("enable_joint_force_control", mEnableJointTorqueControl);

	succ &= ParseCharTypes(parser, mCharTypes);
	succ &= ParseCharParams(parser, mCharParams);
	succ &= ParseCharCtrlParams(parser, mCtrlParams);
	if (mCharParams.size() != mCtrlParams.size())
	{
		printf("Char and ctrl file mismatch, %zi vs %zi\n", mCharParams.size(), mCtrlParams.size());
		assert(false);
	}

	std::string sim_mode_str = "";
	parser->ParseInt("num_sim_substeps", mWorldParams.mNumSubsteps);
	parser->ParseDouble("world_scale", mWorldParams.mScale);
	parser->ParseVector("gravity", mWorldParams.mGravity);

	parser->ParseBool("enable_rand_perturbs", mPerturbParams.mEnableRandPerturbs);
	parser->ParseDouble("perturb_time_min", mPerturbParams.mTimeMin);
	parser->ParseDouble("perturb_time_max", mPerturbParams.mTimeMax);
	parser->ParseDouble("min_perturb", mPerturbParams.mMinPerturb);
	parser->ParseDouble("max_perturb", mPerturbParams.mMaxPerturb);
	parser->ParseDouble("min_pertrub_duration", mPerturbParams.mMinDuration);
	parser->ParseDouble("max_perturb_duration", mPerturbParams.mMaxDuration);
	parser->ParseInts("perturb_part_ids", mPerturbParams.mPerturbPartIDs);


	parser->ParseInts("fall_contact_bodies", mFallContactBodies);	// 哪几个link检测掉落?

	ParseGroundParams(parser, mGroundParams);

	// parse inverse dynamics
	mEnableID = false;
	mIDInfoPath = "";
	mArgParser->ParseBool("enable_inverse_dynamic_solving", mEnableID);
	mArgParser->ParseString("inverse_dynamic_config_file", mIDInfoPath);
	if(mEnableID == true && false == cFileUtil::ExistsFile(mIDInfoPath))
	{
		std::cout <<"[error] cSceneSimChar::ParseArgs failed for enable id but conf path is illegal: " << mIDInfoPath << std::endl;;
		exit(1); 
	}
	
}

void cSceneSimChar::Init()
{
	cScene::Init();

	if (mPerturbParams.mEnableRandPerturbs)
	{
		ResetRandPertrub();
	}

	// 在这里建造地面　+ 角色结构
	BuildWorld();

	// 建造地面(地面有什么可建造的呢)
	BuildGround();

	// 建造角色结构
	BuildCharacters();

	// 初始化角色位置
	// auto & sim_char = GetCharacter(0);
	// std::cout <<"begin init char pos \n";
	// std::cout << "pose = " << sim_char->GetPose().transpose() << std::endl;
	// std::cout << "root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;

	InitCharacterPos();
	// std::cout <<"after init char pos ";
	// std::cout << "pose = " << sim_char->GetPose().transpose() << std::endl;
	// std::cout << "root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;

	// exit(1);
	ResolveCharGroundIntersect();

	// build inverse dynamic
	BuildInverseDynamic();

	ClearObjs();

	// std::cout << "pose = " << sim_char->GetPose().transpose() << std::endl;
	// std::cout << "root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;
	// exit(1);
}

void cSceneSimChar::Clear()
{
	cScene::Clear();

	mChars.clear();
	mGround.reset();
	mFallContactBodies.clear();
	ClearJoints();
	ClearObjs();
}

void cSceneSimChar::Update(double time_elapsed)
{
	// std::cout <<"------------cSceneSimChar::Update------------" << this->GetTime() << std::endl;;
	auto & sim_char = GetCharacter();
	// std::cout <<"[scene] error root pos = " << sim_char->GetRootPos().transpose() << std::endl;
	// std::cout <<"[scene] error root rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;
	// std::cout <<"[scene] error pose = " << sim_char->GetPose().transpose() << std::endl;

	cScene::Update(time_elapsed);

	if (time_elapsed < 0)
	{
		return;
	}


	if (mPerturbParams.mEnableRandPerturbs)
	{
		UpdateRandPerturb(time_elapsed);
	}

	PreUpdate(time_elapsed);		// clear joint torque
	// 显示一下速度：是不是最开始的时候设置的速度太大了?

	
	// order matters!
	if(true == mEnableID && mIDSolver!=nullptr)
	{
		if(eIDSolverType::Online == mIDSolver->GetType())
		{
			auto online_solver = std::dynamic_pointer_cast<cOnlineIDSolver>(mIDSolver);
			online_solver->SetTimestep(time_elapsed);	// record new frame，在重新计算torque以后，更新位移和速度之前...

			// calc & apply torque in this function
			UpdateCharacters(time_elapsed);	// calculate all joint torques, then apply them in bullet
			online_solver->PreSim();

			UpdateWorld(time_elapsed);
			UpdateGround(time_elapsed);
			UpdateObjs(time_elapsed);
			UpdateJoints(time_elapsed);

			PostUpdateCharacters(time_elapsed);
			PostUpdate(time_elapsed);

			online_solver->PostSim();
		}
		else if(eIDSolverType::Offline == mIDSolver->GetType())
		{
			// mIDInfo->SetTimestep(time_elapsed);	// record new frame，在重新计算torque以后，更新位移和速度之前...
			auto offline_solver = std::dynamic_pointer_cast<cOfflineIDSolver>(mIDSolver);
			// calc & apply torque in this function
			if(eOfflineSolverMode::Save == offline_solver->GetOfflineSolverMode())
			{
				offline_solver->SetTimestep(time_elapsed);	// record new frame，在重新计算torque以后，更新位移和速度之前...
				UpdateCharacters(time_elapsed);	// calculate all joint torques, then apply them in bullet
				mIDSolver->PreSim();

				UpdateWorld(time_elapsed);
				UpdateGround(time_elapsed);
				UpdateObjs(time_elapsed);
				UpdateJoints(time_elapsed);

				PostUpdateCharacters(time_elapsed);
				PostUpdate(time_elapsed);

				mIDSolver->PostSim();	
			}
			else if(eOfflineSolverMode::Display == offline_solver->GetOfflineSolverMode())
			{
				// std::cout <<"display!\n";
				offline_solver->DisplaySet();
				// auto & sim_char = GetCharacter(0);
				// std::cout <<"error rot = " << sim_char->GetRootRotation().coeffs().transpose() << std::endl;


				// mIDSolver->PostSim();	
			}
			else if(eOfflineSolverMode::Solve == offline_solver->GetOfflineSolverMode())
			{
				offline_solver->OfflineSolve();
			}
			else
			{
				std::cout <<"[error] cSceneSimChar::Update IDSolver error mode = " << offline_solver->GetOfflineSolverMode();
				exit(1);
			}
		}
		else
		{
			std::cout <<"[error] cSceneSimChar::Update IDSolver Type illegal = " << mIDSolver->GetType() << std::endl;
		}

	}
	else
	{
		// calc & apply torque in this function
		UpdateCharacters(time_elapsed);	// calculate all joint torques, then apply them in bullet


		UpdateWorld(time_elapsed);
		UpdateGround(time_elapsed);
		UpdateObjs(time_elapsed);
		UpdateJoints(time_elapsed);

		PostUpdateCharacters(time_elapsed);
		PostUpdate(time_elapsed);
	
	}
	
	
}

int cSceneSimChar::GetNumChars() const
{
	return static_cast<int>(mChars.size());
}

const std::shared_ptr<cSimCharacter>& cSceneSimChar::GetCharacter()  const
{
	return GetCharacter(gDefaultCharID);
}

const std::shared_ptr<cSimCharacter>& cSceneSimChar::GetCharacter(int char_id) const
{
	return mChars[char_id];
}

const std::shared_ptr<cWorld>& cSceneSimChar::GetWorld() const
{
	return mWorld;
}

tVector cSceneSimChar::GetCharPos() const
{
	return GetCharacter()->GetRootPos();
}

const std::shared_ptr<cGround>& cSceneSimChar::GetGround() const
{
	return mGround;
}

const tVector& cSceneSimChar::GetGravity() const
{
	return mWorldParams.mGravity;
}

bool cSceneSimChar::LoadControlParams(const std::string& param_file, const std::shared_ptr<cSimCharacter>& out_char)
{
	const auto& ctrl = out_char->GetController();
	bool succ = ctrl->LoadParams(param_file);
	return succ;
}

void cSceneSimChar::AddPerturb(const tPerturb& perturb)
{
	mWorld->AddPerturb(perturb);
}

void cSceneSimChar::ApplyRandForce(double min_force, double max_force, 
									double min_dur, double max_dur, cSimObj* obj)
{
	assert(obj != nullptr);
	tPerturb perturb = tPerturb::BuildForce();
	perturb.mObj = obj;
	perturb.mLocalPos.setZero();
	perturb.mPerturb[0] = mRand.RandDouble(-1, 1);
	perturb.mPerturb[1] = mRand.RandDouble(-1, 1);
	perturb.mPerturb[2] = mRand.RandDouble(-1, 1);
	perturb.mPerturb = mRand.RandDouble(min_force, max_force) * perturb.mPerturb.normalized();
	perturb.mDuration = mRand.RandDouble(min_dur, max_dur);

	AddPerturb(perturb);
}

void cSceneSimChar::ApplyRandForce()
{
	for (int i = 0; i < GetNumChars(); ++i)
	{
		ApplyRandForce(i);
	}
}

void cSceneSimChar::ApplyRandForce(int char_id)
{
	const std::shared_ptr<cSimCharacter>& curr_char = GetCharacter(char_id);
	int num_parts = curr_char->GetNumBodyParts();
	int part_idx = GetRandPerturbPartID(curr_char);
	assert(part_idx != gInvalidIdx);
	const auto& part = curr_char->GetBodyPart(part_idx);
	ApplyRandForce(mPerturbParams.mMinPerturb, mPerturbParams.mMaxPerturb, mPerturbParams.mMinDuration, mPerturbParams.mMaxDuration, part.get());
}

int cSceneSimChar::GetRandPerturbPartID(const std::shared_ptr<cSimCharacter>& character)
{
	int rand_id = gInvalidIdx;
	int num_part_ids = static_cast<int>(mPerturbParams.mPerturbPartIDs.size());
	if (num_part_ids > 0)
	{
		int idx = mRand.RandInt(0, num_part_ids);
		rand_id = mPerturbParams.mPerturbPartIDs[idx];
	}
	else
	{
		int num_parts = character->GetNumBodyParts();
		rand_id = mRand.RandInt(0, num_parts);
	}
	return rand_id;
}

void cSceneSimChar::RayTest(const tVector& beg, const tVector& end, cWorld::tRayTestResult& out_result) const
{
	cWorld::tRayTestResults results;
	mWorld->RayTest(beg, end, results);

	out_result.mObj = nullptr;
	if (results.size() > 0)
	{
		out_result = results[0];
	}
}

void cSceneSimChar::SetGroundParamBlend(double lerp)
{
	mGround->SetParamBlend(lerp);
}

int cSceneSimChar::GetNumParamSets() const
{
	return static_cast<int>(mGroundParams.mParamArr.rows());
}

void cSceneSimChar::OutputCharState(const std::string& out_file) const
{
	const auto& char0 = GetCharacter();
	tVector root_pos = char0->GetRootPos();
	double ground_h = mGround->SampleHeight(root_pos);
	tMatrix trans = char0->BuildOriginTrans();
	trans(1, 3) -= ground_h;

	char0->WriteState(out_file, trans);
}

void cSceneSimChar::OutputGround(const std::string& out_file) const
{
	mGround->Output(out_file);
}

std::string cSceneSimChar::GetName() const
{
	return "Sim Character";
}

bool cSceneSimChar::BuildCharacters()
{
	/*
		最关键的character build过程终于找到了！
	 */
	bool succ = true;
	mChars.clear();

	int num_chars = static_cast<int>(mCharParams.size());
	for (int i = 0; i < num_chars; ++i)
	{
		// 对于每个角色，都由SimCharacter管理(这是一个类)
		// 根据mCharParams来创建角色模型(character parameters)
		const cSimCharacter::tParams& curr_params = mCharParams[i];	// 这叫做当前参数curr_params
		std::shared_ptr<cSimCharacter> curr_char;

		// 为什么角色还有一个builder?
		cSimCharBuilder::eCharType char_type = cSimCharBuilder::cCharGeneral;
		if (mCharTypes.size() > i)
		{
			char_type = mCharTypes[i];
		}
		cSimCharBuilder::CreateCharacter(char_type, curr_char);

		succ &= curr_char->Init(mWorld, curr_params);
		// std::cout <<"init1 pose = " << curr_char->GetPose().transpose() << std::endl;
		if (succ)
		{
			SetFallContacts(mFallContactBodies, *curr_char);
			// std::cout << "[SceneSimChar.cpp buildCharacter] set fall contacts, mFallContactBodies = " ;
			// for(auto i: mFallContactBodies)
			// 	std::cout << i <<" ";
			// std::cout << std::endl;

			curr_char->RegisterContacts(cWorld::eContactFlagCharacter, cWorld::eContactFlagEnvironment);

			InitCharacterPos(curr_char);
			// std::cout <<"init2 pose = " << curr_char->GetPose().transpose() << std::endl;

			if (i < mCtrlParams.size())
			{
				auto ctrl_params = mCtrlParams[i];
				ctrl_params.mChar = curr_char;
				ctrl_params.mGravity = GetGravity();
				ctrl_params.mGround = mGround;

				std::shared_ptr<cCharController> ctrl;
				succ = BuildController(ctrl_params, ctrl);
				if (succ && ctrl != nullptr)
				{
					// 设置角色的控制器，一共5种，继承关系复杂。
					curr_char->SetController(ctrl);
				}
			}
			mChars.push_back(curr_char);
		}
	}
	
	mChars[0]->SetEnablejointTorqueControl(mEnableJointTorqueControl);
	return succ;
}

bool cSceneSimChar::ParseCharTypes(const std::shared_ptr<cArgParser>& parser, std::vector<cSimCharBuilder::eCharType>& out_types) const
{
	bool succ = true;
	std::vector<std::string> char_type_strs;
	succ = parser->ParseStrings("char_types", char_type_strs);

	int num = static_cast<int>(char_type_strs.size());
	out_types.clear();
	for (int i = 0; i < num; ++i)
	{
		std::string str = char_type_strs[i];
		cSimCharBuilder::eCharType char_type = cSimCharBuilder::eCharNone;
		cSimCharBuilder::ParseCharType(str, char_type);

		if (char_type != cSimCharBuilder::eCharNone)
		{
			out_types.push_back(char_type);
		}
	}

	return succ;
}

bool cSceneSimChar::ParseCharParams(const std::shared_ptr<cArgParser>& parser, std::vector<cSimCharacter::tParams>& out_params) const
{
	bool succ = true;

	std::vector<std::string> char_files;
	succ = parser->ParseStrings("character_files", char_files);

	std::vector<std::string> state_files;
	parser->ParseStrings("state_files", state_files);

	std::vector<double> init_pos_xs;
	parser->ParseDoubles("char_init_pos_xs", init_pos_xs);
	
	int num_files = static_cast<int>(char_files.size());
	out_params.resize(num_files);

	for (int i = 0; i < num_files; ++i)
	{
		cSimCharacter::tParams& params = out_params[i];
		params.mID = i;
		params.mCharFile = char_files[i];
		
		params.mEnableContactFall = mEnableContactFall;

		if (state_files.size() > i)
		{
			params.mStateFile = state_files[i];
		}

		if (init_pos_xs.size() > i)
		{
			params.mInitPos[0] = init_pos_xs[i];
		}
	}

	if (!succ)
	{
		printf("No valid character file specified.\n");
	}

	return succ;
}

bool cSceneSimChar::ParseCharCtrlParams(const std::shared_ptr<cArgParser>& parser, std::vector<cCtrlBuilder::tCtrlParams>& out_params) const
{
	bool succ = true;

	std::vector<std::string> ctrl_files;
	parser->ParseStrings("char_ctrl_files", ctrl_files);

	int num_ctrls = static_cast<int>(ctrl_files.size());

	std::vector<std::string> char_ctrl_strs;
	parser->ParseStrings("char_ctrls", char_ctrl_strs);

	out_params.resize(num_ctrls);
	for (int i = 0; i < num_ctrls; ++i)
	{
		auto& ctrl_params = out_params[i];
		const std::string& type_str = char_ctrl_strs[i];
		cCtrlBuilder::ParseCharCtrl(type_str, ctrl_params.mCharCtrl);
		ctrl_params.mCtrlFile = ctrl_files[i];
	}

	return succ;
}

void cSceneSimChar::BuildWorld()
{
	//std::cout <<"void cSceneSimChar::BuildWorld() 建造bullet世界" <<  std::endl;
	mWorld = std::shared_ptr<cWorld>(new cWorld());
	mWorld->Init(mWorldParams);
}

void cSceneSimChar::BuildGround()
{
	mGroundParams.mHasRandSeed = mHasRandSeed;
	mGroundParams.mRandSeed = mRandSeed;
	cGroundBuilder::BuildGround(mWorld, mGroundParams, mGround);
}

bool cSceneSimChar::BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl)
{
	bool succ = cCtrlBuilder::BuildController(ctrl_params, out_ctrl);
	return succ;
}

void cSceneSimChar::SetFallContacts(const std::vector<int>& fall_bodies, cSimCharacter& out_char) const
{
	// 这个函数负责注册fall bodies
	int num_fall_bodies = static_cast<int>(fall_bodies.size());
	if (num_fall_bodies > 0)	// 如果这个这个目标不是空的
	{
		for (int i = 0; i < out_char.GetNumBodyParts(); ++i)
		{
			out_char.SetBodyPartFallContact(i, false);
		}

		for (int i = 0; i < num_fall_bodies; ++i)
		{
			int b = fall_bodies[i];
			out_char.SetBodyPartFallContact(b, true);	// 那么就把这些数字都定义为有contact
		}
	}
}

void cSceneSimChar::InitCharacterPos()
{
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		InitCharacterPos(mChars[i]);
	}
}

void cSceneSimChar::InitCharacterPos(const std::shared_ptr<cSimCharacter>& out_char)
{
	if (mEnableRandCharPlacement)
	{
		SetCharRandPlacement(out_char);
	}
	else
	{
		InitCharacterPosFixed(out_char);
	}
}

void cSceneSimChar::InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char)
{
	tVector root_pos = out_char->GetRootPos();
	int char_id = out_char->GetID();
	root_pos[0] = mCharParams[char_id].mInitPos[0];

	double h = mGround->SampleHeight(root_pos);
	root_pos[1] += h;

	out_char->SetRootPos(root_pos);
}

void cSceneSimChar::BuildInverseDynamic()
{
	if(!mEnableID)
	{
		mIDSolver = nullptr;
		return;
	}

	// build inverse dynamics
	auto sim_char = this->GetCharacter(0);
	
	mIDSolver = BuildIDSolver(mIDInfoPath, sim_char.get(), sim_char->GetWorld()->GetInternalWorld().get());
	// mOnlineIDSolver = std::shared_ptr<cOnlineIDSolver>(new cOnlineIDSolver(sim_char.get(), sim_char->GetWorld()->GetInternalWorld().get()));
	// mOfflineIDSolver = std::shared_ptr<cOfflineIDSolver>(new cOfflineIDSolver(sim_char.get(), sim_char->GetWorld()->GetInternalWorld().get(), "./args/0311/id_conf_offline.json"));
	// std::shared_ptr<cOnlineIDSolver>(new cOnlineIDSolver(sim_char.get(), sim_char->GetWorld()->GetInternalWorld().get()));

	// from json vec to Eigen::VectorXd
	// auto JsonVec2Eigen = [](Json::Value root)->Eigen::VectorXd
	// {
	// 	Eigen::VectorXd vec(root.size());
	// 	for (int i = 0; i < vec.size(); i++) vec[i] = root[i].asDouble();
	// 	return vec;
	// };

	// offline mode: read trajectory from files, then solve it.
	// online mode for debug: start with the simulation at the same time, record each state and solve them at onece
	// then compare the desired ID result and the ideal one. It will be very easy to debug.
	if(eIDSolverType::Offline == mIDSolver->GetType())
		std::cout << "[log] Inverse Dynamics runs in offline mode." << std::endl;
	else if(eIDSolverType::Online == mIDSolver->GetType())
		std::cout << "[log] Inverse Dynamics runs in online mode." << std::endl;
	else
	{
		std::cout <<"unrecognized ID solver mode = " << mIDSolver->GetType() << std::endl;
		exit(1);
	}
	//if (mEnableID)
	//{
	//	// parse ID info
	//	std::ifstream f_stream(mIDInfoPath);
	//	Json::Reader reader;
	//	Json::Value root;
	//	bool succ = reader.parse(f_stream, root);
	//	f_stream.close();
	//	if (!succ)
	//	{
	//		std::cout << "[error] void cSceneSimChar::InitInverseDynamic() parse failed " << std::endl;
	//		exit(1);
	//	}
	//	Json::Value states = root["states"], poses = root["poses"], actions = root["actions"], contact_info = root["contact_info"];
	//	int num_pairs = states.size();
	//	for (int i = 0; i < num_pairs; i++)
	//	{
	//		// for the final state there is no action
	//		Json::Value cur_state = states[i], cur_pose = poses[i], cur_action = actions[i], cur_contact_info = contact_info[i];
	//		//printf("[debug] total pairs = %d, state size = %d, pose size = %d, action size = %d, contact_info size = %d\n",
	//		//	num_pairs,
	//		//	cur_state.size(),
	//		//	cur_pose.size(),
	//		//	cur_action.size(),
	//		//	cur_contact_info.size());
	//		Eigen::VectorXd state = JsonVec2Eigen(cur_state), pose = JsonVec2Eigen(cur_pose),
	//			action = JsonVec2Eigen(cur_action), contact_info = JsonVec2Eigen(cur_contact_info);
	//		mIDInfo->AddNewFrame(state, pose, action, contact_info);
	//	}
	//	std::cout << "[log] load ID items number = " << mIDInfo->GetNumOfFrames() << std::endl;
	//	// remodeling the Timer
	//	mTimer.SetMaxTime(mIDInfo->GetMaxTime());
	//	// compute the position, velocity, acceleration for each link in each frame by forward euler
	//	mIDInfo->SolveInverseDynamics();
	//}
	//else
	//{
	//	std::cout << "[warning] InverseDynamics disabled!" << std::endl;
	//}
}

void cSceneSimChar::SetCharRandPlacement(const std::shared_ptr<cSimCharacter>& out_char)
{
	tVector rand_pos = tVector::Zero();
	tQuaternion rand_rot = tQuaternion::Identity();
	CalcCharRandPlacement(out_char, rand_pos, rand_rot);
	out_char->SetRootTransform(rand_pos, rand_rot);
}

void cSceneSimChar::CalcCharRandPlacement(const std::shared_ptr<cSimCharacter>& out_char, tVector& out_pos, tQuaternion& out_rot)
{
	tVector char_pos = out_char->GetRootPos();
	tQuaternion char_rot = out_char->GetRootRotation();

	tVector rand_pos;
	tQuaternion rand_rot;
	mGround->SamplePlacement(tVector::Zero(), rand_pos, rand_rot);

	out_pos = rand_pos;
	out_pos[1] += char_pos[1];
	out_rot = rand_rot * char_rot;
}

void cSceneSimChar::ResolveCharGroundIntersect()
{
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		ResolveCharGroundIntersect(mChars[i]);
	}
}

void cSceneSimChar::ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const
{
	// 为了防止初始状态和地面有碰撞，加上去。
	const double pad = 0.001;

	int num_parts = out_char->GetNumBodyParts();
	double min_violation = 0;
	for (int b = 0; b < num_parts; ++b)
	{
		if (out_char->IsValidBodyPart(b))
		{
			tVector aabb_min;
			tVector aabb_max;
			const auto& part = out_char->GetBodyPart(b);
			part->CalcAABB(aabb_min, aabb_max);

			tVector mid = 0.5 * (aabb_min + aabb_max);
			tVector sw = tVector(aabb_min[0], 0, aabb_min[2], 0);
			tVector nw = tVector(aabb_min[0], 0, aabb_max[2], 0);
			tVector ne = tVector(aabb_max[0], 0, aabb_max[2], 0);
			tVector se = tVector(aabb_max[0], 0, aabb_min[2], 0);

			double max_ground_height = 0;
			max_ground_height = mGround->SampleHeight(aabb_min);
			max_ground_height = std::max(max_ground_height, mGround->SampleHeight(mid));
			max_ground_height = std::max(max_ground_height, mGround->SampleHeight(sw));
			max_ground_height = std::max(max_ground_height, mGround->SampleHeight(nw));
			max_ground_height = std::max(max_ground_height, mGround->SampleHeight(ne));
			max_ground_height = std::max(max_ground_height, mGround->SampleHeight(se));
			max_ground_height += pad;

			double min_height = aabb_min[1];
			min_violation = std::min(min_violation, min_height - max_ground_height);
		}
	}

	if (min_violation < 0)
	{
		tVector root_pos = out_char->GetRootPos();
		root_pos[1] += -min_violation;
		out_char->SetRootPos(root_pos);
	}
}

void cSceneSimChar::UpdateWorld(double time_step)
{
	mWorld->Update(time_step);
}

void cSceneSimChar::UpdateCharacters(double time_step)
{
	/*
		1. compute torque by PD target
		2. apply these torques to joints, then bullet links
	*/
	// std::cout << "void cSceneSimChar::UpdateCharacters(double time_step)" << std::endl;
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		// 角色更新
		const auto& curr_char = GetCharacter(i);
		curr_char->Update(time_step);
		
		// print torque info
		if (true == mEnableTorqueRecord && mTorqueRecordFile.size() > 0)
		{
			std::ofstream fout;
			fout.open(mTorqueRecordFile.c_str(), std::ios::app);
			if (true == fout.fail())
			{
				std::cout << "[cSceneSimChar] open torque record file " << mTorqueRecordFile << " failed! abort..." << std::endl;
				abort();
			}

			int joints_num = curr_char->GetNumJoints();
			for (int id = 0; id < joints_num; id++)
			{
				const cSimBodyJoint & joint = curr_char->GetJoint(id);
				const tVector & torque = joint.GetTotalTorque();
				fout << "joint " << id << ", torque = " << torque.transpose() << std::endl;
			}
		}
	}
}

void cSceneSimChar::PostUpdateCharacters(double time_step)
{
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		const auto& curr_char = GetCharacter(i);
		curr_char->PostUpdate(time_step);
	}
}

void cSceneSimChar::UpdateGround(double time_elapsed)
{
	tVector view_min;
	tVector view_max;
	GetViewBound(view_min, view_max);
	mGround->Update(time_elapsed, view_min, view_max);
}

void cSceneSimChar::UpdateRandPerturb(double time_step)
{
	mPerturbParams.mTimer += time_step;
	if (mPerturbParams.mTimer >= mPerturbParams.mNextTime)
	{
		ApplyRandForce();
		ResetRandPertrub();
	}
}

void cSceneSimChar::ResetScene()
{
	cScene::ResetScene();

	if (mPerturbParams.mEnableRandPerturbs)
	{
		ResetRandPertrub();
	}
	
	ResetWorld();
	ResetCharacters();
	ResetGround();
	CleanObjs();

	InitCharacterPos();
	ResolveCharGroundIntersect();

	if(mEnableID)mIDSolver->Reset();
	// mOnlineIDSolver->Reset();
	// mOfflineIDSolver->Reset();
}

void cSceneSimChar::ResetCharacters()
{
	int num_chars = GetNumChars();
	for (int i = 0; i < num_chars; ++i)
	{
		const auto& curr_char = GetCharacter(i);
		curr_char->Reset();
	}
}

void cSceneSimChar::ResetWorld()
{
	mWorld->Reset();
}

void cSceneSimChar::ResetGround()
{
	mGround->Clear();

	tVector view_min;
	tVector view_max;
	GetViewBound(view_min, view_max);

	tVector view_size = view_max - view_min;
	view_min = -view_size;
	view_max = view_size;

	view_min[0] += gGroundSpawnOffset;
	view_max[0] += gGroundSpawnOffset;
	view_min[2] += gGroundSpawnOffset;
	view_max[2] += gGroundSpawnOffset;

	mGround->Update(0, view_min, view_max);
}

void cSceneSimChar::PreUpdate(double timestep)
{
	ClearJointForces();
}

void cSceneSimChar::PostUpdate(double timestep)
{
}

void cSceneSimChar::GetViewBound(tVector& out_min, tVector& out_max) const
{
	const std::shared_ptr<cSimCharacter>& character = GetCharacter();
	const cDeepMimicCharController* ctrl = reinterpret_cast<cDeepMimicCharController*>(character->GetController().get());

	out_min.setZero();
	out_max.setZero();
	if (ctrl != nullptr)
	{
		ctrl->GetViewBound(out_min, out_max);
	}
	else
	{
		character->CalcAABB(out_min, out_max);
	}

	out_min += tVector(-gCharViewDistPad, 0, -gCharViewDistPad, 0);
	out_max += tVector(gCharViewDistPad, 0, gCharViewDistPad, 0);
}

void cSceneSimChar::ParseGroundParams(const std::shared_ptr<cArgParser>& parser, cGround::tParams& out_params) const
{
	std::string terrain_file = "";
	parser->ParseString("terrain_file", terrain_file);
	parser->ParseDouble("terrain_blend", out_params.mBlend);

	if (terrain_file != "")
	{
		bool succ = cGroundBuilder::ParseParamsJson(terrain_file, out_params);
		if (!succ)
		{
			printf("Failed to parse terrain params from %s\n", terrain_file.c_str());
			assert(false);
		}
	}
}


void cSceneSimChar::UpdateObjs(double time_step)
{
	int num_objs = GetNumObjs();
	for (int i = 0; i < num_objs; ++i)
	{
		const tObjEntry& obj = mObjs[i];
		if (obj.IsValid() && obj.mEndTime <= GetTime())
		{
			RemoveObj(i);
		}
	}
}

void cSceneSimChar::UpdateJoints(double timestep)
{
	int num_joints = GetNumJoints();
	for (int j = 0; j < num_joints; ++j)
	{
		const tJointEntry& joint = mJoints[j];
		if (joint.IsValid())
		{
			joint.mJoint->ApplyTau();
		}
	}
}

void cSceneSimChar::ClearJointForces()
{
	int num_joints = GetNumJoints();
	for (int j = 0; j < num_joints; ++j)
	{
		const tJointEntry & joint = mJoints[j];
		if (joint.IsValid())
		{
			joint.mJoint->ClearTau();
		}
	}
}

void cSceneSimChar::ClearObjs()
{
	mObjs.Clear();
}

void cSceneSimChar::CleanObjs()
{
	int idx = 0;
	for (int i = 0; i < GetNumObjs(); ++i)
	{
		const tObjEntry& entry = mObjs[i];
		if (entry.IsValid() && !entry.mPersist)
		{
			RemoveObj(i);
		}
	}
}

int cSceneSimChar::AddObj(const tObjEntry& obj_entry)
{
	int handle = static_cast<int>(mObjs.Add(obj_entry));
	return handle;
}

void cSceneSimChar::RemoveObj(int handle)
{
	assert(handle != gInvalidIdx);
	mObjs[handle].mObj.reset();
	mObjs.Free(handle);
}


void cSceneSimChar::ClearJoints()
{
	mJoints.Clear();
}

int cSceneSimChar::AddJoint(const tJointEntry& joint_entry)
{
	int handle = static_cast<int>(mJoints.Add(joint_entry));
	return handle;
}

void cSceneSimChar::RemoveJoint(int handle)
{
	assert(handle != gInvalidIdx);
	mJoints[handle].mJoint.reset();
	mJoints.Free(handle);
}

int cSceneSimChar::GetNumJoints() const
{
	return static_cast<int>(mJoints.GetSize());
}

bool cSceneSimChar::HasFallen(const cSimCharacter& sim_char) const
{
	// 判断准则: 
	// std::cout <<"bool cSceneSimChar::HasFallen(const cSimCharacter& sim_char) const called "<<std::endl;
	bool fallen = sim_char.HasFallen();

	tVector root_pos = sim_char.GetRootPos();
	tVector ground_aabb_min;	// 地面aabb的最小 vec4d
	tVector ground_aabb_max;	// 地面aabb的最大 vec4d
	mGround->CalcAABB(ground_aabb_min, ground_aabb_max);	// 计算地面的aabb包围盒
	ground_aabb_min[1] = -std::numeric_limits<double>::infinity();
	ground_aabb_max[1] = std::numeric_limits<double>::infinity();
	bool in_aabb = cMathUtil::ContainsAABB(root_pos, ground_aabb_min, ground_aabb_max);
	if(false == in_aabb)
	{
		std::cout <<"[end] contact with groung, judged from AABB box " << std::endl;
	}
	fallen |= !in_aabb;

	return fallen;
}

void cSceneSimChar::SpawnProjectile()
{
	double density = 100;
	double min_size = 0.1;
	double max_size = 0.3;
	double min_speed = 10;
	double max_speed = 20;
	double life_time = 2;
	double y_offset = 0;
	SpawnProjectile(density, min_size, max_size, min_speed, max_speed, y_offset, life_time);
}

void cSceneSimChar::SpawnBigProjectile()
{
	double density = 100;
	double min_size = 1.25;
	double max_size = 1.75;
	double min_speed = 11;
	double max_speed = 12;
	double life_time = 2;
	double y_offset = 0.5;
	SpawnProjectile(density, min_size, max_size, min_speed, max_speed, y_offset, life_time);
}

int cSceneSimChar::GetNumObjs() const
{
	return static_cast<int>(mObjs.GetCapacity());
}

const std::shared_ptr<cSimRigidBody>& cSceneSimChar::GetObj(int id) const
{
	return mObjs[id].mObj;
}

const cSceneSimChar::tObjEntry& cSceneSimChar::GetObjEntry(int id) const
{
	return mObjs[id];
}

void cSceneSimChar::SetRandSeed(unsigned long seed)
{
	cScene::SetRandSeed(seed);
	if (mGround != nullptr)
	{
		mGround->SeedRand(seed);
	}
}

void cSceneSimChar::SpawnProjectile(double density, double min_size, double max_size,
									double min_speed, double max_speed, double y_offset,
									double life_time)
{
	double min_dist = 1;
	double max_dist = 2;
	tVector aabb_min;
	tVector aabb_max;

	int char_id = mRand.RandInt(0, GetNumChars());
	const auto& curr_char = GetCharacter(char_id);
	curr_char->CalcAABB(aabb_min, aabb_max);

	tVector aabb_center = (aabb_min + aabb_max) * 0.5;
	tVector obj_size = tVector(1, 1, 1, 0) * mRand.RandDouble(min_size, max_size);
	
	double rand_theta = mRand.RandDouble(0, M_PI);
	double rand_dist = mRand.RandDouble(min_dist, max_dist);

	double aabb_size_x = (aabb_max[0] - aabb_min[0]);
	double aabb_size_z = (aabb_max[2] - aabb_min[2]);
	double buffer_dist = std::sqrt(aabb_size_x * aabb_size_x + aabb_size_z * aabb_size_z);

	double rand_x = 0.5 * buffer_dist + rand_dist * std::cos(rand_theta);
	rand_x *= mRand.RandSign();
	rand_x += aabb_center[0];
	double rand_y = mRand.RandDouble(aabb_min[1], aabb_max[1]) + obj_size[1] * 0.5;
	rand_y += y_offset;

	double rand_z = aabb_center[2];
	rand_z = 0.5 * buffer_dist + rand_dist * std::sin(rand_theta);
	rand_z *= mRand.RandSign();
	rand_z += aabb_center[2];

	tVector pos = tVector(rand_x, rand_y, rand_z, 0);
	tVector target = tVector(mRand.RandDouble(aabb_min[0], aabb_max[0]),
		mRand.RandDouble(aabb_min[1], aabb_max[1]), aabb_center[2], 0);

	tVector com_vel = curr_char->CalcCOMVel();
	tVector vel = (target - pos).normalized();
	vel *= mRand.RandDouble(min_speed, max_speed);
	vel[0] += com_vel[0];
	vel[2] += com_vel[2];

	cSimBox::tParams params;
	params.mSize = obj_size;
	params.mPos = pos;
	params.mVel = vel;
	params.mFriction = 0.7;
	params.mMass = density * params.mSize[0] * params.mSize[1] * params.mSize[2];
	std::shared_ptr<cSimBox> box = std::shared_ptr<cSimBox>(new cSimBox());
	box->Init(mWorld, params);
	box->UpdateContact(cWorld::eContactFlagObject, cContactManager::gFlagNone);

	tObjEntry obj_entry;
	obj_entry.mObj = box;
	obj_entry.mEndTime = GetTime() + life_time;
	
	AddObj(obj_entry);
}

void cSceneSimChar::ResetRandPertrub()
{
	mPerturbParams.mTimer = 0;
	mPerturbParams.mNextTime = mRand.RandDouble(mPerturbParams.mTimeMin, mPerturbParams.mTimeMax);
}
