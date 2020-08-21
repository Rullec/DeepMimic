﻿#include "SceneSimChar.h"
#include "SceneImitate.h"
#include "sim/Controller/CtPDGenController.h"
#include "sim/Controller/DeepMimicCharController.h"
#include "sim/SimItems/SimBox.h"
#include "sim/SimItems/SimCharacter.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/TrajManager/BuildIDSolver.hpp"
#include "sim/TrajManager/TrajRecorder.h"
#include "sim/World/GroundBuilder.h"
#include "sim/World/GroundPlane.h"
#include "sim/World/WorldBuilder.h"
#include "util/FileUtil.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <util/BulletUtil.h>
#include <util/TimeUtil.hpp>
using namespace std;

const int gDefaultCharID = 0;
const double gCharViewDistPad = 1;
const double cSceneSimChar::gGroundSpawnOffset =
    -1; // some padding to prevent parts of character from getting spawned
        // inside obstacles

const size_t gInitGroundUpdateCount = std::numeric_limits<size_t>::max();

cSceneSimChar::tObjEntry::tObjEntry()
{
    mObj = nullptr;
    mEndTime = std::numeric_limits<double>::infinity();
    mColor = tVector(0.5, 0.5, 0.5, 1);
    mPersist = false;
}

bool cSceneSimChar::tObjEntry::IsValid() const { return mObj != nullptr; }

cSceneSimChar::tJointEntry::tJointEntry() { mJoint = nullptr; }

bool cSceneSimChar::tJointEntry::IsValid() const { return mJoint != nullptr; }

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
    mEnablePDTargetSolveTest = false;
    mTorqueRecordFile = "";
    mEnableJointTorqueControl = true;
    mEnableID = false;
    mEnableTrajRecord = false;
    mTrajRecorder = nullptr;

    mWorldParams.mGenWorldConfig = "args/world_config/sim_config.json";
    mWorldParams.mNumSubsteps = 1;
    mWorldParams.mScale = 1;
    mWorldParams.mGravity = gGravity;
}

cSceneSimChar::~cSceneSimChar() { Clear(); }

void cSceneSimChar::ParseArgs(const std::shared_ptr<cArgParser> &parser)
{
    cScene::ParseArgs(parser);

    bool succ = true;

    parser->ParseBool(
        "enable_char_contact_fall",
        mEnableContactFall); // 打开角色接触掉落, 这东西默认是开启的。
    parser->ParseBool("enable_rand_char_placement", mEnableRandCharPlacement);
    parser->ParseBool("enable_torque_record", mEnableTorqueRecord);
    parser->ParseString("torque_record_file", mTorqueRecordFile);
    parser->ParseBool("enable_joint_force_control", mEnableJointTorqueControl);
    parser->ParseBool("enable_pdtarget_solve_test", mEnablePDTargetSolveTest);
    parser->ParseBool("pause_at_first", mPauseAtFirst);

    succ &= ParseCharTypes(parser, mCharTypes);
    succ &= ParseCharParams(parser, mCharParams);
    succ &= ParseCharCtrlParams(parser, mCtrlParams);
    if (mCharParams.size() != mCtrlParams.size())
    {
        printf("Char and ctrl file mismatch, %zi vs %zi\n", mCharParams.size(),
               mCtrlParams.size());
        assert(false);
    }

    std::string sim_mode_str = "";
    parser->ParseInt("num_sim_substeps", mWorldParams.mNumSubsteps);
    MIMIC_ASSERT(
        mWorldParams.mNumSubsteps == 1 &&
        "In order to do InverseDynamics, sim_substeps are forced to be 1");
    parser->ParseDouble("world_scale", mWorldParams.mScale);
    parser->ParseVector("gravity", mWorldParams.mGravity);
    parser->ParseString("world_type", mWorldParams.mWorldType);

    parser->ParseBool("enable_rand_perturbs",
                      mPerturbParams.mEnableRandPerturbs);
    parser->ParseDouble("perturb_time_min", mPerturbParams.mTimeMin);
    parser->ParseDouble("perturb_time_max", mPerturbParams.mTimeMax);
    parser->ParseDouble("min_perturb", mPerturbParams.mMinPerturb);
    parser->ParseDouble("max_perturb", mPerturbParams.mMaxPerturb);
    parser->ParseDouble("min_pertrub_duration", mPerturbParams.mMinDuration);
    parser->ParseDouble("max_perturb_duration", mPerturbParams.mMaxDuration);
    parser->ParseInts("perturb_part_ids", mPerturbParams.mPerturbPartIDs);

    parser->ParseInts("fall_contact_bodies",
                      mFallContactBodies); // 哪几个link检测掉落?

    ParseGroundParams(parser, mGroundParams);

    mEnableID = false;
    mIDInfoPath = "";
    mArgParser->ParseBool("enable_inverse_dynamic_solving", mEnableID);
    mArgParser->ParseString("inverse_dynamic_config_file", mIDInfoPath);
    if (mEnableID == true && false == cFileUtil::ExistsFile(mIDInfoPath))
    {
        std::cout << "[error] cSceneSimChar::ParseArgs failed for enable "
                     "id but conf path is illegal: "
                  << mIDInfoPath << std::endl;
        ;
        exit(1);
    }

    mEnableTrajRecord = false;
    mTrajRecorderConfig = "";
    mArgParser->ParseBool("enable_traj_recoder", mEnableTrajRecord);
    mArgParser->ParseString("traj_recorder_config", mTrajRecorderConfig);
    if (mEnableTrajRecord == true &&
        false == cFileUtil::ExistsFile(mTrajRecorderConfig))
    {
        MIMIC_ERROR("traj recorder config {} doesn't exist",
                    mTrajRecorderConfig);
    }

    // enable guided control or not
    mArgParser->ParseBool("enable_guided_control", mEnableGuidedControl);
    mArgParser->ParseString("guided_traj_file", mGuidedTrajFile);
}

void cSceneSimChar::Init()
{
    cScene::Init();

    if (mPerturbParams.mEnableRandPerturbs)
    {
        ResetRandPertrub();
    }

    BuildWorld();
    BuildGround();
    BuildCharacters();

    auto &cur_char = GetCharacter();
    auto multibody = dynamic_cast<cSimCharacterGen *>(cur_char.get());
    // Init the position of our character, accoridng to the ref motion
    InitCharacterPos();
    ResolveCharGroundIntersect();
    BuildTrajManager();
    ClearObjs();

    // std::cout << "root rot = " <<
    // sim_char->GetRootRotation().coeffs().transpose() << std::endl; std::cout
    // << "root pos = " << sim_char->GetRootPos().transpose() << std::endl;
    // exit(1);
}

void cSceneSimChar::Clear()
{
    cScene::Clear();

    mChars.clear();
    mGround.reset();
    mFallContactBodies.clear();
    ClearObjs();
}

void cSceneSimChar::Update(double time_elapsed)
{
    // MIMIC_DEBUG("timestep = {}", time_elapsed);
    auto &sim_char = GetCharacter();
    auto multibody = dynamic_cast<cSimCharacterGen *>(sim_char.get());
    // std::cout << "[update] pose 0 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    // std::cout << "[update] q 0 = " << multibody->Getq().transpose()
    //           << std::endl;
    // std::cout << "[update] qdot 0 = " << multibody->Getqdot().transpose()
    //           << std::endl;
    // exit(0);
    cScene::Update(time_elapsed);
    // std::cout << "[update] pose 1 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    if (time_elapsed < 0)
    {
        return;
    }

    if (mPerturbParams.mEnableRandPerturbs)
    {
        UpdateRandPerturb(time_elapsed);
    }

    // std::cout << "[update] pose 2 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    PreUpdate(time_elapsed); // clear joint torque
    // 显示一下速度：是不是最开始的时候设置的速度太大了?
    // std::cout << "[update] pose 3 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    UpdateCharacters(time_elapsed);
    // std::cout << "[update] pose 4 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    if (mEnableID)
    {
        mIDSolver->SetTimestep(time_elapsed);
        mIDSolver->PreSim();
    }
    if (mEnableTrajRecord)
    {
        mTrajRecorder->SetTimestep(time_elapsed);
        mTrajRecorder->PreSim();
    }
    // std::cout << "[update] pose 5 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    UpdateWorld(time_elapsed);
    // std::cout << "[update] pose 6 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    UpdateGround(time_elapsed);
    UpdateObjs(time_elapsed);
    // std::cout << "[update] pose 7 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    // std::cout << "[update] q 7 = " << multibody->Getq().transpose()
    //           << std::endl;
    PostUpdateCharacters(time_elapsed);
    // std::cout << "[update] pose 7.5 = " << sim_char->GetPose().transpose()
    //           << std::endl;
    // std::cout << "[update] q 7.5 = " << multibody->Getq().transpose()
    //           << std::endl;
    // exit(1);
    if (mEnableID)
        mIDSolver->PostSim();
    if (mEnableTrajRecord)
    {
        mTrajRecorder->PostSim();
    }
    PostUpdate(time_elapsed);
    // std::cout << "[update] pose 8 = " << sim_char->GetPose().transpose()
    //           << std::endl;

    // cTimeUtil::End("sim update");
}

int cSceneSimChar::GetNumChars() const
{
    return static_cast<int>(mChars.size());
}

const std::shared_ptr<cSimCharacterBase> &cSceneSimChar::GetCharacter() const
{
    return GetCharacter(gDefaultCharID);
}

const std::shared_ptr<cSimCharacterBase> &
cSceneSimChar::GetCharacter(int char_id) const
{
    return mChars[char_id];
}

const std::shared_ptr<cWorldBase> &cSceneSimChar::GetWorld() const
{
    return mWorld;
}

tVector cSceneSimChar::GetCharPos() const
{
    return GetCharacter()->GetRootPos();
}

const std::shared_ptr<cGround> &cSceneSimChar::GetGround() const
{
    return mGround;
}

const tVector &cSceneSimChar::GetGravity() const
{
    return mWorldParams.mGravity;
}

bool cSceneSimChar::LoadControlParams(
    const std::string &param_file,
    const std::shared_ptr<cSimCharacterBase> &out_char)
{
    const auto &ctrl = out_char->GetController();
    bool succ = ctrl->LoadParams(param_file);
    return succ;
}

void cSceneSimChar::AddPerturb(const tPerturb &perturb)
{
    mWorld->AddPerturb(perturb);
}

void cSceneSimChar::ApplyRandForce(double min_force, double max_force,
                                   double min_dur, double max_dur, cSimObj *obj)
{
    assert(obj != nullptr);
    tPerturb perturb = tPerturb::BuildForce();
    perturb.mObj = obj;
    perturb.mLocalPos.setZero();
    perturb.mPerturb[0] = mRand.RandDouble(-1, 1);
    perturb.mPerturb[1] = mRand.RandDouble(-1, 1);
    perturb.mPerturb[2] = mRand.RandDouble(-1, 1);
    perturb.mPerturb =
        mRand.RandDouble(min_force, max_force) * perturb.mPerturb.normalized();
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
    const std::shared_ptr<cSimCharacterBase> &curr_char =
        Downcast(GetCharacter(char_id));
    int num_parts = curr_char->GetNumBodyParts();
    int part_idx = GetRandPerturbPartID(curr_char);
    assert(part_idx != gInvalidIdx);
    const auto &part = curr_char->GetBodyPart(part_idx);
    ApplyRandForce(mPerturbParams.mMinPerturb, mPerturbParams.mMaxPerturb,
                   mPerturbParams.mMinDuration, mPerturbParams.mMaxDuration,
                   part.get());
}

int cSceneSimChar::GetRandPerturbPartID(
    const std::shared_ptr<cSimCharacterBase> &character)
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

void cSceneSimChar::RayTest(const tVector &beg, const tVector &end,
                            cFeaWorld::tRayTestResult &out_result) const
{
    cFeaWorld::tRayTestResults results;
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

void cSceneSimChar::OutputCharState(const std::string &out_file) const
{
    const auto &char0 = GetCharacter();
    tVector root_pos = char0->GetRootPos();
    double ground_h = mGround->SampleHeight(root_pos);
    tMatrix trans = char0->BuildOriginTrans();
    trans(1, 3) -= ground_h;

    char0->WriteState(out_file, trans);
}

void cSceneSimChar::OutputGround(const std::string &out_file) const
{
    mGround->Output(out_file);
}

std::string cSceneSimChar::GetName() const { return "Sim Character"; }

#include "sim/Controller/CtPDController.h"
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
        const cSimCharacterBase::tParams &curr_params =
            mCharParams[i]; // 这叫做当前参数curr_params
        std::shared_ptr<cSimCharacterBase> curr_char;

        // 为什么角色还有一个builder?
        cSimCharBuilder::eCharType char_type = cSimCharBuilder::eCharInvalid;
        if (mCharTypes.size() > i)
        {
            char_type = mCharTypes[i];
        }
        cSimCharBuilder::CreateCharacter(char_type, curr_char);

        succ &= curr_char->Init(mWorld, curr_params);
        if (succ)
        {
            SetFallContacts(mFallContactBodies, curr_char);

            curr_char->RegisterContacts(cFeaWorld::eContactFlagCharacter,
                                        cFeaWorld::eContactFlagEnvironment);

            // std::cout << "[init] pose 0 = " <<
            // curr_char->GetPose().transpose()
            //           << std::endl;
            InitCharacterPos(curr_char);
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
                    auto ct_pd_controller =
                        std::dynamic_pointer_cast<cCtPDController>(ctrl);
                    if (ct_pd_controller != nullptr)
                    {
                        ct_pd_controller->GetImpPDController()
                            .SetEnableSolvePDTargetTest(
                                mEnablePDTargetSolveTest);
                        MIMIC_WARN("set EnablePDTargetSolveTest = {}",
                                   std::to_string(mEnablePDTargetSolveTest));
                    }
                    else
                    {
                        MIMIC_WARN("ignore Flag EnablePDTargetSolveTest");
                    }
                }
            }
            mChars.push_back(curr_char);
        }

        // set up other setting
        curr_char->SetEnablejointTorqueControl(mEnableJointTorqueControl);

        if (mEnableGuidedControl == true)
        {
            MIMIC_INFO("guided control enabled");
            auto gen_ctrl = std::dynamic_pointer_cast<cCtPDGenController>(
                curr_char->GetController());
            MIMIC_ASSERT(gen_ctrl != nullptr);
            gen_ctrl->SetGuidedControlInfo(mEnableGuidedControl,
                                           mGuidedTrajFile);
        }
    }
    // std::cout << "[init] pose 1 = " << GetCharacter()->GetPose().transpose()
    //           << std::endl;
    return succ;
}

bool cSceneSimChar::ParseCharTypes(
    const std::shared_ptr<cArgParser> &parser,
    std::vector<cSimCharBuilder::eCharType> &out_types) const
{
    bool succ = true;
    std::vector<std::string> char_type_strs;
    succ = parser->ParseStrings("char_types", char_type_strs);

    int num = static_cast<int>(char_type_strs.size());
    out_types.clear();
    for (int i = 0; i < num; ++i)
    {
        std::string str = char_type_strs[i];
        cSimCharBuilder::eCharType char_type = cSimCharBuilder::eCharInvalid;
        cSimCharBuilder::ParseCharType(str, char_type);

        if (char_type != cSimCharBuilder::eCharInvalid)
        {
            out_types.push_back(char_type);
        }
    }

    return succ;
}

bool cSceneSimChar::ParseCharParams(
    const std::shared_ptr<cArgParser> &parser,
    std::vector<cSimCharacterBase::tParams> &out_params) const
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
        cSimCharacterBase::tParams &params = out_params[i];
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

bool cSceneSimChar::ParseCharCtrlParams(
    const std::shared_ptr<cArgParser> &parser,
    std::vector<cCtrlBuilder::tCtrlParams> &out_params) const
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
        auto &ctrl_params = out_params[i];
        const std::string &type_str = char_ctrl_strs[i];
        cCtrlBuilder::ParseCharCtrl(type_str, ctrl_params.mCharCtrl);
        ctrl_params.mCtrlFile = ctrl_files[i];
    }

    return succ;
}

void cSceneSimChar::BuildWorld()
{
    cWorldBuilder::BuildWorld(mWorld, mWorldParams);
    mWorld->Init(mWorldParams);
}

void cSceneSimChar::BuildGround()
{
    mGroundParams.mHasRandSeed = mHasRandSeed;
    mGroundParams.mRandSeed = mRandSeed;
    cGroundBuilder::BuildGround(mWorld, mGroundParams, mGround);
}

bool cSceneSimChar::BuildController(
    const cCtrlBuilder::tCtrlParams &ctrl_params,
    std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = cCtrlBuilder::BuildController(ctrl_params, out_ctrl);
    return succ;
}

void cSceneSimChar::SetFallContacts(
    const std::vector<int> &fall_bodies,
    std::shared_ptr<cSimCharacterBase> &out_char) const
{
    // 这个函数负责注册fall bodies
    for (int i = 0; i < out_char->GetNumBodyParts(); ++i)
    {
        out_char->SetBodyPartFallContact(i, false);
    }
    int num_fall_bodies = static_cast<int>(fall_bodies.size());
    if (num_fall_bodies > 0) // 如果这个这个目标不是空的
    {
        for (int i = 0; i < num_fall_bodies; ++i)
        {
            int b = fall_bodies[i];
            out_char->SetBodyPartFallContact(
                b, true); // 那么就把这些数字都定义为有contact
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

void cSceneSimChar::InitCharacterPos(
    const std::shared_ptr<cSimCharacterBase> &out_char)
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

void cSceneSimChar::InitCharacterPosFixed(
    const std::shared_ptr<cSimCharacterBase> &out_char)
{
    tVector root_pos = out_char->GetRootPos();
    int char_id = out_char->GetID();
    root_pos[0] = mCharParams[char_id].mInitPos[0];

    double h = mGround->SampleHeight(root_pos);
    root_pos[1] += h;

    out_char->SetRootPos(root_pos);
}

/**
 * \brief               Build Inverse Dynamic Solver or Trajecotry manager
 */
void cSceneSimChar::BuildTrajManager()
{
    if (true == mEnableID)
    {
        // build inverse dynamics
        auto sim_char = this->GetCharacter(0);

        auto scene_imitate_ptr = dynamic_cast<cSceneImitate *>(this);
        if (scene_imitate_ptr == nullptr)
        {
            std::cout << "[error] cSceneSimChar::BuildTrajManager can only "
                         "be finished when cSceneImitate is instanced\n";
            exit(1);
        }
        auto kin_char = scene_imitate_ptr->GetKinChar();
        mIDSolver = BuildIDSolver(mIDInfoPath, scene_imitate_ptr);

        // offline mode: read trajectory from files, then solve it.
        // online mode for debug: start with the simulation at the same time,
        // record each state and solve them at onece then compare the desired ID
        // result and the ideal one. It will be very easy to debug.
        if (eIDSolverType::OfflineSolve == mIDSolver->GetType())
            std::cout << "[log] Inverse Dynamics runs in offlineSolve mode."
                      << std::endl;
        else if (eIDSolverType::Online == mIDSolver->GetType())
            std::cout << "[log] Inverse Dynamics runs in online mode."
                      << std::endl;
        else if (eIDSolverType::Display == mIDSolver->GetType())
            std::cout << "[log] Inverse Dynamics runs in display mode."
                      << std::endl;
        else
        {
            std::cout << "unrecognized ID solver mode = "
                      << mIDSolver->GetType() << std::endl;
            exit(1);
        }
    }

    if (true == mEnableTrajRecord)
    {
        mTrajRecorder = new cTrajRecorder(dynamic_cast<cSceneImitate *>(this),
                                          mTrajRecorderConfig);
    }
    //
}

void cSceneSimChar::SetCharRandPlacement(
    const std::shared_ptr<cSimCharacterBase> &out_char)
{
    tVector rand_pos = tVector::Zero();
    tQuaternion rand_rot = tQuaternion::Identity();
    CalcCharRandPlacement(out_char, rand_pos, rand_rot);
    // MIMIC_DEBUG("SetCharRandPlacement, root pos {}, root rot {}",
    //             rand_pos.transpose(), rand_rot.coeffs().transpose());
    out_char->SetRootTransform(rand_pos, rand_rot);
}

void cSceneSimChar::CalcCharRandPlacement(
    const std::shared_ptr<cSimCharacterBase> &out_char, tVector &out_pos,
    tQuaternion &out_rot)
{
    tVector char_pos = out_char->GetRootPos();
    tQuaternion char_rot = out_char->GetRootRotation();

    tVector rand_pos;
    tQuaternion rand_rot;
    mGround->SamplePlacement(tVector::Zero(), rand_pos, rand_rot);
    // std::cout << "ground sample random pos = " << rand_pos.transpose()
    //           << std::endl;
    // std::cout << "raw out pos = " << out_pos.transpose() << std::endl;
    // std::cout << "get char pos = " << char_pos.transpose() << std::endl;
    out_pos = rand_pos;
    out_pos[1] += char_pos[1];
    out_rot = rand_rot * char_rot;
    // std::cout << "new out pos = " << out_pos.transpose() << std::endl;
}

void cSceneSimChar::ResolveCharGroundIntersect()
{
    // for characters
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto &sim_char =
            std::dynamic_pointer_cast<cSimCharacterBase>(mChars[i]);
        ResolveCharGroundIntersect(sim_char);
    }
}

void cSceneSimChar::ResolveCharGroundIntersect(
    const std::shared_ptr<cSimCharacterBase> &out_char) const
{
    // 为了防止初始状态和地面有碰撞，加上去。
    const double pad = 0.001;

    int num_parts = out_char->GetNumBodyParts();
    double min_violation = 0;
    for (int b = 0; b < num_parts; ++b)
    {
        // if this body part is valid
        if (out_char->IsValidBodyPart(b))
        {
            tVector aabb_min; // smallest values
            tVector aabb_max; // biggest values
            const auto &part = out_char->GetBodyPart(b);
            part->CalcAABB(aabb_min,
                           aabb_max); // calculate the AABB at this
                                      // momentum (reply on bullet API)

            tVector mid =
                0.5 * (aabb_min + aabb_max); // find the center of this box
            tVector sw =
                tVector(aabb_min[0], 0, aabb_min[2],
                        0); // ignore y axis, find 4 corner points in XOZ plane
            tVector nw = tVector(aabb_min[0], 0, aabb_max[2], 0);
            tVector ne = tVector(aabb_max[0], 0, aabb_max[2], 0);
            tVector se = tVector(aabb_max[0], 0, aabb_min[2], 0);

            double max_ground_height = 0;
            max_ground_height =
                mGround->SampleHeight(aabb_min); // find the max ground height
            max_ground_height =
                std::max(max_ground_height, mGround->SampleHeight(mid));
            max_ground_height =
                std::max(max_ground_height, mGround->SampleHeight(sw));
            max_ground_height =
                std::max(max_ground_height, mGround->SampleHeight(nw));
            max_ground_height =
                std::max(max_ground_height, mGround->SampleHeight(ne));
            max_ground_height =
                std::max(max_ground_height, mGround->SampleHeight(se));
            max_ground_height += pad; // avoid collision gap

            double min_height = aabb_min[1]; // it is the lowest height
                                             // for character bodies
            min_violation = std::min(
                min_violation,
                min_height - max_ground_height); // get a "minus" biggest value
                                                 // for violation. -999
        }
    }

    // it violation occurs
    if (min_violation < 0)
    {
        // here is a root pos, uplift our body
        tVector root_pos = out_char->GetRootPos();
        root_pos[1] += -min_violation;
        out_char->SetRootPos(root_pos);
    }
}

void cSceneSimChar::UpdateWorld(double time_step) { mWorld->Update(time_step); }

void cSceneSimChar::UpdateCharacters(double time_step)
{
    /*
            1. compute torque by PD target
            2. apply these torques to joints, then bullet links
    */
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        // 角色更新
        const auto &curr_char = GetCharacter(i);
        curr_char->Update(time_step);

        // print torque info
        if (true == mEnableTorqueRecord && mTorqueRecordFile.size() > 0)
        {
            std::ofstream fout;
            fout.open(mTorqueRecordFile.c_str(), std::ios::app);
            MIMIC_ASSERT(true != fout.fail());

            int joints_num = curr_char->GetNumJoints();
            for (int id = 0; id < joints_num; id++)
            {
                const cSimBodyJoint &joint = curr_char->GetJoint(id);
                const tVector &torque = joint.GetTotalTorque();
                fout << "joint " << id << ", torque = " << torque.transpose()
                     << std::endl;
            }
        }
    }
}

void cSceneSimChar::PostUpdateCharacters(double time_step)
{
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto &curr_char = GetCharacter(i);
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

// extern int reset_cnt;
void cSceneSimChar::ResetScene()
{
    MIMIC_DEBUG("reset scene!");
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
    if (mEnableID)
        mIDSolver->Reset();
    if (mEnableTrajRecord)
        mTrajRecorder->Reset();
}

void cSceneSimChar::ResetCharacters()
{
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto &curr_char = GetCharacter(i);
        curr_char->Reset();
    }
}

void cSceneSimChar::ResetWorld() { mWorld->Reset(); }

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
    // ClearJointForces();
}

extern bool gAnimating;
void cSceneSimChar::PostUpdate(double timestep)
{

    mWorld->PostUpdate();
    // MIMIC_WARN("poseupdate, get time {}, timestep {}", GetTime(), timestep);
    if (mPauseAtFirst == true && std::fabs(GetTime() - timestep) < 1e-10)
        gAnimating = false;
}

void cSceneSimChar::GetViewBound(tVector &out_min, tVector &out_max) const
{
    const std::shared_ptr<cSimCharacterBase> &character = GetCharacter();
    const cDeepMimicCharController *ctrl =
        reinterpret_cast<cDeepMimicCharController *>(
            character->GetController().get());

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

void cSceneSimChar::ParseGroundParams(const std::shared_ptr<cArgParser> &parser,
                                      cGround::tParams &out_params) const
{
    std::string terrain_file = "";
    parser->ParseString("terrain_file", terrain_file);
    parser->ParseDouble("terrain_blend", out_params.mBlend);

    if (terrain_file != "")
    {
        bool succ = cGroundBuilder::ParseParamsJson(terrain_file, out_params);
        if (!succ)
        {
            printf("Failed to parse terrain params from %s\n",
                   terrain_file.c_str());
            assert(false);
        }
    }
}

void cSceneSimChar::UpdateObjs(double time_step)
{
    int num_objs = GetNumObjs();
    for (int i = 0; i < num_objs; ++i)
    {
        const tObjEntry &obj = mObjs[i];
        if (obj.IsValid() && obj.mEndTime <= GetTime())
        {
            RemoveObj(i);
        }
    }
}

void cSceneSimChar::ClearObjs() { mObjs.Clear(); }

void cSceneSimChar::CleanObjs()
{
    int idx = 0;
    for (int i = 0; i < GetNumObjs(); ++i)
    {
        const tObjEntry &entry = mObjs[i];
        if (entry.IsValid() && !entry.mPersist)
        {
            RemoveObj(i);
        }
    }
}

int cSceneSimChar::AddObj(const tObjEntry &obj_entry)
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

bool cSceneSimChar::HasFallen(const cSimCharacterBase &sim_char) const
{
    // 判断准则:
    // std::cout <<"bool cSceneSimChar::HasFallen(const cSimCharacterBase&
    // sim_char) const called "<<std::endl;
    bool fallen = sim_char.HasFallen();

    tVector root_pos = sim_char.GetRootPos();
    tVector ground_aabb_min; // 地面aabb的最小 vec4d
    tVector ground_aabb_max; // 地面aabb的最大 vec4d
    mGround->CalcAABB(ground_aabb_min, ground_aabb_max); // 计算地面的aabb包围盒
    ground_aabb_min[1] = -std::numeric_limits<double>::infinity();
    ground_aabb_max[1] = std::numeric_limits<double>::infinity();
    bool in_aabb =
        cMathUtil::ContainsAABB(root_pos, ground_aabb_min, ground_aabb_max);
    if (false == in_aabb)
    {
        std::cout << "[end] contact with groung, judged from AABB box "
                  << std::endl;
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
    SpawnProjectile(density, min_size, max_size, min_speed, max_speed, y_offset,
                    life_time);
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
    SpawnProjectile(density, min_size, max_size, min_speed, max_speed, y_offset,
                    life_time);
}

int cSceneSimChar::GetNumObjs() const
{
    return static_cast<int>(mObjs.GetCapacity());
}

const std::shared_ptr<cSimRigidBody> &cSceneSimChar::GetObj(int id) const
{
    return mObjs[id].mObj;
}

const cSceneSimChar::tObjEntry &cSceneSimChar::GetObjEntry(int id) const
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

void cSceneSimChar::SpawnProjectile(double density, double min_size,
                                    double max_size, double min_speed,
                                    double max_speed, double y_offset,
                                    double life_time)
{
    double min_dist = 1;
    double max_dist = 2;
    tVector aabb_min;
    tVector aabb_max;

    int char_id = mRand.RandInt(0, GetNumChars());
    const auto &curr_char = GetCharacter(char_id);
    curr_char->CalcAABB(aabb_min, aabb_max);

    tVector aabb_center = (aabb_min + aabb_max) * 0.5;
    tVector obj_size =
        tVector(1, 1, 1, 0) * mRand.RandDouble(min_size, max_size);

    double rand_theta = mRand.RandDouble(0, M_PI);
    double rand_dist = mRand.RandDouble(min_dist, max_dist);

    double aabb_size_x = (aabb_max[0] - aabb_min[0]);
    double aabb_size_z = (aabb_max[2] - aabb_min[2]);
    double buffer_dist =
        std::sqrt(aabb_size_x * aabb_size_x + aabb_size_z * aabb_size_z);

    double rand_x = 0.5 * buffer_dist + rand_dist * std::cos(rand_theta);
    rand_x *= mRand.RandSign();
    rand_x += aabb_center[0];
    double rand_y =
        mRand.RandDouble(aabb_min[1], aabb_max[1]) + obj_size[1] * 0.5;
    rand_y += y_offset;

    double rand_z = aabb_center[2];
    rand_z = 0.5 * buffer_dist + rand_dist * std::sin(rand_theta);
    rand_z *= mRand.RandSign();
    rand_z += aabb_center[2];

    tVector pos = tVector(rand_x, rand_y, rand_z, 0);
    tVector target =
        tVector(mRand.RandDouble(aabb_min[0], aabb_max[0]),
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
    params.mMass =
        density * params.mSize[0] * params.mSize[1] * params.mSize[2];
    std::shared_ptr<cSimBox> box = std::shared_ptr<cSimBox>(new cSimBox());
    box->Init(mWorld, params);
    box->UpdateContact(cFeaWorld::eContactFlagObject,
                       cContactManager::gFlagNone);

    tObjEntry obj_entry;
    obj_entry.mObj = box;
    obj_entry.mEndTime = GetTime() + life_time;

    AddObj(obj_entry);
}

void cSceneSimChar::ResetRandPertrub()
{
    mPerturbParams.mTimer = 0;
    mPerturbParams.mNextTime =
        mRand.RandDouble(mPerturbParams.mTimeMin, mPerturbParams.mTimeMax);
}
