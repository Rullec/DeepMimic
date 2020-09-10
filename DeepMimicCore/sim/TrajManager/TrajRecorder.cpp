#include "TrajRecorder.h"
#include "InteractiveIDSolver.hpp"
#include "scenes/SceneImitate.h"
#include "sim/Controller/CtPDController.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/World/FeaWorld.h"
#include "sim/World/GenWorld.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>

tLoadInfo::tLoadInfo()
{
    mLoadPath = "";
    mLoadMode = eLoadMode::INVALID;
    mPoseMat.resize(0, 0);
    mVelMat.resize(0, 0);
    mAccelMat.resize(0, 0);
    mPDTargetMat.resize(0, 0);
    mCharPoseMat.resize(0, 0);
    mMotion = nullptr;
    mTimesteps.resize(0);
    mRewards.resize(0);
    mMotionRefTime.resize(0);
    mContactForces.clear();
    mLinkRot.clear();
    mLinkPos.clear();
    mExternalForces.clear();
    mExternalTorques.clear();
    mTotalFrame = 0;
    mCurFrame = 0;
    mEnableOutputMotionInfo = false;
    mOutputMotionInfoPath = "";
}

void tLoadInfo::LoadTrajV1(cSimCharacterBase *sim_char, const std::string &path)
{
    auto &raw_pose = sim_char->GetPose();
    Json::Value data_json, list_json;
    bool succ = cJsonUtil::LoadJson(path, data_json);
    mLoadPath = path;
    if (!succ)
    {
        MIMIC_ERROR("LoadTrajV1 parse json {} failed", path);
        exit(0);
    }

    list_json = data_json["list"];
    assert(list_json.isNull() == false);
    // std::cout <<"get list json, begin set up pramas\n";
    const int &target_version = cJsonUtil::ParseAsInt("version", data_json);

#ifndef IGNORE_TRAJ_VERISION
    if (1 != target_version)
    {
        MIMIC_ERROR(
            "LoadTrajV1 is called but the spcified fileversion in {} is {}",
            path.c_str(), target_version);
        exit(1);
    }
#endif

    int num_of_frames = list_json.size();
    int num_of_links = sim_char->GetNumBodyParts();
    int pose_size = sim_char->GetPose().size();
    auto ctrl =
        dynamic_cast<cCtPDController *>(sim_char->GetController().get());
    {
        mTotalFrame = num_of_frames;
        mPoseMat.resize(num_of_frames, pose_size), mPoseMat.setZero();
        mVelMat.resize(num_of_frames, pose_size), mVelMat.setZero();
        mAccelMat.resize(num_of_frames, pose_size), mAccelMat.setZero();
        mActionMat.resize(num_of_frames, ctrl->GetActionSize()),
            mActionMat.setZero();
        mPDTargetMat.resize(num_of_frames, ctrl->GetActionSize()),
            mPDTargetMat.setZero();
        mContactForces.resize(num_of_frames);
        for (auto &x : mContactForces)
            x.clear();
        mLinkRot.resize(num_of_frames);
        for (auto &x : mLinkRot)
            x.resize(num_of_links);
        mLinkPos.resize(num_of_frames);
        for (auto &x : mLinkPos)
            x.resize(num_of_links);
        mExternalForces.resize(num_of_frames);
        for (auto &x : mExternalForces)
            x.resize(num_of_links);
        mExternalTorques.resize(num_of_frames);
        for (auto &x : mExternalTorques)
            x.resize(num_of_links);
        mTruthJointForces.resize(num_of_frames);
        for (auto &x : mTruthJointForces)
            x.resize(num_of_links - 1);
        mTimesteps.resize(num_of_frames), mTimesteps.setZero();
        mRewards.resize(num_of_frames), mRewards.setZero();
        mMotionRefTime.resize(num_of_frames), mMotionRefTime.setZero();
    }

    for (int frame_id = 0; frame_id < num_of_frames; frame_id++)
    {
        auto &cur_frame = list_json[frame_id];
        auto &cur_pose = cur_frame["pos"];
        auto &cur_vel = cur_frame["vel"];
        auto &cur_accel = cur_frame["accel"];
        auto &cur_timestep = cur_frame["timestep"];
        auto &cur_ref_time = cur_frame["motion_ref_time"];
        auto &cur_reward = cur_frame["reward"];
        auto &cur_contact_num = cur_frame["contact_num"];
        auto &cur_contact_info = cur_frame["contact_info"];
        auto &cur_ext_force = cur_frame["external_force"];
        auto &cur_ext_torque = cur_frame["external_torque"];
        auto &cur_truth_joint_force = cur_frame["truth_joint_force"];
        auto &cur_truth_action = cur_frame["truth_action"];
        auto &cur_truth_pd_target = cur_frame["truth_pd_target"];
        assert(cur_pose.isNull() == false && cur_pose.size() == pose_size);
        assert(cur_vel.isNull() == false);
        if (frame_id >= 1)
            assert(cur_vel.size() == pose_size);
        assert(cur_accel.isNull() == false);
        if (frame_id >= 2)
            assert(cur_accel.size() == pose_size);
        assert(cur_timestep.isNull() == false && cur_timestep.asDouble() > 0);
        assert(cur_ref_time.isNull() == false);
        assert(cur_contact_info.size() == cur_contact_num.asInt());
        assert(cur_truth_joint_force.isNull() == false &&
               cur_truth_joint_force.size() == (num_of_links - 1) * 4);
        // std::cout <<"load pd target size = " << cur_truth_action.size()
        // << std::endl; std::cout <<"action space size = " <<
        // ctrl->GetActionSize() << std::endl;
        assert(cur_truth_action.isNull() == false &&
               cur_truth_action.size() == ctrl->GetActionSize());
        assert(cur_truth_pd_target.isNull() == false &&
               cur_truth_pd_target.size() == ctrl->GetActionSize());

        // 1. pos, vel, accel
        for (int j = 0; j < pose_size; j++)
            mPoseMat(frame_id, j) = cur_pose[j].asDouble();
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() <<
        // std::endl;
        cIDSolver::SetGeneralizedPos(sim_char, mPoseMat.row(frame_id));
        cIDSolver::RecordMultibodyInfo(sim_char, mLinkRot[frame_id],
                                       mLinkPos[frame_id]);
        for (int j = 0; j < pose_size && frame_id >= 1; j++)
            mVelMat(frame_id, j) = cur_vel[j].asDouble();
        for (int j = 0; j < pose_size && frame_id >= 1; j++)
            mAccelMat(frame_id, j) = cur_accel[j].asDouble();

        // 2. timestep and reward
        mTimesteps[frame_id] = cur_timestep.asDouble();
        mRewards[frame_id] = cur_reward.asDouble();
        mMotionRefTime[frame_id] = cur_ref_time.asDouble();

        // 3. contact info
        // mContactForces
        mContactForces[frame_id].resize(cur_contact_num.asInt());
        for (int c_id = 0; c_id < cur_contact_info.size(); c_id++)
        {
            for (int i = 0; i < 4; i++)
            {
                /*

           single_contact["force_pos"] = Json::arrayValue;
            for(int i=0; i<3; i++)
           single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for(int i=0; i<3; i++)
           single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
                */
                mContactForces[frame_id][c_id].mPos[i] =
                    cur_contact_info[c_id]["force_pos"][i].asDouble();
                mContactForces[frame_id][c_id].mForce[i] =
                    cur_contact_info[c_id]["force_value"][i].asDouble();
            }
            mContactForces[frame_id][c_id].mId =
                cur_contact_info[c_id]["force_link_id"].asInt();
            mContactForces[frame_id][c_id].mIsSelfCollision =
                cur_contact_info[c_id]["is_self_collision"].asBool();
        }

        // std::cout <<"[load file] frame " << frame_id <<" contact num = "
        // << cur_contact_info.size() << std::endl;

        // 4. load external forces
        // mExternalForces[frame_id].resize(num_of_links);
        // mExternalTorques[frame_id].resize(num_of_links);
        for (int idx = 0; idx < num_of_links; idx++)
        {
            // auto & cur_ext_force = ;
            // auto & cur_ext_torque =
            // mExternalTorques[frame_id][idx];
            for (int i = 0; i < 4; i++)
            {
                mExternalForces[frame_id][idx][i] =
                    cur_ext_force[idx * 4 + i].asDouble();
                mExternalTorques[frame_id][idx][i] =
                    cur_ext_torque[idx * 4 + i].asDouble();
            }
            assert(mExternalForces[frame_id][idx].norm() < 1e-10);
            assert(mExternalTorques[frame_id][idx].norm() < 1e-10);
        }

        // 5. truth joint forces
        for (int idx = 0; idx < num_of_links - 1; idx++)
        {
            for (int j = 0; j < 4; j++)
                mTruthJointForces[frame_id][idx][j] =
                    cur_truth_joint_force[idx * 4 + j].asDouble();
        }

        // 6. truth actions and pd targes
        for (int idx = 0; idx < ctrl->GetActionSize(); idx++)
        {
            mActionMat(frame_id, idx) = cur_truth_action[idx].asDouble();
            mPDTargetMat(frame_id, idx) = cur_truth_pd_target[idx].asDouble();
        }
    }
#ifdef VERBOSE
    std::cout << "[debug] cInteractiveIDSolver::LoadTraj " << path
              << ", number of frames = " << num_of_frames << std::endl;
#endif
    assert(num_of_frames > 0);
    sim_char->SetPose(raw_pose);
}
void tLoadInfo::LoadTrajV2(cSimCharacterBase *sim_char, const std::string &path)
{
    MIMIC_DEBUG("begin to load trajv2 {}", path);
    // only load the critical keywords:
    // 1. char_pose
    // 2. timestep
    // 3. frame id
    // 4. contact num
    // 5. contact_info
    // 6. rewards
    // 7. motion ref time
    // 8. actions
    int num_of_links = sim_char->GetNumBodyParts();
    auto ctrl = dynamic_cast<cCtController *>(sim_char->GetController().get());
    MIMIC_ASSERT(ctrl != nullptr);
    // and we also need to restore pose, vel, accel, link rot, link pos,
    tVectorXd raw_pose = sim_char->GetPose();
    Json::Value data_json, list_json;
    bool succ = cJsonUtil::LoadJson(path, data_json);
    mLoadPath = path;
    if (!succ)
    {
        MIMIC_ERROR("LoadTrajV2 parse json {} failed", path);
        exit(0);
    }

    list_json = data_json["list"];
    assert(list_json.isNull() == false);
    // std::cout <<"get list json, begin set up pramas\n";
    const int version = cJsonUtil::ParseAsInt("version", data_json);
#ifndef IGNORE_TRAJ_VERISION
    if (2 != version)
    {
        MIMIC_ERROR(
            "LoadTrajV2 is called but the spcified fileversion in {} is {}",
            path, version);
        exit(1);
    }
#endif

    int num_of_frames = list_json.size();
    int action_size = ctrl->GetActionSize();
    if (eSimCharacterType::Generalized == sim_char->GetCharType())
    {
        MIMIC_WARN("LoadTrajV2 for generalized force hacked");
        action_size += 6;
    }

    int pose_size = sim_char->GetPose().size();
    int q_dof_size = 0;
    if (sim_char->GetCharType() == eSimCharacterType::Featherstone)
    {
        auto mMultibody =
            dynamic_cast<cSimCharacter *>(sim_char)->GetMultiBody();
        q_dof_size = mMultibody->getNumDofs();
        bool mFloatingBase = mMultibody->hasFixedBase() == false;
        if (mFloatingBase)
            q_dof_size += 6;
    }

    {
        mTotalFrame = num_of_frames;
        mPoseMat.resize(num_of_frames, q_dof_size), mPoseMat.setZero();
        mVelMat.resize(num_of_frames, q_dof_size), mVelMat.setZero();
        mAccelMat.resize(num_of_frames, q_dof_size), mAccelMat.setZero();
        mCharPoseMat.resize(num_of_frames, pose_size), mCharPoseMat.setZero();
        mActionMat.resize(num_of_frames, action_size),
            MIMIC_INFO("action size {}", action_size);

        mActionMat.setZero();
        mPDTargetMat.resize(num_of_frames, action_size), mPDTargetMat.setZero();
        mContactForces.resize(num_of_frames);
        for (auto &x : mContactForces)
            x.clear();
        mLinkRot.resize(num_of_frames);
        for (auto &x : mLinkRot)
            x.resize(num_of_links);
        mLinkPos.resize(num_of_frames);
        for (auto &x : mLinkPos)
            x.resize(num_of_links);
        mExternalForces.resize(num_of_frames);
        for (auto &x : mExternalForces)
        {
            x.resize(num_of_links);
            for (auto &j : x)
                j.setZero();
        };
        mExternalTorques.resize(num_of_frames);
        for (auto &x : mExternalTorques)
        {
            x.resize(num_of_links);
            for (auto &j : x)
                j.setZero();
        };
        mTruthJointForces.resize(num_of_frames);
        for (auto &x : mTruthJointForces)
        {
            x.resize(num_of_links - 1);
            for (auto &j : x)
                j.setZero();
        };
        mTimesteps.resize(num_of_frames), mTimesteps.setZero();
        mRewards.resize(num_of_frames), mRewards.setZero();
        mMotionRefTime.resize(num_of_frames), mMotionRefTime.setZero();
    }

    // begin to load pose, timestep, motion ref time, contact infos, rewards
    tVectorXd buffer_pos, buffer_vel;

    for (int frame_id = 0; frame_id < num_of_frames; frame_id++)
    {
        // std::cout <<"frame " << frame_id << std::endl;
        const Json::Value &cur_frame = list_json[frame_id];
        const Json::Value &cur_char_pose =
            cJsonUtil::ParseAsValue("char_pose", cur_frame);
        const Json::Value &cur_contact_info =
            cJsonUtil::ParseAsValue("contact_info", cur_frame);
        const Json::Value &cur_truth_action =
            cJsonUtil::ParseAsValue("truth_action", cur_frame);
        const int cur_contact_num =
            cJsonUtil::ParseAsInt("contact_num", cur_frame);

        // 1. restore 0 order info pos from char pose
        if (cur_char_pose.isNull() == true || cur_char_pose.size() != pose_size)
        {
            MIMIC_ERROR("LoadTrajV2: char pose shape is empty or "
                        "invalid {} != {}",
                        cur_char_pose.size(), pose_size);
            exit(0);
        }
        if (cur_contact_info.isNull() == true ||
            cur_contact_info.size() != cur_contact_num)
        {
            MIMIC_ERROR("LoadTrajV2: contact info is empty or invalid {} != {}",
                        cur_contact_info.size(), cur_contact_num);
            exit(0);
        }

        for (int j = 0; j < pose_size; j++)
            mCharPoseMat(frame_id, j) = cur_char_pose[j].asDouble();
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() <<
        // std::endl;
        sim_char->SetPose(mCharPoseMat.row(frame_id));
        cIDSolver::RecordGeneralizedInfo(sim_char, buffer_pos, buffer_vel);
        mPoseMat.row(frame_id) = buffer_pos;
        cIDSolver::RecordMultibodyInfo(sim_char, mLinkRot[frame_id],
                                       mLinkPos[frame_id]);

        // 2. timestep, reward and ref time
        mTimesteps[frame_id] = cJsonUtil::ParseAsDouble("timestep", cur_frame);
        mRewards[frame_id] = cJsonUtil::ParseAsDouble("reward", cur_frame);
        mMotionRefTime[frame_id] =
            cJsonUtil::ParseAsDouble("motion_ref_time", cur_frame);

        // 3. contact info
        mContactForces[frame_id].resize(cur_contact_num);
        for (int c_id = 0; c_id < cur_contact_info.size(); c_id++)
        {
            for (int i = 0; i < 4; i++)
            {
                mContactForces[frame_id][c_id].mPos[i] =
                    cur_contact_info[c_id]["force_pos"][i].asDouble();
                mContactForces[frame_id][c_id].mForce[i] =
                    cur_contact_info[c_id]["force_value"][i].asDouble();
            }
            mContactForces[frame_id][c_id].mId =
                cur_contact_info[c_id]["force_link_id"].asInt();
            mContactForces[frame_id][c_id].mIsSelfCollision =
                cur_contact_info[c_id]["is_self_collision"].asBool();
        }

        // 4. actions
        for (int idx = 0; idx < action_size; idx++)
        {
            mActionMat(frame_id, idx) = cur_truth_action[idx].asDouble();
        }
    }

#ifdef VERBOSE
    std::cout << "[debug] cInteractiveIDSolver::LoadTraj " << path
              << ", number of frames = " << num_of_frames << std::endl;
#endif
    assert(num_of_frames > 0);
    sim_char->SetPose(raw_pose);
}
cTrajRecorder::cTrajRecorder(cSceneImitate *scene, const std::string &conf)
{
    mScene = scene;
    mSimChar = scene->GetCharacter().get();
    ParseConfig(conf);
    MIMIC_TRACE("build traj recoder by {}", conf);
}

cTrajRecorder::~cTrajRecorder() {}

/**
 * \brief               Record info before simulation
 */
void cTrajRecorder::PreSim()
{
    mSaveInfo.mCharPoses[mSaveInfo.mCurFrameId] = mSimChar->GetPose();

    if (eSimCharacterType::Generalized == mSimChar->GetCharType())
    {
        RecordActiveForceGen();
    }

    auto gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
    if (gen_char != nullptr)
    {
        std::cout << "-----------------------------frame "
                  << mSaveInfo.mCurFrameId << "----------------------\n";
        std::cout << "q = " << gen_char->Getq().transpose() << std::endl;
        std::cout << "qdot = " << gen_char->Getqdot().transpose() << std::endl;
        // std::endl; std::cout << "M = \n" << gen_char->GetMassMatrix() <<
        // std::endl; std::cout << "C = \n" << gen_char->GetCoriolisMatrix() <<
        // std::endl;
    }
}

/**
 * \brief               Record info after simulation
 */
void cTrajRecorder::PostSim()
{
    ReadContactInfo();

    mSaveInfo.mCurFrameId++;
    if (mRecordMaxFrame == mSaveInfo.mCurFrameId)
    {
        MIMIC_INFO("record frames exceed the upper bound {}, reset",
                   mSaveInfo.mCurFrameId);
        Reset();
    }
}

/**
 * \brief               Reset & save current trajectory per episode
 */
void cTrajRecorder::Reset()
{
    Json::Value root;
    cInteractiveIDSolver::SaveTrajV2(mSaveInfo, root);

    cJsonUtil::WriteJson(mTrajSavePath, root, true);
    MIMIC_INFO("cTrajRecoder::Reset finished, save traj to {}, exit",
               mTrajSavePath);
    VerifyDynamicsEquation();
    exit(1);
}

/**
 *
 */
void cTrajRecorder::SetTimestep(double t)
{
    mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = t;
}

/**
 * \brief               Parse config
 *          This function will parse the trajectory recoder's config to control
 * its behavior.
 *          1. save place?
 *          2. save version?
 */
void cTrajRecorder::ParseConfig(const std::string &conf)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    mTrajSavePath = cJsonUtil::ParseAsString("traj_save_filename", root);
    mRecordMaxFrame = cJsonUtil::ParseAsInt("record_max_frame", root);

    MIMIC_DEBUG("trajectory recoder parse config: get traj save path{}",
                mTrajSavePath);
}

void cTrajRecorder::ReadContactInfoGen()
{
    int num_of_links = mSimChar->GetNumBodyParts();
    auto &cur_contact_info = mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    cur_contact_info.clear();

    tContactForceInfo pt_info;
    for (int i = 0; i < num_of_links; i++)
    {
        for (auto &pt : mSimChar->GetBodyPart(i)->GetContactPts())
        {
            pt_info.mId = i;
            pt_info.mIsSelfCollision = pt.mIsSelfCollision;
            pt_info.mForce = pt.mForce;
            pt_info.mPos = pt.mPos;
            cur_contact_info.push_back(pt_info);
        }
    }
    MIMIC_INFO("record {} contact points in traj recorder",
               cur_contact_info.size());

    MIMIC_WARN("link id += 1 in order to get matched with MIMICControl");
    for (auto &x : cur_contact_info)
    {
        x.mId += 1;
    }
}
void cTrajRecorder::ReadContactInfoRaw()
{

    MIMIC_ERROR("hasn't been implemented")
}
void cTrajRecorder::ReadContactInfo()
{
    switch (mSimChar->GetCharType())
    {
    case eSimCharacterType::Featherstone:
        ReadContactInfoRaw();
        break;
    case eSimCharacterType::Generalized:
        ReadContactInfoGen();
        break;
    default:
        break;
    }

    const auto &current_forces =
        mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    int frame = mSaveInfo.mCurFrameId;
    if (current_forces.size())
    {
        MIMIC_DEBUG("frame {} num of contacts {}", frame,
                    current_forces.size());
        for (int i = 0; i < current_forces.size(); i++)
        {
            MIMIC_DEBUG("contact {} force = {}", i,
                        current_forces[i].mForce.transpose());
        }
    }
}

/**
 * \brief           Record the active force of gen
 */
void cTrajRecorder::RecordActiveForceGen()
{
    auto gen = dynamic_cast<cSimCharacterGen *>(mSimChar);
    mSaveInfo.mTruthAction[mSaveInfo.mCurFrameId] = gen->GetGeneralizedForce();
}

void cTrajRecorder::VerifyDynamicsEquation()
{
    tVectorXd qbefore, qcur, qnext, qdot, qddot;
    cSimCharacterGen *gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
    double dt = mSaveInfo.mTimesteps[0];
    int dof = gen_char->GetNumOfFreedom();
    int num_of_links = gen_char->GetNumOfLinks();
    for (int frame_id = 1; frame_id < mSaveInfo.mCurFrameId - 1; frame_id++)
    {
        // MIMIC_DEBUG("frame id {}", frame_id);
        tVectorXd cur_pose = mSaveInfo.mCharPoses[frame_id];
        qcur = gen_char->ConvertPoseToq(cur_pose);
        qbefore = gen_char->ConvertPoseToq(mSaveInfo.mCharPoses[frame_id - 1]);
        qnext = gen_char->ConvertPoseToq(mSaveInfo.mCharPoses[frame_id + 1]);
        // gen_char->SetPose(cur_pose);

        qdot = (qcur - qbefore) / dt;
        qddot = (qbefore + qnext - 2 * qcur) / (std::pow(dt, 2));
        tVectorXd qddot_num = (qbefore + qnext - 2 * qcur);
        gen_char->SetqAndqdot(qcur, qdot);
        tMatrixXd M = gen_char->GetMassMatrix(),
                  C = gen_char->GetCoriolisMatrix();

        tVectorXd LHS = M * qddot + C * qdot;

        // get the contact gen force
        const auto &cur_contact_forces = mSaveInfo.mContactForces[frame_id];
        int num_of_contacts = cur_contact_forces.size();
        tVectorXd Q_contact = tVectorXd::Zero(dof);
        for (int c_id = 0; c_id < num_of_contacts; c_id++)
        {
            int link_id = cur_contact_forces[c_id].mId;
            tVector force = cur_contact_forces[c_id].mForce;
            tVector pos = cur_contact_forces[c_id].mPos;
            // MIMIC_DEBUG("contact {} link {} force {} pos {} ", c_id, link_id,
            //             force.transpose(), pos.transpose());
            tMatrixXd jac;
            gen_char->ComputeJacobiByGivenPointTotalDOFWorldFrame(
                link_id - 1, pos.segment(0, 3), jac);
            Q_contact += jac.transpose() * force.segment(0, 3);
        }
        tVectorXd Q_G = tVectorXd::Zero(dof);
        for (int link_id = 0; link_id < num_of_links; link_id++)
        {
            Q_G += gen_char->GetLinkById(link_id)->GetJKv().transpose() *
                   gGravity.segment(0, 3) *
                   gen_char->GetBodyPart(link_id)->GetMass();
        }
        tVectorXd active_force = LHS - Q_G - Q_contact;
        // tVectorXd active_diff = mSaveInfo.mTruthAction[frame_id] -
        // active_force;

        // if (frame_id >= 322)
        // {
        //     // MIMIC_DEBUG("active diff {}", active_diff.transpose());
        //     // MIMIC_OUTPUT("frame {} M norm = {}", frame_id, M.norm());
        //     // MIMIC_OUTPUT("frame {} C norm = {}", frame_id, C.norm());
        //     // MIMIC_OUTPUT("frame {} q = {}", frame_id, qcur.norm());
        //     // MIMIC_OUTPUT("frame {} qdot = {}", frame_id, qdot.norm());
        //     MIMIC_OUTPUT("frame {} qddot norm = {}", frame_id, qddot.norm());
        //     MIMIC_OUTPUT("frame {} qddot = {}", frame_id, qddot.transpose());
        //     MIMIC_OUTPUT("frame {} qddot num = {}", frame_id,
        //                  qddot_num.transpose());
        //     MIMIC_OUTPUT("frame {} qbefore {} qcur {} qnext {} ", frame_id,
        //                  qbefore[4], qcur[4], qnext[4]);
        //     // MIMIC_OUTPUT("frame {} Mqddot = {}", frame_id, (M *
        //     // qddot).norm()); MIMIC_OUTPUT("frame {} dt = {}", frame_id,
        //     dt);

        //     // MIMIC_OUTPUT("frame {} LHS = {}", frame_id, LHS.transpose());
        //     // MIMIC_OUTPUT("frame {} QG {}", frame_id, Q_G.transpose());
        //     // MIMIC_OUTPUT("frame {} active force {}", frame_id,
        //     //              active_force.transpose());
        // }
    }
    // MIMIC_ERROR("Verify Dynamics equ done");
    exit(1);
}