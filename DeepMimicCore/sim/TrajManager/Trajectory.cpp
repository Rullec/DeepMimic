#include "Trajectory.h"
#include "sim/Controller/CtPDFeaController.h"
#include "sim/SimItems/SimCharacter.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/TrajManager/IDSolver.h"
#include "util/BulletUtil.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>

/**
 * \brief                   Save trajectories to disk
 * \param mSaveInfo         the trajecoty we want to save
 * \param traj_dir          the storaged directory
 * \param traj_root_name    root name for storaged.
 */
std::string tSaveInfo::SaveTraj(const std::string &traj_dir,
                                const std::string &traj_rootname,
                                eTrajFileVersion mTrajFileVersion)
{

    if (cFileUtil::ExistsDir(traj_dir) == false)
    {
        MIMIC_ERROR("SaveTraj directory {} doesn't exist", traj_dir);
        exit(0);
    }

    Json::Value root;
    switch (mTrajFileVersion)
    {
    case eTrajFileVersion::V1:
        SaveTrajV1(root);
        break;
    case eTrajFileVersion::V2:
        SaveTrajV2(root);
        break;
    default:
        MIMIC_ERROR("SaveTraj unsupported traj file version {}",
                    static_cast<int>(mTrajFileVersion));
        exit(1);
        break;
    }
    std::string traj_name, final_name;
    do
    {
        traj_name = cFileUtil::GenerateRandomFilename(traj_rootname);
        final_name = cFileUtil::ConcatFilename(traj_dir, traj_name);
    } while (cFileUtil::ExistsFile(final_name) == true);

    cFileUtil::AddLock(final_name);
    if (false == cFileUtil::ValidateFilePath(final_name))
    {
        MIMIC_ERROR("SaveTraj path {} illegal", final_name);
        exit(1);
    }
    cJsonUtil::WriteJson(final_name, root, false);
    cFileUtil::DeleteLock(final_name);
    return traj_name;
}
/**
 * \brief                   save trajectories for full version
 * \param mSaveInfo
 * \param path              target save location
 */
void tSaveInfo::SaveTrajV1(Json::Value &root)
{

    Json::Value single_frame;
    root["epoch"] = mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 1;
    for (int frame_id = 0; frame_id < mCurFrameId; frame_id++)
    {
        /* set up single frame, including:
            - frame id
            - timestep
            - generalized coordinate, pos, vel, accel
            - contact info
            - external forces
        */

        SaveCommonInfo(single_frame, frame_id);
        SavePosVelAccel(single_frame, frame_id);
        // set up contact info
        SaveContactInfo(single_frame, frame_id);
        // set up external force & torques info
        SaveExtForce(single_frame, frame_id);

        // set up truth joint torques
        SaveTruthJointForce(single_frame, frame_id);
        // set up truth action
        SaveTruthAction(single_frame, frame_id);
        // set up truth pd target
        SaveTruthPDTarget(single_frame, frame_id);
        // set up character poses
        SaveCharPose(single_frame, frame_id);

        // append to the whole list
        root["list"].append(single_frame);
    }
}

/**
 * \brief                   Save simplified trajectries which are called "V2"
 * \param mSaveInfo
 * \param path              target save location
 *
 */
void tSaveInfo::SaveTrajV2(Json::Value &root)
{

    /*  .traj v2 is simplified version compared with v1.
    Only save these keywords in v2:
    1. char_pose
    2. timestep
    3. frame id
    4. contact num
    5. contact_info
    6. rewards
    7. motion ref time
    8. action
*/

    Json::Value single_frame;
    root["epoch"] = mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 2;

    for (int frame_id = 0; frame_id < mCurFrameId; frame_id++)
    {
        SaveCommonInfo(single_frame, frame_id);

        // single_frame["frame_id"] = frame_id;
        SaveCharPose(single_frame, frame_id);

        // contact info
        SaveContactInfo(single_frame, frame_id);

        // actions
        SaveTruthAction(single_frame, frame_id);

        root["list"].append(single_frame);
    }
}

void tSaveInfo::SaveCommonInfo(Json::Value &value, int frame_id) const
{
    value["frame_id"] = frame_id;
    value["timestep"] = mTimesteps[frame_id];
    value["motion_ref_time"] = mRefTime[frame_id];
    value["reward"] = mRewards[frame_id];
}

void tSaveInfo::SavePosVelAccel(Json::Value &value, int frame_id) const
{
    value["pos"] = Json::Value(Json::arrayValue);
    value["vel"] = Json::Value(Json::arrayValue);
    value["accel"] = Json::Value(Json::arrayValue);
    for (int dof = 0; dof < mBuffer_q[frame_id].size(); dof++)
    {
        value["pos"].append(mBuffer_q[frame_id][dof]);
        if (frame_id >= 1)
        {
            value["vel"].append(mBuffer_u[frame_id][dof]);
            value["accel"].append(mBuffer_u_dot[frame_id][dof]);
        }
    }
}
void tSaveInfo::SaveContactInfo(Json::Value &value, int frame_id) const
{
    value["contact_info"] = Json::arrayValue;
    // value["contact_num"] =
    // mContactForces[frame_id].size();
    value["contact_num"] = static_cast<int>(mContactForces[frame_id].size());
    for (int c_id = 0; c_id < mContactForces[frame_id].size(); c_id++)
    {
        Json::Value single_contact;
        const tContactForceInfo &force = mContactForces[frame_id][c_id];
        single_contact["force_pos"] = Json::arrayValue;
        for (int i = 0; i < 4; i++)
            single_contact["force_pos"].append(force.mPos[i]);
        single_contact["force_value"] = Json::arrayValue;
        for (int i = 0; i < 4; i++)
            single_contact["force_value"].append(force.mForce[i]);
        single_contact["force_link_id"] = force.mId;
        single_contact["is_self_collision"] = force.mIsSelfCollision;
        value["contact_info"].append(single_contact);
    }
}
void tSaveInfo::SaveTruthJointForce(Json::Value &value, int frame_id) const
{
    value["truth_joint_force"] = Json::arrayValue;
    int num_links = mTruthJointForces[frame_id].size() + 1;
    for (int i = 0; i < num_links - 1; i++)
    {
        for (int j = 0; j < 4; j++)
            value["truth_joint_force"].append(
                mTruthJointForces[frame_id][i][j]);
    }
}
void tSaveInfo::SaveTruthAction(Json::Value &value, int frame_id) const
{
    value["truth_action"] = Json::arrayValue;
    for (int i = 0; i < mTruthAction[frame_id].size(); i++)
    {
        value["truth_action"].append(mTruthAction[frame_id][i]);
    }
}
void tSaveInfo::SaveTruthPDTarget(Json::Value &value, int frame_id) const
{
    value["truth_pd_target"] = Json::arrayValue;
    for (int i = 0; i < mTruthPDTarget[frame_id].size(); i++)
    {
        value["truth_pd_target"].append(mTruthPDTarget[frame_id][i]);
    }
}
void tSaveInfo::SaveCharPose(Json::Value &value, int frame_id) const
{
    value["char_pose"] = Json::arrayValue;
    for (int i = 0; i < mCharPoses[frame_id].size(); i++)
    {
        value["char_pose"].append(mCharPoses[frame_id][i]);
    }
}

void tSaveInfo::SaveExtForce(Json::Value &value, int frame_id) const
{
    value["external_force"] = Json::arrayValue;
    value["external_torque"] = Json::arrayValue;
    int num_links = mExternalForces[frame_id].size();
    for (int link_id = 0; link_id < num_links; link_id++)
    {
        for (int idx = 0; idx < 4; idx++)
        {
            value["external_force"].append(
                mExternalForces[frame_id][link_id][idx]);
            value["external_torque"].append(
                mExternalTorques[frame_id][link_id][idx]);
        }
    }
}

tLoadInfo::tLoadInfo()
{
    mLoadPath = "";
    mLoadMode = eLoadMode::INVALID;
    mPosMat.resize(0, 0);
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
    mVersion = eTrajFileVersion::UNSET;
}

/*
    @Function: LoadMotion
    @params: path Type const std::string &, the filename of specified motion
    @params: motion Type cMotion *, the targeted motion storaged.
*/
void LoadMotion(const std::string &path, cMotion *motion)
{
    MIMIC_ASSERT(cFileUtil::ExistsFile(path));

    cMotion::tParams params;
    params.mMotionFile = path;
    motion->Load(params);
    MIMIC_DEBUG("Load Motion from {}, frame_nums = {}, dof = {}", path,
                motion->GetNumFrames(), motion->GetNumDof());
}

/**
 * \brief
 */
void tLoadInfo::LoadTraj(cSimCharacterBase *sim_char, const std::string &path)
{
    // fetch the trajectory version
    mLoadPath = path;
    Json::Value root;
    MIMIC_ASSERT(cJsonUtil::LoadJson(path, root));

    switch (CalcTrajVersion(cJsonUtil::ParseAsInt("version", root)))
    {
    case eTrajFileVersion::UNSET:
        MIMIC_ERROR("LoadTraj: Please point out the version of traj file {}",
                    path);
        exit(0);
        break;
    case eTrajFileVersion::V1:
        LoadTrajV1(sim_char, root);
        mVersion = eTrajFileVersion::V1;
        break;
    case eTrajFileVersion::V2:
        LoadTrajV2(sim_char, root);
        mVersion = eTrajFileVersion::V2;
        break;
    default:
        MIMIC_ERROR("Unsupported version");
    }
}

void tLoadInfo::LoadTrajV1(cSimCharacterBase *sim_char,
                           const Json::Value &data_json)
{
    auto &raw_pose = sim_char->GetPose();
    Json::Value list_json = data_json["list"];
    MIMIC_ASSERT(list_json.isNull() == false);
    const int target_version = cJsonUtil::ParseAsInt("version", data_json);
    int num_of_frames = list_json.size();

    auto ctrl =
        dynamic_cast<cCtPDController *>(sim_char->GetController().get());
    MIMIC_ASSERT(ctrl != nullptr);
    {
        int num_of_links = 0;
        int pos_size = 0;
        if (sim_char->GetCharType() == eSimCharacterType::Featherstone)
        {
            num_of_links = sim_char->GetNumBodyParts() + 1;
            auto multibody =
                dynamic_cast<cSimCharacter *>(sim_char)->GetMultiBody();
            pos_size = multibody->getNumDofs();
            if (multibody->hasFixedBase() == false)
                pos_size += 6;
        }
        else if (sim_char->GetCharType() == eSimCharacterType::Generalized)
        {
            pos_size = sim_char->GetNumDof();
            num_of_links = sim_char->GetNumBodyParts();
            // MIMIC_ERROR("pos size {}, num of links {}", pos_size,
            // num_of_links);
        }

        mTotalFrame = num_of_frames;
        mPosMat.resize(num_of_frames, pos_size), mPosMat.setZero();
        mVelMat.resize(num_of_frames, pos_size), mVelMat.setZero();
        mAccelMat.resize(num_of_frames, pos_size), mAccelMat.setZero();
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

        auto &cur_contact_num = cur_frame["contact_num"];
        auto &cur_contact_info = cur_frame["contact_info"];
        auto &cur_ext_force = cur_frame["external_force"];
        auto &cur_ext_torque = cur_frame["external_torque"];

        auto &cur_truth_action = cur_frame["truth_action"];
        auto &cur_truth_pd_target = cur_frame["truth_pd_target"];

        MIMIC_ASSERT(cur_contact_info.size() == cur_contact_num.asInt());
        // MIMIC_ASSERT(cur_truth_joint_force.size() == (num_of_links - 1) * 4);
        // std::cout <<"load pd target size = " << cur_truth_action.size()
        // << std::endl; std::cout <<"action space size = " <<
        // ctrl->GetActionSize() << std::endl;
        MIMIC_ASSERT(cur_truth_pd_target.size() == ctrl->GetActionSize());
        MIMIC_ASSERT(cur_truth_action.size() == ctrl->GetActionSize());

        // 1. pos, vel, accel
        tVectorXd pos, vel, accel;
        LoadPosVelAndAccel(cur_frame, frame_id, mPosMat, mVelMat, mAccelMat);

        cIDSolver::SetGeneralizedPos(sim_char, mPosMat.row(frame_id));
        cIDSolver::RecordMultibodyInfo(sim_char, mLinkRot[frame_id],
                                       mLinkPos[frame_id]);

        // 2. timestep and reward
        LoadCommonInfo(cur_frame, mTimesteps[frame_id], mRewards[frame_id],
                       mMotionRefTime[frame_id]);
        // 3. contact info
        LoadContactInfo(cur_contact_info, mContactForces[frame_id]);

        // std::cout <<"[load file] frame " << frame_id <<" contact num = "
        // << cur_contact_info.size() << std::endl;

        // 4. load external forces
        // mExternalForces[frame_id].resize(num_of_links);
        // mExternalTorques[frame_id].resize(num_of_links);
        LoadExternalForceTorque(cur_frame, mExternalForces[frame_id],
                                mExternalTorques[frame_id]);
        // 5. truth joint forces
        LoadJointTorques(cur_frame, mTruthJointForces[frame_id]);

        // 6. truth actions and pd targes
        mActionMat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_truth_action);
        mPDTargetMat.row(frame_id) =
            cJsonUtil::ReadVectorJson(cur_truth_pd_target);
    }
#ifdef VERBOSE
    MIMIC_DEBUG("LoadTraj {}, num of frames = {}", path, num_of_frames);
#endif
    assert(num_of_frames > 0);
    sim_char->SetPose(raw_pose);
}
void tLoadInfo::LoadTrajV2(cSimCharacterBase *sim_char,
                           const Json::Value &data_json)
{
    // only load the critical keywords:
    // 1. char_pose
    // 2. timestep
    // 3. frame id
    // 4. contact num
    // 5. contact_info
    // 6. rewards
    // 7. motion ref time
    // 8. actions

    // int pos_size = 0;
    tVectorXd raw_pose = sim_char->GetPose();
    Json::Value list_json = data_json["list"];
    MIMIC_ASSERT(list_json.isNull() == false);
    const int target_version = cJsonUtil::ParseAsInt("version", data_json);
    MIMIC_ASSERT(target_version == 2);
    int num_of_frames = list_json.size();

    auto ctrl =
        dynamic_cast<cCtPDController *>(sim_char->GetController().get());
    MIMIC_ASSERT(ctrl != nullptr);
    // and we also need to restore pose, vel, accel, link rot, link pos,

    // std::cout <<"get list json, begin set up pramas\n";

    // int num_of_links = 0;
    // int pose_size = sim_char->GetPose().size();
    // int q_dof_size = 0;
    // if (sim_char->GetCharType() == eSimCharacterType::Featherstone)
    // {
    //     num_of_links = sim_char->GetNumBodyParts() + 1;
    //     auto mMultibody =
    //         dynamic_cast<cSimCharacter *>(sim_char)->GetMultiBody();
    //     q_dof_size = mMultibody->getNumDofs();
    //     bool mFloatingBase = mMultibody->hasFixedBase() == false;
    //     if (mFloatingBase)
    //         q_dof_size += 6;
    // }
    // else if (sim_char->GetCharType() == eSimCharacterType::Generalized)
    // {
    //     q_dof_size = sim_char->GetNumDof();
    //     num_of_links = sim_char->GetNumBodyParts();
    //     // MIMIC_ERROR("pos size {}, num of links {}", pos_size,
    //     // num_of_links);
    // }
    {
        int num_of_links = 0;
        int pos_size = 0;
        if (sim_char->GetCharType() == eSimCharacterType::Featherstone)
        {
            num_of_links = sim_char->GetNumBodyParts() + 1;
            auto multibody =
                dynamic_cast<cSimCharacter *>(sim_char)->GetMultiBody();
            pos_size = multibody->getNumDofs();
            if (multibody->hasFixedBase() == false)
                pos_size += 6;
        }
        else if (sim_char->GetCharType() == eSimCharacterType::Generalized)
        {
            pos_size = sim_char->GetNumDof();
            num_of_links = sim_char->GetNumBodyParts();
            // MIMIC_ERROR("pos size {}, num of links {}", pos_size,
            // num_of_links);
        }
        int action_size = ctrl->GetActionSize();
        int pose_size = sim_char->GetPose().size();
        mTotalFrame = num_of_frames;
        mPosMat.resize(num_of_frames, pos_size), mPosMat.setZero();
        mVelMat.resize(num_of_frames, pos_size), mVelMat.setZero();
        mAccelMat.resize(num_of_frames, pos_size), mAccelMat.setZero();
        mCharPoseMat.resize(num_of_frames, pose_size), mCharPoseMat.setZero();
        mActionMat.resize(num_of_frames, action_size);
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
        mCharPoseMat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_char_pose);
        // 1. restore 0 order info pos from char pose
        MIMIC_ASSERT(cur_contact_info.size() == cur_contact_num);
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() <<
        // std::endl;
        sim_char->SetPose(mCharPoseMat.row(frame_id));
        cIDSolver::RecordGeneralizedInfo(sim_char, buffer_pos, buffer_vel);
        mPosMat.row(frame_id) = buffer_pos;
        cIDSolver::RecordMultibodyInfo(sim_char, mLinkRot[frame_id],
                                       mLinkPos[frame_id]);

        // 2. timestep, reward and ref time
        LoadCommonInfo(cur_frame, mTimesteps[frame_id], mRewards[frame_id],
                       mMotionRefTime[frame_id]);

        // 3. contact info
        LoadContactInfo(cur_contact_info, mContactForces[frame_id]);

        // 4. actions
        mActionMat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_truth_action);
    }

#ifdef VERBOSE
    MIMIC_DEBUG("LoadTraj {}, num of frames = {}", path, num_of_frames);
#endif
    assert(num_of_frames > 0);
    sim_char->SetPose(raw_pose);
}

void tLoadInfo::PrintLoadInfo(cSimCharacterBase *sim_char,
                              const std::string &filename,
                              bool disable_root /* = true*/) const
{
    if (sim_char->GetCharType() == eSimCharacterType::Featherstone)
    {
        auto mMultibody =
            dynamic_cast<cSimCharacter *>(sim_char)->GetMultiBody();
        int mNumLinks = mMultibody->getNumLinks() + 1;
        std::ofstream fout(filename, std::ios::app);
        if (mCurFrame == 0)
            return;

        if (disable_root == false)
        {
            // fetch mass
            std::vector<double> mk_lst(0);
            double total_mass = 0;
            mk_lst.resize(mNumLinks);
            for (int i = 0; i < mNumLinks; i++)
            {
                if (i == 0)
                    mk_lst[i] = mMultibody->getBaseMass();
                else
                    mk_lst[i] = mMultibody->getLinkMass(i - 1);
                total_mass += mk_lst[i];
            }

            // output to file
            fout << "---------- frame " << mCurFrame << " -----------\n";
            fout << "num_of_links = " << mNumLinks << std::endl;
            fout << "timestep = " << mTimesteps[mCurFrame] << std::endl;
            fout << "total_mass = " << total_mass << std::endl;

            for (int i = 0; i < mNumLinks; i++)
            {
                tVector vel =
                    (mLinkPos[mCurFrame][i] - mLinkPos[mCurFrame - 1][i]) /
                    mTimesteps[mCurFrame];
                tVector omega = cMathUtil::CalcQuaternionVel(
                    cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame - 1][i]),
                    cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame][i]),
                    mTimesteps[mCurFrame]);
                fout << "--- link " << i << " ---\n";
                fout << "mass = " << mk_lst[i] << std::endl;
                fout << "pos = "
                     << mLinkPos[mCurFrame][i].transpose().segment(0, 3)
                     << std::endl;
                fout << "rot = "
                     << cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame][i])
                            .coeffs()
                            .transpose()
                     << std::endl;
                fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
                fout << "omega = " << omega.transpose().segment(0, 3)
                     << std::endl;
                // fout << "COM cur = " << COM_cur.transpose() << std::endl;
            }
        }
        else
        {
            // disable the root link for DeepMimic, mass is zero and inertia is
            // zero fetch mass
            std::vector<double> mk_lst(0);
            double total_mass = 0;
            tVector COM_cur = tVector::Zero();
            mk_lst.resize(mNumLinks - 1);
            for (int i = 0; i < mNumLinks - 1; i++)
            {
                mk_lst[i] = mMultibody->getLinkMass(i);
                total_mass += mk_lst[i];
                COM_cur += mk_lst[i] * mLinkPos[mCurFrame][i + 1];
            }
            COM_cur /= total_mass;

            // output to file
            fout << "---------- frame " << mCurFrame << " -----------\n";
            fout << "num_of_links = " << mNumLinks - 1 << std::endl;
            fout << "timestep = " << mTimesteps[mCurFrame] << std::endl;
            fout << "total_mass = " << total_mass << std::endl;
            tVector lin_mom = tVector::Zero(), ang_mom = tVector::Zero();

            for (int i = 1; i < mNumLinks; i++)
            {
                tVector vel =
                    (mLinkPos[mCurFrame][i] - mLinkPos[mCurFrame - 1][i]) /
                    mTimesteps[mCurFrame];
                tVector omega = cMathUtil::CalcQuaternionVel(
                    cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame - 1][i]),
                    cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame][i]),
                    mTimesteps[mCurFrame]);
                fout << "--- link " << i - 1 << " ---\n";
                fout << "mass = " << mk_lst[i - 1] << std::endl;
                fout << "pos = "
                     << mLinkPos[mCurFrame][i].transpose().segment(0, 3)
                     << std::endl;
                fout << "rot = "
                     << cMathUtil::RotMatToQuaternion(mLinkRot[mCurFrame][i])
                            .coeffs()
                            .transpose()
                     << std::endl;
                fout << "rotmat = \n"
                     << mLinkRot[mCurFrame][i].block(0, 0, 3, 3) << std::endl;
                fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
                fout << "omega = " << omega.transpose().segment(0, 3)
                     << std::endl;
                fout << "inertia = "
                     << cBulletUtil::btVectorTotVector0(
                            mMultibody->getLinkInertia(i - 1))
                            .transpose()
                            .segment(0, 3)
                     << std::endl;
                lin_mom += mk_lst[i - 1] * vel;
                ang_mom += mk_lst[i - 1] *
                               (mLinkPos[mCurFrame][i] - COM_cur).cross3(vel) +
                           mLinkRot[mCurFrame][i] *
                               cBulletUtil::btVectorTotVector0(
                                   mMultibody->getLinkInertia(i - 1))
                                   .asDiagonal() *
                               mLinkRot[mCurFrame][i].transpose() * omega;
            }

            tVector contact_force_impulse = tVector::Zero();
            for (auto &pt : mContactForces[mCurFrame])
            {
                contact_force_impulse += pt.mForce * mTimesteps[mCurFrame];
            }
            fout << "COM cur = " << COM_cur.transpose().segment(0, 3)
                 << std::endl;
            fout << "linear momentum = " << lin_mom.transpose().segment(0, 3)
                 << std::endl;
            fout << "contact pt num = " << mContactForces[mCurFrame].size()
                 << std::endl;
            fout << "contact pt impulse = " << contact_force_impulse.transpose()
                 << std::endl;
            fout << "angular momentum = " << ang_mom.transpose().segment(0, 3)
                 << std::endl;
        }
    }
}

eTrajFileVersion CalcTrajVersion(int version_int)
{
    eTrajFileVersion version;
    switch (version_int)
    {
    case 1:
        version = eTrajFileVersion::V1;
        break;
    case 2:
        version = eTrajFileVersion::V2;
        break;
    default:
        MIMIC_ERROR("CalcTrajVersion unsupported type");
    }
    return version;
}
// eTrajFileVersion tLoadInfo::CalcTrajVersion(const std::string &filename)
// const

// {
//     Json::Value data_json;
//     MIMIC_ASSERT(cJsonUtil::LoadJson(filename, data_json));
//     return CalcTrajVersion(cJsonUtil::ParseAsInt("version", data_json));
// }

void tLoadInfo::LoadCommonInfo(const Json::Value &cur_frame, double &timestep,
                               double &motion_ref_time, double &reward) const
{
    timestep = cJsonUtil::ParseAsDouble("timestep", cur_frame);
    motion_ref_time = cJsonUtil::ParseAsDouble("motion_ref_time", cur_frame);
    reward = cJsonUtil::ParseAsDouble("reward", cur_frame);
}

void tLoadInfo::LoadContactInfo(
    const Json::Value &cur_contact_info,
    tEigenArr<tContactForceInfo> &contact_array) const
{
    contact_array.resize(cur_contact_info.size());
    for (int c_id = 0; c_id < cur_contact_info.size(); c_id++)
    {
        for (int i = 0; i < 4; i++)
        {
            contact_array[c_id].mPos[i] =
                cur_contact_info[c_id]["force_pos"][i].asDouble();
            contact_array[c_id].mForce[i] =
                cur_contact_info[c_id]["force_value"][i].asDouble();
        }
        contact_array[c_id].mId =
            cur_contact_info[c_id]["force_link_id"].asInt();
        contact_array[c_id].mIsSelfCollision =
            cur_contact_info[c_id]["is_self_collision"].asBool();
    }
}

void tLoadInfo::LoadPosVelAndAccel(const Json::Value &cur_frame, int frame_id,
                                   tMatrixXd &pos_mat, tMatrixXd &vel_mat,
                                   tMatrixXd &accel_mat) const
{
    const auto &cur_pose = cJsonUtil::ParseAsValue("pos", cur_frame);
    const auto &cur_vel = cJsonUtil::ParseAsValue("vel", cur_frame);
    const auto &cur_accel = cJsonUtil::ParseAsValue("accel", cur_frame);
    // std::cout << "pose mat size: " << pos_mat.row(frame_id).size() <<
    // std::endl;

    pos_mat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_pose);

    // std::cout << "cur pose size: " << jpos.size() << std::endl;
    if (frame_id >= 1)
        vel_mat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_vel);
    if (frame_id >= 2)
        accel_mat.row(frame_id) = cJsonUtil::ReadVectorJson(cur_accel);
}

void tLoadInfo::LoadJointTorques(const Json::Value &cur_frame,
                                 tEigenArr<tVector> &joint_torques)
{
    auto &cur_truth_joint_force = cur_frame["truth_joint_force"];
    int num_of_links = joint_torques.size() + 1;
    MIMIC_ASSERT((num_of_links - 1) == cur_truth_joint_force.size() / 4);
    for (int idx = 0; idx < num_of_links - 1; idx++)
    {
        for (int j = 0; j < 4; j++)
            joint_torques[idx][j] =
                cur_truth_joint_force[idx * 4 + j].asDouble();
    }
}

void tLoadInfo::LoadExternalForceTorque(const Json::Value &cur_frame,
                                        tEigenArr<tVector> &forces,
                                        tEigenArr<tVector> &torques) const
{
    int num_of_links = forces.size();
    auto cur_force = cJsonUtil::ParseAsValue("external_force", cur_frame);
    auto cur_torque = cJsonUtil::ParseAsValue("external_torque", cur_frame);
    // MIMIC_DEBUG("num of link {}, force {}, torque {}", num_of_links,
    //             cur_force.size() / 4, cur_torque.size() / 4);
    MIMIC_ASSERT(num_of_links == (cur_force.size() / 4));
    MIMIC_ASSERT(num_of_links == (cur_torque.size() / 4));
    for (int idx = 0; idx < num_of_links; idx++)
    {
        // auto & cur_ext_force = ;
        // auto & cur_ext_torque =
        // mExternalTorques[frame_id][idx];
        for (int i = 0; i < 4; i++)
        {
            forces[idx][i] = cur_force[idx * 4 + i].asDouble();
            torques[idx][i] = cur_torque[idx * 4 + i].asDouble();
        }
        // assert(mExternalForces[frame_id][idx].norm() < 1e-10);
        // assert(mExternalTorques[frame_id][idx].norm() < 1e-10);
    }
}

/**
 * \brief               Write current summary table to the disk
 * \param path          specified location
 * \param is_append     Generally, this option is used with MPI.
 *                      When here are multiple process sampling or solving ID
 * together, they need to write a single summary table. In this case, all
 * processes need to load json from disk first, and append its own content in
 * it, finally write back. If you want to enable this feature, simply set
 * is_append to true.
 */
void tSummaryTable::WriteToDisk(const std::string &path, bool is_append)
{

    // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk is_append=" <<
    // is_append << std::endl;
    if (cFileUtil::AddLock(path) == false)
    {
        std::cout << "[error] tSummaryTable::WriteToDisk add lock failed for "
                  << path << std::endl;
        exit(1);
    }
    if (cFileUtil::ValidateFilePath(path) == false)
    {
        std::cout << "[error] tSummaryTable::WriteToDisk path invalid " << path
                  << std::endl;
        exit(0);
    }
    // std::cout <<"-----------------------------------------------" << mpi_rank
    // << std::endl; std::cout <<"[log] write table to " << path << std::endl;
    Json::Value root;
    if (is_append == true && cFileUtil::ExistsFile(path) == true)
    {
        // therer exists a file
        if (false == cJsonUtil::LoadJson(path, root))
        {
            std::cout << "[error] tSummaryTable WriteToDisk: Parse json "
                      << path << " failed\n";
            exit(0);
        }
        root["sample_num_of_trajs"] =
            mTotalEpochNum + root["sample_num_of_trajs"].asInt();

        if (root["item_list"].isArray() == false)
        {
            std::cout << "[error] tSummaryTable WriteToDisk single "
                         "traj list is not an array\n";
            exit(0);
        }
        Json::Value single_epoch;
        for (auto &x : mEpochInfos)
        {
            // std::cout << "sample traj file name = " << x.sample_traj_filename
            //           << std::endl;
            single_epoch["num_of_frame"] = x.frame_num;
            single_epoch["length_second"] = x.length_second;
            single_epoch["sample_traj_filename"] = x.sample_traj_filename;
            single_epoch["ID_train_filename"] = x.train_filename;
            single_epoch["mr_traj_filename"] = x.mr_traj_filename;
            root["item_list"].append(single_epoch);
        }
        // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk append size =
        // " << mEpochInfos.size() << std::endl;;
    }
    else
    {
        // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk begin to
        // overwrite\n";
        root["sample_char_file"] = mSampleCharFile;
        root["sample_controller_file"] = mSampleControllerFile;
        root["sample_num_of_trajs"] = mTotalEpochNum;
        root["sample_timestamp"] = mTimeStamp;
        root["item_list"] = Json::arrayValue;
        root["sample_traj_dir"] = mSampleTrajDir;
        root["ID_traindata_dir"] = mIDTraindataDir;
        root["mr_traj_dir"] = mMrTrajDir;
        Json::Value single_epoch;
        for (auto &x : mEpochInfos)
        {
            single_epoch["num_of_frame"] = x.frame_num;
            single_epoch["length_second"] = x.length_second;
            single_epoch["sample_traj_filename"] = x.sample_traj_filename;
            single_epoch["ID_train_filename"] = x.train_filename;
            single_epoch["mr_traj_filename"] = x.mr_traj_filename;
            root["item_list"].append(single_epoch);
        }
        // root["sample_action_theta_dist_file"] = mActionThetaDistFile;
    }

    // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk ready to write trajs
    // num = " << root["item_list"].size() << std::endl;;
    cJsonUtil::WriteJson(path, root, true);
    MIMIC_INFO("WriteToDisk {}", path);
    if (cFileUtil::DeleteLock(path) == false)
    {
        MIMIC_ERROR("delete lock failed for {}", path);
        exit(1);
    }
}

void tSummaryTable::LoadFromDisk(const std::string &path)
{
    // cFileUtil::AddLock(path);
    if (cFileUtil::ValidateFilePath(path) == false)
    {
        std::cout << "[error] tSummaryTable::LoadFromDisk path invalid " << path
                  << std::endl;
        exit(0);
    }

    if (cFileUtil::ExistsFile(path) == false &&
        cFileUtil::ExistsFile(path + ".bak") == true)
    {
        MIMIC_WARN("LoadFromDisk: path {} doesn't exist, but {} found, "
                   "rename and use this backup file",
                   path.c_str(), (path + ".bak").c_str());
        cFileUtil::RenameFile(path + ".bak", path);
    }

    Json::Value root;
    cJsonUtil::LoadJson(path, root);

    // overview infos
    mSampleCharFile = cJsonUtil::ParseAsString("sample_char_file", root);
    mSampleControllerFile =
        cJsonUtil::ParseAsString("sample_controller_file", root);
    mTotalEpochNum = cJsonUtil::ParseAsInt("sample_num_of_trajs", root);
    // mTotalLengthTime = root["total_second"].asDouble();
    // mTotalLengthFrame = root["total_frame"].asInt();
    mTimeStamp = cJsonUtil::ParseAsString("sample_timestamp", root);
    // mActionThetaDistFile =
    //     cJsonUtil::ParseAsString("sample_action_theta_dist_file", root);
    mSampleTrajDir = cJsonUtil::ParseAsString("sample_traj_dir", root);
    mIDTraindataDir = cJsonUtil::ParseAsString("ID_traindata_dir", root);
    mMrTrajDir = cJsonUtil::ParseAsString("mr_traj_dir", root);

    auto &trajs_lst = root["item_list"];
    if (mTotalEpochNum != trajs_lst.size())
    {
        std::cout
            << "[warn] tSummaryTable::LoadFromDisk trajs num doesn't match "
            << mTotalEpochNum << " " << trajs_lst.size() << ", correct it\n";
        mTotalEpochNum = trajs_lst.size();
    }

    std::cout << "mtotal epoch num = " << mTotalEpochNum << std::endl;
    // resize and load all trajs info
    mEpochInfos.resize(mTotalEpochNum);
    for (int i = 0; i < mTotalEpochNum; i++)
    {
        const auto &cur_traj_json = trajs_lst[i];
        auto &cur_epoch_info = mEpochInfos[i];
        cur_epoch_info.frame_num =
            cJsonUtil::ParseAsInt("num_of_frame", cur_traj_json);
        cur_epoch_info.length_second =
            cJsonUtil::ParseAsDouble("length_second", cur_traj_json);
        cur_epoch_info.sample_traj_filename =
            cJsonUtil::ParseAsString("sample_traj_filename", cur_traj_json);
        cur_epoch_info.mr_traj_filename =
            cJsonUtil::ParseAsString("mr_traj_filename", cur_traj_json);
        cur_epoch_info.train_filename =
            trajs_lst[i]["ID_train_filename"].asString();
    }
    MIMIC_INFO("LoadFromDisk " + path);
}

tSummaryTable::tSingleEpochInfo::tSingleEpochInfo()
{
    frame_num = 0;
    length_second = 0;
    sample_traj_filename = "";
    mr_traj_filename = "";
    train_filename = "";
}

tSummaryTable::tSummaryTable()
{
    mSampleCharFile = "";
    mSampleControllerFile = "";
    mTimeStamp = "";
    // mActionThetaDistFile = "";
    mTotalEpochNum = 0;
    // mTotalLengthTime = 0;
    // mTotalLengthFrame = 0;
    mSampleTrajDir = "";
    mEpochInfos.clear();
    // mLogger = cLogUtil::CreateLogger("tSummaryTable");
}