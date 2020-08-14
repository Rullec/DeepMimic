#include "InteractiveIDSolver.hpp"
#include "sim/Controller/CtPDController.h"
#include "sim/SimItems/SimCharacter.h"
#include "util/BulletUtil.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>
// #define IGNORE_TRAJ_VERISION
// #define VERBOSE

std::string controller_details_path;
cInteractiveIDSolver::cInteractiveIDSolver(cSceneImitate *imitate_scene,
                                           eIDSolverType type,
                                           const std::string &conf)
    : cIDSolver(imitate_scene, type)
{
    mLogger = cLogUtil::CreateLogger("InteractiveIDSolver");
    ParseConfig(conf);
}

cInteractiveIDSolver::~cInteractiveIDSolver() {}

void cInteractiveIDSolver::LoadTraj(tLoadInfo &load_info,
                                    const std::string &path)
{
    switch (mTrajFileVersion)
    {
    case eTrajFileVersion::UNSET:
        mLogger->error("LoadTraj: Please point out the version of traj file %s",
                       path.c_str());
        exit(0);
        break;
    case eTrajFileVersion::V1:
        load_info.LoadTrajV1(mSimChar, path);
        break;
    case eTrajFileVersion::V2:
        load_info.LoadTrajV2(mSimChar, path);
        break;
    }
}

void cInteractiveIDSolver::PrintLoadInfo(tLoadInfo &load_info,
                                         const std::string &filename,
                                         bool disable_root /*= true*/) const
{
    std::ofstream fout(filename, std::ios::app);
    if (mLoadInfo.mCurFrame == 0)
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
        fout << "---------- frame " << mLoadInfo.mCurFrame << " -----------\n";
        fout << "num_of_links = " << mNumLinks << std::endl;
        fout << "timestep = " << mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]
             << std::endl;
        fout << "total_mass = " << total_mass << std::endl;

        for (int i = 0; i < mNumLinks; i++)
        {
            tVector vel = (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] -
                           mLoadInfo.mLinkPos[mLoadInfo.mCurFrame - 1][i]) /
                          mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
            tVector omega = cMathUtil::CalcQuaternionVel(
                cMathUtil::RotMatToQuaternion(
                    mLoadInfo.mLinkRot[mLoadInfo.mCurFrame - 1][i]),
                cMathUtil::RotMatToQuaternion(
                    mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]),
                mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]);
            fout << "--- link " << i << " ---\n";
            fout << "mass = " << mk_lst[i] << std::endl;
            fout << "pos = "
                 << mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i]
                        .transpose()
                        .segment(0, 3)
                 << std::endl;
            fout << "rot = "
                 << cMathUtil::RotMatToQuaternion(
                        mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i])
                        .coeffs()
                        .transpose()
                 << std::endl;
            fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
            fout << "omega = " << omega.transpose().segment(0, 3) << std::endl;
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
            COM_cur +=
                mk_lst[i] * mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i + 1];
        }
        COM_cur /= total_mass;

        // output to file
        fout << "---------- frame " << mLoadInfo.mCurFrame << " -----------\n";
        fout << "num_of_links = " << mNumLinks - 1 << std::endl;
        fout << "timestep = " << mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]
             << std::endl;
        fout << "total_mass = " << total_mass << std::endl;
        tVector lin_mom = tVector::Zero(), ang_mom = tVector::Zero();

        for (int i = 1; i < mNumLinks; i++)
        {
            tVector vel = (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] -
                           mLoadInfo.mLinkPos[mLoadInfo.mCurFrame - 1][i]) /
                          mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
            tVector omega = cMathUtil::CalcQuaternionVel(
                cMathUtil::RotMatToQuaternion(
                    mLoadInfo.mLinkRot[mLoadInfo.mCurFrame - 1][i]),
                cMathUtil::RotMatToQuaternion(
                    mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]),
                mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]);
            fout << "--- link " << i - 1 << " ---\n";
            fout << "mass = " << mk_lst[i - 1] << std::endl;
            fout << "pos = "
                 << mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i]
                        .transpose()
                        .segment(0, 3)
                 << std::endl;
            fout << "rot = "
                 << cMathUtil::RotMatToQuaternion(
                        mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i])
                        .coeffs()
                        .transpose()
                 << std::endl;
            fout << "rotmat = \n"
                 << mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i].block(0, 0, 3, 3)
                 << std::endl;
            fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
            fout << "omega = " << omega.transpose().segment(0, 3) << std::endl;
            fout << "inertia = "
                 << cBulletUtil::btVectorTotVector0(
                        mMultibody->getLinkInertia(i - 1))
                        .transpose()
                        .segment(0, 3)
                 << std::endl;
            lin_mom += mk_lst[i - 1] * vel;
            ang_mom +=
                mk_lst[i - 1] *
                    (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] - COM_cur)
                        .cross3(vel) +
                mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i] *
                    cBulletUtil::btVectorTotVector0(
                        mMultibody->getLinkInertia(i - 1))
                        .asDiagonal() *
                    mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i].transpose() *
                    omega;
        }

        tVector contact_force_impulse = tVector::Zero();
        for (auto &pt : mLoadInfo.mContactForces[mLoadInfo.mCurFrame])
        {
            contact_force_impulse +=
                pt.mForce * mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
        }
        fout << "COM cur = " << COM_cur.transpose().segment(0, 3) << std::endl;
        fout << "linear momentum = " << lin_mom.transpose().segment(0, 3)
             << std::endl;
        fout << "contact pt num = "
             << mLoadInfo.mContactForces[mLoadInfo.mCurFrame].size()
             << std::endl;
        fout << "contact pt impulse = " << contact_force_impulse.transpose()
             << std::endl;
        fout << "angular momentum = " << ang_mom.transpose().segment(0, 3)
             << std::endl;
    }
}

/*
    @Function: LoadMotion
    @params: path Type const std::string &, the filename of specified motion
    @params: motion Type cMotion *, the targeted motion storaged.
*/
void cInteractiveIDSolver::LoadMotion(const std::string &path,
                                      cMotion *motion) const
{
    assert(cFileUtil::ExistsFile(path));

    cMotion::tParams params;
    params.mMotionFile = path;
    motion->Load(params);
    mLogger->debug("Load Motion from %s, frame_nums = %d, dof = %d", path,
                   motion->GetNumFrames(), motion->GetNumDof());
}

void cInteractiveIDSolver::SaveMotion(const std::string &path_root,
                                      cMotion *motion) const
{
    assert(nullptr != motion);
    if (false == cFileUtil::ValidateFilePath(path_root))
    {
        mLogger->error("SaveMotion: path root invalid %s", path_root);
        exit(1);
    }
    std::string filename = cFileUtil::RemoveExtension(path_root) + "_" +
                           std::to_string(mSaveInfo.mCurEpoch) + "." +
                           cFileUtil::GetExtension(path_root);
    mLogger->info("SaveMotion for epoch %d to %s", mSaveInfo.mCurEpoch,
                  filename);
    motion->FinishAddFrame();
    motion->Output(filename);
    motion->Clear();
}

/**
 * \brief                   Save Train Data "*.train"
 * \param dir               storaged directory
 * \param info              a info struct for what we need to save.
 */
void cInteractiveIDSolver::SaveTrainData(
    const std::string &dir, const std::string &filename,
    std::vector<tSingleFrameIDResult> &info) const
{
    if (cFileUtil::ExistsDir(dir) == false)
    {
        mLogger->error("SaveTrainData target dir {} doesn't exist", dir);
        exit(0);
    }
    Json::Value root;
    root["num_of_frames"] = static_cast<int>(info.size());
    root["data_list"] = Json::arrayValue;
    int num_of_frame = info.size();
    Json::Value single_frame;
    for (int i = 0; i < num_of_frame; i++)
    {
        single_frame["frame_id"] = i;
        single_frame["state"] = Json::arrayValue;
        single_frame["action"] = Json::arrayValue;
        for (int j = 0; j < info[i].state.size(); j++)
            single_frame["state"].append(info[i].state[j]);
        for (int j = 0; j < info[i].action.size(); j++)
            single_frame["action"].append(info[i].action[j]);
        single_frame["reward"] = info[i].reward;
        root["data_list"].append(single_frame);
    }
    cJsonUtil::WriteJson(cFileUtil::ConcatFilename(dir, filename), root, false);
#ifdef VERBOSE
    std::cout << "[log] cInteractiveIDSolver::SaveTrainData to " << path
              << std::endl;
#endif
}

/**
 * \brief       Init Action Theta Dist.
 * For more details please check the comment for var "mActionThetaDist"
 */
void cInteractiveIDSolver::InitActionThetaDist(cSimCharacter *sim_char,
                                               tMatrixXd &mat) const
{
    // [links except root, Granularity]
    mat.resize(mSimChar->GetMultiBody()->getNumLinks(),
               mActionThetaGranularity);
    mat.setZero();
}

void cInteractiveIDSolver::LoadActionThetaDist(const std::string &path,
                                               tMatrixXd &mat) const
{
    Json::Value root;
    if (cJsonUtil::LoadJson(path, root) == false)
    {
        mLogger->error("LoadActionThetaDist failed for " + path);
        exit(1);
    }

    int num_of_joints = mSimChar->GetNumJoints();
    mat.resize(num_of_joints, mActionThetaGranularity);
    mat.setZero();

    if (num_of_joints != root.size())
    {
        mLogger->error(
            "LoadActionThetaDist from %s expected %d items but get %d",
            path.c_str(), num_of_joints, root.size());
        exit(1);
    }
    tVectorXd row;
    auto &multibody = mSimChar->GetMultiBody();
    for (int i = 0; i < multibody->getNumLinks(); i++)
    {
        cJsonUtil::ReadVectorJson(root[std::to_string(i)], row);
        mat.row(i) = row;
    }
    mLogger->info("LoadActionThetaDist from " + path);
}

void cInteractiveIDSolver::SaveActionThetaDist(const std::string &path,
                                               tMatrixXd &mat) const
{
    if (cFileUtil::ValidateFilePath(path) == false)
    {
        mLogger->error("SaveActionThetaDist to %s failed", path.c_str());
        exit(1);
    }

    Json::Value root;
    for (int i = 0; i < mat.rows(); i++)
    {
        root[std::to_string(i)] = Json::arrayValue;
        for (int j = 0; j < mActionThetaGranularity; j++)
            root[std::to_string(i)].append(mat(i, j));
    }
    cJsonUtil::WriteJson(path, root, true);
    mLogger->info("SaveActionThetaDist to " + path);
}

void cInteractiveIDSolver::ParseConfig(const std::string &path)
{
    if (cFileUtil::ExistsFile(path) == false)
    {
        mLogger->info("ParseConfig %s doesn't exist", path.c_str());
        exit(1);
    }

    Json::Value root;
    cJsonUtil::LoadJson(path, root);

    mTrajFileVersion = static_cast<eTrajFileVersion>(
        cJsonUtil::ParseAsInt("traj_file_version", root));
}

/**
 * \brief                   Save trajectories to disk
 * \param mSaveInfo         the trajecoty we want to save
 * \param traj_dir          the storaged directory
 * \param traj_root_name    root name for storaged.
 */
std::string
cInteractiveIDSolver::SaveTraj(tSaveInfo &mSaveInfo,
                               const std::string &traj_dir,
                               const std::string &traj_rootname) const
{
    if (cFileUtil::ExistsDir(traj_dir) == false)
    {
        mLogger->error("SaveTraj directory {} doesn't exist", traj_dir);
        exit(0);
    }

    Json::Value root;
    switch (mTrajFileVersion)
    {
    case eTrajFileVersion::V1:
        SaveTrajV1(mSaveInfo, root);
        break;
    case eTrajFileVersion::V2:
        SaveTrajV2(mSaveInfo, root);
        break;
    default:
        mLogger->error("SaveTraj unsupported traj file version %d",
                       static_cast<int>(mTrajFileVersion));
        exit(1);
        break;
    }

    std::string traj_name = cFileUtil::GenerateRandomFilename(traj_rootname);
    std::string final_name = cFileUtil::ConcatFilename(traj_dir, traj_name);
    cFileUtil::AddLock(final_name);
    if (false == cFileUtil::ValidateFilePath(final_name))
    {
        mLogger->error("SaveTraj path %s illegal", final_name);
        exit(1);
    }
    cJsonUtil::WriteJson(final_name, root, false);
    cFileUtil::DeleteLock(final_name);
#ifdef VERBOSE
    std::cout << "[log] cInteractiveIDSolver::SaveTraj for epoch "
              << mSaveInfo.mCurEpoch << " to " << final_name << std::endl;
#endif
    return traj_name;
}

/**
 * \brief                   save trajectories for full version
 * \param mSaveInfo
 * \param path              target save location
 */
void cInteractiveIDSolver::SaveTrajV1(tSaveInfo &mSaveInfo, Json::Value &root)
{

    Json::Value single_frame;
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 1;
    for (int frame_id = 0; frame_id < mSaveInfo.mCurFrameId; frame_id++)
    {
        /* set up single frame, including:
            - frame id
            - timestep
            - generalized coordinate, pos, vel, accel
            - contact info
            - external forces
        */
        single_frame["frame_id"] = frame_id;
        single_frame["timestep"] = mSaveInfo.mTimesteps[frame_id];
        single_frame["motion_ref_time"] = mSaveInfo.mRefTime[frame_id];
        single_frame["reward"] = mSaveInfo.mRewards[frame_id];
        single_frame["pos"] = Json::Value(Json::arrayValue);
        single_frame["vel"] = Json::Value(Json::arrayValue);
        single_frame["accel"] = Json::Value(Json::arrayValue);
        for (int dof = 0; dof < mSaveInfo.mBuffer_q[frame_id].size(); dof++)
        {
            single_frame["pos"].append(mSaveInfo.mBuffer_q[frame_id][dof]);
            if (frame_id >= 1)
            {
                single_frame["vel"].append(mSaveInfo.mBuffer_u[frame_id][dof]);
                single_frame["accel"].append(
                    mSaveInfo.mBuffer_u_dot[frame_id][dof]);
            }
        }

        // set up contact info
        single_frame["contact_info"] = Json::arrayValue;
        // single_frame["contact_num"] =
        // mSaveInfo.mContactForces[frame_id].size();
        single_frame["contact_num"] =
            static_cast<int>(mSaveInfo.mContactForces[frame_id].size());
        for (int c_id = 0; c_id < mSaveInfo.mContactForces[frame_id].size();
             c_id++)
        {
            Json::Value single_contact;
            const tContactForceInfo &force =
                mSaveInfo.mContactForces[frame_id][c_id];
            single_contact["force_pos"] = Json::arrayValue;
            for (int i = 0; i < 4; i++)
                single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for (int i = 0; i < 4; i++)
                single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
            single_contact["is_self_collision"] = force.mIsSelfCollision;
            single_frame["contact_info"].append(single_contact);
        }

        // set up external force & torques info
        single_frame["external_force"] = Json::arrayValue;
        single_frame["external_torque"] = Json::arrayValue;
        int num_links = mSaveInfo.mExternalForces[frame_id].size();
        for (int link_id = 0; link_id < num_links; link_id++)
        {
            for (int idx = 0; idx < 4; idx++)
            {
                single_frame["external_force"].append(
                    mSaveInfo.mExternalForces[frame_id][link_id][idx]);
                single_frame["external_torque"].append(
                    mSaveInfo.mExternalTorques[frame_id][link_id][idx]);
            }
        }

        // set up truth joint torques
        single_frame["truth_joint_force"] = Json::arrayValue;
        for (int i = 0; i < num_links - 1; i++)
        {
            for (int j = 0; j < 4; j++)
                single_frame["truth_joint_force"].append(
                    mSaveInfo.mTruthJointForces[frame_id][i][j]);
        }

        // set up truth action
        single_frame["truth_action"] = Json::arrayValue;
        for (int i = 0; i < mSaveInfo.mTruthAction[frame_id].size(); i++)
        {
            single_frame["truth_action"].append(
                mSaveInfo.mTruthAction[frame_id][i]);
        }

        // set up truth pd target
        single_frame["truth_pd_target"] = Json::arrayValue;
        for (int i = 0; i < mSaveInfo.mTruthPDTarget[frame_id].size(); i++)
        {
            single_frame["truth_pd_target"].append(
                mSaveInfo.mTruthPDTarget[frame_id][i]);
        }

        // set up character poses
        single_frame["char_pose"] = Json::arrayValue;
        for (int i = 0; i < mSaveInfo.mCharPoses[frame_id].size(); i++)
        {
            single_frame["char_pose"].append(mSaveInfo.mCharPoses[frame_id][i]);
        }

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
void cInteractiveIDSolver::SaveTrajV2(
    tSaveInfo &mSaveInfo,
    Json::Value &root) // save trajectories for simplified version
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
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 2;

    for (int frame_id = 0; frame_id < mSaveInfo.mCurFrameId; frame_id++)
    {
        single_frame["frame_id"] = frame_id;
        single_frame["char_pose"] = Json::arrayValue;
        for (int i = 0; i < mSaveInfo.mCharPoses[frame_id].size(); i++)
        {
            single_frame["char_pose"].append(mSaveInfo.mCharPoses[frame_id][i]);
        }
        single_frame["timestep"] = mSaveInfo.mTimesteps[frame_id];

        // contact info
        single_frame["contact_info"] = Json::arrayValue;
        // single_frame["contact_num"] =
        // mSaveInfo.mContactForces[frame_id].size();
        single_frame["contact_num"] =
            static_cast<int>(mSaveInfo.mContactForces[frame_id].size());
        for (int c_id = 0; c_id < mSaveInfo.mContactForces[frame_id].size();
             c_id++)
        {
            Json::Value single_contact;
            const tContactForceInfo &force =
                mSaveInfo.mContactForces[frame_id][c_id];
            single_contact["force_pos"] = Json::arrayValue;
            for (int i = 0; i < 4; i++)
                single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for (int i = 0; i < 4; i++)
                single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
            single_contact["is_self_collision"] = force.mIsSelfCollision;
            single_frame["contact_info"].append(single_contact);
        }

        // rewards
        single_frame["reward"] = mSaveInfo.mRewards[frame_id];

        // motion ref time
        single_frame["motion_ref_time"] = mSaveInfo.mRefTime[frame_id];

        // actions
        single_frame["truth_action"] = Json::arrayValue;
        for (int i = 0; i < mSaveInfo.mTruthAction[frame_id].size(); i++)
        {
            single_frame["truth_action"].append(
                mSaveInfo.mTruthAction[frame_id][i]);
        }

        root["list"].append(single_frame);
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
void cInteractiveIDSolver::tSummaryTable::WriteToDisk(const std::string &path,
                                                      bool is_append)
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
        root["sample_action_theta_dist_file"] = mActionThetaDistFile;
    }

    // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk ready to write trajs
    // num = " << root["item_list"].size() << std::endl;;
    cJsonUtil::WriteJson(path, root, true);
    mLogger->info("WriteToDisk {}", path);
    if (cFileUtil::DeleteLock(path) == false)
    {
        mLogger->error("delete lock failed for {}", path);
        exit(1);
    }
}

void cInteractiveIDSolver::tSummaryTable::LoadFromDisk(const std::string &path)
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
        mLogger->error("LoadFromDisk: path %s doesn't exist, but %s found, "
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
    mActionThetaDistFile =
        cJsonUtil::ParseAsString("sample_action_theta_dist_file", root);
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
    mLogger->info("LoadFromDisk " + path);
}

cInteractiveIDSolver::tSummaryTable::tSingleEpochInfo::tSingleEpochInfo()
{
    frame_num = 0;
    length_second = 0;
    sample_traj_filename = "";
    mr_traj_filename = "";
    train_filename = "";
}

cInteractiveIDSolver::tSummaryTable::tSummaryTable()
{
    mSampleCharFile = "";
    mSampleControllerFile = "";
    mTimeStamp = "";
    mActionThetaDistFile = "";
    mTotalEpochNum = 0;
    // mTotalLengthTime = 0;
    // mTotalLengthFrame = 0;
    mSampleTrajDir = "";
    mEpochInfos.clear();
    mLogger = cLogUtil::CreateLogger("tSummaryTable");
}