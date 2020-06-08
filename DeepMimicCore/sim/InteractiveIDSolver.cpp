#include "InteractiveIDSolver.hpp"
#include <sim/SimCharacter.h>
#include "sim/CtPDController.h"
#include "../util/JsonUtil.h"
#include "../util/BulletUtil.h"
#include "../util/FileUtil.h"
#include <iostream>
// #define IGNORE_TRAJ_VERISION
// #define VERBOSE

std::string controller_details_path;
cInteractiveIDSolver::cInteractiveIDSolver(cSceneImitate * imitate_scene, eIDSolverType type, const std::string & conf)
    :cIDSolver(imitate_scene, type)
{
    mLogger = cLogUtil::CreateLogger("InteractiveIDSolver");
    ParseConfig(conf);
}

cInteractiveIDSolver::~cInteractiveIDSolver()
{

}


void cInteractiveIDSolver::LoadTraj(tLoadInfo & load_info, const std::string & path)
{
    switch(mTrajFileVersion)
    {
        case eTrajFileVersion::UNSET: mLogger->error("LoadTraj: Please point out the version of traj file %s", path.c_str()); exit(0); break;
        case eTrajFileVersion::V1: LoadTrajV1(load_info, path); break;
        case eTrajFileVersion::V2: LoadTrajV2(load_info, path); break;
    }
}


void cInteractiveIDSolver::PrintLoadInfo(tLoadInfo & load_info, const std::string & filename, bool disable_root /*= true*/) const
{
    std::ofstream fout(filename, std::ios::app);
    if(mLoadInfo.mCurFrame == 0) return;

    if(disable_root == false)
    {
        // fetch mass
        std::vector<double> mk_lst(0);
        double total_mass = 0;
        mk_lst.resize(mNumLinks);
        for(int i=0; i<mNumLinks; i++)
        {
            if(i ==0 )
                mk_lst[i] = mMultibody->getBaseMass();
            else
                mk_lst[i] = mMultibody->getLinkMass(i - 1);
            total_mass += mk_lst[i];
        }

        // output to file
        fout << "---------- frame " << mLoadInfo.mCurFrame << " -----------\n";
        fout << "num_of_links = " << mNumLinks << std::endl;
        fout << "timestep = " << mLoadInfo.mTimesteps[mLoadInfo.mCurFrame] << std::endl;
        fout << "total_mass = " << total_mass << std::endl;

        for(int i=0; i < mNumLinks; i++)
        {
            tVector vel = (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] - mLoadInfo.mLinkPos[mLoadInfo.mCurFrame-1][i]) / mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
            tVector omega = cMathUtil::CalcQuaternionVel(
                cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame-1][i]),
                cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]),
                mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]);
            fout << "--- link " << i <<" ---\n";
            fout << "mass = " << mk_lst[i] << std::endl;
            fout << "pos = " << mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i].transpose().segment(0, 3) << std::endl;
            fout << "rot = " << cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]).coeffs().transpose() << std::endl;
            fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
            fout << "omega = " << omega.transpose().segment(0, 3) << std::endl;
            // fout << "COM cur = " << COM_cur.transpose() << std::endl;
        }
    }
    else
    {
        // disable the root link for DeepMimic, mass is zero and inertia is zero fetch mass
        std::vector<double> mk_lst(0);
        double total_mass = 0;
        tVector COM_cur = tVector::Zero();
        mk_lst.resize(mNumLinks - 1);
        for(int i=0; i<mNumLinks - 1; i++)
        {
            mk_lst[i] = mMultibody->getLinkMass(i);
            total_mass += mk_lst[i];
            COM_cur += mk_lst[i] * mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i + 1];
        }
        COM_cur /= total_mass;
        
        // output to file
        fout << "---------- frame " << mLoadInfo.mCurFrame << " -----------\n";
        fout << "num_of_links = " << mNumLinks-1 << std::endl;
        fout << "timestep = " << mLoadInfo.mTimesteps[mLoadInfo.mCurFrame] << std::endl;
        fout << "total_mass = " << total_mass << std::endl;
        tVector lin_mom = tVector::Zero(), ang_mom = tVector::Zero();

        for(int i=1; i < mNumLinks; i++)
        {
            tVector vel = (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] - mLoadInfo.mLinkPos[mLoadInfo.mCurFrame-1][i]) / mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
            tVector omega = cMathUtil::CalcQuaternionVel(
                cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame-1][i]),
                cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]),
                mLoadInfo.mTimesteps[mLoadInfo.mCurFrame]);
            fout << "--- link " << i-1 <<" ---\n";
            fout << "mass = " << mk_lst[i-1] << std::endl;
            fout << "pos = " << mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i].transpose().segment(0, 3) << std::endl;
            fout << "rot = " << cMathUtil::RotMatToQuaternion(mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i]).coeffs().transpose() << std::endl;
            fout << "rotmat = \n" << mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i].block(0, 0, 3, 3) << std::endl;
            fout << "vel = " << vel.transpose().segment(0, 3) << std::endl;
            fout << "omega = " << omega.transpose().segment(0, 3) << std::endl;
            fout << "inertia = " << cBulletUtil::btVectorTotVector0(mMultibody->getLinkInertia(i-1)).transpose().segment(0, 3) << std::endl;
            lin_mom += mk_lst[i-1] * vel;
            ang_mom += mk_lst[i-1] * (mLoadInfo.mLinkPos[mLoadInfo.mCurFrame][i] - COM_cur).cross3(vel)
                        + mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i] 
                            * 
                        cBulletUtil::btVectorTotVector0(mMultibody->getLinkInertia(i-1)).asDiagonal()
                        *mLoadInfo.mLinkRot[mLoadInfo.mCurFrame][i].transpose()
                        * omega;
            
        }

        tVector contact_force_impulse = tVector::Zero();
        for(auto & pt : mLoadInfo.mContactForces[mLoadInfo.mCurFrame])
        {
            contact_force_impulse += pt.mForce * mLoadInfo.mTimesteps[mLoadInfo.mCurFrame];
        }
        fout << "COM cur = " << COM_cur.transpose().segment(0, 3) << std::endl;
        fout << "linear momentum = " << lin_mom.transpose().segment(0, 3) << std::endl;
        fout << "contact pt num = " << mLoadInfo.mContactForces[mLoadInfo.mCurFrame].size() << std::endl;
        fout << "contact pt impulse = " << contact_force_impulse.transpose() << std::endl;
        fout << "angular momentum = " << ang_mom.transpose().segment(0, 3) << std::endl;
    }
    
}


/*
    @Function: LoadMotion
    @params: path Type const std::string &, the filename of specified motion
    @params: motion Type cMotion *, the targeted motion storaged.
*/
void cInteractiveIDSolver::LoadMotion(const std::string & path, cMotion * motion) const
{
    assert(cFileUtil::ExistsFile(path));
    
    cMotion::tParams params;
    params.mMotionFile = path;
    motion->Load(params);
    DebugPrintf(mLogger, "Load Motion from %s, frame_nums = %d, dof = %d", path, motion->GetNumFrames(), motion->GetNumDof());
}

void cInteractiveIDSolver::SaveMotion(const std::string & path_root, cMotion * motion) const
{
    assert(nullptr != motion);
    if(false == cFileUtil::ValidateFilePath(path_root))
    {
        ErrorPrintf(mLogger, "SaveMotion: path root invalid %s", path_root);
        exit(1);
    }
    std::string filename = cFileUtil::RemoveExtension(path_root) + "_" + std::to_string(mSaveInfo.mCurEpoch) + "." + cFileUtil::GetExtension(path_root);
    InfoPrintf(mLogger, "SaveMotion for epoch %d to %s", mSaveInfo.mCurEpoch, filename);
    motion->FinishAddFrame();
    motion->Output(filename);
    motion->Clear();
}

/**
 * \brief                   Save Train Data "*.train"
 * \param path              target filename
 * \param info              a info struct for what we need to save.
*/
void cInteractiveIDSolver::SaveTrainData(const std::string & path, std::vector<tSingleFrameIDResult> & info) const
{
    if(cFileUtil::ValidateFilePath(path) == false)
    {
        mLogger->error("SaveTrainData path {} invalid", path);
        exit(0);
    }
    Json::Value root;
    root["num_of_frames"] = static_cast<int>(info.size());
    root["data_list"] = Json::arrayValue;
    int num_of_frame = info.size();
    Json::Value single_frame;
    for(int i=0; i<num_of_frame; i++)
    {
        single_frame["frame_id"] = i;
        single_frame["state"] = Json::arrayValue;
        single_frame["action"] = Json::arrayValue;
        for(int j = 0; j < info[i].state.size(); j++) single_frame["state"].append(info[i].state[j]);
        for(int j = 0; j < info[i].action.size(); j++) single_frame["action"].append(info[i].action[j]);
        single_frame["reward"] = info[i].reward;
        root["data_list"].append(single_frame);
    }
    cJsonUtil::WriteJson(path, root, false);
#ifdef VERBOSE
    std::cout <<"[log] cInteractiveIDSolver::SaveTrainData to " << path << std::endl;
#endif
}


/**
 * \brief       Init Action Theta Dist. 
 * For more details please check the comment for var "mActionThetaDist"
*/
void cInteractiveIDSolver::InitActionThetaDist(cSimCharacter * sim_char, tMatrixXd & mat) const
{
    // [links except root, Granularity]
    mat.resize(mSimChar->GetMultiBody()->getNumLinks(), mActionThetaGranularity);
    mat.setZero();
}

void cInteractiveIDSolver::LoadActionThetaDist(const std::string & path, tMatrixXd & mat) const
{
    Json::Value root;
    if(cJsonUtil::LoadJson(path, root) == false)
    {
        mLogger->error("LoadActionThetaDist failed for " + path);
        exit(1);
    }
    
    int num_of_joints = mSimChar->GetNumJoints();
    mat.resize(num_of_joints, mActionThetaGranularity);
    mat.setZero();

    if(num_of_joints != root.size())
    {
        ErrorPrintf(mLogger, "LoadActionThetaDist from %s expected %d items but get %d", path.c_str(), num_of_joints, root.size());
        exit(1);
    }
    tVectorXd row;
    auto & multibody = mSimChar->GetMultiBody();
    for (int i = 0; i<multibody->getNumLinks(); i++)
    {
        cJsonUtil::ReadVectorJson(root[std::to_string(i)], row);
        mat.row(i) = row;
    }
    mLogger->info("LoadActionThetaDist from " + path);
}

void cInteractiveIDSolver::SaveActionThetaDist(const std::string & path, tMatrixXd & mat) const
{
    if(cFileUtil::ValidateFilePath(path) == false)
    {
        ErrorPrintf(mLogger, "SaveActionThetaDist to %s failed", path.c_str());
        exit(1);
    }

    Json::Value root;
    for (int i=0; i<mat.rows(); i++)
    {
        root[std::to_string(i)] = Json::arrayValue;
        for(int j =0; j<mActionThetaGranularity; j++) root[std::to_string(i)].append(mat(i, j));
    }
    cJsonUtil::WriteJson(path, root, true);
    mLogger->info("SaveActionThetaDist to " + path);
}

void cInteractiveIDSolver::ParseConfig(const std::string & path)
{
    if(cFileUtil::ExistsFile(path) == false)
    {
        ErrorPrintf(mLogger, "ParseConfig %s doesn't exist", path.c_str());
        exit(1);
    }

    Json::Value root;
    cJsonUtil::LoadJson(path, root);

    mTrajFileVersion = static_cast<eTrajFileVersion>(cJsonUtil::ParseAsInt("traj_file_version", root));
}

/**
 * \brief                   Save trajectories to disk
*/
std::string cInteractiveIDSolver::SaveTraj(tSaveInfo & mSaveInfo, const std::string & path_raw) const
{
    if(cFileUtil::ValidateFilePath(path_raw) == false)
    {
        ErrorPrintf(mLogger, "SaveTraj path %s invalid", path_raw);
        exit(0);
    } 

    Json::Value root;
    switch (mTrajFileVersion)
    {
    case eTrajFileVersion::V1: root = SaveTrajV1(mSaveInfo); break;
    case eTrajFileVersion::V2: root = SaveTrajV2(mSaveInfo); break;
    default:
        mLogger->error("SaveTraj unsupported traj file version %d", static_cast<int>(mTrajFileVersion));
        exit(1);
        break;
    }

    std::string final_name = cFileUtil::GenerateRandomFilename(path_raw);
    cFileUtil::AddLock(final_name);
    if(false == cFileUtil::ValidateFilePath(final_name))
    {
        ErrorPrintf(mLogger, "SaveTraj path %s illegal", final_name);
        exit(1);
    }
    cJsonUtil::WriteJson(final_name, root, false);
    cFileUtil::DeleteLock(final_name);
#ifdef VERBOSE
    std::cout <<"[log] cInteractiveIDSolver::SaveTraj for epoch " << mSaveInfo.mCurEpoch <<" to " << final_name << std::endl;
#endif
    return final_name;
}

/**
 * \brief                   save trajectories for full version
 * \param mSaveInfo
 * \param path              target save location
*/
const Json::Value cInteractiveIDSolver::SaveTrajV1(tSaveInfo & mSaveInfo) const
{

#ifndef IGNORE_TRAJ_VERISION
    if (mTrajFileVersion != eTrajFileVersion::V1)
    {
        mLogger->error("SaveTrajV1 is called but not the spcified traj file version is %d", static_cast<int>(mTrajFileVersion));
        exit(1);
    }
#endif

    Json::Value root, single_frame;
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 1;
    for(int frame_id=0; frame_id<mSaveInfo.mCurFrameId; frame_id++)
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
        for(int dof = 0; dof < mSaveInfo.mBuffer_q[frame_id].size(); dof++)
        {
            single_frame["pos"].append(mSaveInfo.mBuffer_q[frame_id][dof]);
            if(frame_id>=1)
            {
                single_frame["vel"].append(mSaveInfo.mBuffer_u[frame_id][dof]);
                single_frame["accel"].append(mSaveInfo.mBuffer_u_dot[frame_id][dof]);
            }
        }

        // set up contact info
        single_frame["contact_info"] = Json::arrayValue;
        // single_frame["contact_num"] = mSaveInfo.mContactForces[frame_id].size();
        single_frame["contact_num"] = static_cast<int>(mSaveInfo.mContactForces[frame_id].size());
        for(int c_id = 0; c_id < mSaveInfo.mContactForces[frame_id].size(); c_id++)
        {
            Json::Value single_contact;
            const tContactForceInfo & force = mSaveInfo.mContactForces[frame_id][c_id];
            single_contact["force_pos"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
            single_contact["is_self_collision"] = force.mIsSelfCollision;
            single_frame["contact_info"].append(single_contact);
        }

        // set up external force & torques info
        single_frame["external_force"] = Json::arrayValue;
        single_frame["external_torque"] = Json::arrayValue;
        for(int link_id = 0; link_id<mNumLinks; link_id++)
        {
            for(int idx = 0; idx<4; idx++)
            {
                 single_frame["external_force"].append(mSaveInfo.mExternalForces[frame_id][link_id][idx]);
                 single_frame["external_torque"].append(mSaveInfo.mExternalTorques[frame_id][link_id][idx]);
            }
        }

        // set up truth joint torques
        single_frame["truth_joint_force"] = Json::arrayValue;
        assert(mNumLinks-1 == mSaveInfo.mTruthJointForces[frame_id].size());
        for(int i=0; i< mNumLinks-1;i++)
        {
            for(int j=0; j<4; j++)
            single_frame["truth_joint_force"].append(mSaveInfo.mTruthJointForces[frame_id][i][j]);
        }

        // set up truth action
        single_frame["truth_action"] = Json::arrayValue;
        for(int i=0; i<mSaveInfo.mTruthAction[frame_id].size(); i++)
        {
            single_frame["truth_action"].append(mSaveInfo.mTruthAction[frame_id][i]);
        }

        // set up truth pd target
        single_frame["truth_pd_target"] = Json::arrayValue;
        for(int i=0; i<mSaveInfo.mTruthPDTarget[frame_id].size(); i++)
        {
            single_frame["truth_pd_target"].append(mSaveInfo.mTruthPDTarget[frame_id][i]);
        }

        // set up character poses
        single_frame["char_pose"] = Json::arrayValue;
        for(int i=0; i < mSaveInfo.mCharPoses[frame_id].size(); i++)
        {
            single_frame["char_pose"].append(mSaveInfo.mCharPoses[frame_id][i]);
        }

        // append to the whole list
        root["list"].append(single_frame);
    }
    return root;
}

/**
 * \brief                   Save simplified trajectries which are called "V2"
 * \param mSaveInfo
 * \param path              target save location
 * 
*/
const Json::Value cInteractiveIDSolver::SaveTrajV2(tSaveInfo & mSaveInfo) const    // save trajectories for simplified version
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

#ifndef IGNORE_TRAJ_VERISION
    if(eTrajFileVersion::V2 != mTrajFileVersion)
    {
        mLogger->error("SaveTrajV2 is called but now the mTrajFileVersion is %{}", static_cast<int>(mTrajFileVersion));
        exit(1);
    }
#endif 

    Json::Value root, single_frame;
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    root["version"] = 2;

    for(int frame_id=0; frame_id<mSaveInfo.mCurFrameId; frame_id++)
    {
        single_frame["frame_id"] = frame_id;
        single_frame["char_pose"] = Json::arrayValue;
        for(int i=0; i < mSaveInfo.mCharPoses[frame_id].size(); i++)
        {
            single_frame["char_pose"].append(mSaveInfo.mCharPoses[frame_id][i]);
        }
        single_frame["timestep"] = mSaveInfo.mTimesteps[frame_id];

        // contact info
        single_frame["contact_info"] = Json::arrayValue;
        // single_frame["contact_num"] = mSaveInfo.mContactForces[frame_id].size();
        single_frame["contact_num"] = static_cast<int>(mSaveInfo.mContactForces[frame_id].size());
        for(int c_id = 0; c_id < mSaveInfo.mContactForces[frame_id].size(); c_id++)
        {
            Json::Value single_contact;
            const tContactForceInfo & force = mSaveInfo.mContactForces[frame_id][c_id];
            single_contact["force_pos"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_value"].append(force.mForce[i]);
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
        for(int i=0; i<mSaveInfo.mTruthAction[frame_id].size(); i++)
        {
            single_frame["truth_action"].append(mSaveInfo.mTruthAction[frame_id][i]);
        }

        root["list"].append(single_frame);
    }

    return root;
}

void cInteractiveIDSolver::tSummaryTable::WriteToDisk(const std::string & path, bool is_append)
{ 
    
    // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk is_append=" << is_append << std::endl;
    if(cFileUtil::AddLock(path) == false)
    {
        std::cout <<"[error] tSummaryTable::WriteToDisk add lock failed for " << path << std::endl;
        exit(1);
    }
    if(cFileUtil::ValidateFilePath(path) == false)
    {
        std::cout <<"[error] tSummaryTable::WriteToDisk path invalid " << path << std::endl;
        exit(0);
    }
    // std::cout <<"-----------------------------------------------" << mpi_rank << std::endl;
    // std::cout <<"[log] write table to " << path << std::endl;
    Json::Value root;
    if(is_append == true && cFileUtil::ExistsFile(path) == true)
    {
        // therer exists a file
        if(false == cJsonUtil::LoadJson(path, root))
        {
            std::cout <<"[error] tSummaryTable WriteToDisk: Parse json " << path <<" failed\n";
            exit(0);
        }
        root["num_of_trajs"] = mTotalEpochNum + root["num_of_trajs"].asInt();
        root["total_second"] = mTotalLengthTime + root["total_second"].asDouble();
        root["total_frame"] = mTotalLengthFrame + root["total_frame"].asInt();
        // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk load trajs num = " << root["single_trajs_lst"].size() << std::endl;;
        if(root["single_trajs_lst"].isArray() == false)
        {
            std::cout << "[error] tSummaryTable WriteToDisk single traj list is not an array\n";
            exit(0);
        }
        Json::Value single_epoch;
        for (auto & x : mEpochInfos)
        {
            single_epoch["num_of_frame"] = x.frame_num;
            single_epoch["length_second"] = x.length_second;
            single_epoch["traj_filename"] = x.traj_filename;
            single_epoch["train_data_filename"] = x.train_data_filename;
            root["single_trajs_lst"].append(single_epoch);
        }
        // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk append size = " << mEpochInfos.size() << std::endl;;
    }
    else
    {
        // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk begin to overwrite\n";
        root["char_file"] = mSampleCharFile;
        root["controller_file"] = mSampleControllerFile;
        root["num_of_trajs"] = mTotalEpochNum;
        root["total_second"] = mTotalLengthTime;
        root["total_frame"] = mTotalLengthFrame;
        root["timestamp"] = mTimeStamp;
        root["single_trajs_lst"] = Json::arrayValue;
        Json::Value single_epoch;
        for (auto & x : mEpochInfos)
        {
            single_epoch["num_of_frame"] = x.frame_num;
            single_epoch["length_second"] = x.length_second;
            single_epoch["traj_filename"] = x.traj_filename;
            single_epoch["train_data_filename"] = x.train_data_filename;
            root["single_trajs_lst"].append(single_epoch);
        }
        root["action_theta_dist_file"] = mActionThetaDistFile;
    }

    // std::cout <<"[debug] " << mpi_rank <<" WriteToDisk ready to write trajs num = " << root["single_trajs_lst"].size() << std::endl;;
    cJsonUtil::WriteJson(path, root, true);
    InfoPrintf(mLogger, "WriteToDisk %s", path.c_str());
    if(cFileUtil::DeleteLock(path) == false)
    {
        std::cout <<"[error] tSummaryTable::WriteToDisk delete lock failed for " << path << std::endl;
        exit(1);
    }
}

void cInteractiveIDSolver::tSummaryTable::LoadFromDisk(const std::string & path)
{
    // cFileUtil::AddLock(path);
    if(cFileUtil::ValidateFilePath(path) == false)
    {
        std::cout <<"[error] tSummaryTable::LoadFromDisk path invalid " << path << std::endl;
        exit(0);
    }
    
    if(cFileUtil::ExistsFile(path) == false && cFileUtil::ExistsFile(path + ".bak") == true)
    {
        InfoPrintf(mLogger, "LoadFromdDisk: path %s doesn't exist, but %s found, rename and use this backup file", path.c_str(), (path+".bak").c_str());
        cFileUtil::RenameFile(path + ".bak", path);
    }
    
    Json::Value root;
    cJsonUtil::LoadJson(path, root);

    // overview infos
    mSampleCharFile = root["char_file"].asString();
    mSampleControllerFile = root["controller_file"].asString();
    mTotalEpochNum = root["num_of_trajs"].asInt();
    mTotalLengthTime = root["total_second"].asDouble();
    mTotalLengthFrame = root["total_frame"].asInt();
    mTimeStamp = root["timestamp"].asString();
    mActionThetaDistFile = root["action_theta_dist_file"].asString();


    auto & trajs_lst = root["single_trajs_lst"];
    if(mTotalEpochNum != trajs_lst.size())
    {
        std::cout << "[warn] tSummaryTable::LoadFromDisk trajs num doesn't match " << mTotalEpochNum <<" " << trajs_lst.size() << ", correct it\n";
        mTotalEpochNum = trajs_lst.size();
    }

    // resize and load all trajs info
    mEpochInfos.resize(mTotalEpochNum);
    for(int i=0; i<mTotalEpochNum; i++)
    {
        mEpochInfos[i].frame_num = trajs_lst[i]["num_of_frame"].asInt();
        mEpochInfos[i].length_second = trajs_lst[i]["length_second"].asDouble();
        mEpochInfos[i].traj_filename = trajs_lst[i]["traj_filename"].asString();
    }
    mLogger->info("LoadFromDisk " + path);
}

cInteractiveIDSolver::tSummaryTable::tSingleEpochInfo::tSingleEpochInfo()
{
    frame_num = 0;
    length_second = 0;
    traj_filename = "";
    train_data_filename = "";
}

cInteractiveIDSolver::tSummaryTable::tSummaryTable()
{
    mSampleCharFile = "";
    mSampleControllerFile = "";
    mTimeStamp = "";
    mActionThetaDistFile = "";
    mTotalEpochNum = 0;
    mTotalLengthTime = 0;
    mTotalLengthFrame = 0;
    mEpochInfos.clear();
    mLogger = cLogUtil::CreateLogger("tSummaryTable");
}


/**
 * \brief               load full version of trajectories
*/
void cInteractiveIDSolver::LoadTrajV1(tLoadInfo & load_info, const std::string & path)
{
    auto & raw_pose = mSimChar->GetPose();
    Json::Value data_json, list_json;
    bool succ = cJsonUtil::LoadJson(path, data_json);
    load_info.mLoadPath = path;
    if(!succ)
    {
        ErrorPrintf(mLogger, "LoadTraj parse json %s failed", path);
        exit(0);
    } 
    

    list_json = data_json["list"];
    assert(list_json.isNull() == false);
    // std::cout <<"get list json, begin set up pramas\n";
    const int & target_version = cJsonUtil::ParseAsInt("version", data_json);

#ifndef IGNORE_TRAJ_VERISION
    if (1 != target_version)
    {
        mLogger->error("LoadTrajV1 is called but the spcified fileversion in {} is {}", path.c_str(), target_version);
        exit(1);
    }
#endif

    int num_of_frames = list_json.size();
    {
        load_info.mTotalFrame = num_of_frames;
        load_info.mPoseMat.resize(num_of_frames, mDof), load_info.mPoseMat.setZero();
        load_info.mVelMat.resize(num_of_frames, mDof), load_info.mVelMat.setZero();
        load_info.mAccelMat.resize(num_of_frames, mDof), load_info.mAccelMat.setZero();
        load_info.mActionMat.resize(num_of_frames, mCharController->GetActionSize()), load_info.mActionMat.setZero();
        load_info.mPDTargetMat.resize(num_of_frames, mCharController->GetActionSize()), load_info.mPDTargetMat.setZero();
        load_info.mContactForces.resize(num_of_frames); for(auto & x : load_info.mContactForces) x.clear();
        load_info.mLinkRot.resize(num_of_frames); for(auto & x : load_info.mLinkRot) x.resize(mNumLinks);
        load_info.mLinkPos.resize(num_of_frames); for(auto & x : load_info.mLinkPos) x.resize(mNumLinks);
        load_info.mExternalForces.resize(num_of_frames); for(auto & x : load_info.mExternalForces) x.resize(mNumLinks);
        load_info.mExternalTorques.resize(num_of_frames); for(auto & x : load_info.mExternalTorques) x.resize(mNumLinks);
        load_info.mTruthJointForces.resize(num_of_frames); for(auto & x : load_info.mTruthJointForces) x.resize(mNumLinks - 1);
        load_info.mTimesteps.resize(num_of_frames), load_info.mTimesteps.setZero();
        load_info.mRewards.resize(num_of_frames), load_info.mRewards.setZero();
        load_info.mMotionRefTime.resize(num_of_frames), load_info.mMotionRefTime.setZero();
    }

    for(int frame_id = 0; frame_id<num_of_frames; frame_id++)
    {
        auto & cur_frame = list_json[frame_id];
        auto & cur_pose = cur_frame["pos"];
        auto & cur_vel = cur_frame["vel"];
        auto & cur_accel = cur_frame["accel"];
        auto & cur_timestep = cur_frame["timestep"];
        auto & cur_ref_time = cur_frame["motion_ref_time"];
        auto & cur_reward = cur_frame["reward"];
        auto & cur_contact_num = cur_frame["contact_num"];
        auto & cur_contact_info = cur_frame["contact_info"];
        auto & cur_ext_force = cur_frame["external_force"];
        auto & cur_ext_torque = cur_frame["external_torque"];
        auto & cur_truth_joint_force = cur_frame["truth_joint_force"];
        auto & cur_truth_action = cur_frame["truth_action"];
        auto & cur_truth_pd_target = cur_frame["truth_pd_target"];
        assert(cur_pose.isNull() == false && cur_pose.size() == mDof);
        assert(cur_vel.isNull() == false); if(frame_id>=1) assert(cur_vel.size() == mDof);
        assert(cur_accel.isNull() == false);if(frame_id>=2) assert(cur_accel.size() == mDof);
        assert(cur_timestep.isNull() == false && cur_timestep.asDouble() > 0);
        assert(cur_ref_time.isNull() == false);
        assert(cur_contact_info.size() == cur_contact_num.asInt());
        assert(cur_truth_joint_force.isNull() == false && cur_truth_joint_force.size() == (mNumLinks-1) * 4);
        // std::cout <<"load pd target size = " << cur_truth_action.size() << std::endl;
        // std::cout <<"action space size = " << mCharController->GetActionSize() << std::endl;
        assert(cur_truth_action.isNull() == false && cur_truth_action.size() == mCharController->GetActionSize());
        assert(cur_truth_pd_target.isNull() == false && cur_truth_pd_target.size() == mCharController->GetActionSize());

        // 1. pos, vel, accel
        for(int j=0; j<mDof; j++) load_info.mPoseMat(frame_id, j) = cur_pose[j].asDouble();
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() << std::endl;
        SetGeneralizedPos(load_info.mPoseMat.row(frame_id));
        RecordMultibodyInfo(load_info.mLinkRot[frame_id], load_info.mLinkPos[frame_id]);
        for(int j=0; j<mDof && frame_id>=1; j++) load_info.mVelMat(frame_id, j) = cur_vel[j].asDouble();
        for(int j=0; j<mDof && frame_id>=1; j++) load_info.mAccelMat(frame_id, j) = cur_accel[j].asDouble();

        // 2. timestep and reward
        load_info.mTimesteps[frame_id] = cur_timestep.asDouble();
        load_info.mRewards[frame_id] = cur_reward.asDouble();
        load_info.mMotionRefTime[frame_id] = cur_ref_time.asDouble();

        // 3. contact info
        // load_info.mContactForces
        load_info.mContactForces[frame_id].resize(cur_contact_num.asInt());
        for(int c_id = 0; c_id < cur_contact_info.size(); c_id++)
        {
            for(int i=0; i<4; i++)
            {
                /*

           single_contact["force_pos"] = Json::arrayValue;
            for(int i=0; i<3; i++) single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for(int i=0; i<3; i++) single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
                */
                load_info.mContactForces[frame_id][c_id].mPos[i] = cur_contact_info[c_id]["force_pos"][i].asDouble();
                load_info.mContactForces[frame_id][c_id].mForce[i] = cur_contact_info[c_id]["force_value"][i].asDouble();
            }
            load_info.mContactForces[frame_id][c_id].mId = cur_contact_info[c_id]["force_link_id"].asInt();
            load_info.mContactForces[frame_id][c_id].mIsSelfCollision = cur_contact_info[c_id]["is_self_collision"].asBool();
        }

        // std::cout <<"[load file] frame " << frame_id <<" contact num = " << cur_contact_info.size() << std::endl;

        // 4. load external forces
        // load_info.mExternalForces[frame_id].resize(mNumLinks);
        // load_info.mExternalTorques[frame_id].resize(mNumLinks);
        for(int idx = 0; idx < mNumLinks; idx++)
        {
            // auto & cur_ext_force = ;
            // auto & cur_ext_torque = load_info.mExternalTorques[frame_id][idx];
            for(int i= 0; i< 4; i++)
            {
                load_info.mExternalForces[frame_id][idx][i] = cur_ext_force[idx * 4 + i].asDouble();
                load_info.mExternalTorques[frame_id][idx][i] = cur_ext_torque[idx * 4 + i].asDouble();
            }
            assert(load_info.mExternalForces[frame_id][idx].norm() < 1e-10);
            assert(load_info.mExternalTorques[frame_id][idx].norm() < 1e-10);
        }

        // 5. truth joint forces
        for(int idx = 0; idx < mNumLinks - 1; idx++)
        {
            for(int j=0; j<4; j++)
            load_info.mTruthJointForces[frame_id][idx][j] = cur_truth_joint_force[idx * 4 + j].asDouble();
        }

        // 6. truth actions and pd targes
        for(int idx = 0; idx < mCharController->GetActionSize(); idx++)
        {
            load_info.mActionMat(frame_id, idx) = cur_truth_action[idx].asDouble();
            load_info.mPDTargetMat(frame_id, idx) = cur_truth_pd_target[idx].asDouble();
        }

    }
#ifdef VERBOSE
    std::cout <<"[debug] cInteractiveIDSolver::LoadTraj " << path <<", number of frames = " << num_of_frames << std::endl;
#endif
    assert(num_of_frames > 0);
    mSimChar->SetPose(raw_pose);
}

void cInteractiveIDSolver::LoadTrajV2(tLoadInfo & load_info, const std::string & path)
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

    // and we also need to restore pose, vel, accel, link rot, link pos, 
    auto & raw_pose = mSimChar->GetPose();
    Json::Value data_json, list_json;
    bool succ = cJsonUtil::LoadJson(path, data_json);
    load_info.mLoadPath = path;
    if(!succ)
    {
        ErrorPrintf(mLogger, "LoadTraj parse json %s failed", path);
        exit(0);
    } 
    

    list_json = data_json["list"];
    assert(list_json.isNull() == false);
    // std::cout <<"get list json, begin set up pramas\n";
    
#ifndef IGNORE_TRAJ_VERISION
    if (2 != cJsonUtil::ParseAsInt("version", data_json))
    {
        mLogger->error("LoadTrajV2 is called but the spcified fileversion in {} is {}", path, cJsonUtil::ParseAsInt("version", data_json));
        exit(1);
    }
#endif

    int num_of_frames = list_json.size();
    int pose_size = mSimChar->GetPose().size();
    {
        load_info.mTotalFrame = num_of_frames;
        load_info.mPoseMat.resize(num_of_frames, mDof), load_info.mPoseMat.setZero();
        load_info.mVelMat.resize(num_of_frames, mDof), load_info.mVelMat.setZero();
        load_info.mAccelMat.resize(num_of_frames, mDof), load_info.mAccelMat.setZero();
        load_info.mCharPoseMat.resize(num_of_frames, pose_size), load_info.mCharPoseMat.setZero();
        load_info.mActionMat.resize(num_of_frames, mCharController->GetActionSize()), load_info.mActionMat.setZero();
        load_info.mPDTargetMat.resize(num_of_frames, mCharController->GetActionSize()), load_info.mPDTargetMat.setZero();
        load_info.mContactForces.resize(num_of_frames); for(auto & x : load_info.mContactForces) x.clear();
        load_info.mLinkRot.resize(num_of_frames); for(auto & x : load_info.mLinkRot) x.resize(mNumLinks);
        load_info.mLinkPos.resize(num_of_frames); for(auto & x : load_info.mLinkPos) x.resize(mNumLinks);
        load_info.mExternalForces.resize(num_of_frames); for(auto & x : load_info.mExternalForces) {x.resize(mNumLinks); for(auto & j :x) j.setZero();};
        load_info.mExternalTorques.resize(num_of_frames); for(auto & x : load_info.mExternalTorques) {x.resize(mNumLinks); for(auto & j :x) j.setZero();};
        load_info.mTruthJointForces.resize(num_of_frames); for(auto & x : load_info.mTruthJointForces) {x.resize(mNumLinks - 1); for(auto & j :x) j.setZero();};
        load_info.mTimesteps.resize(num_of_frames), load_info.mTimesteps.setZero();
        load_info.mRewards.resize(num_of_frames), load_info.mRewards.setZero();
        load_info.mMotionRefTime.resize(num_of_frames), load_info.mMotionRefTime.setZero();
    }
    
    // begin to load pose, timestep, motion ref time, contact infos, rewards
    tVectorXd buffer_pos, buffer_vel;
    
    for(int frame_id = 0; frame_id<num_of_frames; frame_id++)
    {
        // std::cout <<"frame " << frame_id << std::endl;
        const Json::Value & cur_frame = list_json[frame_id];
        const Json::Value & cur_char_pose = cJsonUtil::ParseAsValue("char_pose", cur_frame);
        const Json::Value & cur_contact_info = cJsonUtil::ParseAsValue("contact_info", cur_frame);
        const Json::Value & cur_truth_action = cJsonUtil::ParseAsValue("truth_action", cur_frame);
        const int cur_contact_num = cJsonUtil::ParseAsInt("contact_num", cur_frame);

        // 1. restore 0 order info pos from char pose
        if(cur_char_pose.isNull() == true || cur_char_pose.size() != pose_size)
        {
            mLogger->error("LoadTrajV2: char pose shape is empty or invalid {} != {}", cur_char_pose.size(), pose_size); 
            exit(0);
        }
        if(cur_contact_info.isNull() == true || cur_contact_info.size() != cur_contact_num)
        {
            mLogger->error("LoadTrajV2: contact info is empty or invalid {} != {}", cur_contact_info.size(), cur_contact_num); 
            exit(0);
        }

        for(int j=0; j<pose_size; j++) load_info.mCharPoseMat(frame_id, j) = cur_char_pose[j].asDouble();
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() << std::endl;
        mSimChar->SetPose(load_info.mCharPoseMat.row(frame_id));
        RecordGeneralizedInfo(buffer_pos, buffer_vel);
        load_info.mPoseMat.row(frame_id) = buffer_pos;
        RecordMultibodyInfo(load_info.mLinkRot[frame_id], load_info.mLinkPos[frame_id]);

        // 2. timestep, reward and ref time
        load_info.mTimesteps[frame_id] = cJsonUtil::ParseAsDouble("timestep", cur_frame);
        load_info.mRewards[frame_id] = cJsonUtil::ParseAsDouble("reward", cur_frame);
        load_info.mMotionRefTime[frame_id] = cJsonUtil::ParseAsDouble("motion_ref_time", cur_frame);

        // 3. contact info
        load_info.mContactForces[frame_id].resize(cur_contact_num);
        for(int c_id = 0; c_id < cur_contact_info.size(); c_id++)
        {
            for(int i=0; i<4; i++)
            {
                load_info.mContactForces[frame_id][c_id].mPos[i] = cur_contact_info[c_id]["force_pos"][i].asDouble();
                load_info.mContactForces[frame_id][c_id].mForce[i] = cur_contact_info[c_id]["force_value"][i].asDouble();
            }
            load_info.mContactForces[frame_id][c_id].mId = cur_contact_info[c_id]["force_link_id"].asInt();
            load_info.mContactForces[frame_id][c_id].mIsSelfCollision = cur_contact_info[c_id]["is_self_collision"].asBool();
        }

        // 4. actions
        for(int idx = 0; idx < mCharController->GetActionSize(); idx++)
        {
            load_info.mActionMat(frame_id, idx) = cur_truth_action[idx].asDouble();
        }

    }

#ifdef VERBOSE
    std::cout <<"[debug] cInteractiveIDSolver::LoadTraj " << path <<", number of frames = " << num_of_frames << std::endl;
#endif
    assert(num_of_frames > 0);
    mSimChar->SetPose(raw_pose);
}