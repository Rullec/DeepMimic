#include "OfflineIDSolver.h"
#include "anim/KinCharacter.h"
#include "sim/TrajManager/Trajectory.h"
#include "util/BulletUtil.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include "util/MPIUtil.h"
#include "util/TimeUtil.hpp"
cOfflineIDSolver::cOfflineIDSolver(cSceneImitate *imi,
                                   const std::string &config)
    : cIDSolver(imi, eIDSolverType::OfflineSolve)
{
    // controller_details_path =
    //     "logs/controller_logs/controller_details_offlinesolve.txt";
    // gRewardInfopath = "reward_info_solve.txt";
    mEnableActionVerfied = true;
    mEnableRewardRecalc = true;
    mEnableDiscreteVerified = false;
    mEnableTorqueVerified = false;
    // mEnableRestoreThetaByActionDist = false;
    // mEnableRestoreThetaByGT = true;
    ParseConfig(config);
}

void cOfflineIDSolver::ParseConfig(const std::string &conf)
{
    Json::Value root_;
    if (false == cJsonUtil::LoadJson(conf, root_))
    {
        MIMIC_ERROR("ParseConfig {} failed", conf);
        exit(1);
    }
    Json::Value root = root_["SolveModeInfo"];

    mRefMotionPath = cJsonUtil::ParseAsString("ref_motion_path", root);
    mRetargetCharPath = cJsonUtil::ParseAsString("retargeted_char_path", root);
    mEnableRewardRecalc = cJsonUtil::ParseAsBool("enable_reward_recal", root);
    mEnableActionVerfied =
        cJsonUtil::ParseAsBool("enable_action_verified", root);
    mEnableTorqueVerified =
        cJsonUtil::ParseAsBool("enable_torque_verified", root);
    mEnableDiscreteVerified =
        cJsonUtil::ParseAsBool("enable_discrete_verified", root);
    // mEnableRestoreThetaByActionDist =
    //     cJsonUtil::ParseAsBool("enable_restore_theta_by_action_dist", root);
    // mEnableRestoreThetaByGT =
    //     cJsonUtil::ParseAsBool("enable_restore_theta_by_ground_truth", root);

    // if (mEnableRestoreThetaByGT == mEnableRestoreThetaByActionDist)
    // {
    //     MIMIC_ERROR("Please select a restoration policy between GT {} "
    //                 "and ActionDsit {}",
    //                 mEnableRestoreThetaByGT,
    //                 mEnableRestoreThetaByActionDist);
    //     exit(1);
    // }

    // 2. load solving mode
    mSolveMode = ParseSolvemode(cJsonUtil::ParseAsString("solve_mode", root));
    switch (mSolveMode)
    {
    case eSolveMode::SingleTrajSolveMode:
        ParseSingleTrajConfig(root["SingleTrajSolveInfo"]);
        break;
    case eSolveMode::BatchTrajSolveMode:
        ParseBatchTrajConfig(root["BatchTrajSolveInfo"]);
        break;
    default:
        exit(0);
        break;
    }

    // verify that the retarget char path is the same as the simchar skeleton
    // path
    if (mRetargetCharPath != mSimChar->GetCharFilename())
    {
        MIMIC_ERROR("retarget path {} != loaded simchar path {}",
                    mRetargetCharPath.c_str(),
                    mSimChar->GetCharFilename().c_str());
        exit(0);
    }

    // verify that the ref motion is the same as kinchar motion
    if (mRefMotionPath != mKinChar->GetMotion().GetMotionFile())
    {
        MIMIC_ERROR("ID ref motion path {} != loaded motion path {}",
                    mRefMotionPath.c_str(),
                    mKinChar->GetMotion().GetMotionFile().c_str());
        exit(0);
    }
}

void cOfflineIDSolver::ParseSingleTrajConfig(
    const Json::Value &single_traj_config)
{
    assert(single_traj_config.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSolve(const Json::Value &
    // save_value)\n";
    const Json::Value &solve_traj_path = single_traj_config["solve_traj_path"];
    assert(solve_traj_path.isNull() == false);
    mSingleTrajSolveConfig.mSolveTrajPath = solve_traj_path.asString();
    /*
    "export_train_data_path_meaning" :
    "训练数据的输出路径，后缀名为.train，里面存放了state, action,
    reward三个键值", "export_train_data_path" :
    "data/batch_train_data/0424/leftleg_0.train", "ref_motion_meaning" :
    "重新计算reward所使用到的motion", "ref_motion" :
    "data/0424/motions/walk_motion_042401_leftleg.txt"
*/
    const Json::Value &export_train_data_path =
        single_traj_config["export_train_data_path"];
    assert(export_train_data_path.isNull() == false);

    mSingleTrajSolveConfig.mExportDataPath = export_train_data_path.asString();

    if (false ==
        cFileUtil::ValidateFilePath(mSingleTrajSolveConfig.mExportDataPath))
    {
        MIMIC_ERROR("ParseSingleTrajConfig export train data path illegal: {}",
                    mSingleTrajSolveConfig.mExportDataPath);
        exit(0);
    }
    MIMIC_INFO("working in SingleTrajSolve mode");
}

void cOfflineIDSolver::ParseBatchTrajConfig(
    const Json::Value &batch_traj_config)
{
    assert(batch_traj_config.isNull() == false);
    mBatchTrajSolveConfig.mOriSummaryTableFile =
        cJsonUtil::ParseAsString("summary_table_filename", batch_traj_config);
    mBatchTrajSolveConfig.mExportDataDir =
        cJsonUtil::ParseAsString("export_train_data_dir", batch_traj_config);
    mBatchTrajSolveConfig.mDestSummaryTableFile = cFileUtil::ConcatFilename(
        mBatchTrajSolveConfig.mExportDataDir,
        cFileUtil::GetFilename(mBatchTrajSolveConfig.mOriSummaryTableFile));
    mBatchTrajSolveConfig.mSolveTarget = ParseSolveTargetInBatchMode(
        cJsonUtil::ParseAsString("solve_target", batch_traj_config));

    if (cFileUtil::ExistsDir(mBatchTrajSolveConfig.mExportDataDir) == false)
    {
        MIMIC_WARN("Train datａ aoutput folder {} doesn't exist, created.",
                   mBatchTrajSolveConfig.mExportDataDir);
        cFileUtil::CreateDir(mBatchTrajSolveConfig.mExportDataDir.c_str());
    }
    MIMIC_INFO("working in BatchTrajSolve mode");
}

cOfflineIDSolver::eSolveTarget cOfflineIDSolver::ParseSolveTargetInBatchMode(
    const std::string &solve_target_str) const
{
    eSolveTarget mSolveTarget = eSolveTarget::INVALID_SOLVETARGET;
    for (int i = 0; i < eSolveTarget::SolveTargetNum; i++)
    {
        if (solve_target_str == SolveTargetstr[i])
        {
            mSolveTarget = static_cast<eSolveTarget>(i);
        }
    }
    if (eSolveTarget::INVALID_SOLVETARGET == mSolveTarget)
    {
        MIMIC_ERROR("Invalid Solve Target : {}", solve_target_str);
        exit(0);
    }
    // std::cout << solve_target_str << std::endl;
    return mSolveTarget;
}

cOfflineIDSolver::eSolveMode
cOfflineIDSolver::ParseSolvemode(const std::string &name) const
{
    eSolveMode mSolveMode = eSolveMode::INVALID_SOLVEMODE;
    for (int i = 0; i < eSolveMode::SolveModeNum; i++)
    {
        if (SolveModeStr[i] == name)
        {
            mSolveMode = static_cast<eSolveMode>(i);
            break;
        }
    }
    if (mSolveMode == eSolveMode::INVALID_SOLVEMODE)
    {
        MIMIC_ERROR("parse solve mode failed {}", name);
        exit(0);
    }
    return mSolveMode;
}

/**
 * \brief                   Save Train Data "*.train"
 * \param dir               storaged directory
 * \param info              a info struct for what we need to save.
 */
void cOfflineIDSolver::SaveTrainData(
    const std::string &dir, const std::string &filename,
    const std::vector<tSingleFrameIDResult> &info) const
{
    if (cFileUtil::ExistsDir(dir) == false)
    {
        MIMIC_ERROR("SaveTrainData target dir {} doesn't exist", dir);
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
    MIMIC_INFO("SaveTrainData to {}" path);
#endif
}

/**
 * \brief               Load batch summary table info in MPI environment
 * used before the BatchTrajSolve
 */
void cOfflineIDSolver::LoadBatchInfoMPI(
    const std::string &path, std::vector<int> &solved_traj_ids,
    std::vector<std::string> &solved_traj_names)
{
    cMPIUtil::InitMPI();

    // mpi_rank = world_rank;

    // 2. load the json: ensure that all process can load the summary table
    // correctly.
    cFileUtil::AddLock(path);
    mSummaryTable.LoadFromDisk(path);
    mSummaryTable.mIDTraindataDir = mBatchTrajSolveConfig.mExportDataDir;
    cFileUtil::DeleteLock(path);
    cMPIUtil::SetBarrier();

    // 2.1 load action distribution from files
    // if (mEnableRestoreThetaByActionDist == true)
    // {
    //     InitActionThetaDist(mSimChar, mActionThetaDist);
    //     LoadActionThetaDist(mSummaryTable.mActionThetaDistFile,
    //                         mActionThetaDist);
    // }

    // 3. rename this summary table: after that here is a MPI_Barrier, which
    // ensures that our process will not delete other processes' result.
    cFileUtil::AddLock(path);
    if (cFileUtil::ExistsFile(path))
        cFileUtil::RenameFile(path, path + ".bak");
    if (cFileUtil::ExistsDir(mBatchTrajSolveConfig.mExportDataDir))
        cFileUtil::ClearDir(mBatchTrajSolveConfig.mExportDataDir.c_str());
    cFileUtil::DeleteLock(path);
    cMPIUtil::SetBarrier();

    // 4. select all wait-to-solved trajectory
    solved_traj_ids.clear();
    solved_traj_names.clear();

    int total_traj_num = mSummaryTable.mEpochInfos.size();
    int world_size = cMPIUtil::GetCommSize(),
        world_rank = cMPIUtil::GetWorldRank();
    for (int i = world_rank, id = 0; i < total_traj_num; i += world_size, id++)
    {
        std::string target_traj_filename_full = "";
        switch (mBatchTrajSolveConfig.mSolveTarget)
        {
        case eSolveTarget::MRedTraj:
            target_traj_filename_full = cFileUtil::ConcatFilename(
                mSummaryTable.mMrTrajDir,
                mSummaryTable.mEpochInfos[i].mr_traj_filename);
            break;
        case eSolveTarget::SampledTraj:
            target_traj_filename_full = cFileUtil::ConcatFilename(
                mSummaryTable.mSampleTrajDir,
                mSummaryTable.mEpochInfos[i].sample_traj_filename);
            break;
        default:
            MIMIC_ASSERT(mBatchTrajSolveConfig.mSolveTarget !=
                         eSolveTarget::INVALID_SOLVETARGET);
            break;
        }
        solved_traj_ids.push_back(i);
        solved_traj_names.push_back(target_traj_filename_full);
        if (cFileUtil::ExistsFile(target_traj_filename_full) == false)
        {
            MIMIC_ERROR("BatchTrajsSolve: the traj {} to be solved "
                        "does not exist",
                        target_traj_filename_full);
            exit(1);
        }
    }
}

void cOfflineIDSolver::AddBatchInfoMPI(
    int global_traj_id, const std::string target_traj_filename_full,
    const std::vector<tSingleFrameIDResult> &mResult,
    const std::vector<tSummaryTable::tSingleEpochInfo> &old_epoch_info,
    double total_time)
{
    tSummaryTable::tSingleEpochInfo single_epoch_info;
    std::string export_name = cFileUtil::RemoveExtension(cFileUtil::GetFilename(
                                  target_traj_filename_full)) +
                              ".train";
    cFileUtil::AddLock(export_name);
    SaveTrainData(mBatchTrajSolveConfig.mExportDataDir, export_name, mResult);
    MIMIC_INFO("Save traindata to {}",
               cFileUtil::ConcatFilename(
                   mBatchTrajSolveConfig.mExportDataDir,
                   cFileUtil::RemoveExtension(export_name) + ".train"));
    cFileUtil::DeleteLock(export_name);

    single_epoch_info.frame_num = mResult.size();
    single_epoch_info.length_second = total_time;
    // mLoadInfo.mTotalFrame * mLoadInfo.mTimesteps[1];
    single_epoch_info.sample_traj_filename =
        old_epoch_info[global_traj_id].sample_traj_filename;
    single_epoch_info.train_filename = export_name;
    single_epoch_info.mr_traj_filename =
        old_epoch_info[global_traj_id].mr_traj_filename;
    mSummaryTable.mEpochInfos.push_back(single_epoch_info);
    mSummaryTable.mTotalEpochNum += 1;
}