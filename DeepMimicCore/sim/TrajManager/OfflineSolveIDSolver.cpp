#include "OfflineSolveIDSolver.h"
#include "anim/KinCharacter.h"
#include "anim/Motion.h"
#include "scenes/SceneImitate.h"
#include "sim/Controller/CtPDFeaController.h"
#include "sim/SimItems/SimCharacter.h"
#include "util/BulletUtil.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include "util/MPIUtil.h"
#include "util/TimeUtil.hpp"
#include <iostream>

// extern std::string controller_details_path;
// extern std::string gRewardInfopath;
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

cOfflineIDSolver::~cOfflineIDSolver()
{
    cLogUtil::DropLogger("OfflineIDSolver");
}

void cOfflineIDSolver::PreSim()
{
    cTimeUtil::Begin("ID Solving");
    if (mSolveMode == eSolveMode::SingleTrajSolveMode)
    {
        std::vector<tSingleFrameIDResult> mResult;
        MIMIC_INFO("begin to solve traj {}",
                   mSingleTrajSolveConfig.mSolveTrajPath);
        mLoadInfo.LoadTraj(mSimChar, mSingleTrajSolveConfig.mSolveTrajPath);
        SingleTrajSolve(mResult);
        std::string export_dir =
            cFileUtil::GetDir(mSingleTrajSolveConfig.mExportDataPath);
        std::string export_name =
            cFileUtil::GetFilename(mSingleTrajSolveConfig.mExportDataPath);
        SaveTrainData(export_dir, export_name, mResult);
        MIMIC_INFO("save train data to {}",
                   mSingleTrajSolveConfig.mExportDataPath);
    }
    else if (mSolveMode == eSolveMode::BatchTrajSolveMode)
    {
        MIMIC_INFO("Batch solve, summary table = {}",
                   mBatchTrajSolveConfig.mOriSummaryTableFile);
        BatchTrajsSolve(mBatchTrajSolveConfig.mOriSummaryTableFile);
    }
    else
    {
        MIMIC_ERROR("PreSim invalid mode {}", mSolveMode);
        exit(0);
    }
    cTimeUtil::End("ID Solving");
    exit(0);
}

void cOfflineIDSolver::PostSim() {}

void cOfflineIDSolver::Reset()
{
    std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
    exit(0);
}

void cOfflineIDSolver::SetTimestep(double) {}

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

void cOfflineIDSolver::SingleTrajSolve(
    std::vector<tSingleFrameIDResult> &IDResults)
{
    // cTimeUtil::Begin("OfflineSolve");
    assert(mLoadInfo.mTotalFrame > 0);
    IDResults.resize(mLoadInfo.mTotalFrame);
    // std::cout <<"[debug] cOfflineIDSolver::OfflineSolve: motion total frame =
    // " << mLoadInfo.mTotalFrame << std::endl;
    tVectorXd old_q, old_u;
    RecordGeneralizedInfo(mSimChar, old_q, old_u);

    mCharController->SetInitTime(mLoadInfo.mMotionRefTime[0] -
                                 mLoadInfo.mTimesteps[0]);
    // 1. calc vel and accel from pos
    /*
    double cur_timestep = mSaveInfo.mTimesteps[cur_frame - 1],
            last_timestep = mSaveInfo.mTimesteps[cur_frame - 2];
                        tVectorXd old_vel_after =
    mSaveInfo.mBuffer_u[cur_frame]; tVectorXd old_vel_before =
    mSaveInfo.mBuffer_u[cur_frame - 1]; tVectorXd old_accel = (old_vel_after -
    old_vel_before) / cur_timestep; mSaveInfo.mBuffer_u[cur_frame - 1] =
    CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 2],
    mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
                        mSaveInfo.mBuffer_u[cur_frame] =
    CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1],
    mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
                        mSaveInfo.mBuffer_u_dot[cur_frame - 1] =
    (mSaveInfo.mBuffer_u[cur_frame] - mSaveInfo.mBuffer_u[cur_frame - 1]) /
    cur_timestep;
    */
    for (int frame_id = 0; frame_id < mLoadInfo.mTotalFrame - 1; frame_id++)
    {
        tVectorXd cur_vel =
            CalcGeneralizedVel(mLoadInfo.mPosMat.row(frame_id),
                               mLoadInfo.mPosMat.row(frame_id + 1),
                               mLoadInfo.mTimesteps[frame_id]);
        // std::cout <<"cur vel size = " << cur_vel.size() << std::endl;
        // std::cout <<"com vel size = " << mLoadInfo.mVelMat.row(frame_id +
        // 1).size() << std::endl; std::cout <<" cur vel = " <<
        // cur_vel.transpose() << std::endl; std::cout <<"vel comp = " <<
        // mLoadInfo.mVelMat.row(frame_id + 1) << std::endl;
        if (mEnableDiscreteVerified == true)
        {
            tVectorXd diff =
                mLoadInfo.mVelMat.row(frame_id + 1).transpose() - cur_vel;
            assert(diff.norm() < 1e-10);
        }

        mLoadInfo.mVelMat.row(frame_id + 1) = cur_vel;
        // std::cout <<"vel = " << cur_vel.transpose() << std::endl;
        // std::cout <<"frame id " << frame_id << " vel diff = " <<
        // diff.norm() << std::endl;
    }

    for (int frame_id = 1; frame_id < mLoadInfo.mTotalFrame - 1; frame_id++)
    {
        tVectorXd cur_accel = (mLoadInfo.mVelMat.row(frame_id + 1) -
                               mLoadInfo.mVelMat.row(frame_id)) /
                              mLoadInfo.mTimesteps[frame_id];
        // std::cout <<"cur accel size = " << cur_accel.size() << std::endl;
        // std::cout <<"com accel size = " <<
        // mLoadInfo.mAccelMat.row(frame_id).size() << std::endl; std::cout
        // <<"cur accel = " << cur_accel.transpose() << std::endl; std::cout
        // <<"accel comp = " << mLoadInfo.mAccelMat.row(frame_id) <<
        // std::endl;
        if (mEnableDiscreteVerified == true)
        {
            tVectorXd diff =
                mLoadInfo.mAccelMat.row(frame_id).transpose() - cur_accel;
            assert(diff.norm() < 1e-7);
        }

        mLoadInfo.mAccelMat.row(frame_id) = cur_accel;

        // std::cout <<"frame id " << frame_id << " accel diff = " <<
        // diff.norm() << std::endl;
    }

    // 2. calculate link pos and link rot from generalized info
    for (int frame_id = 0; frame_id < mLoadInfo.mTotalFrame; frame_id++)
    {
        SetGeneralizedPos(mSimChar, mLoadInfo.mPosMat.row(frame_id));
        RecordMultibodyInfo(mSimChar, mLoadInfo.mLinkRot[frame_id],
                            mLoadInfo.mLinkPos[frame_id]);
    }
    // exit(0);

    // 3. Init the status for mKinChar and mSimChar according to mLoadInfo
    // ATTENTION: most of these codes are used to gurantee the validity of
    // reward we get from the following Inverse Dynamics procedure.
    mKinChar->SetTime(mLoadInfo.mMotionRefTime[0] -
                      mLoadInfo.mTimesteps[0]); // training policy: random init
    mKinChar->Update(0); // update mKinChar inner status by Update(0)
    SetGeneralizedPos(
        mSimChar,
        mLoadInfo.mPosMat.row(0)); // set up the sim char pos from mLoadInfo
    mSimChar->PostUpdate(0);
    mScene
        ->ResolveCharGroundIntersectInverseDynamic(); // Resolve intersection
                                                      // between char and the
                                                      // ground. Sync to KinChar
                                                      // is also included.
    mKinChar->Update(mLoadInfo.mTimesteps[0]);        // Go for another timestep
    mCharController->Update(mLoadInfo.mTimesteps[0]);

    // 4. solve ID for each frame
    double ID_torque_err = 0, ID_action_err = 0, reward_err = 0;
    tVectorXd torque = tVectorXd::Zero(mSimChar->GetPose().size()),
              pd_target = tVectorXd::Zero(mSimChar->GetPose().size());

    for (int cur_frame = 1; cur_frame < mLoadInfo.mTotalFrame; cur_frame++)
    {
        auto &cur_ID_res = IDResults[cur_frame];

        // 4.1 update the sim char
        ClearID();
        SetGeneralizedPos(mSimChar, mLoadInfo.mPosMat.row(cur_frame));
        SetGeneralizedVelFea(mLoadInfo.mVelMat.row(cur_frame));
        mSimChar->PostUpdate(0);

        // 4.2 record state at this moment
        mCharController->RecordState(cur_ID_res.state);

        // 4.3 solve Inverse Dynamic for joint torques
        tEigenArr<tVector> result;
        // SetGeneralizedInfo(mLoadInfo.mPosMat.row(frame_id));
        // std::cout <<"log frame = " << cur_frame << std::endl;
        cIDSolver::SolveIDSingleStep(
            result, mLoadInfo.mContactForces[cur_frame],
            mLoadInfo.mLinkPos[cur_frame], mLoadInfo.mLinkRot[cur_frame],
            mLoadInfo.mPosMat.row(cur_frame), mLoadInfo.mVelMat.row(cur_frame),
            mLoadInfo.mAccelMat.row(cur_frame), cur_frame,
            mLoadInfo.mExternalForces[cur_frame],
            mLoadInfo.mExternalTorques[cur_frame]);

        ID_torque_err = cIDSolver::CalcAssembleJointForces(
            result, torque, mLoadInfo.mTruthJointForces[cur_frame]);
        // 4.4 convert the result joint torques into PD Target
        // the vector "torque" has the same shape as pose and vel
        /*
            root joint: occupy the first 7 DOF in the vector, all are set to
           zero revolute: occupy 1 DOF in the vector, the value of torque
            spherical: occupy 4 DOF in the vector, [torque_x, torque_y,
           torque_z, 0] fixed: No sapce
        */

        double timestep = mLoadInfo.mTimesteps[cur_frame];
        mCharController->CalcPDTargetByTorque(timestep, mSimChar->GetPose(),
                                              mSimChar->GetVel(), torque,
                                              pd_target);

        // std::cout << "load pd = " <<
        // mLoadInfo.mPDTargetMat.row(cur_frame) << std::endl; std::cout <<
        // "solved pd = " << pd_target.transpose() << std::endl; exit(1);

        // 4.5 convert PD target to Neural Network action
        // action is different from fetched pd_target!
        // For ball joints, their action is represented in axis angle; but
        // their pd_Target is quaternion we still need to have a convert
        // here. pd_target = [x, y, z, w], axis angle = [angle, ax, ay, az]
        tVectorXd action = pd_target;
        mCharController->CalcActionByTargetPose(action);

        // 4.6 sometimes, the action of spherical joints can be zero, which
        // doesn't make sense. We will given thess zero action the same
        // value as its previous one
        if (cur_frame >= 2)
            PostProcessAction(action, IDResults[cur_frame - 1].action);

        // 5. verified the ID result action if possible
        // the loaded action hasn't been normalized, we need to preprocess
        // it before comparing...
        if (true == mEnableActionVerfied)
        {
            tVectorXd truth_action = mLoadInfo.mActionMat.row(cur_frame);
            double total_action_err = 0, single_action_err = 0;
            assert(truth_action.size() == mCharController->GetActionSize());
            total_action_err = cIDSolver::CalcActionError(action, truth_action);
            if (total_action_err > 1e-4)
            {
                MIMIC_ERROR("SingleTrajSolve {} frame {} action err = %.3f",
                            mLoadInfo.mLoadPath.c_str(), cur_frame,
                            total_action_err);
            }
            ID_action_err += total_action_err;
        }
        cur_ID_res.action = action;

        double prev_phase = mKinChar->GetPhase();
        if (true == mEnableRewardRecalc)
        {
            // 4.9 recalculate the reward according to current motion
            // you must confirm that the simchar skeleton file is the
            // same as trajectories skeleton file accordly (Now it has
            // been guranteed in ParseConfig)
            cur_ID_res.reward = mScene->CalcReward(0);
        }

        // update kinchar and mcharcontroller to maintain the recorded state
        // phase is correct
        mKinChar->Update(mLoadInfo.mTimesteps[cur_frame]);
        mCharController->UpdateTimeOnly(mLoadInfo.mTimesteps[cur_frame]);

        // recalcualte the reward?
        if (true == mEnableRewardRecalc)
        {
            double curr_phase = mKinChar->GetPhase();
            // 4.10 judging whether we should jump to the next cycle and
            // update/sync the kinChar according to the simchar.
            if (curr_phase < prev_phase)
            {
                (dynamic_cast<cSceneImitate *>(mScene))
                    ->SyncKinCharNewCycleInverseDynamic(*mSimChar, *mKinChar);
            }

            reward_err +=
                std::fabs(cur_ID_res.reward - mLoadInfo.mRewards[cur_frame]);
            // std::cout <<"frame " << cur_frame <<" truth action = " <<
            // mLoadInfo.mActionMat.row(cur_frame) << std::endl;
            // std::cout <<"frame " << cur_frame <<" solved action = "
            // << action.transpose() << std::endl; std::cout <<"frame "
            // << cur_frame <<" cur reward = " << cur_ID_res.reward <<
            // ", load reward = " << mLoadInfo.mRewards[cur_frame] <<
            // std::endl; std::cout <<"frame " << cur_frame <<" cur
            // reward = " << cur_ID_res.reward << ", load reward = " <<
            // mLoadInfo.mRewards[cur_frame] <<  std::endl;
        }
        else
        {
            cur_ID_res.reward = mLoadInfo.mRewards[cur_frame];
        }
    }
    if (ID_torque_err < 1e-3 && ID_action_err < 1e-3 && reward_err < 1e-3)
    {
        MIMIC_INFO(
            "SingleTrajSolve ID succ {}, total ID torque err = {}, total "
            "ID Action error = {}, total ID reward error = {}",
            mLoadInfo.mLoadPath, ID_torque_err, ID_action_err, reward_err);
    }
    else
    {
        MIMIC_ERROR(
            "SingleTrajSolve ID failed {}, total ID torque err = {}, total "
            "ID Action error = {}, total ID reward error = {}",
            mLoadInfo.mLoadPath, ID_torque_err, ID_action_err, reward_err);
    }

    // post process

    // if (mEnableRestoreThetaByActionDist == true)
    //     RestoreActionByThetaDist(IDResults);
    // if (mEnableRestoreThetaByGT == true)
    //     RestoreActionByGroundTruth(IDResults);
    // cTimeUtil::End("OfflineSolve");
}

/**
 * \brief       Given A summary files, this function will solve their ID
 * very quick under the help of MPI
 */
void cOfflineIDSolver::BatchTrajsSolve(const std::string &path)
{
    // 1. MPI init
    cMPIUtil::InitMPI();
    int world_size = cMPIUtil::GetCommSize();

    int world_rank = cMPIUtil::GetWorldRank();
    // mpi_rank = world_rank;

    // 2. load the json: ensure that all process can load the summary table
    // correctly.
    cFileUtil::AddLock(path);
    mSummaryTable.LoadFromDisk(path);
    auto &summary_table = mSummaryTable;
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

    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
    // exit(0);
    // 4. now remember all info in summary_table and clear them at all
    int total_traj_num = summary_table.mEpochInfos.size();
    int my_own_task_num = 0;
    auto full_epoch_infos = summary_table.mEpochInfos;
    summary_table.mEpochInfos.clear();
    summary_table.mTotalEpochNum = 0;
    // summary_table.mTotalLengthFrame = 0;
    // summary_table.mTotalLengthTime = 0;

    std::vector<tSingleFrameIDResult> mResult(0);
    tSummaryTable::tSingleEpochInfo single_epoch_info;
    for (int i = world_rank; i < total_traj_num; i += world_size)
        my_own_task_num++;
    // std::cout <<"total traj num = " << total_traj_num <<", my own task
    // num = " << my_own_task_num << std::endl;

    for (int i = world_rank, id = 0; i < total_traj_num; i += world_size, id++)
    {
        std::string target_traj_filename_full = "";

        switch (mBatchTrajSolveConfig.mSolveTarget)
        {
        case eSolveTarget::MRedTraj:
            target_traj_filename_full = cFileUtil::ConcatFilename(
                summary_table.mMrTrajDir, full_epoch_infos[i].mr_traj_filename);
            break;
        case eSolveTarget::SampledTraj:
            target_traj_filename_full = cFileUtil::ConcatFilename(
                summary_table.mSampleTrajDir,
                full_epoch_infos[i].sample_traj_filename);
            break;
        default:
            MIMIC_ASSERT(mBatchTrajSolveConfig.mSolveTarget !=
                         eSolveTarget::INVALID_SOLVETARGET);
            break;
        }
        if (cFileUtil::ExistsFile(target_traj_filename_full) == false)
        {
            MIMIC_ERROR("BatchTrajsSolve: the traj {} to be solved "
                        "does not exist",
                        target_traj_filename_full);
            exit(1);
        }
        // 4.1 load a single traj and solve ID for it.
        // const std::string & sample_traj_filename_full =
        // cFileUtil::ConcatFilename(summary_table.mSampleTrajDir,
        // full_epoch_infos[i].sample_traj_filename);
        mLoadInfo.LoadTraj(mSimChar, target_traj_filename_full);
        SingleTrajSolve(mResult);

        std::string export_name =
            cFileUtil::RemoveExtension(
                cFileUtil::GetFilename(target_traj_filename_full)) +
            ".train";
        cFileUtil::AddLock(export_name);
        SaveTrainData(mBatchTrajSolveConfig.mExportDataDir, export_name,
                      mResult);
        MIMIC_INFO("Save traindata to {}",
                   cFileUtil::ConcatFilename(
                       mBatchTrajSolveConfig.mExportDataDir,
                       cFileUtil::RemoveExtension(export_name) + ".train"));
        cFileUtil::DeleteLock(export_name);

        single_epoch_info.frame_num = mLoadInfo.mTotalFrame;
        single_epoch_info.length_second =
            mLoadInfo.mTotalFrame * mLoadInfo.mTimesteps[1];
        single_epoch_info.sample_traj_filename =
            full_epoch_infos[i].sample_traj_filename;
        single_epoch_info.train_filename = export_name;
        single_epoch_info.mr_traj_filename =
            full_epoch_infos[i].mr_traj_filename;
        summary_table.mEpochInfos.push_back(single_epoch_info);
        summary_table.mTotalEpochNum += 1;
        // summary_table.mTotalLengthTime +=
        // single_epoch_info.length_second; summary_table.mTotalLengthFrame
        // += single_epoch_info.frame_num;
        MIMIC_INFO("proc {} progress {}/{}", world_rank, id + 1,
                   my_own_task_num);
    }

    cMPIUtil::SetBarrier();
    MIMIC_INFO("proc {} tasks size = {}, expected size = {}", world_rank,
               my_own_task_num, summary_table.mEpochInfos.size());
    summary_table.WriteToDisk(mBatchTrajSolveConfig.mDestSummaryTableFile,
                              true);

    cMPIUtil::Finalize();
}

// void cOfflineIDSolver::RestoreActionByThetaDist(
//     std::vector<tSingleFrameIDResult> &IDResult)
// {
//     int num_of_joints = mSimChar->GetNumJoints();
//     if (mActionThetaDist.rows() != num_of_joints ||
//         mActionThetaDist.cols() != mActionThetaGranularity)
//     {
//         MIMIC_ERROR("RestoreActionThetaDist cur action theta shape ({}, "
//                     "{}) != ({}, {})",
//                     mActionThetaDist.rows(), mActionThetaDist.cols(),
//                     num_of_joints, mActionThetaGranularity);
//         exit(1);
//     }
//     auto &multibody = mSimChar->GetMultiBody();
//     for (int frame_id = 1; frame_id < IDResult.size(); frame_id++)
//     {
//         auto &cur_res = IDResult[frame_id];
//         int phase;

//         // TODO: Temporialy, in order to keep the consistency for action
//         // symbols between python agent and C++ ID result, we do this
//         trick. if (frame_id == IDResult.size() - 1)
//             phase =
//                 static_cast<int>(cur_res.state[0] *
//                 mActionThetaGranularity);
//         else
//             phase = static_cast<int>(IDResult[frame_id + 1].state[0] *
//                                      mActionThetaGranularity);

//         for (int i = 0, f_cnt = 0; i < multibody->getNumLinks(); i++)
//         {

//             switch (mSimChar->GetMultiBody()->getLink(i).m_jointType)
//             {
//             case btMultibodyLink::eFeatherstoneJointType::eSpherical:
//             {
//                 int sgn = cMathUtil::Sign(mActionThetaDist(i, phase));

//                 if
//                 (static_cast<int>(cMathUtil::Sign(cur_res.action[f_cnt]))
//                 !=
//                     sgn)
//                     cur_res.action.segment(f_cnt, 4) *= -1;

//                 const auto &axis = cur_res.action.segment(f_cnt + 1, 3);
//                 double debug_norm = axis.norm();
//                 if (std::fabs(debug_norm - 1) > 1e-6)
//                 {
//                     std::cout << "[error] frame " << frame_id << " joint
//                     " <<
//                     i
//                               << " axis norm = 0 " << debug_norm
//                               << ", axis = " << axis.transpose() <<
//                               std::endl;
//                 }
//                 f_cnt += 4;
//             };
//             break;
//             case btMultibodyLink::eFeatherstoneJointType::eRevolute:
//                 f_cnt++;
//                 break;
//             case btMultibodyLink::eFeatherstoneJointType::eFixed:
//                 break;
//             default:
//                 MIMIC_ERROR("RestoreActionThetaDist "
//                             "unsupporeted joint type");
//                 exit(1);
//                 break;
//             }
//         }
//     }
// }

// void cOfflineIDSolver::RestoreActionByGroundTruth(
//     std::vector<tSingleFrameIDResult> &IDResult)
// {
//     int num_of_joints = mSimChar->GetNumJoints();
//     auto &multibody = mSimChar->GetMultiBody();

//     for (int frame_id = 1; frame_id < IDResult.size() - 1; frame_id++)
//     {
//         auto &cur_res = IDResult[frame_id];
//         const tVectorXd &ground_truth_action =
//             mLoadInfo.mActionMat.row(frame_id);

//         for (int i = 0, f_cnt = 0; i < multibody->getNumLinks(); i++)
//         {

//             switch (mSimChar->GetMultiBody()->getLink(i).m_jointType)
//             {
//             case btMultibodyLink::eFeatherstoneJointType::eSpherical:
//             {
//                 int sgn = cMathUtil::Sign(ground_truth_action[f_cnt]);
//                 if
//                 (static_cast<int>(cMathUtil::Sign(cur_res.action[f_cnt]))
//                 !=
//                     sgn)
//                     cur_res.action.segment(f_cnt, 4) *= -1;
//                 f_cnt += 4;
//             };
//             break;
//             case btMultibodyLink::eFeatherstoneJointType::eRevolute:
//                 f_cnt++;
//                 break;
//             case btMultibodyLink::eFeatherstoneJointType::eFixed:
//                 break;
//             default:
//                 MIMIC_ERROR("RestoreActionThetaDist "
//                             "unsupporeted joint type");
//                 exit(1);
//                 break;
//             }
//         }
//     }
// }

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
    std::vector<tSingleFrameIDResult> &info) const
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
