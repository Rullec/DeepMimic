#include "OfflineFeaIDSolver.h"
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
cOfflineFeaIDSolver::cOfflineFeaIDSolver(cSceneImitate *imi,
                                         const std::string &config)
    : cOfflineIDSolver(imi, config)
{
}

void cOfflineFeaIDSolver::PreSim()
{
    cTimeUtil::Begin("ID Solving");
    if (mSolveMode == eSolveMode::SingleTrajSolveMode)
    {
        std::vector<tSingleFrameIDResult> mResult;
        MIMIC_INFO("begin to solve traj {}",
                   mSingleTrajSolveConfig.mSolveTrajPath);
        mLoadInfo.LoadTraj(mSimChar, mSingleTrajSolveConfig.mSolveTrajPath);
        MIMIC_ASSERT(mLoadInfo.mIntegrationScheme ==
                     GetIntegrationSchemeWorld());
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

void cOfflineFeaIDSolver::PostSim() {}

void cOfflineFeaIDSolver::Reset()
{
    std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
    exit(0);
}

void cOfflineFeaIDSolver::SetTimestep(double) {}

void cOfflineFeaIDSolver::SingleTrajSolve(
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

    // Resolve intersection between char and the ground. Sync to KinChar is also included.
    mScene->ResolveCharGroundIntersectInverseDynamic();

    mKinChar->Update(mLoadInfo.mTimesteps[0]); // Go for another timestep
    mCharController->Update(mLoadInfo.mTimesteps[0]);

    // 4. solve ID for each frame
    double ID_torque_err = 0, ID_action_err = 0, reward_err = 0;
    tVectorXd torque = tVectorXd::Zero(mSimChar->GetPose().size()),
              pd_target = tVectorXd::Zero(mSimChar->GetPose().size());

    for (int cur_frame = 1; cur_frame < mLoadInfo.mTotalFrame - 1; cur_frame++)
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

        if (mLoadInfo.mVersion == eTrajFileVersion::V1)
        {
            ID_torque_err = cIDSolver::CalcAssembleJointForces(
                result, torque, mLoadInfo.mTruthJointForces[cur_frame]);
        }

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
void cOfflineFeaIDSolver::BatchTrajsSolve(const std::string &table_path)
{
    // 1. MPI init
    std::vector<int> traj_id_array;
    std::vector<std::string> traj_name_array;

    LoadBatchInfoMPI(table_path, traj_id_array, traj_name_array);
    int world_size = cMPIUtil::GetCommSize(),
        world_rank = cMPIUtil::GetWorldRank();

    // 4. now remember all info in summary_table and clear them at all
    // cl
    auto old_batch_info = mSummaryTable.mEpochInfos;
    mSummaryTable.mEpochInfos.clear();
    mSummaryTable.mTotalEpochNum = 0;

    std::vector<tSingleFrameIDResult> mResult(0);
    for (int local_id = 0; local_id < traj_name_array.size(); local_id++)
    {
        int global_id = traj_id_array[local_id];
        std::string target_traj_filename_full = traj_name_array[local_id];
        mLoadInfo.LoadTraj(mSimChar, target_traj_filename_full);
        MIMIC_ASSERT(mLoadInfo.mIntegrationScheme ==
                     GetIntegrationSchemeWorld());
        SingleTrajSolve(mResult);
        AddBatchInfoMPI(global_id, target_traj_filename_full, mResult,
                        old_batch_info,
                        mLoadInfo.mTotalFrame * mLoadInfo.mTimesteps[1]);
        // mSummaryTable.mTotalLengthTime +=
        // single_epoch_info.length_second; mSummaryTable.mTotalLengthFrame
        // += single_epoch_info.frame_num;
        MIMIC_INFO("proc {} progress {}/{}", world_rank, local_id + 1,
                   traj_name_array.size());
    }

    cMPIUtil::SetBarrier();
    MIMIC_INFO("proc {} tasks size = {}, expected size = {}", world_rank,
               traj_id_array.size(), mSummaryTable.mEpochInfos.size());
    mSummaryTable.WriteToDisk(mBatchTrajSolveConfig.mDestSummaryTableFile,
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
