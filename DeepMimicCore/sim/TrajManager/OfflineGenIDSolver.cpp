#include "OfflineGenIDSolver.h"
#include "BulletGenDynamics/btGenController/btGenContactAwareAdviser.h"
#include "BulletGenDynamics/btGenController/btTraj.h"
#include "BulletGenDynamics/btGenModel/RobotModelDynamics.h"
#include "BulletGenDynamics/btGenWorld.h"
#include "scenes/SceneImitate.h"
#include "sim/Controller/CtPDController.h"
#include "sim/Controller/CtPDGenController.h"
#include "sim/World/GenWorld.h"
#include "util/FileUtil.h"
#include "util/LogUtil.h"
#include "util/MPIUtil.h"
#include <iostream>

cOfflineGenIDSolver::cOfflineGenIDSolver(cSceneImitate *imitate,
                                         const std::string &config)
    : cOfflineIDSolver(imitate, config)
{
    mInited = false;
    mIDResult.clear();
    if (this->mSolveMode == eSolveMode::SingleTrajSolveMode)
    {
        mCurrentTrajPath = mSingleTrajSolveConfig.mSolveTrajPath;
        mCurrentOutputPath = mSingleTrajSolveConfig.mExportDataPath;
    }
    else if (mSolveMode == eSolveMode::BatchTrajSolveMode)
    {
        // 1. get all traj files which need to solve
        LoadBatchInfoMPI(mBatchTrajSolveConfig.mOriSummaryTableFile,
                         mBatchTrajIdArray, mBatchNameArray);
        mOldEpochInfos = mSummaryTable.mEpochInfos;
        mSummaryTable.mEpochInfos.clear();
        mBatchCurLocalTrajId = 0;

        // 2. set the first traj file as the current traj path
        mCurrentTrajPath = mBatchNameArray[mBatchCurLocalTrajId];
        mCurrentOutputPath = "";
        // this->mAdviser->SetTraj(mCurrentTrajPath, "", false);
    }
}
void cOfflineGenIDSolver::Reset()
{
    if (this->mSolveMode == eSolveMode::SingleTrajSolveMode)
    {
        mAdviser->Reset();
        MIMIC_INFO("OfflineGenIDsolver done");
        std::string export_dir =
            cFileUtil::GetDir(mSingleTrajSolveConfig.mExportDataPath);
        std::string export_name =
            cFileUtil::GetFilename(mSingleTrajSolveConfig.mExportDataPath);
        SaveTrainData(export_dir, export_name, this->mIDResult);
        MIMIC_INFO("save train data to {}",
                   mSingleTrajSolveConfig.mExportDataPath);
        exit(0);
    }
    else if (mSolveMode == eSolveMode::BatchTrajSolveMode)
    {
        // 1. save the current train data (take care of the race condition)
        mAdviser->Reset();
        int global_id = mBatchTrajIdArray[this->mBatchCurLocalTrajId];

        AddBatchInfoMPI(global_id, mCurrentTrajPath, mIDResult, mOldEpochInfos,
                        mCurTimestep * mIDResult.size());

        int world_rank = cMPIUtil::GetWorldRank();
        MIMIC_INFO("proc {} progress {}/{}", world_rank,
                   mBatchCurLocalTrajId + 1, mBatchTrajIdArray.size());
        mBatchCurLocalTrajId++;
        // 2. check whether there is something left we need to solve

        if (mBatchCurLocalTrajId >= mBatchTrajIdArray.size())
        {
            // 3. if not, write to the summary table incrementaly and exit
            cMPIUtil::SetBarrier();
            MIMIC_INFO("proc {} tasks size = {}, expected size = {}",
                       world_rank, mBatchTrajIdArray.size(),
                       mSummaryTable.mEpochInfos.size());
            mSummaryTable.WriteToDisk(
                mBatchTrajSolveConfig.mDestSummaryTableFile, true);
            cMPIUtil::Finalize();
            exit(0);
        }
        else
        {
            // 4. if so, set up the new trajs
            mCurrentTrajPath = mBatchNameArray[mBatchCurLocalTrajId];
            mCurrentOutputPath = "";
            mAdviser->SetTraj(mCurrentTrajPath, mCurrentOutputPath);
            // 5. reset the timer, also check the left time of timer
            double timer_max_time = mScene->GetTimer().GetMaxTime();
            double traj_length = mAdviser->GetRefTraj()->GetTimeLength();
            MIMIC_ASSERT(timer_max_time > traj_length);
            mScene->GetTimer().Reset();
        }
        mIDResult.clear();
    }
}

/**
 * \brief                   In Gen ID Solver, we need to record the trajectory
 * the same as before
 *
 *  1. Record State
 *  2. Record Action
 */
void cOfflineGenIDSolver::PreSim()
{
    if (mInited == false)
        Init();
    if (this->mAdviser->IsEnd() == true)
    {
        Reset();
    }

    // 1. remove active force applied by the controller
    auto model = dynamic_cast<cRobotModelDynamics *>(mSimChar);
    // model->ClearForce();

    // 2. record state
    tSingleFrameIDResult res;
    mCharController->RecordState(res.state);
    res.reward = mScene->CalcReward(0);
    this->mIDResult.push_back(res);
    // std::cout << "pre sim in offline gen solver\n";
    mPosePre = mSimChar->GetPose();
    mVelPre = mSimChar->GetVel();
    // std::cout << "cur gen force = " << model->GetGenera1lizedForce().norm()
    //           << std::endl;
}
void cOfflineGenIDSolver::PostSim()
{

    // record current action
    // gen torque -> PD target -> action
    // 1. get gen torque
    tVectorXd gen_force = tVectorXd::Zero(this->mSimChar->GetNumDof());
    gen_force.segment(6, mSimChar->GetNumDof() - 6) =
        mAdviser->GetPrevControlForce();

    // 2. to PD target
    auto ctrl = dynamic_cast<cCtPDGenController *>(this->mCharController);
    MIMIC_ASSERT(ctrl != nullptr);
    tVectorXd pd_target;

    // here we should put the pre pose and pre vel
    ctrl->CalcPDTargetByTorque(this->mCurTimestep, mPosePre, mVelPre, gen_force,
                               pd_target);
    // MIMIC_DEBUG("ID calc pd target = {}", pd_target.transpose());
    MIMIC_INFO("Contact-aware Inverse dynamics running for frame {}",
               mAdviser->GetInternalFrameId());
    // 3. from pd target to action
    tVectorXd action = pd_target;
    ctrl->CalcActionByTargetPose(action);
    mIDResult[mIDResult.size() - 1].action = action;
    std::cout << "action = " << action.transpose() << std::endl;
    // tVectorXd pd_target;
    // ctrl->CalcPDTargetByTorque(mCurTimestep, );
    // pd_target = dynamic_cast<this> std::cout
    //             << "post sim in offline gen solver\n";
}
void cOfflineGenIDSolver::SetTimestep(double delta_time)
{
    this->mCurTimestep = delta_time;
}

void cOfflineGenIDSolver::SingleTrajSolve(
    std::vector<tSingleFrameIDResult> &IDResult)
{
    MIMIC_ERROR("this API should not be called");
}

void cOfflineGenIDSolver::BatchTrajsSolve(const std::string &path)
{
    MIMIC_ERROR("this API should not be called");
}

void cOfflineGenIDSolver::Init()
{
    if (mInited == false)
    {
        auto gen_world = dynamic_cast<cGenWorld *>(mWorld);
        MIMIC_ASSERT(gen_world != nullptr);
        auto bt_world = gen_world->GetInternalGenWorld();
        MIMIC_ASSERT(bt_world != nullptr);
        bt_world->SetEnableContacrAwareControl();
        mAdviser = bt_world->GetContactAwareAdviser();
        mAdviser->SetTraj(mCurrentTrajPath, mCurrentOutputPath);
        double timer_max_time = mScene->GetTimer().GetMaxTime();
        double traj_length = mAdviser->GetRefTraj()->GetTimeLength();
        MIMIC_ASSERT(timer_max_time > traj_length);
        mScene->GetTimer().Reset();
    }
    mInited = true;
}