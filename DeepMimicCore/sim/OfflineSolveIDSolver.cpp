#include "OfflineSolveIDSolver.hpp"
#include <anim/KinCharacter.h>
#include <scenes/SceneImitate.h>
#include <anim/Motion.h>
#include "util/TimeUtil.hpp"
#include "../util/JsonUtil.h"
#include "../util/BulletUtil.h"
#include "sim/CtPDController.h"
#include <sim/SimCharacter.h>
#include "../util/FileUtil.h"
#include <iostream>
#ifdef __APPLE__
    #include <mpi.h>
#else
    #include <mpi/mpi.h>
#endif

extern std::string controller_details_path;
// extern std::string gRewardInfopath;
cOfflineIDSolver::cOfflineIDSolver(cSceneImitate * imi, const std::string & config)
:cInteractiveIDSolver(imi, eIDSolverType::OfflineSolve)
{
    mLogger = cLogUtil::CreateLogger("OfflineIDSolver");
    controller_details_path = "logs/controller_logs/controller_details_offlinesolve.txt";
    // gRewardInfopath = "reward_info_solve.txt";
    mEnableActionVerfied = true;
    mEnableRewardRecalc = true;
    ParseConfig(config);
    
}

cOfflineIDSolver::~cOfflineIDSolver()
{
    cLogUtil::DropLogger("OfflineIDSolver");
}

void cOfflineIDSolver::PreSim()
{
    cTimeUtil::Begin("ID Solving");
    if(mOfflineSolveMode == eOfflineSolveMode::SingleTrajSolveMode)
    {
        std::vector<tSingleFrameIDResult> mResult;
        InfoPrintf(mLogger, "begin to solve traj %s ", mSingleTrajSolveConfig.mSolveTrajPath.c_str());
        LoadTraj(mLoadInfo, mSingleTrajSolveConfig.mSolveTrajPath);
        SingleTrajSolve(mResult);
        SaveTrainData(mSingleTrajSolveConfig.mExportDataPath, mResult);
    }
    else if (mOfflineSolveMode == eOfflineSolveMode::BatchTrajSolveMode)
    {
        InfoPrintf(mLogger, "Batch solve, summary table = %s", mBatchTrajSolveConfig.mSummaryTableFile.c_str());
        BatchTrajsSolve(mBatchTrajSolveConfig.mSummaryTableFile);
    }
    else
    {
        ErrorPrintf(mLogger, "PreSim invalid mode %d", mOfflineSolveMode);
        exit(0);
    }
    cTimeUtil::End("ID Solving");
    exit(0);
}

void cOfflineIDSolver::PostSim()
{
   
}

void cOfflineIDSolver::Reset()
{
    std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
    exit(0);
}

void cOfflineIDSolver::SetTimestep(double)
{

}

void cOfflineIDSolver::ParseConfig(const std::string & conf)
{
    Json::Value root_;
    if(false == cJsonUtil::ParseJson(conf, root_))
    {
        ErrorPrintf(mLogger, "ParseConfig %s failed", conf);
        exit(1);
    }
    Json::Value root = root_["SolveModeInfo"];

    // 1. load shared config
    const Json::Value   &   ref_motion_path = root["ref_motion_path"],
                        &   retargeted_char_path = root["retargeted_char_path"],
                        &   recalc_reward = root["enable_reward_recal"],
                        &   enable_action_verified = root["enable_action_verified"],
    mRefMotionPath = ref_motion_path.asString();
    mRetargetCharPath = retargeted_char_path.asString();
    mEnableRewardRecalc = recalc_reward.asBool();
    mEnableActionVerfied = enable_action_verified.asBool();

    // 2. load solving mode
    const Json::Value & solve_mode_json = root["solve_mode"];
    mOfflineSolveMode = eOfflineSolveMode::INVALID;
    for(int i=0; i<eOfflineSolveMode::OfflineSolveModeNum; i++)
    {
        if(mConfPath[i] == solve_mode_json.asString())
        {
            mOfflineSolveMode = static_cast<eOfflineSolveMode>(i);
            break;
        }
    }
    if(mOfflineSolveMode == eOfflineSolveMode::INVALID)
    {
        ErrorPrintf(mLogger, "parse solve mode failed %s", solve_mode_json.asString());
        exit(0);
    }

    switch (mOfflineSolveMode)
    {
    case eOfflineSolveMode::SingleTrajSolveMode: ParseSingleTrajConfig(root["SingleTrajSolveInfo"]); break;
    case eOfflineSolveMode::BatchTrajSolveMode: ParseBatchTrajConfig(root["BatchTrajSolveInfo"]); break;
    default: exit(0); break;
    }

    // verify that the retarget char path is the same as the simchar skeleton path
    if(mRetargetCharPath != mSimChar->GetCharFilename())
    {
        ErrorPrintf(mLogger, "retarget path %s != loaded simchar path %s", mRetargetCharPath, mSimChar->GetCharFilename());
        exit(0);
    }

    // verify that the ref motion is the same as kinchar motion
    if(mRefMotionPath != mKinChar->GetMotion().GetMotionFile())
    {
        ErrorPrintf(mLogger, "ID ref motion path %s != loaded motion path %s", mRefMotionPath, mKinChar->GetMotion().GetMotionFile());
        exit(0);
    }
}

void cOfflineIDSolver::ParseSingleTrajConfig(const Json::Value & single_traj_config)
{
    assert(single_traj_config.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSolve(const Json::Value & save_value)\n";
    const Json::Value & solve_traj_path = single_traj_config["solve_traj_path"];
    assert(solve_traj_path.isNull() == false);
    mSingleTrajSolveConfig.mSolveTrajPath = solve_traj_path.asString();
/*
    "export_train_data_path_meaning" : "训练数据的输出路径，后缀名为.train，里面存放了state, action, reward三个键值",
    "export_train_data_path" : "data/batch_train_data/0424/leftleg_0.train",
    "ref_motion_meaning" : "重新计算reward所使用到的motion",
    "ref_motion" : "data/0424/motions/walk_motion_042401_leftleg.txt"
*/
    const Json::Value   &   export_train_data_path = single_traj_config["export_train_data_path"];
    assert(export_train_data_path.isNull() == false);

    mSingleTrajSolveConfig.mExportDataPath = export_train_data_path.asString();

    if(false == cFileUtil::ValidateFilePath(mSingleTrajSolveConfig.mExportDataPath))
    {
        ErrorPrintf(mLogger, "ParseSingleTrajConfig export train data path illegal: %s", mSingleTrajSolveConfig.mExportDataPath);
        exit(0);
    }
    mLogger->info("working in SingleTrajSolve mode");
}

void cOfflineIDSolver::ParseBatchTrajConfig(const Json::Value & batch_traj_config)
{
    assert(batch_traj_config.isNull() == false);
    mBatchTrajSolveConfig.mSummaryTableFile = batch_traj_config["summary_table_filename"].asString();
    mBatchTrajSolveConfig.mExportDataDir = batch_traj_config["export_train_data_dir"].asString();
    mBatchTrajSolveConfig.mEnableRestoreThetaByActionDist = batch_traj_config["enable_restore_theta_by_action_dist"].asBool();
    mBatchTrajSolveConfig.mEnableRestoreThetaByGT = batch_traj_config["enable_restore_theta_by_ground_truth"].asBool();
    mLogger->info("working in BatchTrajSolve mode");
}

void cOfflineIDSolver::SingleTrajSolve(std::vector<tSingleFrameIDResult> & IDResults)
{
    // cTimeUtil::Begin("OfflineSolve");
    assert(mLoadInfo.mTotalFrame > 0);
    IDResults.resize(mLoadInfo.mTotalFrame);
    // std::cout <<"[debug] cOfflineIDSolver::OfflineSolve: motion total frame = " << mLoadInfo.mTotalFrame << std::endl;
    tVectorXd old_q, old_u;
    RecordGeneralizedInfo(old_q, old_u);

    mCharController->SetInitTime(mLoadInfo.mMotionRefTime[0] - mLoadInfo.mTimesteps[0]);
    // 1. calc vel and accel from pos
    /*
    double cur_timestep = mSaveInfo.mTimesteps[cur_frame - 1],
            last_timestep = mSaveInfo.mTimesteps[cur_frame - 2];
			tVectorXd old_vel_after = mSaveInfo.mBuffer_u[cur_frame];
			tVectorXd old_vel_before = mSaveInfo.mBuffer_u[cur_frame - 1];
			tVectorXd old_accel = (old_vel_after - old_vel_before) / cur_timestep;
			mSaveInfo.mBuffer_u[cur_frame - 1] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 2], mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
			mSaveInfo.mBuffer_u[cur_frame] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1], mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
			mSaveInfo.mBuffer_u_dot[cur_frame - 1] = (mSaveInfo.mBuffer_u[cur_frame] - mSaveInfo.mBuffer_u[cur_frame - 1]) / cur_timestep;
    */
    for(int frame_id = 0; frame_id < mLoadInfo.mTotalFrame - 1; frame_id++)
    {
        tVectorXd cur_vel = CalcGeneralizedVel(mLoadInfo.mPoseMat.row(frame_id), mLoadInfo.mPoseMat.row(frame_id+1), mLoadInfo.mTimesteps[frame_id]);
        // std::cout <<"cur vel size = " << cur_vel.size() << std::endl;
        // std::cout <<"com vel size = " << mLoadInfo.mVelMat.row(frame_id + 1).size() << std::endl;
        // std::cout <<" cur vel = " << cur_vel.transpose() << std::endl;
        // std::cout <<"vel comp = " << mLoadInfo.mVelMat.row(frame_id + 1) << std::endl;
        tVectorXd diff = mLoadInfo.mVelMat.row(frame_id + 1).transpose() - cur_vel;
        mLoadInfo.mVelMat.row(frame_id + 1) = cur_vel;
        // std::cout <<"frame id " << frame_id << " vel diff = " << diff.norm() << std::endl;
        assert(diff.norm() < 1e-10);
    }
    
    for(int frame_id = 1; frame_id < mLoadInfo.mTotalFrame - 1; frame_id++)
    {
        tVectorXd cur_accel = (mLoadInfo.mVelMat.row(frame_id + 1)- mLoadInfo.mVelMat.row(frame_id))/ mLoadInfo.mTimesteps[frame_id];
        // std::cout <<"cur accel size = " << cur_accel.size() << std::endl;
        // std::cout <<"com accel size = " << mLoadInfo.mAccelMat.row(frame_id).size() << std::endl;
        // std::cout <<"cur accel = " << cur_accel.transpose() << std::endl;
        // std::cout <<"accel comp = " << mLoadInfo.mAccelMat.row(frame_id) << std::endl;
        tVectorXd diff = mLoadInfo.mAccelMat.row(frame_id).transpose() - cur_accel;
        mLoadInfo.mAccelMat.row(frame_id) = cur_accel;
        // std::cout <<"frame id " << frame_id << " accel diff = " << diff.norm() << std::endl;
        assert(diff.norm() < 1e-7);
    }
    
    // 2. calculate link pos and link rot from generalized info
    for(int frame_id = 0; frame_id < mLoadInfo.mTotalFrame; frame_id++)
    {
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(frame_id));
        RecordMultibodyInfo(mLoadInfo.mLinkRot[frame_id], mLoadInfo.mLinkPos[frame_id]);
    }

    // 3. Init the status for mKinChar and mSimChar according to mLoadInfo
    // ATTENTION: most of these codes are used to gurantee the validity of reward we get from the following Inverse Dynamics procedure.
    mKinChar->SetTime(mLoadInfo.mMotionRefTime[0] - mLoadInfo.mTimesteps[0]);   // training policy: random init
    mKinChar->Update(0);                                                        // update mKinChar inner status by Update(0)
    SetGeneralizedPos(mLoadInfo.mPoseMat.row(0));                               // set up the sim char pos from mLoadInfo
    mSimChar->PostUpdate(0);
    mScene->ResolveCharGroundIntersectInverseDynamic();                         // Resolve intersection between char and the ground. Sync to KinChar is also included.
    mKinChar->Update(mLoadInfo.mTimesteps[0]);                                  // Go for another timestep
    mCharController->Update(mLoadInfo.mTimesteps[0]);

    // 4. solve ID for each frame
    double ID_torque_err = 0, ID_action_err = 0, reward_err = 0;
    tVectorXd torque = tVectorXd::Zero(mSimChar->GetPose().size()), pd_target = tVectorXd::Zero(mSimChar->GetPose().size());

    for(int cur_frame = 1; cur_frame < mLoadInfo.mTotalFrame; cur_frame++)
    {
        auto & cur_ID_res = IDResults[cur_frame];

        // 4.1 update the sim char
        mInverseModel->clearAllUserForcesAndMoments();
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(cur_frame));
        SetGeneralizedVel(mLoadInfo.mVelMat.row(cur_frame));
        mSimChar->PostUpdate(0);

        // 4.2 record state at this moment
        mCharController->RecordState(cur_ID_res.state);
        // std::cout <<"[debug] cur record state = " << cur_ID_res.state[0] << std::endl;

        // 4.3 solve Inverse Dynamic for joint torques
        std::vector<tVector> result;
        // SetGeneralizedInfo(mLoadInfo.mPoseMat.row(frame_id));
        // std::cout <<"log frame = " << cur_frame << std::endl;
        cIDSolver::SolveIDSingleStep(
            result, 
            mLoadInfo.mContactForces[cur_frame], 
            mLoadInfo.mLinkPos[cur_frame], 
            mLoadInfo.mLinkRot[cur_frame], 
            mLoadInfo.mPoseMat.row(cur_frame), 
            mLoadInfo.mVelMat.row(cur_frame), 
            mLoadInfo.mAccelMat.row(cur_frame), 
            cur_frame, 
            mLoadInfo.mExternalForces[cur_frame], 
            mLoadInfo.mExternalTorques[cur_frame]
            );

        int f_cnt = 7;
        torque.segment(0, 7).setZero();
        for(int j=0; j<mNumLinks-1; j++)
        {
            switch (this->mMultibody->getLink(j).m_jointType)
            {
            case btMultibodyLink::eFeatherstoneJointType::eRevolute:
                torque[f_cnt++] = result[j].dot(cBulletUtil::btVectorTotVector0(mMultibody->getLink(j).getAxisTop(0)));
                // std::cout <<"revolute " << j << "force = " << result[j].transpose() << std::endl;
                /* code */
                break;
            case btMultibodyLink::eFeatherstoneJointType::eSpherical:
            {
                // attention: we need to convert this result torque into child space.
                // For more details, see void cSimBodyJoint::ApplyTauSpherical()
                tVector local_torque, local_torque_child;
                local_torque = result[j];
                // std::cout << "spherical joint " << j <<" local torque = " << local_torque.transpose() << std::endl;
                local_torque_child = cMathUtil::QuatRotVec(mSimChar->GetJoint(j).GetChildRot().conjugate(), local_torque);
                torque.segment(f_cnt, 4) = local_torque_child;
                f_cnt+=4;
                // std::cout << "spherical joint " << j <<" local torque child = " << local_torque_child.transpose() << std::endl;
                // std::cout <<"spherical " << j << "force = " << result[j].transpose() << std::endl;
                break;
            }
                
            case btMultibodyLink::eFeatherstoneJointType::eFixed:
                break;
            default:
                std::cout <<"cOfflineIDSolver::OfflineSolve joint " << j <<" type " << mMultibody->getLink(j).m_jointType << std::endl;
                exit(1);
                break;
            }
            double single_error = (result[j] - mLoadInfo.mTruthJointForces[cur_frame][j]).norm();
            if(single_error > 1e-6)
            {
                std::cout <<"[error] cOfflineIDSolver::OfflineSolve for frame " << cur_frame <<" link " << j << ":\n";
                std::cout <<"offline solved = " << result[j].transpose() << std::endl;
                std::cout <<"standard = " << mLoadInfo.mTruthJointForces[cur_frame][j].transpose() << std::endl;
                std::cout <<"error = " << single_error << std::endl;
                assert(0);
            }
            ID_torque_err += single_error;
        }

        // 4.4 convert the result joint torques into PD Target
        // the vector "torque" has the same shape as pose and vel
        /*
            root joint: occupy the first 7 DOF in the vector, all are set to zero
            revolute: occupy 1 DOF in the vector, the value of torque
            spherical: occupy 4 DOF in the vector, [torque_x, torque_y, torque_z, 0]
            fixed: No sapce 
        */
        
        double timestep = mLoadInfo.mTimesteps[cur_frame];
        auto & imp_controller = mCharController->GetImpPDController();
        imp_controller.SolvePDTargetByTorque(timestep,
            mSimChar->GetPose(), mSimChar->GetVel(),
            torque, pd_target);
        // std::cout <<"now pd target before clip = " << pd_target.transpose() << std::endl;
        if(mMultibody->hasFixedBase() == false)
        {
            assert((pd_target.size() - 7) == mCharController->GetActionSize());
            // cut the first 7 DOF of root. They are outside of action space and should not be counted in PD target.
            const tVectorXd short_pd_target = pd_target.segment(7, pd_target.size() - 7);
            pd_target = short_pd_target;
        }
        // std::cout <<"now pd target after clip = " << pd_target.transpose() << std::endl;

        // std::cout << "load pd = " << mLoadInfo.mPDTargetMat.row(cur_frame) << std::endl;
        // std::cout << "solved pd = " << pd_target.transpose() << std::endl;
        // exit(1);
        

        // 4.5 convert PD target to Neural Network action
        // action is different from fetched pd_target! 
        // For ball joints, their action is represented in axis angle; but their pd_Target is quaternion
        // we still need to have a convert here. pd_target = [x, y, z, w], axis angle = [angle, ax, ay, az]
        tVectorXd action = pd_target;
        mCharController->ConvertTargetPoseToActionFullsize(action);

        // 4.6 sometimes, the action of spherical joints can be zero, which doesn't make sense.
        // We will given thess zero action the same value as its previous one
        {
            f_cnt = 0;
            for(int j=0; j<mNumLinks-1; j++)
            {           
                switch (this->mMultibody->getLink(j).m_jointType)
                {
                case btMultibodyLink::eFeatherstoneJointType::eRevolute:
                {
                    f_cnt++;
                    break;
                }
                case btMultibodyLink::eFeatherstoneJointType::eSpherical:
                {
                    if(std::fabs(action.segment(f_cnt, 4).norm()) < 1e-10 && cur_frame >= 2)
                    {
                        // this action is zero
                        action.segment(f_cnt, 4) = IDResults[cur_frame - 1].action.segment(f_cnt, 4);
                        WarnPrintf(mLogger, "SingleTrajSolve for %s: frame %d joint %d action is zero, overwrite it with previsou result.", \
                            mLoadInfo.mLoadPath.c_str(), cur_frame, j);
                    }
                    f_cnt+=4;
                    break;
                }
                case btMultibodyLink::eFeatherstoneJointType::eFixed:
                    break;
                default:
                    ErrorPrintf(mLogger, "SingleTrajSolve joint %d type %d unsupported!", j, mMultibody->getLink(j).m_jointType);
                    exit(1);
                    break;
                }
            }
        }

        // 5. verified the ID result action if possible 
        // the loaded action hasn't been normalized, we need to preprocess it before comparing...
        if(true == mEnableActionVerfied)
        {
            tVectorXd truth_action = mLoadInfo.mActionMat.row(cur_frame);
            double total_action_err = 0, single_action_err = 0;
            assert(truth_action.size() == mCharController->GetActionSize());
            f_cnt = 0;
            for(int j=0; j<mNumLinks-1; j++)
            {           
                switch (this->mMultibody->getLink(j).m_jointType)
                {
                case btMultibodyLink::eFeatherstoneJointType::eRevolute:
                {
                    // 6.1 compare the ideal action and solved action
                    single_action_err = std::fabs( truth_action[f_cnt] - action[f_cnt]);
                    if(single_action_err > 1e-7)
                    {
                        ErrorPrintf(mLogger, "revo joint %d true joint force %lf, solved joint force %lf, diff %lf", j, truth_action[f_cnt], action[f_cnt], single_action_err);
                        total_action_err += single_action_err;
                    }
                    f_cnt++;
                    break;
                }
                case btMultibodyLink::eFeatherstoneJointType::eSpherical:
                {
                    truth_action.segment(f_cnt + 1, 3).normalize();
                    if(std::fabs(truth_action[f_cnt]) <1e-10) truth_action.segment(f_cnt, 4).setZero();

                    // it is because that, the axis angle represention is ambiguous
                    single_action_err = std::min(
                        (truth_action.segment(f_cnt, 4) + action.segment(f_cnt, 4)).norm(),
                        (truth_action.segment(f_cnt, 4) - action.segment(f_cnt, 4)).norm()
                        );
                    if(single_action_err > 1e-7)
                    {

                        // std::cout <<"[debug] OfflineSOlvejoint " << j << " type eSpherical, truth joint force " << \
                        // truth_action.segment(f_cnt, 4).transpose() <<", solved joint force = " \
                        // << action.segment(f_cnt, 4).transpose() <<", diff = " << single_action_err << std::endl;
                        total_action_err += single_action_err;
                    }
                    f_cnt+=4;
                    break;
                }
                case btMultibodyLink::eFeatherstoneJointType::eFixed:
                    break;
                default:
                    ErrorPrintf(mLogger, "SingleTrajSolve joint %d type %d unsupported!", j, mMultibody->getLink(j).m_jointType);
                    exit(1);
                    break;
                }
            }
            
            if(total_action_err > 1e-4)
            {
                ErrorPrintf(mLogger, "SingleTrajSolve %s frame %d action err = %.3f", mLoadInfo.mLoadPath.c_str(), cur_frame, total_action_err);
            }
            ID_action_err += total_action_err;
        }
        

        cur_ID_res.action = action;


        double prev_phase = mKinChar->GetPhase();
        if(true == mEnableRewardRecalc)
        {
            // 4.9 recalculate the reward according to current motion
            // you must confirm that the simchar skeleton file is the same as trajectories skeleton file accordly (Now it has been guranteed in ParseConfig)
            cur_ID_res.reward = mScene->CalcReward(0);
        }

        // update kinchar and mcharcontroller to maintain the recorded state phase is correct
        mKinChar->Update(mLoadInfo.mTimesteps[cur_frame]);
        mCharController->UpdateTimeOnly(mLoadInfo.mTimesteps[cur_frame]);

        // recalcualte the reward?
        if(true == mEnableRewardRecalc)
        {
            double curr_phase = mKinChar->GetPhase();
            // 4.10 judging whether we should jump to the next cycle and update/sync the kinChar according to the simchar.
            if (curr_phase < prev_phase)
            {
                (dynamic_cast<cSceneImitate *>(mScene))->SyncKinCharNewCycleInverseDynamic(*mSimChar, *mKinChar);
            }
            
            reward_err += std::fabs(cur_ID_res.reward - mLoadInfo.mRewards[cur_frame]);
            // std::cout <<"frame " << cur_frame <<" truth action = " << mLoadInfo.mActionMat.row(cur_frame) << std::endl;
            // std::cout <<"frame " << cur_frame <<" solved action = " << action.transpose() << std::endl;
            // std::cout <<"frame " << cur_frame <<" cur reward = " << cur_ID_res.reward << ", load reward = " << mLoadInfo.mRewards[cur_frame] <<  std::endl;
            // std::cout <<"frame " << cur_frame <<" cur reward = " << cur_ID_res.reward << ", load reward = " << mLoadInfo.mRewards[cur_frame] <<  std::endl;
        }
        else
        {
            cur_ID_res.reward = mLoadInfo.mRewards[cur_frame];
        }
        
    }
    if(ID_torque_err < 1e-3 && ID_action_err < 1e-3 && reward_err < 1e-3)
    {
        // std::cout <<"[log] cOfflineIDSolver::OfflineSolve: succ, total ID torque error = " << ID_torque_err;
        // std::cout <<", total ID Action error = " << ID_action_err;
        // std::cout <<"，total ID reward error = " << reward_err << std::endl;
    }
    else
    {
        std::cout <<"[error] OfflineSolve failed " << mLoadInfo.mLoadPath << ", total ID torque error = " << ID_torque_err << ", ";
        std::cout <<"total ID Action error = " << ID_action_err << ", ";
        std::cout <<"total ID reward error = " << reward_err << std::endl;
    }
    // cTimeUtil::End("OfflineSolve");
}

/**
 * \brief       Given A summary files, this function will solve their ID very quick under the help of MPI
*/
void cOfflineIDSolver::BatchTrajsSolve(const std::string & path)
{
    // 1. MPI init
    int is_mpi_init;
    MPI_Initialized(&is_mpi_init);
    if(0 == is_mpi_init) MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // mpi_rank = world_rank;

    // 2. load the json: ensure that all process can load the summary table correctly.
    cFileUtil::AddLock(path);
    mSummaryTable.LoadFromDisk(path);
    auto & summary_table = mSummaryTable;
    cFileUtil::DeleteLock(path);
    MPI_Barrier(MPI_COMM_WORLD);

    // 2.1 load action distribution from files
    if(mBatchTrajSolveConfig.mEnableRestoreThetaByActionDist == true)
    {
        InitActionThetaDist(mSimChar, mActionThetaDist);
        LoadActionThetaDist(mSummaryTable.mActionThetaDistFile, mActionThetaDist);
    }

    // 3. rename this summary table: after that here is a MPI_Barrier, which ensures that our process will not delete other processes' result.
    cFileUtil::AddLock(path);
    if(cFileUtil::ExistsFile(path)) cFileUtil::RenameFile(path, path + ".bak");
    if(cFileUtil::ExistsDir(mBatchTrajSolveConfig.mExportDataDir)) cFileUtil::ClearDir(mBatchTrajSolveConfig.mExportDataDir.c_str());
    cFileUtil::DeleteLock(path);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4. determine my own tasks
    int total_traj_num = summary_table.mEpochInfos.size();
    int st = -1, ed = -1;   // from where to where?
    bool enable_single_thread = false;
    if(total_traj_num < world_size || world_size == 1)
    {
        st = 0, ed = total_traj_num -1;
        enable_single_thread = true;
    }
    else
    {
        int unit = std::floor(total_traj_num * 1.0 / world_size);
        if (unit * world_size < total_traj_num) unit++;
        
        if(unit * world_size < total_traj_num)
        {
            ErrorPrintf(mLogger, "BatchTrajSolve MPI divide unit %d is not enough", unit);
            exit(1);
        }
        st = world_rank * unit;
        ed = st + unit - 1;
    }
    InfoPrintf(mLogger, "proc %d task from %d to %d, size %d/%d", world_rank, st, ed, (ed-st+1), total_traj_num);

    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
    // exit(0);
    // 4. now remember all info in summary_table and clear them at all
    auto full_epoch_infos = summary_table.mEpochInfos;
    summary_table.mEpochInfos.clear();
    summary_table.mTotalEpochNum = 0;
    summary_table.mTotalLengthFrame = 0;
    summary_table.mTotalLengthTime = 0;    

    std::vector<tSingleFrameIDResult> mResult(0);
    cInteractiveIDSolver::tSummaryTable::tSingleEpochInfo single_epoch_info;
    for(int i=st; i <= ed && i < total_traj_num; i++)
    {
        // 4.1 load a single traj and solve ID for it.
        LoadTraj(mLoadInfo, full_epoch_infos[i].traj_filename);
        SingleTrajSolve(mResult);

        // 4.2 
        if(mBatchTrajSolveConfig.mEnableRestoreThetaByActionDist == true)   RestoreActionByThetaDist(mResult);
        if(mBatchTrajSolveConfig.mEnableRestoreThetaByGT == true)   RestoreActionByGroundTruth(mResult);

        std::string export_name = cFileUtil::GetFilename(full_epoch_infos[i].traj_filename);
        export_name = mBatchTrajSolveConfig.mExportDataDir + cFileUtil::RemoveExtension(export_name) + ".train";
        cFileUtil::AddLock(export_name);
        SaveTrainData(export_name, mResult);
        cFileUtil::DeleteLock(export_name);

        single_epoch_info.frame_num = mLoadInfo.mTotalFrame;
        single_epoch_info.length_second = mLoadInfo.mTotalFrame * mLoadInfo.mTimesteps[1];
        single_epoch_info.traj_filename = full_epoch_infos[i].traj_filename;
        single_epoch_info.train_data_filename = export_name;
        summary_table.mEpochInfos.push_back(single_epoch_info);
        summary_table.mTotalEpochNum += 1;
        summary_table.mTotalLengthTime += single_epoch_info.length_second;
        summary_table.mTotalLengthFrame += single_epoch_info.frame_num;
        InfoPrintf(mLogger, "proc %d progress %d/%d", world_rank, i-st+1, ed-st+1);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    InfoPrintf(mLogger, "proc %d tasks size = %d, expected size = %d", world_rank, ed - st+1, summary_table.mEpochInfos.size());
    if(enable_single_thread == true)
    {
        if(0 == world_rank)
        {
            summary_table.WriteToDisk(mBatchTrajSolveConfig.mSummaryTableFile, true);
        }
        
    }
    else
    {
        summary_table.WriteToDisk(mBatchTrajSolveConfig.mSummaryTableFile, true);
    }
    
    MPI_Finalize();
}

void cOfflineIDSolver::RestoreActionByThetaDist(std::vector<tSingleFrameIDResult> & IDResult)
{
    int num_of_joints = mSimChar->GetNumJoints();
    if(mActionThetaDist.rows() != num_of_joints || mActionThetaDist.cols() != mActionThetaGranularity)
    {
        ErrorPrintf(mLogger, "RestoreActionThetaDist cur action theta shape (%d, %d) != (%d, %d)", mActionThetaDist.rows(), mActionThetaDist.cols(),
            num_of_joints, mActionThetaGranularity);
        exit(1);
    }
    auto & multibody = mSimChar->GetMultiBody();
    for(int frame_id=1; frame_id < IDResult.size(); frame_id++)
    {
        auto & cur_res = IDResult[frame_id];
        int phase;
        
        // TODO: Temporialy, in order to keep the consistency for action symbols between python agent and C++ ID result, we do this trick.
        if(frame_id == IDResult.size()-1)phase = static_cast<int>(cur_res.state[0] * mActionThetaGranularity);
        else phase = static_cast<int>(IDResult[frame_id+1].state[0] * mActionThetaGranularity);
         
        for(int i=0, f_cnt=0; i<multibody->getNumLinks(); i++)
        {

            switch (mSimChar->GetMultiBody()->getLink(i).m_jointType)
            {
            case btMultibodyLink::eFeatherstoneJointType::eSpherical:
            {
                int sgn = cMathUtil::Sign(mActionThetaDist(i, phase));
                
                if(static_cast<int>(cMathUtil::Sign(cur_res.action[f_cnt])) != sgn)cur_res.action.segment(f_cnt, 4) *= -1;

                const auto & axis = cur_res.action.segment(f_cnt+1, 3);
                double debug_norm = axis.norm();
                if(std::fabs(debug_norm -1) > 1e-6)
                {
                    std::cout <<"[error] frame " << frame_id <<" joint " << i <<" axis norm = 0 " << debug_norm <<", axis = " << axis.transpose() << std::endl;
                }
                f_cnt += 4;
                
            };break;
            case btMultibodyLink::eFeatherstoneJointType::eRevolute: f_cnt++; break;
            case btMultibodyLink::eFeatherstoneJointType::eFixed: break;
            default: mLogger->error("RestoreActionThetaDist unsupporeted joint type"); exit(1);
                break;
            }

        }
    }

}

void cOfflineIDSolver::RestoreActionByGroundTruth(std::vector<tSingleFrameIDResult> & IDResult)
{
    int num_of_joints = mSimChar->GetNumJoints();
    auto & multibody = mSimChar->GetMultiBody();
    
    for(int frame_id=1; frame_id < IDResult.size(); frame_id++)
    {
        auto & cur_res = IDResult[frame_id];
        const tVectorXd & ground_truth_action = mLoadInfo.mActionMat.row(frame_id);
         
        for(int i=0, f_cnt=0; i<multibody->getNumLinks(); i++)
        {

            switch (mSimChar->GetMultiBody()->getLink(i).m_jointType)
            {
            case btMultibodyLink::eFeatherstoneJointType::eSpherical:
            {
                int sgn = cMathUtil::Sign(ground_truth_action[f_cnt]);
                if(static_cast<int>(cMathUtil::Sign(cur_res.action[f_cnt])) != sgn) cur_res.action.segment(f_cnt, 4) *= -1;
                f_cnt += 4;
                
            };break;
            case btMultibodyLink::eFeatherstoneJointType::eRevolute: f_cnt++; break;
            case btMultibodyLink::eFeatherstoneJointType::eFixed: break;
            default: mLogger->error("RestoreActionThetaDist unsupporeted joint type"); exit(1);
                break;
            }
        }
    }
}