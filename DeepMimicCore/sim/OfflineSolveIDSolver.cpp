#include "OfflineSolveIDSolver.hpp"
#include <anim/KinCharacter.h>
#include <scenes/SceneImitate.h>
#include <anim/Motion.h>
#include "util/cTimeUtil.hpp"
#include "../util/JsonUtil.h"
#include "../util/BulletUtil.h"
#include "sim/CtPDController.h"
#include <sim/SimCharacter.h>
#include "../util/FileUtil.h"
#include <iostream>

extern std::string controller_details_path;
extern std::string gRewardInfopath;
cOfflineSolveIDSolver::cOfflineSolveIDSolver(cSceneImitate * imi, const std::string & config)
:cInteractiveIDSolver(imi, eIDSolverType::OfflineSolve)
{
    controller_details_path = "logs/controller_logs/controller_details_offlinesolve.txt";
    gRewardInfopath = "reward_info_solve.txt";
    Parseconfig(config);
}

cOfflineSolveIDSolver::~cOfflineSolveIDSolver()
{

}

void cOfflineSolveIDSolver::PreSim()
{
    std::vector<tSingleFrameIDResult> mResult;
    OfflineSolve(mResult);
}

void cOfflineSolveIDSolver::PostSim()
{
   
}

void cOfflineSolveIDSolver::Reset()
{
    std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
    exit(0);
}

void cOfflineSolveIDSolver::SetTimestep(double)
{

}

void cOfflineSolveIDSolver::Parseconfig(const std::string & conf)
{
    Json::Value root;
    cJsonUtil::ParseJson(conf, root);
    auto solve_value = root["SolveModeInfo"];
    assert(solve_value.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSolve(const Json::Value & save_value)\n";
    const Json::Value & solve_traj_path = solve_value["solve_traj_path"];
    assert(solve_traj_path.isNull() == false);
    LoadTraj(mLoadInfo, solve_traj_path.asString());
/*
    "export_train_data_path_meaning" : "训练数据的输出路径，后缀名为.train，里面存放了state, action, reward三个键值",
    "export_train_data_path" : "data/batch_train_data/0424/leftleg_0.train",
    "ref_motion_meaning" : "重新计算reward所使用到的motion",
    "ref_motion" : "data/0424/motions/walk_motion_042401_leftleg.txt"
*/
    const Json::Value   &   export_train_data_path = solve_value["export_train_data_path"],
                        &   ref_motion_path = solve_value["ref_motion_path"],
                        &   retargeted_char_path = solve_value["retargeted_char_path"];
    assert(export_train_data_path.isNull() == false);
    assert(ref_motion_path.isNull() == false);
    assert(retargeted_char_path.isNull() == false);
    mExportDataPath = export_train_data_path.asString();
    mRefMotionPath = ref_motion_path.asString();
    mRetargetCharPath = retargeted_char_path.asString();

    if(false == cFileUtil::ExistsFile(mRefMotionPath))
    {
        std::cout << "[error] cOfflineSolveIDSolver::Parseconfig ref motion doesn't exists: " << mRefMotionPath << std::endl;
        exit(0);
    }
    if(false == cFileUtil::ValidateFilePath(mExportDataPath))
    {
        std::cout << "[error] cOfflineSolveIDSolver::Parseconfig export train data path illegal: " << mExportDataPath << std::endl;
        exit(0);
    }
    if(false == cFileUtil::ValidateFilePath(mRetargetCharPath))
    {
        std::cout << "[error] cOfflineSolveIDSolver::Parseconfig retarget char path illegal: " << mRetargetCharPath << std::endl;
        exit(0);
    }

    // verify that the retarget char path is the same as the simchar skeleton path
    if(mRetargetCharPath != mSimChar->GetCharFilename())
    {
        std::cout <<"[error] cOfflineSolveIDSolver::Parseconfig retarget path " << mRetargetCharPath <<" != loaded simchar path " << mSimChar->GetCharFilename() << std::endl;
        std::cout <<"it will make the state which we will get in OfflineSolve error\n";
        exit(0);
    }

    // verify that the ref motion is the same as kinchar motion
    if(mRefMotionPath != mKinChar->GetMotion().GetMotionFile())
    {
        std::cout <<"[error] cOfflineSolveIDSolver::Parseconfig ref motion file " << mRefMotionPath <<" != loaded kinchar motion path " << mKinChar->GetMotion().GetMotionFile() << std::endl;
        std::cout <<"it will make the state which we will get in OfflineSolve error\n";
        exit(0);
    }
}


void cOfflineSolveIDSolver::OfflineSolve(std::vector<tSingleFrameIDResult> & IDResults)
{
    std::cout <<"[log] cOfflineIDSolver::OfflineSolve: begin\n";
    cTimeUtil::Begin("OfflineSolve");
    assert(mLoadInfo.mTotalFrame > 0);
    IDResults.resize(mLoadInfo.mTotalFrame);
    std::cout <<"[debug] cOfflineIDSolver::OfflineSolve: motion total frame = " << mLoadInfo.mTotalFrame << std::endl;
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
    for(int frame_id =0; frame_id < mLoadInfo.mTotalFrame - 1; frame_id++)
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

    // 2. solve ID
    // for(int cur_frame = 0; cur_frame < 5; cur_frame++)
    // {
    //     std::cout <<"frame " << cur_frame <<" link pos : " << mLoadInfo.mLinkPos[cur_frame][0].transpose() << std::endl;
    // }
    // exit(1);

    double ID_torque_err = 0, ID_action_err = 0, reward_err = 0;
    tVectorXd torque = tVectorXd::Zero(mSimChar->GetPose().size()), pd_target = tVectorXd::Zero(mSimChar->GetPose().size());
    mKinChar->SetTime(mLoadInfo.mTimesteps[0]);
    mKinChar->Update(0);
    for(int cur_frame = 1; cur_frame < mLoadInfo.mTotalFrame - 1; cur_frame++)
    {
        auto & cur_ID_res = IDResults[cur_frame];
        // if(cur_frame > 10) break;
        // std::cout <<"---------frame " << cur_frame <<"--------------\n";
        
        // 1. set the sim character pose
        mInverseModel->clearAllUserForcesAndMoments();
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(cur_frame));
        SetGeneralizedVel(mLoadInfo.mVelMat.row(cur_frame));
        mSimChar->PostUpdate(0);

        // set the kin char pose
        // mKinChar->SetTime();
        // mKinChar->Pose(mLoadInfo.mMotionRefTime[cur_frame]);
        
        


        // record state at this moment
        mCharController->RecordState(cur_ID_res.state);
        // if(cur_frame == 200)
        // {
        //     std::cout <<"offline kinchar 200 pose = " << mKinChar->GetPose().transpose() << std::endl;
        //     exit(1);
        // }
        std::vector<tVector> result;
        // SetGeneralizedInfo(mLoadInfo.mPoseMat.row(frame_id));
        /*
        
        
                cIDSolver::SolveIDSingleStep(
                    mSaveInfo.mSolvedJointForces[cur_frame], 
                    mSaveInfo.mContactForces[cur_frame], 
                    mSaveInfo.mLinkPos[cur_frame-1], 
                    mSaveInfo.mLinkRot[cur_frame-1], 
                    mSaveInfo.mBuffer_q[cur_frame-1], 
                    mSaveInfo.mBuffer_u[cur_frame-1], 
                    mSaveInfo.mBuffer_u_dot[cur_frame-1],
                    cur_frame, 
                    mSaveInfo.mExternalForces[cur_frame-1], 
                    mSaveInfo.mExternalTorques[cur_frame-1]
                );
        */
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
        // std::cout <<"done\n";
        // exit(1);

        // 4. solve PD target
        // the vector "torque" has the same shape as pose and vel
        /*
            root joint: occupy the first 7 DOF in the vector, all are set to zero
            revolute: occupy 1 DOF in the vector, the value of torque
            spherical: occupy 4 DOF in the vector, [torque_x, torque_y, torque_z, 0]
            fixed: No sapce 
        */
        
        // std::cout <<"dof = " << mSimChar->GetNumDof() << std::endl;
        // std::cout <<"pose size = " << mSimChar->GetPose().size() << std::endl;
        // std::cout <<"vel size = " << mSimChar->GetVel().size() << std::endl;
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
        

        // 5. restore action from pd target pose
        // action is different from fetched pd_target! 
        // For ball joints, their action is represented in axis angle; but their pd_Target is quaternion
        // we still need to have a convert here. pd_target = [x, y, z, w], axis angle = [angle, ax, ay, az]
        tVectorXd action = pd_target;
        mCharController->ConvertTargetPoseToActionFullsize(action);
        
        // 6. the loaded action hasn't been normalized, we need to preprocess it
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
                    std::cout <<"joint " << j << " type eRevolute, truth joint force " <<\
                        truth_action[f_cnt] <<", solved joint force = " << action[f_cnt] <<", diff = "\
                        << single_action_err << std::endl;
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
                    std::cout <<"joint " << j << " type eSpherical, truth joint force " << \
                    truth_action.segment(f_cnt, 4).transpose() <<", solved joint force = " \
                    << action.segment(f_cnt, 4).transpose() <<", diff = " << single_action_err << std::endl;
                    total_action_err += single_action_err;
                }
                f_cnt+=4;
                break;
            }
            case btMultibodyLink::eFeatherstoneJointType::eFixed:
                break;
            default:
                std::cout <<"cOfflineIDSolver::OfflineSolve joint " << j <<" type " << mMultibody->getLink(j).m_jointType << std::endl;
                exit(1);
                break;
            }
        }

        if(total_action_err > 1e-7)
        {
            std::cout <<"[error] OfflineSolveIDSolver: frame " << cur_frame <<" action err = " << total_action_err << std::endl;
            exit(1);
        }
        ID_action_err += total_action_err;

        // 2. record action into ID Result
        cur_ID_res.action = action;

        // 3. recalculate the reward according to current motion
        // you must confirm that the simchar skeleton file is the same as trajectories skeleton file accordly (Now it has been guranteed in ParseConfig)
          
        double prev_phase = mKinChar->GetPhase();
            
        mKinChar->Update(mLoadInfo.mTimesteps[cur_frame]);
        cur_ID_res.reward = mScene->CalcReward(0);

        double curr_phase = mKinChar->GetPhase();
            // 如果之前阶段比当前阶段大，代表进入新循环了
            if (curr_phase < prev_phase)
            {
                // std::cout<<"it updates! now time = " << mKinChar->GetTime() << std::endl;
                // exit(1);
                (dynamic_cast<cSceneImitate *>(mScene))->SyncKinCharNewCycleInverseDynamic(*mSimChar, *mKinChar);
            }
            
        // std::cout <<"Verbose calculated reward here\n";
        // cur_ID_res.reward = mScene->CalcReward(0);
        reward_err += std::fabs(cur_ID_res.reward - mLoadInfo.mRewards[cur_frame]);
        std::cout <<"frame " << cur_frame <<" cur reward = " << cur_ID_res.reward << ", load reward = " << mLoadInfo.mRewards[cur_frame] <<  std::endl;
        // std::cout <<"frame " << cur_frame <<" cur reward = " << cur_ID_res.reward << ", load reward = " << mLoadInfo.mRewards[cur_frame] <<  std::endl;

    }
    if(ID_torque_err < 1e-6 && ID_action_err < 1e-6)
    {
        std::cout <<"[log] cOfflineIDSolver::OfflineSolve: succ, total ID torque error = " << ID_torque_err << std::endl;
        std::cout <<"[log] cOfflineIDSolver::OfflineSolve: succ, total ID Action error = " << ID_action_err << std::endl;
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::OfflineSolve: failed, total ID torque error = " << ID_torque_err << std::endl;
        std::cout <<"[error] cOfflineIDSolver::OfflineSolve: failed, total ID Action error = " << ID_action_err << std::endl;
    }
    std::cout <<"[log] cOfflineIDSolver::OfflineSolve: total reward error = " << reward_err << std::endl;
    cTimeUtil::End("OfflineSolve");
    exit(1);
}