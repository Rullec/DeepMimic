#include "OfflineSolveIDSolver.hpp"
#include "util/cTimeUtil.hpp"
#include "../util/JsonUtil.h"
#include "../util/BulletUtil.h"
#include "sim/CtPDController.h"
#include <sim/SimCharacter.h>
#include "../util/FileUtil.h"
#include <iostream>

extern std::string controller_details_path;
cOfflineSolverIDSolver::cOfflineSolverIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, const std::string & config)
:cInteractiveIDSolver(sim_char, world, eIDSolverType::OfflineSolve)
{
    controller_details_path = "logs/controller_logs/controller_details_offlinesolve.txt";
    Parseconfig(config);
    OfflineSolve();
}

cOfflineSolverIDSolver::~cOfflineSolverIDSolver()
{

}

void cOfflineSolverIDSolver::PreSim()
{
    
}

void cOfflineSolverIDSolver::PostSim()
{
   
}

void cOfflineSolverIDSolver::Reset()
{
    std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
    exit(0);
}

void cOfflineSolverIDSolver::SetTimestep(double)
{

}

void cOfflineSolverIDSolver::Parseconfig(const std::string & conf)
{
    Json::Value root;
    cJsonUtil::ParseJson(conf, root);
    auto solve_value = root["SolveModeInfo"];
    assert(solve_value.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSolve(const Json::Value & save_value)\n";
    const Json::Value & solve_traj_path = solve_value["solve_traj_path"];
    assert(solve_traj_path.isNull() == false);
    LoadTraj(mLoadInfo, solve_traj_path.asString());
}

void cOfflineSolverIDSolver::OfflineSolve()
{
    std::cout <<"[log] cOfflineIDSolver::OfflineSolve: begin\n";
    cTimeUtil::Begin("OfflineSolve");
    assert(mLoadInfo.mTotalFrame > 0);
    std::cout <<"[debug] cOfflineIDSolver::OfflineSolve: motion total frame = " << mLoadInfo.mTotalFrame << std::endl;
    tVectorXd old_q, old_u;
    RecordGeneralizedInfo(old_q, old_u);

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
    double ID_torque_err = 0, ID_action_err = 0;
    tVectorXd torque = tVectorXd::Zero(mSimChar->GetPose().size()), pd_target = tVectorXd::Zero(mSimChar->GetPose().size());
    for(int cur_frame = 1; cur_frame < mLoadInfo.mTotalFrame - 1; cur_frame++)
    {
        // if(cur_frame > 10) break;
        // std::cout <<"---------frame " << cur_frame <<"--------------\n";
        mInverseModel->clearAllUserForcesAndMoments();
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(cur_frame));
        SetGeneralizedVel(mLoadInfo.mVelMat.row(cur_frame));
        mSimChar->PostUpdate(0);

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
        
        // 6. the loaded action hasn't been normalized, we need to preprocess it before comparing...
        tVectorXd truth_action = mLoadInfo.mActionMat.row(cur_frame);
        assert(truth_action.size() == mCharController->GetActionSize());
        f_cnt = 0;
        for(int j=0; j<mNumLinks-1; j++)
        {           
            switch (this->mMultibody->getLink(j).m_jointType)
            {
            case btMultibodyLink::eFeatherstoneJointType::eRevolute:
                f_cnt++;
                break;
            case btMultibodyLink::eFeatherstoneJointType::eSpherical:
            {
                truth_action.segment(f_cnt + 1, 3).normalize();
                if(std::fabs(truth_action[f_cnt]) <1e-10) truth_action.segment(f_cnt, 4).setZero();
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

        tVectorXd diff = (action - truth_action);
        if(diff.norm() > 1e-7)
        {
            cMathUtil::ThresholdOp(diff, 1e-6);
            std::cout <<"truth action = " << truth_action.transpose() << std::endl;
            std::cout <<"solved action = " << action.transpose() << std::endl;
            std::cout << "[log] cOFFlineSolve::OfflineSolve: solving PD target frame " << cur_frame << " error norm = " << diff.norm() << std::endl;
            std::cout << "[log] cOFFlineSolve::OfflineSolve: solving PD target frame " << cur_frame << " error = " << diff.transpose() << std::endl;
            exit(1);
        }
        ID_action_err += diff.norm();
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
    cTimeUtil::End("OfflineSolve");
    exit(1);
}
