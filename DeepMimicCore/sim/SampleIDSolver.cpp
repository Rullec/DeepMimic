#include "SampleIDSolver.hpp"
#include <scenes/SceneImitate.h>
#include "sim/CtPDController.h"
#include <sim/SimCharacter.h>
#include "../util/JsonUtil.h"
#include "../util/FileUtil.h"
#include "../util/cTimeUtil.hpp"
#ifdef __APPLE__
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif
#include <iostream>

extern std::string controller_details_path;
cSampleIDSolver::cSampleIDSolver(cSceneImitate * imitate_scene, const std::string & config)
:cInteractiveIDSolver(imitate_scene, eIDSolverType::Display)
{
    controller_details_path = "logs/controller_logs/controller_details_sample.txt";
    mEnableIDTest = false;
    mClearOldData = false;
    Parseconfig(config);

    // 1. MPI init. We need to check initialized if we call this program by python agent. Second Initialized is prohibited.
    int init_flag;
    MPI_Initialized(&init_flag);
    if(init_flag == false) MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    // clear the data dir
    if(mClearOldData == true)
    {
        cFileUtil::AddLock(mSampleInfo.mSampleTrajsDir);
        if(cFileUtil::ExistsDir(mSampleInfo.mSampleTrajsDir) == true) cFileUtil::ClearDir(mSampleInfo.mSampleTrajsDir.c_str());
        else cFileUtil::CreateDir(mSampleInfo.mSampleTrajsDir.c_str());
        cFileUtil::DeleteLock(mSampleInfo.mSampleTrajsDir);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    std::cout <<"[debug] cSampleIDSolver rank " << world_rank <<"/" << world_size <<" constructed\n";
    cTimeUtil::Begin("sample_traj");
    // MPI_Finalize();
    // exit(0);
}

cSampleIDSolver::~cSampleIDSolver()
{

}

void cSampleIDSolver::PreSim()
{
    mInverseModel->clearAllUserForcesAndMoments();
    const int & cur_frame = mSaveInfo.mCurFrameId;
    // std::cout <<"frame id " << cur_frame  << std::endl;
    // if(0 == cur_frame)
    // {
    //     mMultibody->setBaseVel(btVector3(3, -3, 3));
    //     mMultibody->setBaseOmega(btVector3(10, -10, 10));
    // }

    // clear external force
    mSaveInfo.mExternalForces[cur_frame].resize(mNumLinks);
    for(auto & x : mSaveInfo.mExternalForces[cur_frame]) x.setZero();
    mSaveInfo.mExternalTorques[cur_frame].resize(mNumLinks);
    for(auto & x : mSaveInfo.mExternalTorques[cur_frame]) x.setZero();

    RecordJointForces(mSaveInfo.mTruthJointForces[cur_frame]);
    RecordAction(mSaveInfo.mTruthAction[cur_frame]);
    RecordPDTarget(mSaveInfo.mTruthPDTarget[cur_frame]);

    // for(auto & x : mSaveInfo.mTruthJointForces[cur_frame]) std::cout << x.transpose() << std::endl;
    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);

    // double reward = mScene->CalcReward(0);
    // std::cout << "frame " << cur_frame <<" reward = " << reward << std::endl;

    // only record momentum in PreSim for the first frame
    if(0 == cur_frame)
    {
        RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame], mSaveInfo.mLinkOmega[cur_frame], mSaveInfo.mLinkVel[cur_frame]);
        RecordReward(mSaveInfo.mRewards[cur_frame]);
        RecordRefTime(mSaveInfo.mRefTime[cur_frame]);
        // CalcMomentum(mSaveInfo.mLinkPos[cur_frame],
        //     mSaveInfo.mLinkRot[cur_frame], 
        //     mSaveInfo.mLinkVel[cur_frame], 
        //     mSaveInfo.mLinkOmega[cur_frame],
        //     mSaveInfo.mLinearMomentum[cur_frame],
        //     mSaveInfo.mAngularMomentum[cur_frame]);
        // RecordMomentum(mSaveInfo.mLinearMomentum[cur_frame], mSaveInfo.mAngularMomentum[cur_frame]);
    }
    
    // if(cur_frame == 200)
    // {
    //     double now_time = mKinChar->GetTime();
    //     mKinChar->Pose(0.5);
        
    //     std::cout <<"online kinchar 0.5s pose = " << mKinChar->GetPose().transpose() << std::endl;
    //     // exit(1);
    //     std::cout <<"epoch = " << mSaveInfo.mCurEpoch << std::endl;
    //     mKinChar->Pose(now_time);
    //     // double a;
    //     // std::cin >> a;
    // }

    assert(mSaveInfo.mTimesteps[cur_frame] > 0);
    mSaveInfo.mMotion->AddFrame(mSimChar->GetPose(), mSaveInfo.mTimesteps[cur_frame]);

    mSaveInfo.mCharPoses[cur_frame] = mSimChar->GetPose();
    // if(mSaveInfo.mCurEpoch > 0) exit(1);
    // std::ofstream linea_mom_record("linear_momentum_record_2_obj.txt", std::ios::app);
    // linea_mom_record << "frame " << cur_frame <<" linear momentum = " << mSimChar->GetLinearMomentum().segment(0,3).transpose() <<", norm = " << mSimChar->GetLinearMomentum().segment(0,3).norm() << std::endl;
    
    // if(cur_frame > 10.0)
    // {
    //     std::cout <<"frame = 10, begin test\n";
    //     SetGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame]);
    // }
    // std::cout <<"offline sovler presim record " << cur_frame << std::endl;
    // std::ofstream fout("test2.txt", std::ios::app);
    // fout <<"presim frame id = " << cur_frame;
    // fout << "\n truth joint forces: ";
    // for(auto & x : mSaveInfo.mTruthJointForces[cur_frame]) fout << x.transpose() <<" ";
    // fout << "\n buffer q : ";
    // fout << mSaveInfo.mBuffer_q[cur_frame].transpose() <<" ";
    // fout << "\n buffer u : ";
    // fout << mSaveInfo.mBuffer_u[cur_frame].transpose() <<" ";
    // fout << std::endl;
}

void cSampleIDSolver::PostSim()
{
    // if(mSaveInfo.mCurFrameId > 2) exit(1);
    // std::cout <<"offline post sim frame = " << mSaveInfo.mCurFrameId<<std::endl;
    mSaveInfo.mCurFrameId++;
    const int cur_frame = mSaveInfo.mCurFrameId;

    // record the post generalized info
    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);

    // q dot dot
    if(cur_frame >=2)
    {
        mSaveInfo.mBuffer_u_dot[cur_frame - 1] = (mSaveInfo.mBuffer_u[cur_frame] - mSaveInfo.mBuffer_u[cur_frame - 1]) / mSaveInfo.mTimesteps[cur_frame -1];
    }

    // record contact forces
    RecordContactForces(mSaveInfo.mContactForces[cur_frame-1], mSaveInfo.mTimesteps[cur_frame - 1], mWorldId2InverseId);

    // record multibody info
    RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame], mSaveInfo.mLinkOmega[cur_frame], mSaveInfo.mLinkVel[cur_frame]);

    // record rewards for this frame
    RecordReward(mSaveInfo.mRewards[cur_frame]);
    
    // record reference time now
    RecordRefTime(mSaveInfo.mRefTime[cur_frame]);

    // character pose
    mSaveInfo.mCharPoses[cur_frame] = mSimChar->GetPose();

    if(mEnableIDTest == false)
    {
        return ;
    } 

    // calculate vel and omega from discretion
    CalcDiscreteVelAndOmega(
        mSaveInfo.mLinkPos[cur_frame - 1],
        mSaveInfo.mLinkRot[cur_frame - 1],
        mSaveInfo.mLinkPos[cur_frame],
        mSaveInfo.mLinkRot[cur_frame],
        mSaveInfo.mTimesteps[cur_frame -1],
        mSaveInfo.mLinkDiscretVel[cur_frame],
        mSaveInfo.mLinkDiscretOmega[cur_frame]
        );
    double total_err = 0.0;
    for(int i=0; i<mNumLinks; i++)
    {
        total_err += (mSaveInfo.mLinkDiscretVel[cur_frame][i] - mSaveInfo.mLinkVel[cur_frame][i]).norm();
        total_err += (mSaveInfo.mLinkDiscretOmega[cur_frame][i] - mSaveInfo.mLinkOmega[cur_frame][i]).norm();
        // std::cout <<"frame " << cur_frame <<" link " << i << " vel diff = " << \
        // (mSaveInfo.mLinkDiscretVel[cur_frame][i] - mSaveInfo.mLinkVel[cur_frame][i]).transpose() << std::endl;;
        // std::cout <<"frame " << cur_frame <<" link " << i << " omega diff = " << \
        // (mSaveInfo.mLinkDiscretOmega[cur_frame][i] - mSaveInfo.mLinkOmega[cur_frame][i]).transpose() << std::endl;;
    }
    // if(std::rand() % 100 < 1) std::cout << "[debug] SampleIDSolver PostSim: discrete total err = " << total_err << std::endl; 

    // exit(1);
    // calculate momentum by these discreted values
    CalcMomentum(mSaveInfo.mLinkPos[cur_frame-1],
        mSaveInfo.mLinkRot[cur_frame-1],
        mSaveInfo.mLinkDiscretVel[cur_frame],
        mSaveInfo.mLinkDiscretOmega[cur_frame],
        mSaveInfo.mLinearMomentum[cur_frame - 1],
        mSaveInfo.mAngularMomentum[cur_frame - 1]);

    // calculate the relative generalized velocity, take care of the idx... 
    if (cur_frame >= 2)
    {
        double cur_timestep = mSaveInfo.mTimesteps[cur_frame - 1],
                last_timestep = mSaveInfo.mTimesteps[cur_frame - 2];
        tVectorXd old_vel_after = mSaveInfo.mBuffer_u[cur_frame];
        tVectorXd old_vel_before = mSaveInfo.mBuffer_u[cur_frame - 1];
        tVectorXd old_accel = (old_vel_after - old_vel_before) / cur_timestep;
        mSaveInfo.mBuffer_u[cur_frame - 1] = CalcGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 2], mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
        mSaveInfo.mBuffer_u[cur_frame] = CalcGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1], mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
        mSaveInfo.mBuffer_u_dot[cur_frame - 1] = (mSaveInfo.mBuffer_u[cur_frame] - mSaveInfo.mBuffer_u[cur_frame - 1]) / cur_timestep;
        // std::ofstream fout("test2.txt", std::ios::app);
        // fout <<"offline buffer u dot calc: \n";
        // fout <<"buffer u " << cur_frame -1 <<" = " << mSaveInfo.mBuffer_u[cur_frame - 1].transpose()<< std::endl;
        // fout <<"buffer u " << cur_frame <<" = " << mSaveInfo.mBuffer_u[cur_frame].transpose()<< std::endl;
        // fout <<"buffer u dot " << cur_frame-1 <<" = " << mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose()<< std::endl;

        // test discrete vel integration
        // {
        //     assert(mSimChar->GetNumBodyParts() == mNumLinks - 1);
        //     for(int idx = 0; idx< mNumLinks; idx++)
        //     {   
        //         // mSimChar->GetBodyPart(0)->GetLinearVelocity
        //         tVector cur_vel;
        //         if(idx==0) cur_vel = mSimChar->GetRootVel();
        //         else cur_vel= mSimChar->GetBodyPartVel(idx - 1);
        //         tVector pred_vel = (mSaveInfo.mLinkPos[cur_frame][idx] - mSaveInfo.mLinkPos[cur_frame-1][idx])/cur_timestep;
        //         double diff = (cur_vel - pred_vel).norm();
        //         if(diff > 1e-5)
        //         {
        //             std::cout <<"frame " << cur_frame <<" body " << idx <<" pred vel = " << pred_vel.transpose() <<", tru vel = " << cur_vel.transpose() << std::endl;
        //         }
                
        //     } 
        // }

        double threshold = 1e-5;
        {
            tVectorXd diff = old_vel_after - mSaveInfo.mBuffer_u[cur_frame];
            if (diff.norm() > threshold)
            {
                std::cout << "truth vel = " << old_vel_after.transpose() << std::endl;
                std::cout << "calculated vel = " << mSaveInfo.mBuffer_u[cur_frame].transpose() << std::endl;
                std::cout << "calculate vel after error = " << (diff).transpose() << " | " << (old_vel_after - mSaveInfo.mBuffer_u[cur_frame]).norm() << std::endl;
                //exit(1);
            }

            // check vel
            diff = old_vel_before - mSaveInfo.mBuffer_u[cur_frame - 1];
            if (diff.norm() > threshold)
            {
                std::cout << "truth vel = " << old_vel_after.transpose() << std::endl;
                std::cout << "calculated vel = " << mSaveInfo.mBuffer_u[cur_frame].transpose() << std::endl;
                std::cout << "calculate vel before error = " << (diff).transpose() << " | " << (old_vel_after - mSaveInfo.mBuffer_u[cur_frame]).norm() << std::endl;
                //exit(1);
            }
            
            // check accel
            diff = mSaveInfo.mBuffer_u_dot[cur_frame - 1] - old_accel;
            if (diff.norm() > threshold)
            {
                std::cout << "truth accel =  " << old_accel.transpose() << std::endl;
                std::cout << "calc accel =  " << mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose() << std::endl;
                std::cout << "solved error = " << diff.transpose() << std::endl;
                //exit(1);
            }
        }

        // solve Inverse Dynamics
        {
            // std::cout <<"offline sovler post record " << cur_frame << std::endl;
            // std::ofstream fout("test2.txt", std::ios::app);
            // fout <<"post sim frame id = " << cur_frame;
            // fout << "\n contact forces: ";
            // for(auto & x : mSaveInfo.mContactForces[cur_frame-1]) fout << x.mForce.transpose() <<" ";
            // fout << "\n buffer q : ";
            // fout << mSaveInfo.mBuffer_q[cur_frame].transpose() <<" ";
            // fout << "\n buffer u : ";
            // fout << mSaveInfo.mBuffer_u[cur_frame].transpose() <<" ";
            // fout << std::endl;

            cIDSolver::SolveIDSingleStep(
                mSaveInfo.mSolvedJointForces[cur_frame], 
                mSaveInfo.mContactForces[cur_frame-1], 
                mSaveInfo.mLinkPos[cur_frame-1], 
                mSaveInfo.mLinkRot[cur_frame-1], 
                mSaveInfo.mBuffer_q[cur_frame-1], 
                mSaveInfo.mBuffer_u[cur_frame-1], 
                mSaveInfo.mBuffer_u_dot[cur_frame-1],
                cur_frame, 
                mSaveInfo.mExternalForces[cur_frame-1], 
                mSaveInfo.mExternalTorques[cur_frame-1]
            );
            // exit(1);
            // std::cout <<"offline sovler ID record " << cur_frame << std::endl;
            // fout <<"ID frame id = " << cur_frame;
            // fout << "\n solved forces: ";
            // for(auto & x : mSaveInfo.mSolvedJointForces[cur_frame]) fout << x.transpose() <<" ";
            // fout << "\n buffer u_dot : ";
            // fout << mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose() << " ";
            // fout <<"\n link pos : ";
            // for(auto & x : mSaveInfo.mLinkPos[cur_frame-1]) fout << x.transpose() <<" ";
            // fout <<"\n link rot : ";
            // for(auto & x : mSaveInfo.mLinkRot[cur_frame-1]) fout << x.transpose() <<" ";
            // fout << std::endl;
            // exit(1);
            // check the solved result
            {
                // std::cout <<"Postsim: offline after ID: check solved result\n";
                assert(mSaveInfo.mSolvedJointForces[cur_frame].size() == mSaveInfo.mTruthJointForces[cur_frame-1].size());
                double err = 0;
                for(int id = 0; id< mSaveInfo.mSolvedJointForces[cur_frame].size(); id++)
                {
                    if(cMathUtil::IsSame(mSaveInfo.mSolvedJointForces[cur_frame][id], mSaveInfo.mTruthJointForces[cur_frame-1][id], 1e-5) == false)
                    {
                        err += (mSaveInfo.mSolvedJointForces[cur_frame][id]- mSaveInfo.mTruthJointForces[cur_frame-1][id]).norm();
                        std::cout <<"[error] SampleIDSolver: ID solved error for joint " << id <<" diff : solved = " << mSaveInfo.mSolvedJointForces[cur_frame][id].transpose() <<", truth = " << mSaveInfo.mTruthJointForces[cur_frame-1][id].transpose() << std::endl;
                    }
                }
                // if(err < 1e-5) std::cout << "[log] frame " << cur_frame <<" offline ID solved accurately\n";
                // else std::cout << "[error] frame " << cur_frame <<" offline ID solved error\n";
                if(err > 1e-5) 
                {
                    std::cout <<"[error] SampleIDSolver: ID solved error for frame " << cur_frame << \
                        ", err = " << err << std::endl;
                    // exit(0);
                }
            }
        }
    }
}

void cSampleIDSolver::Reset()
{
    tSummaryTable::tSingleEpochInfo a;
    a.length_second = mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId-1] * mSaveInfo.mCurFrameId;
    a.frame_num = mSaveInfo.mCurFrameId;
    a.traj_filename = SaveTraj(mSaveInfo, mSampleInfo.mSampleTrajsRootName);

    mSummaryTable.mTotalEpochNum++;
    mSummaryTable.mTotalLengthTime += a.length_second;
    mSummaryTable.mTotalLengthFrame += a.frame_num;
    mSummaryTable.mEpochInfos.push_back(a);
    mSaveInfo.mMotion->Clear();

    // clear frame id
    mSaveInfo.mCurEpoch++;
    mSaveInfo.mCurFrameId = 0;

    // judge terminate
    if(mSummaryTable.mTotalEpochNum >= mSampleInfo.mSampleEpoches)
    {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        std::cout <<"[debug] cSampleIDSolver rank " << world_rank <<"/" << world_size <<" sampled done, res num = " << mSummaryTable.mTotalEpochNum << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        // std::cout <<"[log] cSampleIDSolver::Reset sample finished, begin to write summary file\n";
        mSummaryTable.WriteToDisk(mSampleInfo.mSummaryTableFilename);
        // std::cout <<"[log] cSampleIDSolver::Reset write summary file done\n";
        MPI_Finalize();
        cTimeUtil::End("sample_traj");
        exit(0);
    }
}

void cSampleIDSolver::SetTimestep(double timestep)
{
    mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = timestep;
}

void cSampleIDSolver::Parseconfig(const std::string & conf)
{
    Json::Value root;
    cJsonUtil::ParseJson(conf, root);
    auto sample_value = root["SampleModeInfo"];
    assert(sample_value.isNull() == false);
    auto & sample_num_json = sample_value["sample_num"];
    auto & clear_old_data_json = sample_value["clear_old_data"];
    auto & sample_trajs_dir_json = sample_value["sample_trajs_dir"];
    auto & sample_root_json = sample_value["sample_trajs_rootname"];
    auto & summary_table_file = sample_value["summary_table_filename"];
    assert(sample_num_json.isNull() == false);
    assert(sample_trajs_dir_json.isNull() == false);
    assert(sample_root_json.isNull() == false);
    assert(summary_table_file.isNull() == false);

    mSampleInfo.mSampleEpoches = sample_num_json.asInt();
    mSampleInfo.mSampleTrajsDir = sample_trajs_dir_json.asString();
    mSampleInfo.mSampleTrajsRootName = mSampleInfo.mSampleTrajsDir + sample_root_json.asString();
    mSampleInfo.mSummaryTableFilename = mSampleInfo.mSampleTrajsDir + summary_table_file.asString();
    if(sample_value["enable_sample_ID_test"].isNull() == false) mEnableIDTest = sample_value["enable_sample_ID_test"].asBool();
    if(sample_value["clear_old_data_json"].isNull() == false) mClearOldData = sample_value["clear_old_data_json"].asBool();
    // assert(cFileUtil::ValidateFilePath(mSampleInfo.mSampleTrajsRootName));
    // assert(cFileUtil::ValidateFilePath(mSampleInfo.mSummaryTableFilename));
    
    std::cout <<"[log] Inverse Dynamic plugin running in sample mode\n";
    mSaveInfo.mMotion = new cMotion();
    PrintSampleInfo();
    InitSampleSummaryTable();

    // remove the sample trajs dir
    // cFileUtil::AddLock("lock.lock");
    // if(cFileUtil::ExistsDir(mSampleInfo.mSampleTrajsDir) == true) cFileUtil::ClearDir(mSampleInfo.mSampleTrajsDir.c_str());
    // else cFileUtil::CreateDir(mSampleInfo.mSampleTrajsDir.c_str());
    // cFileUtil::DeleteLock("lock.lock");
}

/**
 * \brief               Init the summary table format when ID solver working in "sample" mode
 *  We need to record:
 *  1. current skeleton info
 *  2. current PD control info
 *  3. current ref motion file
*/
void cSampleIDSolver::InitSampleSummaryTable()
{
    mSummaryTable.mSampleCharFile = mSimChar->GetCharFilename();
    mSummaryTable.mSampleControllerFile = mCharController->GetControllerFile();
    mSummaryTable.mTotalEpochNum = 0;
    mSummaryTable.mTotalLengthTime = 0;
    mSummaryTable.mTotalLengthFrame = 0;
    mSummaryTable.mEpochInfos.clear();
    mSummaryTable.mTimeStamp = cTimeUtil::GetSystemTime();
    std::cout <<"[log] cSampleIDSolver::InitSampleSummaryTable: set timestamp = " << mSummaryTable.mTimeStamp << std::endl;
}

void cSampleIDSolver::PrintSampleInfo()
{
    std::cout <<"[log] sample epoches = " << mSampleInfo.mSampleEpoches << std::endl;
    std::cout <<"[log] sample trajs dir = " << mSampleInfo.mSampleTrajsDir << std::endl;
    std::cout <<"[log] sample root name  = " << mSampleInfo.mSampleTrajsRootName << std::endl;
    std::cout <<"[log] sample summart table filename  = " << mSampleInfo.mSummaryTableFilename << std::endl;
}
