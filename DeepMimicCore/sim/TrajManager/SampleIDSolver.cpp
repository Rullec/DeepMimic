#include "SampleIDSolver.h"
#include "scenes/SceneImitate.h"
#include "sim/Controller/CtPDController.h"
#include "sim/SimItems/SimCharacter.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include "util/TimeUtil.hpp"
// #ifdef __APPLE__
// #include <mpi.h>
// #else
// #include <mpi/mpi.h>
// #endif
#include "util/MPIUtil.h"
#include <iostream>
// #define SHOW_SAMPLE_COST

// extern std::string controller_details_path;
cSampleIDSolver::cSampleIDSolver(cSceneImitate *imitate_scene,
                                 const std::string &config)
    : cIDSolver(imitate_scene, eIDSolverType::Display)
{
    // controller_details_path =
    //     "logs/controller_logs/controller_details_sample.txt";
    mEnableIDTest = false;
    mClearOldData = false;
    // mRecordThetaDist = false;
    // mEnableSyncThetaDist = false;
    Parseconfig(config);

    // 1. MPI init. We need to check initialized if we call this program \
    // by python agent. Second Initialized is prohibited
    int init_flag = cMPIUtil::IsInited();

    if (init_flag == false)
        cMPIUtil::InitMPI();

    int world_size = cMPIUtil::GetCommSize();

    int world_rank = cMPIUtil::GetWorldRank();

    // clear the data dir
    if (mClearOldData == true)
    {
        cFileUtil::AddLock(mSampleInfo.mSampleTrajsDir);
        if (cFileUtil::ExistsDir(mSampleInfo.mSampleTrajsDir) == true)
            cFileUtil::ClearDir(mSampleInfo.mSampleTrajsDir.c_str());
        else
            cFileUtil::CreateDir(mSampleInfo.mSampleTrajsDir.c_str());
        cFileUtil::DeleteLock(mSampleInfo.mSampleTrajsDir);
        cMPIUtil::SetBarrier();
    }

    cTimeUtil::Begin("sample_traj");
    // MPI_Finalize();
    // exit(0);
}

cSampleIDSolver::~cSampleIDSolver() { delete mSaveInfo.mMotion; }

void cSampleIDSolver::PreSim()
{
#ifdef SHOW_SAMPLE_COST
    cTimeUtil::BeginLazy("sample_one_epoch");
#endif

    ClearID();
    const int &cur_frame = mSaveInfo.mCurFrameId;
    // if(0 == cur_frame)
    // {
    //     mMultibody->setBaseVel(btVector3(3, -3, 3));
    //     mMultibody->setBaseOmega(btVector3(10, -10, 10));
    // }

    // clear external force
    mSaveInfo.mExternalForces[cur_frame].resize(mNumLinks);
    for (auto &x : mSaveInfo.mExternalForces[cur_frame])
        x.setZero();
    mSaveInfo.mExternalTorques[cur_frame].resize(mNumLinks);
    for (auto &x : mSaveInfo.mExternalTorques[cur_frame])
        x.setZero();

    RecordJointForces(mSaveInfo.mTruthJointForces[cur_frame]);
    RecordAction(mSaveInfo.mTruthAction[cur_frame]);
    // if (mRecordThetaDist == true)
    // {
    //     RecordActionThetaDist(mSaveInfo.mTruthAction[cur_frame],
    //                           mCharController->GetPhase(), mActionThetaDist);
    // }
    RecordPDTarget(mSaveInfo.mTruthPDTarget[cur_frame]);

    // for(auto & x : mSaveInfo.mTruthJointForces[cur_frame]) std::cout <<
    // x.transpose() << std::endl;
    RecordGeneralizedInfo(mSimChar, mSaveInfo.mBuffer_q[cur_frame],
                          mSaveInfo.mBuffer_u[cur_frame]);

    // double reward = mScene->CalcReward(0);
    // std::cout << "frame " << cur_frame <<" reward = " << reward << std::endl;

    // only record momentum in PreSim for the first frame
    if (0 == cur_frame)
    {
        RecordMultibodyInfo(mSimChar, mSaveInfo.mLinkRot[cur_frame],
                            mSaveInfo.mLinkPos[cur_frame],
                            mSaveInfo.mLinkOmega[cur_frame],
                            mSaveInfo.mLinkVel[cur_frame]);
        RecordReward(mSaveInfo.mRewards[cur_frame]);
        RecordRefTime(mSaveInfo.mRefTime[cur_frame]);
        // CalcMomentum(mSaveInfo.mLinkPos[cur_frame],
        //     mSaveInfo.mLinkRot[cur_frame],
        //     mSaveInfo.mLinkVel[cur_frame],
        //     mSaveInfo.mLinkOmega[cur_frame],
        //     mSaveInfo.mLinearMomentum[cur_frame],
        //     mSaveInfo.mAngularMomentum[cur_frame]);
        // RecordMomentum(mSaveInfo.mLinearMomentum[cur_frame],
        // mSaveInfo.mAngularMomentum[cur_frame]);
    }

    // if(cur_frame == 200)
    // {
    //     double now_time = mKinChar->GetTime();
    //     mKinChar->Pose(0.5);

    //     std::cout <<"online kinchar 0.5s pose = " <<
    //     mKinChar->GetPose().transpose() << std::endl;
    //     // exit(1);
    //     std::cout <<"epoch = " << mSaveInfo.mCurEpoch << std::endl;
    //     mKinChar->Pose(now_time);
    //     // double a;
    //     // std::cin >> a;
    // }

    assert(mSaveInfo.mTimesteps[cur_frame] > 0);
    mSaveInfo.mMotion->AddFrame(mSimChar->GetPose(),
                                mSaveInfo.mTimesteps[cur_frame]);

    mSaveInfo.mCharPoses[cur_frame] = mSimChar->GetPose();
    // if(mSaveInfo.mCurEpoch > 0) exit(1);
    // std::ofstream linea_mom_record("linear_momentum_record_2_obj.txt",
    // std::ios::app); linea_mom_record << "frame " << cur_frame <<" linear
    // momentum = " << mSimChar->GetLinearMomentum().segment(0,3).transpose()
    // <<", norm = " << mSimChar->GetLinearMomentum().segment(0,3).norm() <<
    // std::endl;

    // if(cur_frame > 10.0)
    // {
    //     std::cout <<"frame = 10, begin test\n";
    //     SetGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame]);
    // }
    // std::cout <<"offline sovler presim record " << cur_frame << std::endl;
    // std::ofstream fout("test2.txt", std::ios::app);
    // fout <<"presim frame id = " << cur_frame;
    // fout << "\n truth joint forces: ";
    // for(auto & x : mSaveInfo.mTruthJointForces[cur_frame]) fout <<
    // x.transpose() <<" "; fout << "\n buffer q : "; fout <<
    // mSaveInfo.mBuffer_q[cur_frame].transpose() <<" "; fout << "\n buffer u :
    // "; fout << mSaveInfo.mBuffer_u[cur_frame].transpose() <<" "; fout <<
    // std::endl;
    // std::cout << "------------frame " << cur_frame << " begin----------\n";
}

void cSampleIDSolver::PostSim()
{
    // if(mSaveInfo.mCurFrameId > 2) exit(1);
    // std::cout <<"offline post sim frame = " <<
    // mSaveInfo.mCurFrameId<<std::endl;
    mSaveInfo.mCurFrameId++;
    const int cur_frame = mSaveInfo.mCurFrameId;
    // std::cout << "Pose1 = " << mSimChar->GetPose().transpose() << std::endl;
    // record the post generalized info
    RecordGeneralizedInfo(mSimChar, mSaveInfo.mBuffer_q[cur_frame],
                          mSaveInfo.mBuffer_u[cur_frame]);
    // std::cout << "Pose2 = " << mSimChar->GetPose().transpose() << std::endl;
    // q dot dot
    if (cur_frame >= 2)
    {
        mSaveInfo.mBuffer_u_dot[cur_frame - 1] =
            (mSaveInfo.mBuffer_u[cur_frame] -
             mSaveInfo.mBuffer_u[cur_frame - 1]) /
            mSaveInfo.mTimesteps[cur_frame - 1];
    }
    // std::cout << "Pose3 = " << mSimChar->GetPose().transpose() << std::endl;
    // record contact forces
    RecordContactForces(mSaveInfo.mContactForces[cur_frame - 1],
                        mSaveInfo.mTimesteps[cur_frame - 1]);

    // begin to output
    {
        // for (auto &x : mSaveInfo.mContactForces[cur_frame - 1])
        // {
        //     std::cout << x.mForce.transpose() << std::endl;
        // }
        // auto gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
        // if (gen_char != nullptr)
        // {
        //     std::cout << "q = " << mSaveInfo.mBuffer_q[cur_frame - 1].norm()
        //               << std::endl;
        //     std::cout << "qdot = " << mSaveInfo.mBuffer_u[cur_frame - 1].norm()
        //               << std::endl;
        // }
    }

    // std::cout << "Pose4 = " << mSimChar->GetPose().transpose() << std::endl;
    // record multibody info
    RecordMultibodyInfo(
        mSimChar, mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame],
        mSaveInfo.mLinkOmega[cur_frame], mSaveInfo.mLinkVel[cur_frame]);
    // std::cout << "Pose5 = " << mSimChar->GetPose().transpose() << std::endl;
    // record rewards for this frame
    RecordReward(mSaveInfo.mRewards[cur_frame]);
    // std::cout << "Pose6 = " << mSimChar->GetPose().transpose() << std::endl;
    // record reference time now
    RecordRefTime(mSaveInfo.mRefTime[cur_frame]);
    // std::cout << "Pose7 = " << mSimChar->GetPose().transpose() << std::endl;
    // character pose
    tVectorXd pose = mSimChar->GetPose();
    // std::cout << "Pose8 = " << mSimChar->GetPose().transpose() << std::endl;
    // MIMIC_INFO("frame {} pose {} ", cur_frame,
    // mSimChar->GetPose().transpose());
    mSaveInfo.mCharPoses[cur_frame] = mSimChar->GetPose();

    if (mEnableIDTest == false)
    {
#ifdef SHOW_SAMPLE_COST
        cTimeUtil::EndLazy("sample_one_epoch");
#endif
        return;
    }

    // calculate vel and omega from discretion
    CalcDiscreteVelAndOmega(
        mSaveInfo.mLinkPos[cur_frame - 1], mSaveInfo.mLinkRot[cur_frame - 1],
        mSaveInfo.mLinkPos[cur_frame], mSaveInfo.mLinkRot[cur_frame],
        mSaveInfo.mTimesteps[cur_frame - 1],
        mSaveInfo.mLinkDiscretVel[cur_frame],
        mSaveInfo.mLinkDiscretOmega[cur_frame]);
    double total_err = 0.0;
    for (int i = 0; i < mNumLinks; i++)
    {
        total_err += (mSaveInfo.mLinkDiscretVel[cur_frame][i] -
                      mSaveInfo.mLinkVel[cur_frame][i])
                         .norm();
        total_err += (mSaveInfo.mLinkDiscretOmega[cur_frame][i] -
                      mSaveInfo.mLinkOmega[cur_frame][i])
                         .norm();
        // std::cout <<"frame " << cur_frame <<" link " << i << " vel diff = " << \
        // (mSaveInfo.mLinkDiscretVel[cur_frame][i] - mSaveInfo.mLinkVel[cur_frame][i]).transpose() << std::endl;;
        // std::cout <<"frame " << cur_frame <<" link " << i << " omega diff = " << \
        // (mSaveInfo.mLinkDiscretOmega[cur_frame][i] - mSaveInfo.mLinkOmega[cur_frame][i]).transpose() << std::endl;;
    }
    // if(std::rand() % 100 < 1) std::cout << "[debug] SampleIDSolver PostSim:
    // discrete total err = " << total_err << std::endl;

    // exit(1);
    // calculate momentum by these discreted values
    CalcMomentum(mSaveInfo.mLinkPos[cur_frame - 1],
                 mSaveInfo.mLinkRot[cur_frame - 1],
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
        mSaveInfo.mBuffer_u[cur_frame - 1] = CalcGeneralizedVel(
            mSaveInfo.mBuffer_q[cur_frame - 2],
            mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
        mSaveInfo.mBuffer_u[cur_frame] =
            CalcGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1],
                               mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
        mSaveInfo.mBuffer_u_dot[cur_frame - 1] =
            (mSaveInfo.mBuffer_u[cur_frame] -
             mSaveInfo.mBuffer_u[cur_frame - 1]) /
            cur_timestep;
        // std::ofstream fout("test2.txt", std::ios::app);
        // fout <<"offline buffer u dot calc: \n";
        // fout <<"buffer u " << cur_frame -1 <<" = " <<
        // mSaveInfo.mBuffer_u[cur_frame - 1].transpose()<< std::endl; fout
        // <<"buffer u " << cur_frame <<" = " <<
        // mSaveInfo.mBuffer_u[cur_frame].transpose()<< std::endl; fout
        // <<"buffer u dot " << cur_frame-1 <<" = " <<
        // mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose()<< std::endl;

        // test discrete vel integration
        // {
        //     assert(mSimChar->GetNumBodyParts() == mNumLinks - 1);
        //     for(int idx = 0; idx< mNumLinks; idx++)
        //     {
        //         // mSimChar->GetBodyPart(0)->GetLinearVelocity
        //         tVector cur_vel;
        //         if(idx==0) cur_vel = mSimChar->GetRootVel();
        //         else cur_vel= mSimChar->GetBodyPartVel(idx - 1);
        //         tVector pred_vel = (mSaveInfo.mLinkPos[cur_frame][idx] -
        //         mSaveInfo.mLinkPos[cur_frame-1][idx])/cur_timestep; double
        //         diff = (cur_vel - pred_vel).norm(); if(diff > 1e-5)
        //         {
        //             std::cout <<"frame " << cur_frame <<" body " << idx <<"
        //             pred vel = " << pred_vel.transpose() <<", tru vel = " <<
        //             cur_vel.transpose() << std::endl;
        //         }

        //     }
        // }

        double threshold = 1e-5;
        {
            tVectorXd diff = old_vel_after - mSaveInfo.mBuffer_u[cur_frame];
            if (diff.norm() > threshold)
            {
                std::cout << "truth vel = " << old_vel_after.transpose()
                          << std::endl;
                std::cout << "calculated vel = "
                          << mSaveInfo.mBuffer_u[cur_frame].transpose()
                          << std::endl;
                std::cout
                    << "calculate vel after error = " << (diff).transpose()
                    << " | "
                    << (old_vel_after - mSaveInfo.mBuffer_u[cur_frame]).norm()
                    << std::endl;
                // exit(1);
            }

            // check vel
            diff = old_vel_before - mSaveInfo.mBuffer_u[cur_frame - 1];
            if (diff.norm() > threshold)
            {
                std::cout << "truth vel = " << old_vel_after.transpose()
                          << std::endl;
                std::cout << "calculated vel = "
                          << mSaveInfo.mBuffer_u[cur_frame].transpose()
                          << std::endl;
                std::cout
                    << "calculate vel before error = " << (diff).transpose()
                    << " | "
                    << (old_vel_after - mSaveInfo.mBuffer_u[cur_frame]).norm()
                    << std::endl;
                // exit(1);
            }

            // check accel
            diff = mSaveInfo.mBuffer_u_dot[cur_frame - 1] - old_accel;
            if (diff.norm() > threshold)
            {
                std::cout << "truth accel =  " << old_accel.transpose()
                          << std::endl;
                std::cout << "calc accel =  "
                          << mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose()
                          << std::endl;
                std::cout << "solved error = " << diff.transpose() << std::endl;
                // exit(1);
            }
        }

        // solve Inverse Dynamics
        {
            // std::cout <<"offline sovler post record " << cur_frame <<
            // std::endl; std::ofstream fout("test2.txt", std::ios::app); fout
            // <<"post sim frame id = " << cur_frame; fout << "\n contact
            // forces: "; for(auto & x : mSaveInfo.mContactForces[cur_frame-1])
            // fout << x.mForce.transpose() <<" "; fout << "\n buffer q : ";
            // fout << mSaveInfo.mBuffer_q[cur_frame].transpose() <<" ";
            // fout << "\n buffer u : ";
            // fout << mSaveInfo.mBuffer_u[cur_frame].transpose() <<" ";
            // fout << std::endl;

            cIDSolver::SolveIDSingleStep(
                mSaveInfo.mSolvedJointForces[cur_frame - 1],
                mSaveInfo.mContactForces[cur_frame - 1],
                mSaveInfo.mLinkPos[cur_frame - 1],
                mSaveInfo.mLinkRot[cur_frame - 1],
                mSaveInfo.mBuffer_q[cur_frame - 1],
                mSaveInfo.mBuffer_u[cur_frame - 1],
                mSaveInfo.mBuffer_u_dot[cur_frame - 1], cur_frame,
                mSaveInfo.mExternalForces[cur_frame - 1],
                mSaveInfo.mExternalTorques[cur_frame - 1]);
            // exit(1);
            // std::cout <<"offline sovler ID record " << cur_frame <<
            // std::endl; fout <<"ID frame id = " << cur_frame; fout << "\n
            // solved forces: "; for(auto & x :
            // mSaveInfo.mSolvedJointForces[cur_frame-1]) fout << x.transpose()
            // <<" "; fout << "\n buffer u_dot : "; fout <<
            // mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose() << " "; fout
            // <<"\n link pos : "; for(auto & x :
            // mSaveInfo.mLinkPos[cur_frame-1]) fout << x.transpose() <<" ";
            // fout <<"\n link rot : ";
            // for(auto & x : mSaveInfo.mLinkRot[cur_frame-1]) fout <<
            // x.transpose() <<" "; fout << std::endl; exit(1); check the solved
            // result
            {
                // std::cout <<"Postsim: offline after ID: check solved
                // result\n";
                assert(mSaveInfo.mSolvedJointForces[cur_frame - 1].size() ==
                       mSaveInfo.mTruthJointForces[cur_frame - 1].size());
                double err = 0;
                for (int id = 0;
                     id < mSaveInfo.mSolvedJointForces[cur_frame - 1].size();
                     id++)
                {
                    if (cMathUtil::IsSame(
                            mSaveInfo.mSolvedJointForces[cur_frame - 1][id],
                            mSaveInfo.mTruthJointForces[cur_frame - 1][id],
                            1e-5) == false)
                    {
                        err +=
                            (mSaveInfo.mSolvedJointForces[cur_frame - 1][id] -
                             mSaveInfo.mTruthJointForces[cur_frame - 1][id])
                                .norm();
                        std::cout
                            << "[error] SampleIDSolver: ID solved error for "
                               "joint "
                            << id << " diff : solved = "
                            << mSaveInfo.mSolvedJointForces[cur_frame - 1][id]
                                   .transpose()
                            << ", truth = "
                            << mSaveInfo.mTruthJointForces[cur_frame - 1][id]
                                   .transpose()
                            << std::endl;
                    }
                }
                // if(err < 1e-5) std::cout << "[log] frame " << cur_frame <<"
                // offline ID solved accurately\n"; else std::cout << "[error]
                // frame " << cur_frame <<" offline ID solved error\n";
                if (err > 1e-5)
                {
                    std::cout
                        << "[error] SampleIDSolver: ID solved error for frame "
                        << cur_frame << ", err = " << err << std::endl;
                    // exit(0);
                }
            }
        }
    }
}

void cSampleIDSolver::Reset()
{
    tSummaryTable::tSingleEpochInfo a;
    a.length_second =
        mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId - 1] * mSaveInfo.mCurFrameId;
    a.frame_num = mSaveInfo.mCurFrameId;
    mSaveInfo.mIntegrationScheme = GetIntegrationSchemeWorld();
    a.sample_traj_filename = mSaveInfo.SaveTraj(
        mSampleInfo.mSampleTrajsDir, mSampleInfo.mSampleTrajsRootName,
        mSampleInfo.mSampleTrajVer);
    MIMIC_INFO("Sampling: save trajs to {} succ, {}/{}", a.sample_traj_filename,
               mSummaryTable.mTotalEpochNum + 1, mSampleInfo.mSampleEpoches);

    // // test code
    // {
    //     Json::Value v1_root = SaveTrajV1(mSaveInfo),
    //         v2_root = SaveTrajV2(mSaveInfo);

    //     a.sample_traj_filename = "v2.json";
    //     cJsonUtil::WriteJson("v1.json", v1_root);
    //     cJsonUtil::WriteJson("v2.json", v2_root);
    //     // exit(0);
    // }

    mSummaryTable.mTotalEpochNum++;
    // mSummaryTable.mTotalLengthTime += a.length_second;
    // mSummaryTable.mTotalLengthFrame += a.frame_num;
    mSummaryTable.mEpochInfos.push_back(a);
    mSaveInfo.mMotion->Clear();

    // clear frame id
    mSaveInfo.mCurEpoch++;
    mSaveInfo.mCurFrameId = 0;

#ifdef SHOW_SAMPLE_COST
    cTimeUtil::ClearLazy("sample_one_epoch");
#endif
    // judge terminate
    if (mSummaryTable.mTotalEpochNum >= mSampleInfo.mSampleEpoches)
    {
        int world_size = cMPIUtil::GetCommSize();

        int world_rank = cMPIUtil::GetWorldRank();
        MIMIC_INFO("SampleIDSolver rank {}/{} sampled {} epoches done",
                   world_rank, world_size, mSummaryTable.mTotalEpochNum);
        cMPIUtil::SetBarrier();

        // then if possible, write down the theta distribution file
        // if (mRecordThetaDist == true)
        // {
        //     mActionThetaDist /= mSummaryTable.mTotalEpochNum;

        //     // if the sync option was turned off
        //     if (mEnableSyncThetaDist == false)
        //     {
        //         if (world_rank == 0)
        //             SaveActionThetaDist(mSummaryTable.mActionThetaDistFile,
        //                                 mActionThetaDist);
        //         else
        //             SaveActionThetaDist(mSummaryTable.mActionThetaDistFile +
        //                                     "." + std::to_string(world_rank),
        //                                 mActionThetaDist);
        //     }
        //     else
        //     {
        //         // sync between processes by MPI
        //         int h = mActionThetaDist.rows(), w = mActionThetaDist.cols();
        //         // MPI_Status status;
        //         if (world_rank == 0)
        //         {
        //             Eigen::MatrixXd others_give_me_mat =
        //                 Eigen::MatrixXd::Zero(h, w);
        //             for (int j = 1; j < world_size; j++)
        //             {
        //                 others_give_me_mat.data();
        //                 // MPI_Recv(others_give_me_mat.data(), h * w,
        //                 // MPI_DOUBLE,
        //                 //          j, 0, MPI_COMM_WORLD, &status);
        //                 cMPIUtil::GetDoubleData(others_give_me_mat.data(),
        //                                         h * w, j, 0);
        //                 mActionThetaDist += others_give_me_mat;
        //                 // std::cout <<"receve from " << j <<" \n" <<
        //                 // others_give_me_mat << std::endl;
        //             }
        //             // std::cout <<"0 get \n" << my_mat << std::endl;
        //         }
        //         else
        //         {
        //             // MPI_Send(&my_float, 1, MPI_DOUBLE, 0, 0,
        //             MPI_COMM_WORLD);
        //             // MPI_Send(mActionThetaDist.data(), h * w, MPI_DOUBLE,
        //             0,
        //             // 0,
        //             //          MPI_COMM_WORLD);

        //             cMPIUtil::SendDoubleData(mActionThetaDist.data(), h * w,
        //             0,
        //                                      0);
        //             // std::cout << world_rank << " sends \n" << my_mat <<
        //             // std::endl;
        //         }
        //         cMPIUtil::SetBarrier();
        //         // only save this theta dist in process 0
        //         if (world_rank == 0)
        //         {
        //             mActionThetaDist /= world_size;
        //             SaveActionThetaDist(mSummaryTable.mActionThetaDistFile,
        //                                 mActionThetaDist);
        //         }
        //     }
        // }

        mSummaryTable.WriteToDisk(mSampleInfo.mSummaryTableFilename);

        cMPIUtil::Finalize();
        cTimeUtil::End("sample_traj");
        exit(0);
    }
}

void cSampleIDSolver::SetTimestep(double timestep)
{
    mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = timestep;
}

void cSampleIDSolver::Parseconfig(const std::string &conf)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    auto sample_value = root["SampleModeInfo"];

    mEnableIDTest =
        cJsonUtil::ParseAsBool("enable_sample_ID_test", sample_value);
    mClearOldData = cJsonUtil::ParseAsBool("clear_old_data", sample_value);
    // mRecordThetaDist =
    //     cJsonUtil::ParseAsBool("record_theta_distribution", sample_value);
    // mEnableSyncThetaDist =
    //     cJsonUtil::ParseAsBool("enable_sync_theta_dist", sample_value);
    mSampleInfo.mSampleEpoches =
        cJsonUtil::ParseAsInt("sample_num", sample_value);
    mSampleInfo.mSampleTrajsDir =
        cJsonUtil::ParseAsString("sample_trajs_dir", sample_value);
    mSampleInfo.mSampleTrajsRootName =
        cJsonUtil::ParseAsString("sample_trajs_rootname", sample_value);
    mSampleInfo.mSampleTrajVer = CalcTrajVersion(
        cJsonUtil::ParseAsInt("traj_file_version", sample_value));

    std::string tablename =
        cJsonUtil::ParseAsString("summary_table_filename", sample_value);
    mSampleInfo.mSummaryTableFilename =
        cFileUtil::ConcatFilename(mSampleInfo.mSampleTrajsDir, tablename);

    // if (mRecordThetaDist == true)
    // {
    //     mSampleInfo.mActionThetaDistFilename =
    //         mSampleInfo.mSampleTrajsDir + "action_theta_dist.txt";
    // }

    MIMIC_INFO("Inverse Dynamic plugin running in sample mode");
    mSaveInfo.mMotion = new cMotion();
    InitSampleSummaryTable();

    // if (mRecordThetaDist == true)
    // {
    //     InitActionThetaDist(mSimChar, mActionThetaDist);
    //     MIMIC_DEBUG("action theta dist init!");
    // }
}

/**
 * \brief               Init the summary table format when ID solver working in
 * "sample" mode We need to record:
 *  1. current skeleton info
 *  2. current PD control info
 *  3. current ref motion file
 */
void cSampleIDSolver::InitSampleSummaryTable()
{
    mSummaryTable.mSampleCharFile = mSimChar->GetCharFilename();
    mSummaryTable.mSampleControllerFile = mCharController->GetControllerFile();
    // if (mRecordThetaDist == true)
    // mSummaryTable.mActionThetaDistFile =
    //         mSampleInfo.mActionThetaDistFilename;

    mSummaryTable.mTotalEpochNum = 0;
    mSummaryTable.mSampleTrajDir = mSampleInfo.mSampleTrajsDir;
    // mSummaryTable.mTotalLengthTime = 0;
    // mSummaryTable.mTotalLengthFrame = 0;
    mSummaryTable.mEpochInfos.clear();
    mSummaryTable.mTimeStamp = cTimeUtil::GetSystemTime();
    MIMIC_INFO("InitSampleSummaryTable: set timestamp {}",
               mSummaryTable.mTimeStamp);
}

// /**
//  * \brief                           Given the action and cur phase, record
//  the
//  * symbol for thetas in action axis angles. \param cur_action current action
//  of
//  * char \param phase                     phase, motion reference time. [0,1]
//  * \param action_theta_dist_mat     target Action theta distribution matrix.
//  It
//  * will be revisied in this func
//  */
// void cSampleIDSolver::RecordActionThetaDist(
//     const tVectorXd &cur_action, double phase,
//     tMatrixXd &action_theta_dist_mat) const
// {
//     // std::cout <<"------------------------------\n";
//     auto &multibody = mSimChar->GetMultiBody();
//     int int_phase = static_cast<int>(phase * mActionThetaGranularity);
//     // std::cout <<"phase " << phase <<" to " << int_phase << std::endl;
//     if (int_phase < 0 || int_phase > mActionThetaGranularity)
//     {
//         MIMIC_ERROR("RecordActionThetaDist phase {}<-%.3f", int_phase,
//         phase); exit(0);
//     }
//     int num_of_joints = mSimChar->GetNumJoints();
//     if (action_theta_dist_mat.rows() != num_of_joints ||
//         action_theta_dist_mat.cols() != mActionThetaGranularity)
//     {
//         MIMIC_ERROR("RecordActionThetaDist: action theta dist size ({}, {}) "
//                     "!= ({}, {})",
//                     action_theta_dist_mat.rows(),
//                     action_theta_dist_mat.cols(), num_of_joints,
//                     mActionThetaGranularity);
//         exit(0);
//     }

//     int f_cnt = 0;
//     for (int i = 0; i < multibody->getNumLinks(); i++)
//     {
//         switch (multibody->getLink(i).m_jointType)
//         {
//         case btMultibodyLink::eFeatherstoneJointType::eRevolute:
//             f_cnt += 1;
//             break;
//         case btMultibodyLink::eFeatherstoneJointType::eSpherical:
//         {
//             action_theta_dist_mat(i, int_phase) +=
//                 cMathUtil::Sign(cur_action[f_cnt]);
//             if (std::fabs(action_theta_dist_mat(i, int_phase)) > 1e-10 &&
//                 cMathUtil::Sign(action_theta_dist_mat(i, int_phase)) !=
//                     cMathUtil::Sign(cur_action[f_cnt]))
//             {
//                 // DebugPrintf(mLogger, "action theta dist %.5f, cur_action
//                 %.5f",\
//                 //     action_theta_dist_mat(i, int_phase),
//                 cur_action[f_cnt]); MIMIC_WARN(
//                     "RecordActionThetaDist frame {} joint {}: cur action sgn
//                     "
//                     "%.3f != action_theta_dist %.3f. It may drive the ID "
//                     "solving into errors, expanding mActionThetaGranularity
//                     is " "better for this problem.", mSaveInfo.mCurFrameId,
//                     i, cMathUtil::Sign(action_theta_dist_mat(i, int_phase)),
//                     cMathUtil::Sign(cur_action[f_cnt]));
//                 // exit(0);
//             }
//             f_cnt += 4;
//         }
//         break;
//         default:
//             break;
//         }
//     }
// }