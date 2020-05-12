#include "cOfflineIDSolver.hpp"
#include <sim/SimCharacter.h>
#include "sim/CtPDController.h"
#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include "BulletDynamics/Featherstone/btMultiBodyLink.h"
#include <util/JsonUtil.h>
#include <util/FileUtil.h>
#include <util/BulletUtil.h>
#include <iostream>
#include <fstream>

std::string controller_details_path;
cOfflineIDSolver::cOfflineIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world,\
    const std::string & config):cIDSolver(sim_char, world, eIDSolverType::Offline)
{
    ParseConfig(config);
}

void cOfflineIDSolver::ParseConfig(const std::string & path)
{
    Json::Value root, offline_config;
    cJsonUtil::ParseJson(path, root);
    offline_config = root["Offline"];
    assert(offline_config.isNull() == false);
    const std::string offline_mode = offline_config["mode"].asString();
    if("save" == offline_mode)
    {
        controller_details_path = "logs/controller_logs/controller_details_save.txt";
        mMode = eOfflineSolverMode::Save;
        ParseConfigSave(offline_config["SaveModeInfo"]);
    }
    else if("display" == offline_mode)
    {
        controller_details_path = "logs/controller_logs/controller_details_display.txt";
        mMode = eOfflineSolverMode::Display;
        ParseConfigDisplay(offline_config["DisplayModeInfo"]);

    }
    else if("solve" == offline_mode)
    {
        controller_details_path = "logs/controller_logs/controller_details_solve.txt";
        mMode = eOfflineSolverMode::Solve;
        ParseConfigSolve(offline_config["SolveModeInfo"]);
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::ParseConfig mode error " << offline_mode << std::endl;
        exit(1);
    }
    cFileUtil::ClearFile(controller_details_path);
    // std::cout <<"controller details path = " << controller_details_path << std::endl;
    
}

void cOfflineIDSolver::Reset()
{
    if(this->mMode == eOfflineSolverMode::Display)
    {
        std::cout <<"cOfflineIDSolver: displaying reset\n";
        exit(1);
    }
    else if(mMode == eOfflineSolverMode::Save)
    {
        std::cout <<"cOfflineIDSolver: saving reset, have a new saving epoch, need more work to set up epoches\n";
        SaveTraj(mSaveInfo.mSaveTrajRoot);
        SaveMotion(mSaveInfo.mSaveMotionRoot, mSaveInfo.mMotion);

        // output linera momentum to file "ang_mimic.txt"
        //  assert(cFileUtil::OutputVecList(mSaveInfo.mAngularMomentum, mSaveInfo.mCurFrameId, "ang_mimic.txt", cFileUtil::eFileMode::overwrite));
        // std::ofstream fout("ang_mimic.txt");
        // for(int i=0; i<mSaveInfo.mCurFrameId; i++)
        // {
        //     // std::cout << mSaveInfo.mAngularMomentum[i].transpose() << std::endl;
        //     fout << mSaveInfo.mAngularMomentum[i].transpose() << std::endl;
        // }
        // fout.close();
        // exit(1);
        // is impulse-momentum theorem broken? How much? use verify momentum
        VerifyMomentum();
        mSaveInfo.mCurEpoch++;
        mSaveInfo.mCurFrameId = 0;
        exit(1);
    }
    else if(mMode == eOfflineSolverMode::Solve)
    {
        std::cout << "void cOfflineIDSolver::Reset() solve mode solve\n";
        exit(1);
    }
    else
    {
        std::cout << "void cOfflineIDSolver::Reset() solve mode illegal\n";
        exit(1);
    }
}

eOfflineSolverMode cOfflineIDSolver::GetOfflineSolverMode()
{
    return mMode;
}

void cOfflineIDSolver::PreSim()
{
    if(this->mMode == eOfflineSolverMode::Save)
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
	    

        // only record momentum in PreSim for the first frame
        if(0 == cur_frame)
        {
            RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame], mSaveInfo.mLinkOmega[cur_frame], mSaveInfo.mLinkVel[cur_frame]);
            // CalcMomentum(mSaveInfo.mLinkPos[cur_frame],
            //     mSaveInfo.mLinkRot[cur_frame], 
            //     mSaveInfo.mLinkVel[cur_frame], 
            //     mSaveInfo.mLinkOmega[cur_frame],
            //     mSaveInfo.mLinearMomentum[cur_frame],
            //     mSaveInfo.mAngularMomentum[cur_frame]);
            // RecordMomentum(mSaveInfo.mLinearMomentum[cur_frame], mSaveInfo.mAngularMomentum[cur_frame]);
        }
        
        
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
    else
    {
        std::cout <<"[error] cOfflineIDSolver::PreSim error mode = " << mMode << std::endl;
        exit(1);
    }
}

void cOfflineIDSolver::PostSim()
{
    if(this->mMode == eOfflineSolverMode::Save)
    {
        // if(mSaveInfo.mCurFrameId > 2) exit(1);
        // std::cout <<"offline post sim frame = " << mSaveInfo.mCurFrameId<<std::endl;
        mSaveInfo.mCurFrameId++;
        const int cur_frame = mSaveInfo.mCurFrameId;

        // record the post generalized info
	    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);

	    // record contact forces
	    RecordContactForces(mSaveInfo.mContactForces[cur_frame-1], mSaveInfo.mTimesteps[cur_frame - 1], mWorldId2InverseId);

        // record linear momentum
        RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame], mSaveInfo.mLinkOmega[cur_frame], mSaveInfo.mLinkVel[cur_frame]);

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
        std::cout << "[debug] PostSim: discrete total err = " << total_err << std::endl; 

        // exit(1);
        // calculate momentum by these discreted values
        CalcMomentum(mSaveInfo.mLinkPos[cur_frame-1],
            mSaveInfo.mLinkRot[cur_frame-1],
            mSaveInfo.mLinkDiscretVel[cur_frame],
            mSaveInfo.mLinkDiscretOmega[cur_frame],
            mSaveInfo.mLinearMomentum[cur_frame - 1],
            mSaveInfo.mAngularMomentum[cur_frame - 1]);


        // character pose
        mSaveInfo.mCharPoses[cur_frame] = mSimChar->GetPose();

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
                            std::cout <<"[error] offline ID solved error: for joint " << id <<" diff : solved = " << mSaveInfo.mSolvedJointForces[cur_frame][id].transpose() <<", truth = " << mSaveInfo.mTruthJointForces[cur_frame-1][id].transpose() << std::endl;
                        }
                    }
                    if(err < 1e-5) std::cout << "[log] frame " << cur_frame <<" offline ID solved accurately\n";
                    else std::cout << "[error] frame " << cur_frame <<" offline ID solved error\n";
                }
            }
		}

    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::PostSim error mode = " << mMode << std::endl;
        exit(1);
    }
}

void cOfflineIDSolver::SetTimestep(double timestep)
{
    assert(timestep > 0);
    if(mMode == eOfflineSolverMode::Save)
    {
        mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = timestep;
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::SetTimestep invalid working mode = " << mMode << std::endl;
        exit(1);
    }
}

void cOfflineIDSolver::SaveTraj(const std::string & path_raw)
{
    assert(cFileUtil::ValidateFilePath(path_raw));
    std::string path_root = cFileUtil::RemoveExtension(path_raw);
    std::string final_name = path_root + "_" + std::to_string(mSaveInfo.mCurEpoch) + ".json";
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    Json::Value root, single_frame;
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
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
        single_frame["pos"] = Json::Value(Json::arrayValue);
        single_frame["vel"] = Json::Value(Json::arrayValue);
        single_frame["accel"] = Json::Value(Json::arrayValue);
        // std::cout << "----------" << i << std::endl;
        for(int dof = 0; dof < mSaveInfo.mBuffer_q[frame_id].size(); dof++)
        {
            // std::cout << "1----" << dof << std::endl;
            single_frame["pos"].append(mSaveInfo.mBuffer_q[frame_id][dof]);
            // std::cout << "2----" << dof << std::endl;
            if(frame_id>=1)
            single_frame["vel"].append(mSaveInfo.mBuffer_u[frame_id][dof]);
            // std::cout << "3----" << dof << std::endl;
            if(frame_id>=1)
            single_frame["accel"].append(mSaveInfo.mBuffer_u_dot[frame_id][dof]);
            // std::cout << "4----" << dof << std::endl;
        }

        // set up contact info
        single_frame["contact_info"] = Json::arrayValue;
        // single_frame["contact_num"] = mSaveInfo.mContactForces[frame_id].size();
        single_frame["contact_num"] = static_cast<int>(mSaveInfo.mContactForces[frame_id].size());
        for(int c_id = 0; c_id < mSaveInfo.mContactForces[frame_id].size(); c_id++)
        {
            Json::Value single_contact;
            const tForceInfo & force = mSaveInfo.mContactForces[frame_id][c_id];
            single_contact["force_pos"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_pos"].append(force.mPos[i]);
            single_contact["force_value"] = Json::arrayValue;
            for(int i=0; i<4; i++) single_contact["force_value"].append(force.mForce[i]);
            single_contact["force_link_id"] = force.mId;
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
    std::ofstream fout(final_name);
    writer->write(root, &fout);
    std::cout <<"[log] cOfflineIDSolver::SaveTraj " << "for epoch " << mSaveInfo.mCurEpoch <<" to " << final_name << std::endl;
}

void cOfflineIDSolver::DisplaySet()
{
    // if(mLoadInfo.mCurFrame >=1)
    // {
    //     int frame = mLoadInfo.mCurFrame;
    //     std::ofstream fout("linear_momentum_mimic.txt", std::ios::app);
    //     tVector linear_mom = tVector::Zero(), linear_impulse = tVector::Zero();

    //     for(int i=0; i<mNumLinks - 1; i++)
    //     {
    //         auto cur_link = mMultibody->getLink(i);
    //         linear_mom += cur_link.m_mass * (mLoadInfo.mLinkPos[frame][i+1] - mLoadInfo.mLinkPos[frame-1][i+1]) / mLoadInfo.mTimesteps[frame];
    //         linear_impulse += cur_link.m_mass * cBulletUtil::btVectorTotVector0( mWorld->getGravity()) *  mLoadInfo.mTimesteps[frame];
    //         // std::cout << mLoadInfo.mLinkPos[frame][i+1] .transpose()  << std::endl;
    //     }

    //     for(auto & cur : mLoadInfo.mContactForces[frame])
    //     {
    //         linear_impulse+= cur.mForce * mLoadInfo.mTimesteps[frame];
    //     }
    //     fout <<"frame " << frame <<" linear momentum = " << linear_mom.segment(0, 3).transpose() << std::endl;
    //     fout <<"frame " << frame <<" linear impulse = " << linear_impulse.segment(0, 3).transpose() << std::endl;
    // }

    // select and set value for it
    if(mLoadInfo.mLoadMode == eLoadMode::INVALID)
    {
        std::cout <<"[error] cOfflineIDSolver::DisplaySet invalid info mode\n";
        exit(1);
    }
    else if(mLoadInfo.mLoadMode == eLoadMode::LOAD_TRAJ)
    {
        mLoadInfo.mCurFrame++;
        mLoadInfo.mCurFrame %= mLoadInfo.mTotalFrame;
        const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mPoseMat.rows();
        std::cout <<"[log] cOfflineIDSolver display mode: cur frame = " << cur_frame << std::endl;
        const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
        SetGeneralizedPos(q);
        RecordMultibodyInfo(mLoadInfo.mLinkRot[cur_frame], mLoadInfo.mLinkPos[cur_frame]);

        if(mLoadInfo.mEnableOutputMotionInfo == true)
            PrintLoadInfo(mLoadInfo.mOutputMotionInfoPath, true);
        
        // DEBUG: output the contact info
        
        // std::ofstream fout_pt("contact_pt_mimic.txt", std::ios::app);
        // auto contact_pts = mLoadInfo.mContactForces[mLoadInfo.mCurFrame];
        // fout_pt <<"frame " << mLoadInfo.mCurFrame << " contact pts num = " << contact_pts.size() << std::endl;
        // for(auto &x : contact_pts)
        // {
        //     std::string name = "";
        //     if(x.mId == 0) name = mMultibody->getBaseName();
        //     else name = std::string(mMultibody->getLink(x.mId -1).m_linkName);
        //     std::cout <<"name = " << name << std::endl;
        //     fout_pt << "link name : " << mSimChar->GetBodyName(x.mId - 1)\
        //      << ", contact pt pos = " << x.mPos.transpose().segment(0, 3) << std::endl;
        // }
        
        // DEBUG: output the link pos to files
        // std::ofstream fout_pos("pos_verify_mimic.txt", std::ios::app);
        // for(int i=0; i<mNumLinks; i++)
        // {
        //     tVector pos, rot;
        //     if(0 == i)
        //     {
        //         pos = cBulletUtil::btVectorTotVector1(mMultibody->getBasePos());
        //         rot = cBulletUtil::btQuaternionTotQuaternion(mMultibody->getBaseWorldTransform().getRotation()).coeffs();
        //     }
        //     else
        //     {
        //         int multibody_link_id = i - 1;
        //         auto & cur_trans = mMultibody->getLinkCollider(multibody_link_id)->getWorldTransform();
        //         rot = cBulletUtil::btQuaternionTotQuaternion(cur_trans.getRotation()).coeffs();
        //         pos = cBulletUtil::btVectorTotVector1(cur_trans.getOrigin());
        //     }
            
        //     fout_pos <<"link " << i <<" pos = " << pos.segment(0, 3).transpose() << ", rot = " << rot.transpose() << std::endl;
        // }
	
    }
    else if(mLoadInfo.mLoadMode == eLoadMode::LOAD_MOTION)
    {
        mLoadInfo.mCurFrame++;
        const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mMotion->GetNumFrames();
        std::cout <<"[log] cOfflineIDSolver display mode: cur frame = " << cur_frame << std::endl;
        tVectorXd out_pose = mLoadInfo.mMotion->GetFrame(cur_frame);
        // auto & mJointMat = mSimChar->GetJointMat();
        // {
        //     tVector root_delta = tVector::Zero();
        //     tQuaternion root_delta_rot = tQuaternion::Identity();

        //     tVector root_pos = cKinTree::GetRootPos(mJointMat, out_pose);
        //     tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, out_pose);


        //     // root_delta_rot = mOriginRot * root_delta_rot;
        //     root_rot = root_delta_rot * root_rot;
        //     root_pos += root_delta;
        //     root_pos = cMathUtil::QuatRotVec(root_delta_rot, root_pos);
        //     // root_pos += mOrigin;

        //     cKinTree::SetRootPos(mJointMat, root_pos, out_pose);
        //     cKinTree::SetRootRot(mJointMat, root_rot, out_pose);
        // }
        mSimChar->SetPose(out_pose);
        // mSimChar->PostUpdate(0.01);
        // std::cout <<"[id] error root pos = " << mSimChar->GetRootPos().transpose() << std::endl;
        // std::cout <<"[id] error root rot = " << mSimChar->GetRootRotation().coeffs().transpose() << std::endl;
        // std::cout <<"[id] error pose = " << out_pose.transpose() << std::endl;
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::DisplaySet mode invalid: "<< mLoadInfo.mLoadMode << std::endl;
        exit(1);
    }
}

void cOfflineIDSolver::LoadTraj(const std::string & path)
{
    std::cout <<"[debug] offline load trajectory " << path << std::endl; 
    Json::Value data_json, list_json;
    cJsonUtil::ParseJson(path, data_json);
    // std::cout <<"load succ\n";
    list_json = data_json["list"];
    assert(list_json.isNull() == false);
    // std::cout <<"get list json, begin set up pramas\n";
    
    /*
        std::string mLoadPath = "";
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat;
        tVectorXd mTimesteps;
        std::vector<std::vector<tForceInfo>> mContactForces;
        Eigen::MatrixXd mExternalForces, mExternalTorques;
        int mTotalFrame = 0;
        int mCurFrame = 0;
    */
    int num_of_frames = list_json.size();
    {
        mLoadInfo.mTotalFrame = num_of_frames;
        mLoadInfo.mPoseMat.resize(num_of_frames, mDof), mLoadInfo.mPoseMat.setZero();
        mLoadInfo.mVelMat.resize(num_of_frames, mDof), mLoadInfo.mVelMat.setZero();
        mLoadInfo.mAccelMat.resize(num_of_frames, mDof), mLoadInfo.mAccelMat.setZero();
        mLoadInfo.mActionMat.resize(num_of_frames, mCharController->GetActionSize()), mLoadInfo.mActionMat.setZero();
        mLoadInfo.mPDTargetMat.resize(num_of_frames, mCharController->GetActionSize()), mLoadInfo.mPDTargetMat.setZero();
        mLoadInfo.mContactForces.resize(num_of_frames); for(auto & x : mLoadInfo.mContactForces) x.clear();
        mLoadInfo.mLinkRot.resize(num_of_frames); for(auto & x : mLoadInfo.mLinkRot) x.resize(mNumLinks);
        mLoadInfo.mLinkPos.resize(num_of_frames); for(auto & x : mLoadInfo.mLinkPos) x.resize(mNumLinks);
        mLoadInfo.mExternalForces.resize(num_of_frames); for(auto & x : mLoadInfo.mExternalForces) x.resize(mNumLinks);
        mLoadInfo.mExternalTorques.resize(num_of_frames); for(auto & x : mLoadInfo.mExternalTorques) x.resize(mNumLinks);
        mLoadInfo.mTruthJointForces.resize(num_of_frames); for(auto & x : mLoadInfo.mTruthJointForces) x.resize(mNumLinks - 1);
        mLoadInfo.mTimesteps.resize(num_of_frames), mLoadInfo.mTimesteps.setZero();
    }

    for(int frame_id = 0; frame_id<num_of_frames; frame_id++)
    {
        auto & cur_frame = list_json[frame_id];
        auto & cur_pose = cur_frame["pos"];
        auto & cur_vel = cur_frame["vel"];
        auto & cur_accel = cur_frame["accel"];
        auto & cur_timestep = cur_frame["timestep"];
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
        assert(cur_contact_info.size() == cur_contact_num.asInt());
        assert(cur_truth_joint_force.isNull() == false && cur_truth_joint_force.size() == (mNumLinks-1) * 4);
        // std::cout <<"load pd target size = " << cur_truth_action.size() << std::endl;
        // std::cout <<"action space size = " << mCharController->GetActionSize() << std::endl;
        assert(cur_truth_action.isNull() == false && cur_truth_action.size() == mCharController->GetActionSize());
        assert(cur_truth_pd_target.isNull() == false && cur_truth_pd_target.size() == mCharController->GetActionSize());

        // 1. pos, vel, accel
        for(int j=0; j<mDof; j++) mLoadInfo.mPoseMat(frame_id, j) = cur_pose[j].asDouble();
        // std::cout <<cur_pose.size() <<" " << mSimChar->GetNumDof() << std::endl;
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(frame_id));
        RecordMultibodyInfo(mLoadInfo.mLinkRot[frame_id], mLoadInfo.mLinkPos[frame_id]);
        for(int j=0; j<mDof && frame_id>=1; j++) mLoadInfo.mVelMat(frame_id, j) = cur_vel[j].asDouble();
        for(int j=0; j<mDof && frame_id>=1; j++) mLoadInfo.mAccelMat(frame_id, j) = cur_accel[j].asDouble();

        // 2. timestep
        mLoadInfo.mTimesteps[frame_id] = cur_timestep.asDouble();

        // 3. contact info
        // mLoadInfo.mContactForces
        mLoadInfo.mContactForces[frame_id].resize(cur_contact_num.asInt());
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
                mLoadInfo.mContactForces[frame_id][c_id].mPos[i] = cur_contact_info[c_id]["force_pos"][i].asDouble();
                mLoadInfo.mContactForces[frame_id][c_id].mForce[i] = cur_contact_info[c_id]["force_value"][i].asDouble();

            }
            mLoadInfo.mContactForces[frame_id][c_id].mId = cur_contact_info[c_id]["force_link_id"].asInt();
        }

        // std::cout <<"[load file] frame " << frame_id <<" contact num = " << cur_contact_info.size() << std::endl;

        // 4. load external forces
        // mLoadInfo.mExternalForces[frame_id].resize(mNumLinks);
        // mLoadInfo.mExternalTorques[frame_id].resize(mNumLinks);
        for(int idx = 0; idx < mNumLinks; idx++)
        {
            // auto & cur_ext_force = ;
            // auto & cur_ext_torque = mLoadInfo.mExternalTorques[frame_id][idx];
            for(int i= 0; i< 4; i++)
            {
                mLoadInfo.mExternalForces[frame_id][idx][i] = cur_ext_force[idx * 4 + i].asDouble();
                mLoadInfo.mExternalTorques[frame_id][idx][i] = cur_ext_torque[idx * 4 + i].asDouble();
            }
            assert(mLoadInfo.mExternalForces[frame_id][idx].norm() < 1e-10);
            assert(mLoadInfo.mExternalTorques[frame_id][idx].norm() < 1e-10);
        }

        // 5. truth joint forces
        for(int idx = 0; idx < mNumLinks - 1; idx++)
        {
            for(int j=0; j<4; j++)
            mLoadInfo.mTruthJointForces[frame_id][idx][j] = cur_truth_joint_force[idx * 4 + j].asDouble();
        }

        // 6. truth actions and pd targes
        for(int idx = 0; idx < mCharController->GetActionSize(); idx++)
        {
            mLoadInfo.mActionMat(frame_id, idx) = cur_truth_action[idx].asDouble();
            mLoadInfo.mPDTargetMat(frame_id, idx) = cur_truth_pd_target[idx].asDouble();
        }

    }
    std::cout <<"[debug] cOfflineIDSolver::LoadTraj " << path <<", number of frames = " << num_of_frames << std::endl;
    // exit(1);
}

#include "util/cTimeUtil.hpp"
void cOfflineIDSolver::OfflineSolve()
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

/**
    "SaveModeInfo":
    {
        "_comment" : "this section focus on the setting of save mode",
        "save_traj_root" : "data/1209/trajs/traj.json",
        "save_motion_root" :  "data/1209/motion/ID_motion.txt"
    },
*/
void cOfflineIDSolver::ParseConfigSave(const Json::Value & save_value)
{
    assert(save_value.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSave(const Json::Value & save_value)\n";
    mSaveInfo.mCurFrameId = 0;
    const Json::Value save_path_root = save_value["save_traj_root"],
        save_motion_root = save_value["save_motion_root"];

    assert(save_path_root.isNull() == false);
    assert(save_motion_root.isNull() == false);

    mSaveInfo.mSaveTrajRoot = save_path_root.asString();
    mSaveInfo.mSaveMotionRoot = save_motion_root.asString();

    mSaveInfo.mMotion = new cMotion();
    
    assert(cFileUtil::ValidateFilePath(mSaveInfo.mSaveTrajRoot));
    assert(cFileUtil::ValidateFilePath(mSaveInfo.mSaveMotionRoot));
}

/*
    "DisplayModeInfo":
    {
        "_comment" : "this section focus on the setting of display mode",
        "display_traj_path" : "data/1209/trajs/traj_0.json",
        "display_motion_path" : "data/1209/motion/ID_motion_0.txt"
    },
*/
void cOfflineIDSolver::ParseConfigDisplay(const Json::Value & display_value)
{
    assert(display_value.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigDisplay(const Json::Value & save_value)\n";
    const Json::Value & display_traj_path = display_value["display_traj_path"],
        display_motion_path = display_value["display_motion_path"],
        enable_output_motion_info = display_value["enable_output_motion_info"],
        output_motion_info_path = display_value["output_motion_info_path"];

    assert(enable_output_motion_info.isNull() == false);
    assert(output_motion_info_path.isNull() == false);

    mLoadInfo.mEnableOutputMotionInfo = enable_output_motion_info.asBool();
    mLoadInfo.mOutputMotionInfoPath = output_motion_info_path.asString();

    if(mLoadInfo.mEnableOutputMotionInfo)
        cFileUtil::ClearFile(mLoadInfo.mOutputMotionInfoPath);

    // there is only one choice between display_motion_path and display_traj_path
    bool display_motion_path_isnull = display_motion_path.isNull(),
        display_traj_path_isnull = display_traj_path.isNull();
    if(!display_motion_path_isnull || !display_traj_path_isnull)
    {
        if(!display_motion_path_isnull && !display_traj_path_isnull)
        {
            std::cout <<"[error] cOfflineIDSolver::ParseConfigDisplay: there is only one choice between \
                loading motions and loading trajectories\n";
            exit(1);
        }
        if(!display_motion_path_isnull)
        {
            // choose to load motion from files
            mLoadInfo.mLoadMode = eLoadMode::LOAD_MOTION;
            mLoadInfo.mMotion = new cMotion();
            LoadMotion(display_motion_path.asString(), mLoadInfo.mMotion);
            // std::cout <<"[log] offlineIDSolver load motion " << display_motion_path<<", the resulted NOF = " << mLoadInfo.mMotion->GetNumFrames() << std::endl;
        }
        else // choose to load trajectories from files
        {
            mLoadInfo.mLoadMode = eLoadMode::LOAD_TRAJ;
            LoadTraj(display_traj_path.asString());
            // std::cout <<"[log] offlineIDSolver load the trajectory " << display_motion_path<<", the resulted NOF = " << mLoadInfo.mTotalFrame << std::endl;
        }       
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::ParseConfigDisplay: all options are empty\n";
        exit(1);
    }

    // init load info character pose
    const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mPoseMat.rows();
    std::cout <<"[log] cOfflineIDSolver display mode: cur frame = " << cur_frame << std::endl;
    const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
    SetGeneralizedPos(q);
    RecordMultibodyInfo(mLoadInfo.mLinkRot[cur_frame], mLoadInfo.mLinkPos[cur_frame]);
}

/*
    "SolveModeInfo":
    {
        "_comment" : "this section focus on the setting of solve mode, path means full info trajectory but motion is only motion",
        "solve_traj_path" : "data/1209/trajs/traj_0.json"
    }
*/
void cOfflineIDSolver::ParseConfigSolve(const Json::Value & solve_value)
{
    assert(solve_value.isNull() == false);
    // std::cout <<"void cOfflineIDSolver::ParseConfigSolve(const Json::Value & save_value)\n";
    const Json::Value & solve_traj_path = solve_value["solve_traj_path"];
    assert(solve_traj_path.isNull() == false);
    LoadTraj(solve_traj_path.asString());
}

/*
    @Function: LoadMotion
    @params: path Type const std::string &, the filename of specified motion
    @params: motion Type cMotion *, the targeted motion storaged.
*/
void cOfflineIDSolver::LoadMotion(const std::string & path, cMotion * motion) const
{
    assert(cFileUtil::ExistsFile(path));
    std::cout <<"[debug] offline load motion from " << path << std::endl;
    cMotion::tParams params;
    params.mMotionFile = path;
    motion->Load(params);
    
    std::cout <<"[debug] offline load motion frames = " << motion->GetNumFrames() <<", dof = " <<motion->GetNumDof() << std::endl;
}

void cOfflineIDSolver::SaveMotion(const std::string & path_root, cMotion * motion) const
{
    assert(nullptr != motion);
    if(false == cFileUtil::ValidateFilePath(path_root))
    {
        std::cout <<"[error] cOfflineIDSolver::SaveMotion: path root invalid: " << path_root << std::endl;
        exit(1);
    }
    std::string filename = cFileUtil::RemoveExtension(path_root) + "_" + std::to_string(mSaveInfo.mCurEpoch) + "." + cFileUtil::GetExtension(path_root);
    std::cout <<"[log] cOfflineIDSolver::SaveMotion for epoch " << mSaveInfo.mCurEpoch <<" to " << filename << std::endl;
    motion->FinishAddFrame();
    motion->Output(filename);
    motion->Clear();
}

// this function is used to verify the "law of conservation of momentum" when OfflineIDSolver works in "Save" mode
void cOfflineIDSolver::VerifyMomentum()
{
    if(mMode!=eOfflineSolverMode::Save)
    {
        std::cout <<"[error] this function SHOULD NOT be called in mode " << mMode;
        exit(1);
    }

    // std::cout <<"[log] void cOfflineIDSolver::VerifyMomentum begin\n";
    // verify linear momentum
    std::ofstream fout_1("./logs/verify_momentum_pos.txt");
    std::ofstream fout_2("./logs/verify_momentum_vel.txt");
    {
        tVector linear_momentum_now = tVector::Zero(),  // for init linear momentum, it's truly zero now.
                linear_momentum_next = tVector::Zero(),
                COM_pos_now = tVector::Zero(),
                COM_pos_next = tVector::Zero();

        tVector linear_momentum_changes = tVector::Zero(), impulse = tVector::Zero();
        double char_mass = mSimChar->CalcTotalMass(), char_mass_test = 0;
        std::vector<double> link_mass(0);
        for(int id = 0; id< mSimChar->GetNumBodyParts(); id++) // including root
        {
            link_mass.push_back(mSimChar->GetBodyPart(id)->GetMass());
            char_mass_test += link_mass[link_mass.size() - 1];
            COM_pos_now += link_mass[link_mass.size() - 1] * mSaveInfo.mLinkPos[0][id] / char_mass;
        }
        assert(std::abs(char_mass - char_mass_test) < 1e-6);
        
        
        // 3. calculate impulse for this frame
        // 4. compare
        // 5. give another value
        for(int i=0; i< mSaveInfo.mCurFrameId-1; i++)
        {
            const int & next_frame = i+1, & cur_frame = i;
            const double & cur_timestep = mSaveInfo.mTimesteps[cur_frame];
            // 1. calculate momentum for next frame
            COM_pos_next = tVector::Zero();
            for(int id = 0; id< mSimChar->GetNumBodyParts(); id++)
                COM_pos_next += link_mass[id] * mSaveInfo.mLinkPos[next_frame][id] / char_mass;
            
            linear_momentum_next = (COM_pos_next - COM_pos_now) / cur_timestep * char_mass;

            // 2. calculate momentum changes for this frame
            linear_momentum_changes = linear_momentum_next - linear_momentum_now;

            // 3. calculate impulse
            impulse = tVector::Zero();
            impulse += char_mass * mSimChar->GetWorld()->GetGravity() * cur_timestep;

            // mContactForces[i] is the contact force in i frame, and it will generate vel i+1
            for(auto & f: mSaveInfo.mContactForces[i])
            {
                impulse += f.mForce * cur_timestep;
            }

            // 4. compare
            fout_1 <<"frame " << i <<" " << (impulse - linear_momentum_changes).transpose() << std::endl;

            // 5. give another value
            linear_momentum_now = linear_momentum_next;
            COM_pos_now = COM_pos_next;
        }
        
    }
    
    // verify linear momentum from vel
    {
        double total_mass = mSimChar->CalcTotalMass();
        for(int i=0; i<mSaveInfo.mCurFrameId-1; i++)
        {
            tVector diff = mSaveInfo.mLinearMomentum[i+1] - mSaveInfo.mLinearMomentum[i] - mSimChar->GetWorld()->GetGravity() * total_mass * mSaveInfo.mTimesteps[i];
            for(auto & f: mSaveInfo.mContactForces[i])
            {
                diff -= f.mForce * mSaveInfo.mTimesteps[i];
            }
            fout_2 <<"frame " << i <<" " << diff.transpose() << std::endl;
        }

    }
    // verify angular momentum


    // exit(1);
}

void cOfflineIDSolver::PrintLoadInfo(const std::string & filename, bool disable_root /*= true*/)
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