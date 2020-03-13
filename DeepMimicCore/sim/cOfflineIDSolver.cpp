#include "cOfflineIDSolver.hpp"
#include <sim/SimCharacter.h>
#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include <util/JsonUtil.h>
#include <util/FileUtil.h>
#include <iostream>

cOfflineIDSolver::cOfflineIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world,\
    const std::string & config):cIDSolver(sim_char, world, eIDSolverType::Offline)
{
    ParseConfig(config);
}

void cOfflineIDSolver::ParseConfig(const std::string & path)
{
    Json::Value root, offline_config;
    cJsonUtil::ParseJson(path, root);
    offline_config = root["Offline_settings"];
    assert(offline_config.isNull() == false);
    const std::string offline_mode = offline_config["mode"].asString();
    if("save" == offline_mode)
    {
        mMode = eOfflineSolverMode::Save;
        mSaveInfo.mCurFrameId = 0;
    }
    else if("display" == offline_mode)
    {
        mMode = eOfflineSolverMode::Display;
    }
    else if("solve" == offline_mode)
    {
        mMode = eOfflineSolverMode::Solve;
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::ParseConfig mode error " << offline_mode << std::endl;
        exit(1);
    }
    mSaveInfo.mSavePath = offline_config["save_path"].asString();
    assert(cFileUtil::ValidateFilePath(mSaveInfo.mSavePath));
}

void cOfflineIDSolver::Reset()
{
    if(this->mMode == eOfflineSolverMode::Display)
    {
        std::cout <<"cOfflineIDSolver: displaying reset\n";
    }
    else if(mMode == eOfflineSolverMode::Save)
    {
        std::cout <<"cOfflineIDSolver: saving reset, have a new saving epoch, need more work to set up epoches\n";
        SaveToFile(mSaveInfo.mSavePath);
        mSaveInfo.mCurEpoch++;
        mSaveInfo.mCurFrameId = 0;
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
        // clear external force
        mSaveInfo.mExternalForces[cur_frame].resize(mNumLinks);
        for(auto & x : mSaveInfo.mExternalForces[cur_frame]) x.setZero();
        mSaveInfo.mExternalTorques[cur_frame].resize(mNumLinks);
        for(auto & x : mSaveInfo.mExternalTorques[cur_frame]) x.setZero();

        RecordJointForces(mSaveInfo.mTruthJointForces[cur_frame]);
	    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);
	    RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame]);

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
        // std::cout <<"offline post sim frame = " << mSaveInfo.mCurFrameId<<std::endl;
        mSaveInfo.mCurFrameId++;
        const int cur_frame = mSaveInfo.mCurFrameId;

        // record the post generalized info
	    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);

	    // record contact forces
	    RecordContactForces(mSaveInfo.mContactForces[cur_frame], mSaveInfo.mTimesteps[cur_frame - 1], mWorldId2InverseId);

        // calculate the relative generalized velocity, take care of the idx... 
        if (cur_frame >= 2)
		{
            double cur_timestep = mSaveInfo.mTimesteps[cur_frame - 1],
                    last_timestep = mSaveInfo.mTimesteps[cur_frame - 2];
			tVectorXd old_vel_after = mSaveInfo.mBuffer_u[cur_frame];
			tVectorXd old_vel_before = mSaveInfo.mBuffer_u[cur_frame - 1];
			tVectorXd old_accel = (old_vel_after - old_vel_before) / cur_timestep;
			mSaveInfo.mBuffer_u[cur_frame - 1] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 2], mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
			mSaveInfo.mBuffer_u[cur_frame] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1], mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
			mSaveInfo.mBuffer_u_dot[cur_frame - 1] = (mSaveInfo.mBuffer_u[cur_frame] - mSaveInfo.mBuffer_u[cur_frame - 1]) / cur_timestep;
            // std::ofstream fout("test2.txt", std::ios::app);
            // fout <<"offline buffer u dot calc: \n";
            // fout <<"buffer u " << cur_frame -1 <<" = " << mSaveInfo.mBuffer_u[cur_frame - 1].transpose()<< std::endl;
            // fout <<"buffer u " << cur_frame <<" = " << mSaveInfo.mBuffer_u[cur_frame].transpose()<< std::endl;
            // fout <<"buffer u dot " << cur_frame-1 <<" = " << mSaveInfo.mBuffer_u_dot[cur_frame - 1].transpose()<< std::endl;

			double threshold = 1e-6;
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
                // for(auto & x : mSaveInfo.mContactForces[cur_frame]) fout << x.mForce.transpose() <<" ";
                // fout << "\n buffer q : ";
                // fout << mSaveInfo.mBuffer_q[cur_frame].transpose() <<" ";
                // fout << "\n buffer u : ";
                // fout << mSaveInfo.mBuffer_u[cur_frame].transpose() <<" ";
                // fout << std::endl;

                cIDSolver::SolveIDSingleStep(
                    mSaveInfo.mSolvedJointForces[cur_frame], 
                    mSaveInfo.mContactForces[cur_frame], 
                    mSaveInfo.mLinkPos[cur_frame-1], 
                    mSaveInfo.mLinkRot[cur_frame-1], 
                    mSaveInfo.mBuffer_q, 
                    mSaveInfo.mBuffer_u, 
                    mSaveInfo.mBuffer_u_dot, 
                    cur_frame, 
                    mSaveInfo.mExternalForces[cur_frame-1], 
                    mSaveInfo.mExternalTorques[cur_frame-1]
                );

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

// void cOfflineIDSolver::SolveIDSingleStep(std::vector<tVector> & solved_joint_forces,
//     const std::vector<tForceInfo> & contact_forces,
//     const std::vector<tVector> link_pos, 
//     const std::vector<tMatrix> link_rot, 
//     const tVectorXd * mBuffer_q,
//     const tVectorXd * mBuffer_u,
//     const tVectorXd * mBuffer_u_dot,
//     int frame_id,
//     const std::vector<tVector> &mExternalForces,
//     const std::vector<tVector> &mExternalTorques) const
// {
    
// }

void cOfflineIDSolver::SaveToFile(const std::string & path_raw)
{
    assert(cFileUtil::ValidateFilePath(path_raw));
    std::string path_root = cFileUtil::RemoveExtension(path_raw);
    std::string final_name = path_root + "_" + std::to_string(mSaveInfo.mCurEpoch) + ".json";
    Json::Value root, single_frame;
    root["epoch"] = mSaveInfo.mCurEpoch;
    root["list"] = Json::Value(Json::arrayValue);
    for(int i=0; i<mSaveInfo.mCurFrameId; i++)
    {
        /* set up single frame, including:
            - frame id
            - timestep
            - generalized coordinate, pos, vel, accel
            - contact info
            - external forces
        */
        single_frame["frame_id"] = i;
        single_frame["timestep"] = mSaveInfo.mTimesteps[i];
        single_frame["pos"] = Json::Value(Json::arrayValue);
        single_frame["vel"] = Json::Value(Json::arrayValue);
        single_frame["accel"] = Json::Value(Json::arrayValue);
        // std::cout << "----------" << i << std::endl;
        for(int dof = 0; dof < mSaveInfo.mBuffer_q[i].size(); dof++)
        {
            // std::cout << "1----" << dof << std::endl;
            single_frame["pos"].append(mSaveInfo.mBuffer_q[i][dof]);
            // std::cout << "2----" << dof << std::endl;
            if(i>=1)
            single_frame["vel"].append(mSaveInfo.mBuffer_u[i][dof]);
            // std::cout << "3----" << dof << std::endl;
            if(i>=2)
            single_frame["accel"].append(mSaveInfo.mBuffer_u_dot[i][dof]);
            // std::cout << "4----" << dof << std::endl;
        }
        root["list"].append(single_frame);
    }
    
    std::ofstream fout(final_name);
    fout << root << std::endl;
    std::cout <<"[log] cOfflineIDSolver::SaveToFile " << "for epoch " << mSaveInfo.mCurEpoch <<" to " << final_name << std::endl;
}