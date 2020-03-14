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
        mSaveInfo.mSavePath = offline_config["save_path"].asString();
        assert(cFileUtil::ValidateFilePath(mSaveInfo.mSavePath));
    }
    else if("display" == offline_mode)
    {
        mMode = eOfflineSolverMode::Display;
        LoadFromFile(offline_config["load_path"].asString());
    }
    else if("solve" == offline_mode)
    {
        mMode = eOfflineSolverMode::Solve;
        LoadFromFile(offline_config["load_path"].asString());
    }
    else
    {
        std::cout <<"[error] cOfflineIDSolver::ParseConfig mode error " << offline_mode << std::endl;
        exit(1);
    }


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

void cOfflineIDSolver::SaveToFile(const std::string & path_raw)
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

        // append to the whole list
        root["list"].append(single_frame);
    }
    std::ofstream fout(final_name);
    writer->write(root, &fout);
    std::cout <<"[log] cOfflineIDSolver::SaveToFile " << "for epoch " << mSaveInfo.mCurEpoch <<" to " << final_name << std::endl;
}

void cOfflineIDSolver::DisplaySet()
{
    // select and set value for it
    mLoadInfo.mCurFrame++;
    const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mPoseMat.rows();
    std::cout <<"[log] cOfflineIDSolver display mode: cur frame = " << cur_frame << std::endl;
    const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
    SetGeneralizedPos(q);
}

void cOfflineIDSolver::LoadFromFile(const std::string & path)
{
    std::cout <<"[debug] offline load from file " << path << std::endl; 
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
        assert(cur_pose.isNull() == false && cur_pose.size() == mDof);
        assert(cur_vel.isNull() == false); if(frame_id>=1) assert(cur_vel.size() == mDof);
        assert(cur_accel.isNull() == false);if(frame_id>=2) assert(cur_accel.size() == mDof);
        assert(cur_timestep.isNull() == false && cur_timestep.asDouble() > 0);
        assert(cur_contact_info.size() == cur_contact_num.asInt());
        assert(cur_truth_joint_force.isNull() == false && cur_truth_joint_force.size() == (mNumLinks-1) * 4);

        // 1. pos, vel, accel
        for(int j=0; j<mDof; j++) mLoadInfo.mPoseMat(frame_id, j) = cur_pose[j].asDouble();
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
    }
    std::cout <<"[debug] cOfflineIDSolver::LoadFromFile " << path <<", number of frames = " << num_of_frames << std::endl;
    // exit(1);
}

void cOfflineIDSolver::OfflineSolve()
{
    std::cout <<"cOfflineIDSolver::OfflineSolve begin\n";
    assert(mLoadInfo.mTotalFrame > 0);
    std::cout <<"[debug] total frame = " << mLoadInfo.mTotalFrame << std::endl;
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
        tVectorXd cur_vel = CalculateGeneralizedVel(mLoadInfo.mPoseMat.row(frame_id), mLoadInfo.mPoseMat.row(frame_id+1), mLoadInfo.mTimesteps[frame_id]);
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
        // std::cout <<"cur vel size = " << cur_vel.size() << std::endl;
        // std::cout <<"com vel size = " << mLoadInfo.mVelMat.row(frame_id + 1).size() << std::endl;
        // std::cout <<" cur vel = " << cur_vel.transpose() << std::endl;
        // std::cout <<"vel comp = " << mLoadInfo.mVelMat.row(frame_id + 1) << std::endl;
        tVectorXd diff = mLoadInfo.mAccelMat.row(frame_id).transpose() - cur_accel;
        mLoadInfo.mAccelMat.row(frame_id) = cur_accel;
        // std::cout <<"frame id " << frame_id << " accel diff = " << diff.norm() << std::endl;
        assert(diff.norm() < 1e-10);
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
    double err = 0;
    for(int cur_frame = 2; cur_frame < mLoadInfo.mTotalFrame - 1; cur_frame++)
    {
        mInverseModel->clearAllUserForcesAndMoments();
        SetGeneralizedPos(mLoadInfo.mPoseMat.row(cur_frame));
        SetGeneralizedVel(mLoadInfo.mVelMat.row(cur_frame));
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
            mLoadInfo.mLinkPos[cur_frame-1], 
            mLoadInfo.mLinkRot[cur_frame-1], 
            mLoadInfo.mPoseMat.row(cur_frame-1), 
            mLoadInfo.mVelMat.row(cur_frame-1), 
            mLoadInfo.mAccelMat.row(cur_frame-1), 
            cur_frame, 
            mLoadInfo.mExternalForces[cur_frame-1], 
            mLoadInfo.mExternalTorques[cur_frame-1]
            );

        for(int j=0; j<mNumLinks-1; j++)
        {
            double single_error = (result[j] - mLoadInfo.mTruthJointForces[cur_frame-1][j]).norm();
            if(single_error > 1e-6)
            {
                std::cout <<"[error] cOfflineIDSolver::OfflineSolve for frame " << cur_frame <<" link " << j << ":\n";
                std::cout <<"offline solved = " << result[j].transpose() << std::endl;
                std::cout <<"standard = " << mLoadInfo.mTruthJointForces[cur_frame-1][j].transpose() << std::endl;
                std::cout <<"error = " << single_error << std::endl;
                assert(0);
            }
            err += single_error;
        }
        // std::cout <<"done\n";
        // exit(1);
    }
    std::cout <<"[log] offline id solver solved succ, total error = " << err << std::endl;
    exit(1);
}