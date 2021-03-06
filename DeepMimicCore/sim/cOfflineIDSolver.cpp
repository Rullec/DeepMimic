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
    offline_config = root["Offline"];
    assert(offline_config.isNull() == false);
    const std::string offline_mode = offline_config["mode"].asString();
    if("save" == offline_mode)
    {
        mMode = eOfflineSolverMode::Save;
        ParseConfigSave(offline_config["SaveModeInfo"]);
    }
    else if("display" == offline_mode)
    {
        mMode = eOfflineSolverMode::Display;
        ParseConfigDisplay(offline_config["DisplayModeInfo"]);

    }
    else if("solve" == offline_mode)
    {
        mMode = eOfflineSolverMode::Solve;
        ParseConfigSolve(offline_config["SolveModeInfo"]);
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
        SaveTraj(mSaveInfo.mSaveTrajRoot);
        SaveMotion(mSaveInfo.mSaveMotionRoot, mSaveInfo.mMotion);
        VerifyMomentum();
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
        // for(auto & x : mSaveInfo.mTruthJointForces[cur_frame]) std::cout << x.transpose() << std::endl;
	    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);
	    

        // only record momentum in PreSim for the first frame
        if(0 == cur_frame)
        {
            RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame]);
            RecordMomentum(mSaveInfo.mLinearMomentum[cur_frame], mSaveInfo.mAngularMomentum[cur_frame]);
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
        // std::cout <<"offline post sim frame = " << mSaveInfo.mCurFrameId<<std::endl;
        mSaveInfo.mCurFrameId++;
        const int cur_frame = mSaveInfo.mCurFrameId;

        // record the post generalized info
	    RecordGeneralizedInfo(mSaveInfo.mBuffer_q[cur_frame], mSaveInfo.mBuffer_u[cur_frame]);

	    // record contact forces
	    RecordContactForces(mSaveInfo.mContactForces[cur_frame], mSaveInfo.mTimesteps[cur_frame - 1], mWorldId2InverseId);

        // record linear momentum
        RecordMultibodyInfo(mSaveInfo.mLinkRot[cur_frame], mSaveInfo.mLinkPos[cur_frame]);
        RecordMomentum(mSaveInfo.mLinearMomentum[cur_frame], mSaveInfo.mAngularMomentum[cur_frame]);

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
			mSaveInfo.mBuffer_u[cur_frame - 1] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 2], mSaveInfo.mBuffer_q[cur_frame - 1], last_timestep);
			mSaveInfo.mBuffer_u[cur_frame] = CalculateGeneralizedVel(mSaveInfo.mBuffer_q[cur_frame - 1], mSaveInfo.mBuffer_q[cur_frame], cur_timestep);
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
    // select and set value for it
    if(mLoadInfo.mLoadMode == eLoadMode::INVALID)
    {
        std::cout <<"[error] cOfflineIDSolver::DisplaySet invalid info mode\n";
        exit(1);
    }
    else if(mLoadInfo.mLoadMode == eLoadMode::LOAD_TRAJ)
    {
        mLoadInfo.mCurFrame++;
        const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mPoseMat.rows();
        std::cout <<"[log] cOfflineIDSolver display mode: cur frame = " << cur_frame << std::endl;
        const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
        SetGeneralizedPos(q);
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
    std::cout <<"[debug] cOfflineIDSolver::LoadTraj " << path <<", number of frames = " << num_of_frames << std::endl;
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
        display_motion_path = display_value["display_motion_path"];

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

            // mContactForces[i+1] is the contact force in i frame, and it will generate vel i+1
            for(auto & f: mSaveInfo.mContactForces[i+1])
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
            for(auto & f: mSaveInfo.mContactForces[i+1])
            {
                diff -= f.mForce * mSaveInfo.mTimesteps[i];
            }
            fout_2 <<"frame " << i <<" " << diff.transpose() << std::endl;
        }

    }
    // verify angular momentum


    // exit(1);
}