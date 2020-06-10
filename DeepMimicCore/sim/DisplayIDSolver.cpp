#include "DisplayIDSolver.hpp"
#include "../scenes/SceneImitate.h"
#include "../util/JsonUtil.h"
#include <sim/SimCharacter.h>
#include "../util/FileUtil.h"
#include <iostream>

extern std::string controller_details_path;
cDisplayIDSolver::cDisplayIDSolver(cSceneImitate * scene, const std::string & config)
:cInteractiveIDSolver(scene, eIDSolverType::Display, config)
{
    controller_details_path = "logs/controller_logs/controller_details_display.txt";
    Parseconfig(config);
}

cDisplayIDSolver::~cDisplayIDSolver()
{

}

void cDisplayIDSolver::PreSim()
{
    // nothing to do
}

void cDisplayIDSolver::PostSim()
{
    mLoadInfo.mCurFrame++;
    std::cout <<"\r[log] cDisplayIDSolver display mode: cur frame = " << mLoadInfo.mCurFrame;

   if(mLoadInfo.mLoadMode == eLoadMode::INVALID)
    {
        std::cout <<"[error] cDisplayIDSolver::DisplaySet invalid info mode\n";
        exit(1);
    }
    else if(mLoadInfo.mLoadMode == eLoadMode::LOAD_TRAJ)
    {
        const int cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mTotalFrame;
        const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
        // std::cout <<"q = " << q.transpose() << std::endl;
        SetGeneralizedPos(q);
        RecordMultibodyInfo(mLoadInfo.mLinkRot[cur_frame], mLoadInfo.mLinkPos[cur_frame]);

        if(mLoadInfo.mEnableOutputMotionInfo == true)
            PrintLoadInfo(mLoadInfo, mLoadInfo.mOutputMotionInfoPath, true);
        
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
        
        const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mMotion->GetNumFrames();

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
        std::cout <<"[error] cDisplayIDSolver::DisplaySet mode invalid: "<< mLoadInfo.mLoadMode << std::endl;
        exit(1);
    }

    // we are in display mode, so we need to clear the contact info incase something contact with the ground
    mScene->GetWorld()->GetContactManager().Clear();
}

void cDisplayIDSolver::Reset()
{
    std::cout << "[log] cDisplayIDSolver Reset, exiting...\n";
    exit(0);
}

void cDisplayIDSolver::SetTimestep(double)
{

}

void cDisplayIDSolver::Parseconfig(const std::string & conf)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    auto display_value = cJsonUtil::ParseAsValue("DisplayModeInfo", root);
    assert(display_value.isNull() == false);
    // std::cout <<"void cDisplayIDSolver::ParseConfigDisplay(const Json::Value & save_value)\n";
    const Json::Value & display_traj_path = display_value["display_traj_path"],
        display_motion_path = display_value["display_motion_path"];

    // mLoadInfo.mEnableOutputMotionInfo = enable_output_motion_info.asBool();
    mLoadInfo.mEnableOutputMotionInfo = cJsonUtil::ParseAsBool("enable_output_motion_info", display_value);
    mLoadInfo.mOutputMotionInfoPath = cJsonUtil::ParseAsString("output_motion_info_path", display_value);

    if(mLoadInfo.mEnableOutputMotionInfo)
        cFileUtil::ClearFile(mLoadInfo.mOutputMotionInfoPath);

    // there is only one choice between display_motion_path and display_traj_path
    bool display_motion_path_isnull = display_motion_path.isNull(),
        display_traj_path_isnull = display_traj_path.isNull();
    if(!display_motion_path_isnull || !display_traj_path_isnull)
    {
        if(!display_motion_path_isnull && !display_traj_path_isnull)
        {
            std::cout <<"[error] cDisplayIDSolver::ParseConfigDisplay: there is only one choice between \
                loading motions and loading trajectories\n";
            exit(1);
        }
        if(!display_motion_path_isnull)
        {
            // choose to load motion from files
            mLoadInfo.mLoadMode = eLoadMode::LOAD_MOTION;
            mLoadInfo.mMotion = new cMotion();
            LoadMotion(display_motion_path.asString(), mLoadInfo.mMotion);
            mLogger->info("LoadMotion {}", display_motion_path.asString());
            // std::cout <<"[log] offlineIDSolver load motion " << display_motion_path<<", the resulted NOF = " << mLoadInfo.mMotion->GetNumFrames() << std::endl;
        }
        else // choose to load trajectories from files
        {
            mLoadInfo.mLoadMode = eLoadMode::LOAD_TRAJ;
            LoadTraj(mLoadInfo, display_traj_path.asString());
            mLogger->info("LoadTraj {}", display_traj_path.asString());
            // std::cout <<"[log] offlineIDSolver load the trajectory " << display_motion_path<<", the resulted NOF = " << mLoadInfo.mTotalFrame << std::endl;
        }       
    }
    else
    {
        std::cout <<"[error] cDisplayIDSolver::ParseConfigDisplay: all options are empty\n";
        exit(1);
    }

    // init load info character pose
    const int & cur_frame = mLoadInfo.mCurFrame % mLoadInfo.mPoseMat.rows();
    mLogger->info("cDisplayIDSolver display mode: cur frame {}", cur_frame);
    const tVectorXd & q = mLoadInfo.mPoseMat.row(cur_frame);
    SetGeneralizedPos(q);
    RecordMultibodyInfo(mLoadInfo.mLinkRot[cur_frame], mLoadInfo.mLinkPos[cur_frame]);
}