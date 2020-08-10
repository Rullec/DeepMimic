#include "TrajRecorder.h"
#include "InteractiveIDSolver.hpp"
#include "scenes/SceneImitate.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/World/FeaWorld.h"
#include "sim/World/GenWorld.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>

cTrajRecorder::cTrajRecorder(cSceneImitate *scene, const std::string &conf)
{
    mScene = scene;
    mSimChar = scene->GetCharacter().get();
    ParseConfig(conf);
    MIMIC_TRACE("build traj recoder by {}", conf);
}

cTrajRecorder::~cTrajRecorder() {}

/**
 * \brief               Record info before simulation
 */
void cTrajRecorder::PreSim()
{
    mSaveInfo.mCharPoses[mSaveInfo.mCurFrameId] = mSimChar->GetPose();

    // auto gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
    // if (gen_char != nullptr)
    // {
    //     std::cout << "-----------------------------frame "
    //               << mSaveInfo.mCurFrameId << "----------------------\n";
    //     std::cout << "q = " << gen_char->Getq().transpose() << std::endl;
    //     std::cout << "qdot = " << gen_char->Getqdot().transpose() << std::endl;
    //     std::cout << "M = \n" << gen_char->GetMassMatrix() << std::endl;
    //     std::cout << "C = \n" << gen_char->GetCoriolisMatrix() << std::endl;
    // }
}

/**
 * \brief               Record info after simulation
 */
void cTrajRecorder::PostSim()
{
    ReadContactInfo();

    mSaveInfo.mCurFrameId++;
    if (mRecordMaxFrame == mSaveInfo.mCurFrameId)
    {
        MIMIC_INFO("record frames exceed the upper bound {}, reset",
                   mSaveInfo.mCurFrameId);
        Reset();
    }
}

/**
 * \brief               Reset & save current trajectory per episode
 */
void cTrajRecorder::Reset()
{
    Json::Value root;
    cInteractiveIDSolver::SaveTrajV2(mSaveInfo, root);

    cJsonUtil::WriteJson(mTrajSavePath, root, true);
    MIMIC_INFO("cTrajRecoder::Reset finished, save traj to {}, exit",
               mTrajSavePath);
    exit(1);
}

/**
 *
 */
void cTrajRecorder::SetTimestep(double t)
{
    mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = t;
}

/**
 * \brief               Parse config
 *          This function will parse the trajectory recoder's config to control
 * its behavior.
 *          1. save place?
 *          2. save version?
 */
void cTrajRecorder::ParseConfig(const std::string &conf)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    mTrajSavePath = cJsonUtil::ParseAsString("traj_save_filename", root);
    mRecordMaxFrame = cJsonUtil::ParseAsInt("record_max_frame", root);

    MIMIC_DEBUG("trajectory recoder parse config: get traj save path{}",
                mTrajSavePath);
}

void cTrajRecorder::ReadContactInfoGen()
{
    int num_of_links = mSimChar->GetNumBodyParts();
    auto &cur_contact_info = mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    cur_contact_info.clear();

    tContactForceInfo pt_info;
    for (int i = 0; i < num_of_links; i++)
    {
        for (auto &pt : mSimChar->GetBodyPart(i)->GetContactPts())
        {
            pt_info.mId = i;
            pt_info.mIsSelfCollision = pt.mIsSelfCollision;
            pt_info.mForce = pt.mForce;
            pt_info.mPos = pt.mPos;
            cur_contact_info.push_back(pt_info);
        }
    }
    MIMIC_INFO("record {} contact points in traj recorder",
               cur_contact_info.size());

    MIMIC_WARN("link id += 1 in order to get matched with RobotControl");
    for (auto &x : cur_contact_info)
    {
        x.mId += 1;
    }
}
void cTrajRecorder::ReadContactInfoRaw()
{

    MIMIC_ERROR("hasn't been implemented")
}
void cTrajRecorder::ReadContactInfo()
{
    switch (mSimChar->GetCharType())
    {
    case eSimCharacterType::Featherstone:
        ReadContactInfoRaw();
        break;
    case eSimCharacterType::Generalized:
        ReadContactInfoGen();
        break;
    default:
        break;
    }

    const auto &current_forces =
        mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    int frame = mSaveInfo.mCurFrameId;
    if (current_forces.size())
    {
        MIMIC_DEBUG("frame {} num of contacts {}", frame,
                    current_forces.size());
        for (int i = 0; i < current_forces.size(); i++)
        {
            MIMIC_DEBUG("contact {} force = {}", i,
                        current_forces[i].mForce.transpose());
        }
    }
}