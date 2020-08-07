#include "TrajRecorder.h"
#include "InteractiveIDSolver.hpp"
#include "scenes/SceneImitate.h"
#include "sim/World/FeaWorld.h"
#include "sim/World/GenWorld.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"

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
}

/**
 * \brief               Record info after simulation
 */
void cTrajRecorder::PostSim() { mSaveInfo.mCurFrameId++; }

/**
 * \brief               Reset & save current trajectory per episode
 */
void cTrajRecorder::Reset()
{
    Json::Value root;
    cInteractiveIDSolver::SaveTrajV2(mSaveInfo, root);

    std::string name =
        cFileUtil::GetFilename(mSimChar->GetCharFilename()) + "_traj.json";
    cJsonUtil::WriteJson(name, root, true);
    MIMIC_INFO("cTrajRecoder::Reset finished, save traj to {}, exit", name);
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
    MIMIC_DEBUG("trajectory recoder parse config: get traj save path{}",
                mTrajSavePath);
}

void cTrajRecorder::ReadContactInfoGen()
{
    MIMIC_ERROR("hasn't been implemented")
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
    MIMIC_DEBUG("read contact info finished");
}