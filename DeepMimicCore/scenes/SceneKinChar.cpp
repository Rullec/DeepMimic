#include "SceneKinChar.h"
#include "util/LogUtil.h"

cSceneKinChar::cSceneKinChar()
{
    mEnableOutputTraj = false;
    mOuputTrajPath = "";
}

cSceneKinChar::~cSceneKinChar() {}

void cSceneKinChar::Init() { bool succ = BuildCharacters(); }

void cSceneKinChar::ParseArgs(const std::shared_ptr<cArgParser> &parser)
{
    cScene::ParseArgs(parser);
    ParseCharParams(parser, mCharParams);

    // begin to parse the output traj trajectory
    parser->ParseBool("enable_output_traj", mEnableOutputTraj);
    parser->ParseStringCritic("output_traj_path", this->mOuputTrajPath);
    MIMIC_INFO("Enable output traj {} to {}", mEnableOutputTraj,
               mOuputTrajPath);
}

void cSceneKinChar::Reset() { ResetCharacters(); }

void cSceneKinChar::Clear() { mChar.reset(); }

void cSceneKinChar::Update(double time_elapsed)
{
    UpdateCharacters(time_elapsed);

    // output the traj and set false
    if (mEnableOutputTraj == true)
        this->ExportMotionToTraj(), mEnableOutputTraj = false;
}

const std::shared_ptr<cKinCharacter> &cSceneKinChar::GetCharacter() const
{
    return mChar;
}

tVector cSceneKinChar::GetCharPos() const
{
    return GetCharacter()->GetRootPos();
}

double cSceneKinChar::GetTime() const { return GetCharacter()->GetTime(); }

std::string cSceneKinChar::GetName() const { return "Kinematic Char"; }

void cSceneKinChar::ParseCharParams(const std::shared_ptr<cArgParser> &parser,
                                    cKinCharacter::tParams &out_params) const
{
    std::string char_file = "";
    std::string motion_file = "";
    std::string state_file = "";
    double init_pos_xs = 0;

    bool succ = parser->ParseString("character_file", char_file);
    if (succ)
    {
        parser->ParseString("state_file", state_file);
        parser->ParseString("motion_file", motion_file);
        parser->ParseDouble("char_init_pos_x", init_pos_xs);

        out_params.mCharFile = char_file;
        out_params.mMotionFile = motion_file;
        out_params.mStateFile = state_file;
        out_params.mOrigin[0] = init_pos_xs;
    }
    else
    {
        printf("No character file provided\n");
    }
}

bool cSceneKinChar::BuildCharacters()
{
    mChar.reset();

    auto &params = mCharParams;
    params.mID = 0;
    params.mLoadDrawShapes = true;

    bool succ = BuildCharacter(params, mChar);

    return succ;
}

bool cSceneKinChar::BuildCharacter(
    const cKinCharacter::tParams &params,
    std::shared_ptr<cKinCharacter> &out_char) const
{
    out_char = std::shared_ptr<cKinCharacter>(new cKinCharacter());
    bool succ = out_char->Init(params);
    return succ;
}

void cSceneKinChar::ResetCharacters() { mChar->Reset(); }

void cSceneKinChar::UpdateCharacters(double timestep)
{
    mChar->Update(timestep);
}

#include "sim/TrajManager/Trajectory.h"
void cSceneKinChar::ExportMotionToTraj() const
{
    const cMotion &motion = mChar->GetMotion();
    MIMIC_INFO("frame num {}", motion.GetNumFrames());
    tSaveInfo saveinfo;
    saveinfo.SaveTrajV2FromMotion(GetCharacter()->GetNumJoints(), &motion,
                                  this->mOuputTrajPath);
}