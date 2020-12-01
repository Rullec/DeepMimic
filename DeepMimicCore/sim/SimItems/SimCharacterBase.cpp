#include "sim/SimItems/SimCharacterBase.h"
#include "util/LogUtil.h"

cSimCharacterBase::tParams::tParams()
{
    mID = gInvalidIdx;
    mCharFile = "";
    mStateFile = "";
    mInitPos = tVector(0, 0, 0, 0);
    mLoadDrawShapes = true;
    mEnableContactFall = true;
}

cSimCharacterBase::cSimCharacterBase(eSimCharacterType type)
{
    mSimcharType = type;
    mStateStack.clear();
}

cSimCharacterBase::~cSimCharacterBase() {}

eSimCharacterType cSimCharacterBase::GetCharType() const
{
    return mSimcharType;
}

/**
 * \brief           Push the pose vel state
*/
void cSimCharacterBase::PushPoseVelState(std::string name)
{
    tTmpState state;
    state.mName = name;
    state.mPose = mPose;
    state.mVel = mVel;
    mStateStack.push_back(state);
}

/**
 * \brief           Push the pose vel state
*/
void cSimCharacterBase::PopPoseVelState(std::string name)
{
    MIMIC_ASSERT(mStateStack.size() > 0);
    auto &last_state = mStateStack[mStateStack.size() - 1];
    MIMIC_ASSERT(last_state.mName == name);
    mPose = last_state.mPose;
    mVel = last_state.mVel;
    SetPose(mPose);
    SetVel(mVel);
    mStateStack.pop_back();
}