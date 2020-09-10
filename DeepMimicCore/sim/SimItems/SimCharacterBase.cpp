#include "sim/SimItems/SimCharacterBase.h"

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
}

cSimCharacterBase::~cSimCharacterBase() {}

eSimCharacterType cSimCharacterBase::GetCharType() const
{
    return mSimcharType;
}