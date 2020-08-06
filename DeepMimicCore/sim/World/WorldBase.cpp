#include "sim/World/WorldBase.h"

cWorldBase::tParams::tParams()
{
    mWorldType = gWorldType[eWorldType::INVALID_WORLD_TYPE];
    mNumSubsteps = 1;
    mScale = 1;
    mGravity = gGravity;
}

cWorldBase::tConstraintHandle::tConstraintHandle() { mCons = nullptr; }

bool cWorldBase::tConstraintHandle::IsValid() const { return mCons != nullptr; }

void cWorldBase::tConstraintHandle::Clear()
{
    delete mCons;
    mCons = nullptr;
}

cWorldBase::~cWorldBase() {}