#include "sim/World/WorldBase.h"

cWorldBase::tParams::tParams()
{
    mWorldType = gWorldType[eWorldType::INVALID_WORLD_TYPE];
    mGenWorldConfig = "";
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

eWorldType cWorldBase::GetWorldType() const { return mType; }

cWorldBase::cWorldBase(eWorldType type) : mContactManager(this)
{
    mType = type;
}