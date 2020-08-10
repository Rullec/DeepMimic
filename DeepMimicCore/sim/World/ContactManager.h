#pragma once

#include "util/MathUtil.h"
#include <memory>

class cWorldBase;

/**
 * \brief    manage the contact info for character & simobjs
 */
class cContactManager
{
public:
    const static int gInvalidID;
    const static short gFlagAll = -1;
    const static short gFlagNone = 0;
    const static short gFlagRayTest = 1;

    // each bodypart and rigid body has its own contact handle, use to storage the id & flags
    struct tContactHandle
    {
        int mID;
        int mFlags;
        int mFilterFlags;

        tContactHandle();
        bool IsValid() const;
    };

    // info of a single contact point
    struct tContactPt
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        tContactPt();

        bool mIsSelfCollision;  // self collision flag, only works in Multibody case.
        tVector mPos;       // contact position in world frame
        tVector mForce;     // contact force in world frame
    };

    cContactManager(cWorldBase *world);
    virtual ~cContactManager();

    virtual void Init();
    virtual void Reset();
    virtual void Clear();
    virtual void Update();

    virtual tContactHandle RegisterContact(int contact_flags, int filter_flags);
    virtual void UpdateContact(const cContactManager::tContactHandle &handle);
    virtual int GetNumEntries() const;
    virtual int GetNumTotalContactPts() const;
    virtual bool IsInContact(const tContactHandle &handle) const;
    virtual bool IsInContactGenGround(const tContactHandle &handle) const;
    virtual const tEigenArr<tContactPt> &
    GetContactPts(const tContactHandle &handle) const;
    virtual const tEigenArr<tContactPt> &GetContactPts(int handle_id) const;

protected:
    struct tContactEntry
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        tContactEntry();

        int mFlags;
        int mFilterFlags;
        tEigenArr<tContactPt> mContactPts;
    };

    cWorldBase *mWorld;
    tEigenArr<tContactEntry> mContactEntries; // each links has the same entry, collect all these entries together

    virtual int RegisterNewID();
    virtual void ClearContacts();
    virtual bool IsValidContact(const tContactHandle &h0,
                                const tContactHandle &h1) const;
    virtual void UpdateContactInFeaWorld();
    virtual void UpdateContactInGenWorld();
};