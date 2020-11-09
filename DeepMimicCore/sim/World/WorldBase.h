#pragma once

#include <memory>
// #include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include "anim/KinTree.h"
#include "btBulletDynamicsCommon.h"
// #include "sim/SimItems/MultiBody.h"
#include "sim/World/ContactManager.h"
#include "sim/World/PerturbManager.h"
#include "util/MathUtil.h"

class cSimObj;
class cSimJoint;
class cSimBodyLink;
class cSimRigidBody;
class cSimBox;
class cSimCapsule;
class cSimPlane;
class cSimSphere;
class cSimCharacterBase;

class btTypedConstraint;
enum eWorldType
{
    INVALID_WORLD_TYPE,
    FEATHERSTONE_WORLD,
    GENERALIZED_WORLD,
    NUM_WORLD_TYPE
};
const std::string gWorldType[NUM_WORLD_TYPE] = {"invalid", "featherstone",
                                                "generalized"};

class cWorldBase : public std::enable_shared_from_this<cWorldBase>
{
public:
    enum eContactFlag
    {
        eContactFlagNone = 0,
        eContactFlagCharacter = 0x1,
        eContactFlagEnvironment = 0x1 << 1,
        eContactFlagObject = 0x1 << 2,
        eContactFlagAll = cContactManager::gFlagAll
    };

    struct tParams
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        tParams();
        std::string mWorldType;
        std::string mGenWorldConfig;
        int mNumSubsteps;
        double mScale;
        tVector mGravity;
    };

    struct tConstraintHandle
    {
        tConstraintHandle();
        bool IsValid() const;
        void Clear();
        btTypedConstraint *mCons;
    };

    struct tRayTestResult
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        cSimObj *mObj;
        tVector mHitPos;
    };
    typedef tEigenArr<tRayTestResult> tRayTestResults;

    virtual ~cWorldBase();
    virtual void Init(const tParams &params) = 0;
    virtual void Reset() = 0;
    virtual void Update(double time_elapsed) = 0;
    virtual void PostUpdate() = 0;
    virtual eWorldType GetWorldType() const;
    virtual void AddRigidBody(cSimRigidBody &obj) = 0;
    virtual void RemoveRigidBody(cSimRigidBody &obj) = 0;

    virtual void AddCollisionObject(btCollisionObject *col_obj,
                                    int col_filter_group,
                                    int col_filter_mask) = 0;
    virtual void RemoveCollisionObject(btCollisionObject *col_obj) = 0;
    virtual void AddCharacter(cSimCharacterBase *sim_char) = 0;
    virtual void RemoveCharacter(cSimCharacterBase *sim_char) = 0;

    virtual void Constrain(cSimRigidBody &obj) = 0;
    virtual void Constrain(cSimRigidBody &obj, const tVector &linear_factor,
                           const tVector &angular_factor) = 0;
    virtual void RemoveConstraint(tConstraintHandle &handle) = 0;
    virtual void AddJoint(const cSimJoint &joint) = 0;
    virtual void RemoveJoint(cSimJoint &joint) = 0;

    virtual void SetGravity(const tVector &gravity) = 0;

    virtual cContactManager::tContactHandle
    RegisterContact(int contact_flags, int filter_flags) = 0;
    virtual void
    UpdateContact(const cContactManager::tContactHandle &handle) = 0;
    virtual bool
    IsInContact(const cContactManager::tContactHandle &handle) const = 0;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts(const cContactManager::tContactHandle &handle) const = 0;
    virtual cContactManager &GetContactManager() = 0;

    virtual void RayTest(const tVector &beg, const tVector &end,
                         tRayTestResults &results) const = 0;
    virtual void AddPerturb(const tPerturb &perturb) = 0;
    virtual const cPerturbManager &GetPerturbManager() const = 0;

    virtual tVector GetGravity() const = 0;
    virtual double GetScale() const = 0;
    virtual double GetTimeStep() const = 0;
    virtual void SetDefaultLinearDamping(double damping) = 0;
    virtual double GetDefaultLinearDamping() const = 0;
    virtual void SetDefaultAngularDamping(double damping) = 0;
    virtual double GetDefaultAngularDamping() const = 0;
    virtual btDynamicsWorld *GetInternalWorld() = 0;

    // object interface
    virtual btBoxShape *BuildBoxShape(const tVector &box_sizee) const = 0;
    virtual btCapsuleShape *BuildCapsuleShape(double radius,
                                              double height) const = 0;
    virtual btStaticPlaneShape *
    BuildPlaneShape(const tVector &normal, const tVector &origin) const = 0;
    virtual btSphereShape *BuildSphereShape(double radius) const = 0;
    virtual btCylinderShape *BuildCylinderShape(double radius,
                                                double height) const = 0;

    virtual tVector GetSizeBox(const cSimObj &obj) const = 0;
    virtual tVector GetSizeCapsule(const cSimObj &obj) const = 0;
    virtual tVector GetSizePlane(const cSimObj &obj) const = 0;
    virtual tVector GetSizeSphere(const cSimObj &obj) const = 0;
    virtual tVector GetSizeCylinder(const cSimObj &obj) const = 0;

protected:
    struct tConstraintEntry
    {
        cSimObj *mObj0;
        cSimObj *mObj1;
    };

    tParams mParams;
    eWorldType mType;
    double mTimeStep;
    double mDefaultLinearDamping;
    double mDefaultAngularDamping;
    // std::unique_ptr<btMultiBodyDynamicsWorld> mSimWorld;

    std::unique_ptr<btConstraintSolver> mSolver;
    std::unique_ptr<btCollisionDispatcher> mCollisionDispatcher;
    std::unique_ptr<btDefaultCollisionConfiguration> mCollisionConfig;
    std::unique_ptr<btBroadphaseInterface> mBroadPhase;

    cContactManager mContactManager;
    cPerturbManager mPerturbManager;
    // virtual int GetNumConstriants() const;

    cWorldBase(eWorldType type);
};
