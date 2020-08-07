#pragma once

#include <memory>

#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include "btBulletDynamicsCommon.h"

#include "anim/KinTree.h"
#include "sim/SimItems/MultiBody.h"
#include "sim/World/ContactManager.h"
#include "sim/World/PerturbManager.h"
#include "sim/World/WorldBase.h"
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

// Featherstone simulation world
class cFeaWorld : virtual public cWorldBase
{
public:
    cFeaWorld();
    virtual ~cFeaWorld();
    virtual void Init(const tParams &params) override final;
    virtual void Reset() override final;
    virtual void Update(double time_elapsed) override final;

    virtual void AddRigidBody(cSimRigidBody &obj) override final;
    virtual void RemoveRigidBody(cSimRigidBody &obj) override final;

    virtual void AddCollisionObject(btCollisionObject *col_obj,
                                    int col_filter_group,
                                    int col_filter_mask) override final;
    virtual void
    RemoveCollisionObject(btCollisionObject *col_obj) override final;
    virtual void AddCharacter(cSimCharacterBase *sim_char) override final;
    virtual void RemoveCharacter(cSimCharacterBase *sim_char) override final;

    virtual void Constrain(cSimRigidBody &obj) override final;
    virtual void Constrain(cSimRigidBody &obj, const tVector &linear_factor,
                           const tVector &angular_factor) override final;
    virtual void RemoveConstraint(tConstraintHandle &handle) override final;
    virtual void AddJoint(const cSimJoint &joint) override final;
    virtual void RemoveJoint(cSimJoint &joint) override final;

    virtual void SetGravity(const tVector &gravity) override final;

    virtual cContactManager::tContactHandle
    RegisterContact(int contact_flags, int filter_flags) override final;
    virtual void
    UpdateContact(const cContactManager::tContactHandle &handle) override final;
    virtual bool IsInContact(
        const cContactManager::tContactHandle &handle) const override final;
    virtual const tEigenArr<cContactManager::tContactPt> &GetContactPts(
        const cContactManager::tContactHandle &handle) const override final;
    virtual cContactManager &GetContactManager() override final;

    virtual void RayTest(const tVector &beg, const tVector &end,
                         tRayTestResults &results) const override final;
    virtual void AddPerturb(const tPerturb &perturb) override final;
    virtual const cPerturbManager &GetPerturbManager() const override final;

    virtual tVector GetGravity() const override final;
    virtual double GetScale() const override final;
    virtual double GetTimeStep() const override final;
    virtual void SetDefaultLinearDamping(double damping) override final;
    virtual double GetDefaultLinearDamping() const override final;
    virtual void SetDefaultAngularDamping(double damping) override final;
    virtual double GetDefaultAngularDamping() const override final;

    virtual btDynamicsWorld *GetInternalWorld() override;

    // object interface
    virtual btBoxShape *
    BuildBoxShape(const tVector &box_sizee) const override final;
    virtual btCapsuleShape *
    BuildCapsuleShape(double radius, double height) const override final;
    virtual btStaticPlaneShape *
    BuildPlaneShape(const tVector &normal,
                    const tVector &origin) const override final;
    virtual btSphereShape *BuildSphereShape(double radius) const override final;
    virtual btCylinderShape *
    BuildCylinderShape(double radius, double height) const override final;

    virtual tVector GetSizeBox(const cSimObj &obj) const override final;
    virtual tVector GetSizeCapsule(const cSimObj &obj) const override final;
    virtual tVector GetSizePlane(const cSimObj &obj) const override final;
    virtual tVector GetSizeSphere(const cSimObj &obj) const override final;
    virtual tVector GetSizeCylinder(const cSimObj &obj) const override final;

protected:
    btMultiBodyDynamicsWorld *mSimWorld;
    virtual void BuildConsFactor(tVector &out_linear_factor,
                                 tVector &out_angular_factor);
};
