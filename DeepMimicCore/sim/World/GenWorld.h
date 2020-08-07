// #include "BulletDynamics/Dynamics/btDynamicsWorld.h"
#include "sim/World/WorldBase.h"

class btGeneralizeWorld;
class cGenWorld : virtual public cWorldBase
{
public:
    cGenWorld();
    virtual ~cGenWorld();
    virtual void Init(const cWorldBase::tParams &params) override;
    virtual void Reset() override;
    virtual void Update(double time_elapsed) override;

    virtual void AddRigidBody(cSimRigidBody &obj) override;
    virtual void RemoveRigidBody(cSimRigidBody &obj) override;

    virtual void AddCollisionObject(btCollisionObject *col_obj,
                                    int col_filter_group,
                                    int col_filter_mask) override;
    virtual void RemoveCollisionObject(btCollisionObject *col_obj) override;
    virtual void AddCharacter(cSimCharacterBase *sim_char) override;
    virtual void RemoveCharacter(cSimCharacterBase *sim_char) override;

    virtual void Constrain(cSimRigidBody &obj) override;
    virtual void Constrain(cSimRigidBody &obj, const tVector &linear_factor,
                           const tVector &angular_factor) override;
    virtual void RemoveConstraint(tConstraintHandle &handle) override;
    virtual void AddJoint(const cSimJoint &joint) override;
    virtual void RemoveJoint(cSimJoint &joint) override;

    virtual void SetGravity(const tVector &gravity) override;

    virtual cContactManager::tContactHandle
    RegisterContact(int contact_flags, int filter_flags) override;
    virtual void
    UpdateContact(const cContactManager::tContactHandle &handle) override;
    virtual bool
    IsInContact(const cContactManager::tContactHandle &handle) const override;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts(const cContactManager::tContactHandle &handle) const override;
    virtual cContactManager &GetContactManager() override;

    virtual void RayTest(const tVector &beg, const tVector &end,
                         tRayTestResults &results) const override;
    virtual void AddPerturb(const tPerturb &perturb) override;
    virtual const cPerturbManager &GetPerturbManager() const override;

    virtual tVector GetGravity() const override;
    virtual double GetScale() const override;
    virtual double GetTimeStep() const override;
    virtual void SetDefaultLinearDamping(double damping) override;
    virtual double GetDefaultLinearDamping() const override;
    virtual void SetDefaultAngularDamping(double damping) override;
    virtual double GetDefaultAngularDamping() const override;

    virtual btDynamicsWorld * GetInternalWorld() override;

    // object interface
    virtual btBoxShape *BuildBoxShape(const tVector &box_sizee) const override;
    virtual btCapsuleShape *BuildCapsuleShape(double radius,
                                              double height) const override;
    virtual btStaticPlaneShape *
    BuildPlaneShape(const tVector &normal,
                    const tVector &origin) const override;
    virtual btSphereShape *BuildSphereShape(double radius) const override;
    virtual btCylinderShape *BuildCylinderShape(double radius,
                                                double height) const override;

    virtual tVector GetSizeBox(const cSimObj &obj) const override;
    virtual tVector GetSizeCapsule(const cSimObj &obj) const override;
    virtual tVector GetSizePlane(const cSimObj &obj) const override;
    virtual tVector GetSizeSphere(const cSimObj &obj) const override;
    virtual tVector GetSizeCylinder(const cSimObj &obj) const override;

protected:
    btGeneralizeWorld * mbtGenWorld;
    btDynamicsWorld * mbtDynamicsWorldInternal;
};
