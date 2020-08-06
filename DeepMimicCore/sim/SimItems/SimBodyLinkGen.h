#pragma once
#include "BulletGenDynamics/btGenModel/RobotModelDynamics.h"
#include "SimBodyLink.h"

class Link;
class cSimBodyLinkGen : public cSimBodyLink
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    cSimBodyLinkGen();

    virtual ~cSimBodyLinkGen();

    virtual tVector GetPos() const override;
    virtual void SetPos(const tVector &pos) override;
    virtual void GetRotation(tVector &out_axis,
                             double &out_theta) const override;
    virtual tQuaternion GetRotation() const override;
    virtual void SetRotation(const tVector &axis, double theta) override;
    virtual void SetRotation(const tQuaternion &q) override;
    virtual tMatrix GetWorldTransform() const override;
    virtual tMatrix GetLocalTransform() const override;

    virtual void Init(const std::shared_ptr<cWorldBase> &world,
                      cRobotModelDynamics *multibody, int link_id);
    virtual tVector GetSize() const override;

    virtual tVector GetLinearVelocity() const override;
    virtual tVector GetLinearVelocity(const tVector &local_pos) const override;
    virtual void SetLinearVelocity(const tVector &vel) override;
    virtual tVector GetAngularVelocity() const override;
    virtual void SetAngularVelocity(const tVector &vel) override;

    virtual tVector GetInertia() const override;
    virtual double GetMass() const override;
    virtual double GetFriction() const override;
    virtual void SetFriction(double friction) override;
    virtual void ApplyForce(const tVector &force) override;
    virtual void ApplyForce(const tVector &force,
                            const tVector &local_pos) override;
    virtual void ApplyTorque(const tVector &torque) override;
    virtual void ClearForces() override;

    virtual cShape::eShape GetShape() const override;
    virtual void UpdateVel(const tVector &lin_vel,
                           const tVector &ang_vel) override;
    virtual double GetLinkMaxLength() const;
    // virtual const std::shared_ptr<cMultiBody> &GetMultBody() const override;
    virtual int GetJointID() const override;
    virtual Link *GetInternalLink() const;
    virtual void SetEnableFallContact(bool v);
    virtual bool GetEnableFallContact();
    virtual bool IsEndEffector() const;
    virtual bool IsInContact() const;
    
protected:
    cRobotModelDynamics *mRobotModel;
    Link *mLink;
    cRobotCollider *mLinkCollider;
    int mLinkId;

    virtual const std::shared_ptr<cWorldBase> &GetWorld() const override;
    virtual const btCollisionObject *GetCollisionObject() const override;
    virtual btCollisionObject *GetCollisionObject() override;
};
