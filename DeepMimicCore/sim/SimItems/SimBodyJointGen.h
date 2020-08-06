#pragma once

#include "SimBodyJoint.h"
#include "SimCharacterGen.h"

class Joint;
class cSimBodyJointGen : public cSimBodyJoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    cSimBodyJointGen();
    virtual ~cSimBodyJointGen();

    virtual void Init(std::shared_ptr<cWorldBase> &world,
                      cRobotModelDynamics *multibody, int joint_id);
    virtual void Clear() override;
    virtual bool IsValid() const override;

    virtual cKinTree::eJointType GetType() const override;

    virtual void CalcWorldRotation(tVector &out_axis,
                                   double &out_theta) const override;
    virtual tQuaternion CalcWorldRotation() const override;
    virtual tMatrix BuildWorldTrans() const override;
    virtual bool IsRoot() const override;

    virtual void AddTau(const Eigen::VectorXd &tau) override;
    virtual const cSpAlg::tSpVec &GetTau() const override;
    virtual void ApplyTau() override;
    virtual void ClearTau() override;

    virtual tVector CalcWorldPos() const override;
    virtual tVector CalcWorldPos(const tVector &local_pos) const override;
    virtual tVector CalcWorldVel() const override;
    virtual tVector CalcWorldVel(const tVector &local_pos) const override;
    virtual tVector CalcWorldAngVel() const override;
    virtual double GetTorqueLimit() const override;
    virtual double GetForceLimit() const override;
    virtual void SetTorqueLimit(double lim) override;
    virtual void SetForceLimit(double lim) override;

    virtual tVector
    GetParentPos() const override; // in parent link's coordinates
    virtual tVector GetChildPos() const override; // in child link's coordinates
    virtual tQuaternion GetChildRot() const override;
    virtual tMatrix BuildJointChildTrans() const override;
    virtual tMatrix BuildJointParentTrans() const override;
    virtual tVector CalcAxisWorld() const override;
    virtual tVector GetAxisRel() const override;
    virtual int GetPoseSize() const;

    virtual bool HasParent() const override;
    virtual bool HasChild() const override;

    virtual const std::shared_ptr<cSimBodyLink> &GetParent() const override;
    virtual const std::shared_ptr<cSimBodyLink> &GetChild() const override;

    virtual void ClampTotalTorque(tVector &out_torque) const override;
    virtual void ClampTotalForce(tVector &out_force) const override;

    virtual const tVector &GetLimLow() const override;
    virtual const tVector &GetLimHigh() const override;
    virtual bool HasJointLim() const override;

    virtual int GetParamSize() const override;
    virtual void BuildPose(Eigen::VectorXd &out_pose) const override;
    virtual void BuildVel(Eigen::VectorXd &out_vel) const override;
    virtual void SetPose(const Eigen::VectorXd &pose) override;
    virtual void SetVel(const Eigen::VectorXd &vel) override;

    virtual tVector GetTotalTorque() const override;
    virtual tVector GetTotalForce() const override;
    virtual Joint *GetInternalJoint() const;
    virtual double GetJointDiffWeight() const;

protected:
    cKinTree::eJointType mType;
    std::shared_ptr<cWorldBase> mWorld;
    cRobotModelDynamics *mRobotModel;

    tVector mLimLow, mLimHigh;
    double mTorqueLimit, mForceLimit;
    cKinTree::eJointType mJointType;
    Joint *mJoint;
    int mJointId;
    // all torques and forces are in local coordinates
    cSpAlg::tSpVec mTotalTau;

    virtual void InitLimit();
    virtual cKinTree::eJointType FetchJointType() const;

    virtual tVector CalcParentLocalPos(const tVector &local_pos) const override;
    virtual tVector CalcChildLocalPos(const tVector &local_pos) const override;

    virtual void SetTotalTorque(const tVector &torque) override;
    virtual void SetTotalForce(const tVector &force) override;
    virtual void ApplyTauRevolute() override;
    virtual void ApplyTauPlanar() override;
    virtual void ApplyTauPrismatic() override;
    virtual void ApplyTauSpherical() override;
};
