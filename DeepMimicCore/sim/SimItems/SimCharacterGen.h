#pragma once
#include "BulletGenDynamics/btGenModel/RobotModelDynamics.h"
#include "sim/SimItems/SimCharacterBase.h"

class cSimBodyLinkGen;
class cSimBodyJointGen;
class cSimCharacterGen : public cSimCharacterBase, public cRobotModelDynamics
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cSimCharacterGen();
    virtual ~cSimCharacterGen();
    virtual bool Init(const std::shared_ptr<cWorldBase> &world,
                      const cSimCharacterBase::tParams &params) override;
    virtual void Clear() override;
    virtual void Reset() override;
    virtual void Update(double time_step) override;
    virtual void PostUpdate(double time_step) override;

    virtual int GetNumDof() const override;
    virtual int GetParamOffset(int joint_id) const override;
    virtual int GetParamSize(int joint_id) const override;
    virtual int GetRootID() const override;
    virtual tVector GetRootPos() const override;
    virtual void GetRootRotation(tVector &out_axis,
                                 double &out_theta) const override;
    virtual tQuaternion GetRootRotation() const override;
    virtual tVector GetRootVel() const override;
    virtual tVector GetRootAngVel() const override;
    virtual const Eigen::MatrixXd &GetBodyDefs() const override;
    virtual void SetRootPos(const tVector &pos) override;
    virtual void SetRootRotation(const tVector &axis, double theta) override;
    virtual void SetRootRotation(const tQuaternion &q) override;
    virtual void SetRootTransform(const tVector &pos,
                                  const tQuaternion &rot) override;

    virtual void SetRootVel(const tVector &vel) override;
    virtual void SetRootAngVel(const tVector &ang_vel) override;

    virtual tQuaternion CalcHeadingRot() const override;

    virtual int GetNumBodyParts() const override;

    virtual void SetPose(const Eigen::VectorXd &pose) override;
    virtual void SetVel(const Eigen::VectorXd &vel) override;

    virtual tVector CalcJointPos(int joint_id) const override;
    virtual tVector CalcJointVel(int joint_id) const override;
    virtual void CalcJointWorldRotation(int joint_id, tVector &out_axis,
                                        double &out_theta) const override;
    virtual tQuaternion CalcJointWorldRotation(int joint_id) const override;
    virtual tMatrix BuildJointWorldTrans(int joint_id) const override;

    virtual tVector CalcCOM() const override;
    virtual tVector CalcCOMVel() const override;
    virtual void CalcAABB(tVector &out_min, tVector &out_max) const override;
    virtual tVector GetSize() const override;

    virtual const cSimBodyJoint &GetJoint(int joint_id) const override;
    virtual cSimBodyJoint &GetJoint(int joint_id) override;
    virtual void GetChildJoint(int joint_id,
                               Eigen::VectorXd &out_child_id) override;
    virtual int GetParentJoint(int joint_id) override;
    virtual std::shared_ptr<cSimBodyLink> GetBodyPart(int idx) const override;
    virtual tVector GetBodyPartPos(int idx) const override;
    virtual tVector GetBodyPartVel(int idx) const override;
    virtual std::shared_ptr<cSimBodyLink> GetRootPart() const override;

    virtual void RegisterContacts(int contact_flags, int filter_flags) override;
    virtual void UpdateContact(int contact_flags, int filter_flags) override;
    virtual bool IsInContact() const override;
    virtual bool IsInContact(int idx) const override;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts(int idx) const override;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts() const override;
    virtual const cContactManager::tContactHandle &
    GetContactHandle() const override;
    virtual const void GetTotalContactPts(Eigen::VectorXd &) const override;

    virtual bool HasFallen() const override;
    virtual bool HasStumbled() const override;
    virtual bool HasVelExploded(double vel_threshold = 100.0) const override;

    virtual bool IsValidBodyPart(int idx) const override;
    virtual bool EnableBodyPartFallContact(int idx) const override;
    virtual void SetBodyPartFallContact(int idx, bool enable) override;

    virtual void SetController(std::shared_ptr<cCharController> ctrl) override;
    virtual void RemoveController() override;
    virtual bool HasController() const override;
    virtual const std::shared_ptr<cCharController> &GetController() override;
    virtual const std::shared_ptr<cCharController> &
    GetController() const override;
    virtual void EnableController(bool enable) override;

    virtual void ApplyForce(const tVector &force) override;
    virtual void ApplyForce(const tVector &force,
                            const tVector &lo_pos) override;
    virtual void ApplyTorque(const tVector &torque) override;
    virtual void ClearForces() override;
    virtual void ApplyControlForces(const Eigen::VectorXd &tau) override;
    virtual void PlayPossum() override;

    virtual tVector GetPartColor(int part_id) const override;
    virtual double CalcTotalMass() const override;

    virtual void SetLinearDamping(double damping) override;
    virtual void SetAngularDamping(double damping) override;

    virtual const std::shared_ptr<cWorldBase> &GetWorld() const override;

    // cSimObj Interface
    virtual tVector GetPos() const override;
    virtual void SetPos(const tVector &pos) override;
    virtual void GetRotation(tVector &out_axis,
                             double &out_theta) const override;
    virtual tQuaternion GetRotation() const override;
    virtual void SetRotation(const tVector &axis, double theta) override;
    virtual void SetRotation(const tQuaternion &q) override;
    virtual tMatrix GetWorldTransform() const override;

    virtual tVector GetLinearVelocity() const override;
    virtual tVector GetLinearVelocity(const tVector &local_pos) const override;
    virtual void SetLinearVelocity(const tVector &vel) override;
    virtual tVector GetAngularVelocity() const override;
    virtual void SetAngularVelocity(const tVector &vel) override;

    virtual short GetColGroup() const override;
    virtual void SetColGroup(short col_group) override;
    virtual short GetColMask() const override;
    virtual void SetColMask(short col_mask) override;
    virtual void SetEnablejointTorqueControl(bool v_) override;
    virtual std::string GetCharFilename() override;

    // cCharacter Interface rewrite
    virtual void RotateRoot(const tQuaternion &rot) override;
    virtual double CalcHeading() const override;
    virtual tMatrix BuildOriginTrans() const override;
    virtual bool IsEndEffector(int joint_id) const override;
    virtual double CalcJointChainLength(int joint_id) const override;
    virtual int CalcNumEndEffectors() const override;
    virtual double GetJointDiffWeight(int joint_id) const override;
    virtual bool HasDrawShapes() const override;
    virtual const Eigen::MatrixXd &GetDrawShapeDefs() const override;
    virtual const std::shared_ptr<cDrawMesh> &GetMesh(int i) const override;
    virtual int GetNumMeshes() const override;
    virtual std::string GetBodyName(int id) const override;
    virtual std::string GetJointName(int id) const override;
    virtual std::string GetDrawShapeName(int id) const override;
    virtual int GetNumJoints() const override;
    virtual tVectorXd ConvertPoseToq(const tVectorXd &pose) const;
    virtual void CalcCOMAndCOMVel(const tVectorXd &pose,
                                  const tVectorXd &pose_vel, tVector &com,
                                  tVector &com_vel);
    virtual void SetqAndqdot(const tVectorXd &q,
                             const tVectorXd &qdot) override;

protected:
    /// vars
    std::shared_ptr<cCharController> mController;

    /// methods
    virtual void InitDefaultState() override;
    virtual bool LoadBodyDefs(const std::string &char_file,
                              Eigen::MatrixXd &out_body_defs) override;

    virtual bool BuildSimBody(const tParams &params) override;

    virtual bool BuildBodyLinks() override;
    virtual btCollisionShape *
    BuildCollisionShape(const cShape::eShape shape,
                        const tVector &shape_size) override;
    virtual void InitParamMatrix(const std::string &char_file);
    virtual bool BuildJoints() override;
    virtual void BuildConsFactor(int joint_id, tVector &out_linear_factor,
                                 tVector &out_angular_factor) const override;
    virtual void
    BuildRootConsFactor(cKinTree::eJointType joint_type,
                        tVector &out_linear_factor,
                        tVector &out_angular_factor) const override;
    virtual bool FixedBase() const override;
    virtual void RemoveFromWorld() override;

    virtual void ClearJointTorques() override;
    virtual void UpdateJoints() override;
    virtual void UpdateLinkPos() override;
    virtual void UpdateLinkVel() override;

    virtual short GetPartColGroup(int part_id) const override;
    virtual short GetPartColMask(int part_id) const override;

    virtual void BuildJointPose(int joint_id,
                                Eigen::VectorXd &out_pose) const override;
    virtual void BuildJointVel(int joint_id,
                               Eigen::VectorXd &out_vel) const override;

    virtual void BuildPose(Eigen::VectorXd &out_pose) const override;
    virtual void BuildVel(Eigen::VectorXd &out_vel) const override;

    virtual bool CheckFallContact() const override;
    virtual const btCollisionObject *GetCollisionObject() const override;
    virtual btCollisionObject *GetCollisionObject() override;
    void Test();
    eRotationOrder GetRotationOrder() const;
    virtual void Setqdot(const tVectorXd &qdot) override;
    tVectorXd ConvertqToPose(const tVectorXd &pose) const;
    tVectorXd ConvertqdotToPoseVel(const tVectorXd &pose) const;

    tVectorXd ConvertPosevelToqdot(const tVectorXd &pose_vel) const;

    // buffer for storaging all links & joints
    // std::vector<Link *> mLinkArray;
    // std::vector<Joint *> mJointArray;
    std::vector<std::shared_ptr<cSimBodyLinkGen>> mLinkGenArray;
    std::vector<std::shared_ptr<cSimBodyJointGen>> mJointGenArray;
    tMatrixXd mBodyDefs;
    // // used to form the return value in `GetBodyPart`
    // std::vector<std::shared_ptr<cSimBodyLink>> mLinkBaseArray;
    // std::vector<std::shared_ptr<cSimBodyJoint>> mJointBaseArray;
};