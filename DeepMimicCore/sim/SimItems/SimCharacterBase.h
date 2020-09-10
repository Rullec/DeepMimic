#pragma once

// #include "BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h"
// #include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
// #include "MultiBody.h"
#include "anim/Character.h"
#include "sim/Controller/CharController.h"
#include "sim/SimItems/SimBodyJoint.h"
#include "sim/SimItems/SimBodyLink.h"
#include "sim/SimItems/SimObj.h"
#include "sim/World/WorldBase.h"

enum eSimCharacterType
{
    Generalized,
    Featherstone,
    NUM_SIMCHAR_TYPE
};

class cSimCharacterBase : public virtual cCharacter, public virtual cSimObj
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct tParams
    {
        // 一个角色的参数是什么? 就是tParams
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        tParams();

        int mID;
        std::string mCharFile; // 角色文件: character的真正参数存储
        std::string mStateFile;
        tVector mInitPos;
        bool mLoadDrawShapes;
        bool mEnableContactFall;
    };

    cSimCharacterBase(eSimCharacterType type);
    virtual ~cSimCharacterBase() = 0;

    virtual bool Init(const std::shared_ptr<cWorldBase> &world,
                      const tParams &params) = 0;
    virtual void Clear() = 0;
    virtual void Reset() = 0;
    virtual void Update(double time_step) = 0;
    virtual void PostUpdate(double time_step) = 0;

    virtual tVector GetRootPos() const = 0;
    virtual void GetRootRotation(tVector &out_axis,
                                 double &out_theta) const = 0;
    virtual tQuaternion GetRootRotation() const = 0;
    virtual tVector GetRootVel() const = 0;
    virtual tVector GetRootAngVel() const = 0;
    virtual const Eigen::MatrixXd &GetBodyDefs() const = 0;
    virtual void SetRootPos(const tVector &pos) = 0;
    virtual void SetRootRotation(const tVector &axis, double theta) = 0;
    virtual void SetRootRotation(const tQuaternion &q) = 0;
    virtual void SetRootTransform(const tVector &pos,
                                  const tQuaternion &rot) = 0;

    virtual void SetRootVel(const tVector &vel) = 0;
    virtual void SetRootAngVel(const tVector &ang_vel) = 0;

    virtual tQuaternion CalcHeadingRot() const = 0;

    virtual int GetNumBodyParts() const = 0;

    virtual void SetPose(const Eigen::VectorXd &pose) = 0;
    virtual void SetVel(const Eigen::VectorXd &vel) = 0;

    virtual tVector CalcJointPos(int joint_id) const = 0;
    virtual tVector CalcJointVel(int joint_id) const = 0;
    virtual void CalcJointWorldRotation(int joint_id, tVector &out_axis,
                                        double &out_theta) const = 0;
    virtual tQuaternion CalcJointWorldRotation(int joint_id) const = 0;
    virtual tMatrix BuildJointWorldTrans(int joint_id) const = 0;

    virtual tVector CalcCOM() const = 0;
    virtual tVector CalcCOMVel() const = 0;
    virtual void CalcAABB(tVector &out_min, tVector &out_max) const = 0;
    virtual tVector GetSize() const = 0;

    virtual const cSimBodyJoint &GetJoint(int joint_id) const = 0;
    virtual cSimBodyJoint &GetJoint(int joint_id) = 0;
    virtual void GetChildJoint(int joint_id, Eigen::VectorXd &out_child_id) = 0;
    virtual int GetParentJoint(int joint_id) = 0;
    virtual std::shared_ptr<cSimBodyLink> GetBodyPart(int idx) const = 0;
    virtual tVector GetBodyPartPos(int idx) const = 0;
    virtual tVector GetBodyPartVel(int idx) const = 0;
    virtual std::shared_ptr<cSimBodyLink> GetRootPart() const = 0;

    virtual void RegisterContacts(int contact_flags, int filter_flags) = 0;
    virtual void UpdateContact(int contact_flags, int filter_flags) = 0;
    virtual bool IsInContact() const = 0;
    virtual bool IsInContact(int idx) const = 0;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts(int idx) const = 0;
    virtual const tEigenArr<cContactManager::tContactPt> &
    GetContactPts() const = 0;
    virtual const cContactManager::tContactHandle &GetContactHandle() const = 0;
    virtual const void GetTotalContactPts(Eigen::VectorXd &) const = 0;
    virtual eSimCharacterType GetCharType() const;
    virtual bool HasFallen() const = 0;
    virtual bool HasStumbled() const = 0;
    virtual bool HasVelExploded(double vel_threshold = 100.0) const = 0;

    virtual bool IsValidBodyPart(int idx) const = 0;
    virtual bool EnableBodyPartFallContact(int idx) const = 0;
    virtual void SetBodyPartFallContact(int idx, bool enable) = 0;

    virtual void SetController(std::shared_ptr<cCharController> ctrl) = 0;
    virtual void RemoveController() = 0;
    virtual bool HasController() const = 0;
    virtual const std::shared_ptr<cCharController> &GetController() = 0;
    virtual const std::shared_ptr<cCharController> &GetController() const = 0;
    virtual void EnableController(bool enable) = 0;

    virtual void ApplyForce(const tVector &force) = 0;
    virtual void ApplyForce(const tVector &force, const tVector &local_pos) = 0;
    virtual void ApplyTorque(const tVector &torque) = 0;
    virtual void ClearForces() = 0;
    virtual void ApplyControlForces(const Eigen::VectorXd &tau) = 0;
    virtual void PlayPossum() = 0;

    virtual tVector GetPartColor(int part_id) const = 0;
    virtual double CalcTotalMass() const = 0;

    virtual void SetLinearDamping(double damping) = 0;
    virtual void SetAngularDamping(double damping) = 0;

    virtual const std::shared_ptr<cWorldBase> &GetWorld() const = 0;

    // cSimObj Interface
    virtual tVector GetPos() const = 0;
    virtual void SetPos(const tVector &pos) = 0;
    virtual void GetRotation(tVector &out_axis, double &out_theta) const = 0;
    virtual tQuaternion GetRotation() const = 0;
    virtual void SetRotation(const tVector &axis, double theta) = 0;
    virtual void SetRotation(const tQuaternion &q) = 0;
    virtual tMatrix GetWorldTransform() const = 0;

    virtual tVector GetLinearVelocity() const = 0;
    virtual tVector GetLinearVelocity(const tVector &local_pos) const = 0;
    virtual void SetLinearVelocity(const tVector &vel) = 0;
    virtual tVector GetAngularVelocity() const = 0;
    virtual void SetAngularVelocity(const tVector &vel) = 0;

    virtual short GetColGroup() const = 0;
    virtual void SetColGroup(short col_group) = 0;
    virtual short GetColMask() const = 0;
    virtual void SetColMask(short col_mask) = 0;
    virtual void SetEnablejointTorqueControl(bool v_) = 0;
    virtual std::string GetCharFilename() = 0;

protected:
    virtual bool LoadBodyDefs(const std::string &char_file,
                              Eigen::MatrixXd &out_body_defs) = 0;

    virtual bool BuildSimBody(const tParams &params) = 0;

    virtual bool BuildBodyLinks() = 0;
    virtual btCollisionShape *
    BuildCollisionShape(const cShape::eShape shape,
                        const tVector &shape_size) = 0;

    virtual bool BuildJoints() = 0;
    virtual void BuildConsFactor(int joint_id, tVector &out_linear_factor,
                                 tVector &out_angular_factor) const = 0;
    virtual void BuildRootConsFactor(cKinTree::eJointType joint_type,
                                     tVector &out_linear_factor,
                                     tVector &out_angular_factor) const = 0;
    virtual bool FixedBase() const = 0;
    virtual void RemoveFromWorld() = 0;

    virtual void ClearJointTorques() = 0;
    virtual void UpdateJoints() = 0;
    virtual void UpdateLinkPos() = 0;
    virtual void UpdateLinkVel() = 0;

    virtual short GetPartColGroup(int part_id) const = 0;
    virtual short GetPartColMask(int part_id) const = 0;

    virtual void BuildJointPose(int joint_id,
                                Eigen::VectorXd &out_pose) const = 0;
    virtual void BuildJointVel(int joint_id,
                               Eigen::VectorXd &out_vel) const = 0;

    virtual void BuildPose(Eigen::VectorXd &out_pose) const = 0;
    virtual void BuildVel(Eigen::VectorXd &out_vel) const = 0;

    virtual bool CheckFallContact() const = 0;
    virtual const btCollisionObject *GetCollisionObject() const = 0;
    virtual btCollisionObject *GetCollisionObject() = 0;

    eSimCharacterType mSimcharType;
};