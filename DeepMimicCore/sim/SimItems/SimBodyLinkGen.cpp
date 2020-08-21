#include "SimBodyLinkGen.h"
#include "BulletGenDynamics/btGenModel/Link.h"
#include "BulletGenDynamics/btGenModel/RobotCollider.h"
#include "util/LogUtil.h"

cSimBodyLinkGen::cSimBodyLinkGen()
{
    mMultiBody = nullptr;
    mLink = nullptr;
    mLinkCollider = nullptr;
    mLinkId = -1;
    mEnableContactFall = true;
}

cSimBodyLinkGen::~cSimBodyLinkGen() {}
tVector cSimBodyLinkGen::GetPos() const
{

    return cMathUtil::Expand(mLink->GetWorldPos(), 0);
}
void cSimBodyLinkGen::SetPos(const tVector &pos) { MIMIC_ERROR("no"); }
void cSimBodyLinkGen::GetRotation(tVector &out_axis, double &out_theta) const
{
    cMathUtil::QuaternionToAxisAngle(GetRotation(), out_axis, out_theta);
}
tQuaternion cSimBodyLinkGen::GetRotation() const
{
    return cMathUtil::RotMatToQuaternion(
        cMathUtil::ExpandMat(mLink->GetWorldOrientation()));
}
void cSimBodyLinkGen::SetRotation(const tVector &axis, double theta)
{
    MIMIC_ERROR("no");
}
void cSimBodyLinkGen::SetRotation(const tQuaternion &q) { MIMIC_ERROR("no"); }
tMatrix cSimBodyLinkGen::GetWorldTransform() const
{
    return mLink->GetGlobalTransform();
}
tMatrix cSimBodyLinkGen::GetLocalTransform() const
{
    return mLink->GetLocalTransform();
}

/**
 * \brief               Init the wrapper of Link in RobotModelDyanmcis
 */
#include <iostream>
void cSimBodyLinkGen::Init(const std::shared_ptr<cWorldBase> &world,
                           cRobotModelDynamics *multibody, int link_id)
{
    // init all base variables for cSimObj
    mRobotModel = multibody;
    mLinkId = link_id;
    mLink = static_cast<Link *>(multibody->GetLinkById(link_id));
    mLinkCollider = multibody->GetLinkCollider(link_id);
    mLinkCollider->setUserPointer(this);
    // std::cout << "init link gen ptr = " << mLinkCollider->getUserPointer()
    //           << std::endl;

    // dont' need to init variables of cSimBodyLink
    mWorld = world;
    mColShape =
        std::unique_ptr<btCollisionShape>(mLinkCollider->getCollisionShape());
    mType = eType::eTypeDynamic;
    mColGroup = 0;
    mColMask = 0;
}

tVector cSimBodyLinkGen::GetSize() const
{
    return cMathUtil::Expand(mLink->GetMeshScale(), 0);
}

tVector cSimBodyLinkGen::GetLinearVelocity() const
{
    return cMathUtil::Expand(mLink->GetLinkVel(), 0);
}
tVector cSimBodyLinkGen::GetLinearVelocity(const tVector &local_pos) const
{
    tMatrixXd jac;
    mLink->ComputeJacobiByGivenPointTotalDOF(jac, local_pos);
    return cMathUtil::Expand(jac * mRobotModel->Getqdot(), 0);
}
void cSimBodyLinkGen::SetLinearVelocity(const tVector &vel)
{
    MIMIC_ERROR("SetLinearVelocity shouldn't be called");
}
tVector cSimBodyLinkGen::GetAngularVelocity() const
{
    return cMathUtil::Expand(mLink->GetLinkOmega(), 0);
}
void cSimBodyLinkGen::SetAngularVelocity(const tVector &vel)
{

    MIMIC_ERROR("SetAngularVelocity shouldn't be called");
}

tVector cSimBodyLinkGen::GetInertia() const
{
    return cMathUtil::Expand(mLink->GetInertiaTensorBody().diagonal(), 3);
}
double cSimBodyLinkGen::GetMass() const { return mLink->GetMass(); }
double cSimBodyLinkGen::GetFriction() const
{
    MIMIC_ERROR("GetFriction shouldn't be called");
}
void cSimBodyLinkGen::SetFriction(double friction)
{
    MIMIC_ERROR("SetFriction shouldn't be called");
}
void cSimBodyLinkGen::ApplyForce(const tVector &force)
{
    mRobotModel->ApplyForce(mLinkId, force, mLink->GetWorldPos().segment(0, 3));
}
void cSimBodyLinkGen::ApplyForce(const tVector &force, const tVector &local_pos)
{
    MIMIC_ASSERT(std::fabs(local_pos[3] - 1) < 1e-10);
    mRobotModel->ApplyForce(mLinkId, force,
                            mLink->GetGlobalTransform() * local_pos);
}
void cSimBodyLinkGen::ApplyTorque(const tVector &torque)
{
    mRobotModel->ApplyJointTorque(mLinkId, torque);
}
void cSimBodyLinkGen::ClearForces()
{
    MIMIC_ERROR("ClearForces shouldn't be called");
}

cShape::eShape cSimBodyLinkGen::GetShape() const
{
    cShape::eShape shape = cShape::eShape::eShapeNull;
    switch (mLink->GetShapeType())
    {
    case ShapeType::BOX_SHAPE:
        shape = cShape::eShape::eShapeBox;
        break;
    case ShapeType::CAPSULE_SHAPE:
        shape = cShape::eShape::eShapeCapsule;
        break;
    case ShapeType::SPHERE_SHAPE:
        shape = cShape::eShape::eShapeSphere;
        break;
    case ShapeType::CYLINDER:
        shape = cShape::eShape::eShapeCylinder;
        break;
    default:
        MIMIC_ERROR("invalid shape type {}", mLink->GetShapeType());
        break;
    }
    return shape;
}
void cSimBodyLinkGen::UpdateVel(const tVector &lin_vel, const tVector &ang_vel)
{
    mLink->SetLinkVel(lin_vel.segment(0, 3));
    mLink->SetLinkOmega(ang_vel.segment(0, 3));
}
int cSimBodyLinkGen::GetJointID() const { return mLinkId; }

Link *cSimBodyLinkGen::GetInternalLink() const { return mLink; }

void cSimBodyLinkGen::SetEnableFallContact(bool v) { mEnableContactFall = v; }
bool cSimBodyLinkGen::GetEnableFallContact() { return mEnableContactFall; }

const std::shared_ptr<cWorldBase> &cSimBodyLinkGen::GetWorld() const
{
    return mWorld;
}
const btCollisionObject *cSimBodyLinkGen::GetCollisionObject() const
{
    return dynamic_cast<btCollisionObject *>(mLinkCollider);
}

btCollisionObject *cSimBodyLinkGen::GetCollisionObject()
{
    return dynamic_cast<btCollisionObject *>(mLinkCollider);
}
double cSimBodyLinkGen::GetLinkMaxLength() const
{
    MIMIC_WARN("GetLinkMaxLength needs to be tested carefully");
    double length = 0;
    tVector3f mesh_scale = mLink->GetMeshScale();
    switch (mLink->GetShapeType())
    {
    case ShapeType::BOX_SHAPE:
        length = mesh_scale.norm();
        break;
    case ShapeType::CAPSULE_SHAPE:
    {
        length = mesh_scale[1] + 2 * mesh_scale[0];
        break;
    }
    case ShapeType::CYLINDER:
    {
        length =
            std::sqrt(std::pow(mesh_scale[1], 2) + std::pow(mesh_scale[0], 2));
        break;
    }
    case ShapeType::SPHERE_SHAPE:
    {
        length = 2 * mesh_scale[2];
        break;
    }

    default:
        MIMIC_ERROR("invalid link type {}", mLink->GetShapeType());
        break;
    }
    return length;
}

bool cSimBodyLinkGen::IsEndEffector() const
{
    return 0 == mLink->GetNumOfChildren();
}

bool cSimBodyLinkGen::IsInContact() const
{
    return mWorld->IsInContact(mContactHandle);
}

std::string cSimBodyLinkGen::GetName() const { return mLink->GetName(); }