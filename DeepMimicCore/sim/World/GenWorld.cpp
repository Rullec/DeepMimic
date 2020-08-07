#include "sim/World/GenWorld.h"
// #include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"
// #include "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h"
// #include "BulletDynamics/MLCPSolvers/btDantzigSolver.h"
// #include "BulletDynamics/MLCPSolvers/btMLCPSolver.h"

#include "BulletGenDynamics/btGenWorld.h"
#include "sim/SimItems/SimBodyLink.h"
#include "sim/SimItems/SimBox.h"
#include "sim/SimItems/SimCapsule.h"
#include "sim/SimItems/SimCharacter.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/SimItems/SimCylinder.h"
#include "sim/SimItems/SimJoint.h"
#include "sim/SimItems/SimPlane.h"
#include "sim/SimItems/SimSphere.h"
#include "util/LogUtil.hpp"
#include <iostream>
using namespace std;

cGenWorld::cGenWorld() : cWorldBase(eWorldType::GENERALIZED_WORLD)
{
    mDefaultLinearDamping = 0;
    mDefaultAngularDamping = 0;
    mTimeStep = 0;
    mbtGenWorld = new btGeneralizeWorld();
    mbtDynamicsWorldInternal = nullptr;
}

cGenWorld::~cGenWorld() {}

void cGenWorld::Init(const tParams &params)
{
    mParams = params;

    mbtGenWorld->Init(params.mGenWorldConfig);
    mbtDynamicsWorldInternal =
        dynamic_cast<btDynamicsWorld *>(mbtGenWorld->GetInternalWorld());
    SetGravity(params.mGravity);

    mContactManager.Init();
    mPerturbManager.Clear();
}

void cGenWorld::Reset()
{

    mTimeStep = 0;
    mContactManager.Reset();
    mPerturbManager.Clear();
    MIMIC_WARN("world reset hasn't been fully implemented yet");

    mbtGenWorld->Reset();
    // mSimWorld->clearForces();
    // mSolver->reset();
    // mBroadPhase->resetPool(mCollisionDispatcher.get());

    // btOverlappingPairCache *pair_cache =
    //     mSimWorld->getBroadphase()->getOverlappingPairCache();
    // btBroadphasePairArray &pair_array =
    // pair_cache->getOverlappingPairArray(); for (int i = 0; i <
    // pair_array.size(); ++i)
    // {
    //     pair_cache->cleanOverlappingPair(pair_array[i],
    //                                      mSimWorld->getDispatcher());
    // }
}

void cGenWorld::Update(double time_elapsed)
{
    // std::cout <<"void cLCPWorld::Update(double time_elapsed)" << std::endl;
    time_elapsed = std::max(0.0, time_elapsed);
    mPerturbManager.Update(time_elapsed); // 似乎对于扰动有一个统一的管理。

    // multi steps: motion data didn't sync between CollisionObj and btMultiBody
    // btScalar timestep = static_cast<btScalar>(time_elapsed);
    // btScalar subtimestep = timestep / mParams.mNumSubsteps;
    // mSimWorld->stepSimulation(timestep, mParams.mNumSubsteps, subtimestep);
    // mTimeStep = subtimestep;

    btScalar timestep = static_cast<btScalar>(time_elapsed);
    MIMIC_ASSERT(mbtGenWorld != nullptr);
    mbtGenWorld->StepSimulation(timestep); // single step: works well
    mTimeStep = timestep;

    mContactManager.Update();
}

void cGenWorld::AddRigidBody(cSimRigidBody &obj)
{
    // MIMIC_WARN("AddRigidBody hasn't been implemenbted\n");
    mbtGenWorld->AddStaticBody(
        static_cast<btCollisionObject *>(obj.GetSimBody().get()), 0, "ground");
    // const std::unique_ptr<btRigidBody> &body = obj.GetSimBody();

    // short col_group = obj.GetColGroup();
    // short col_mask = obj.GetColMask();
    // col_mask |= cContactManager::gFlagRayTest;

    // obj.SetDamping(mDefaultLinearDamping, mDefaultAngularDamping);
    // mSimWorld->addRigidBody(body.get(), col_group, col_mask);

    // Constrain(obj);
}

void cGenWorld::RemoveRigidBody(cSimRigidBody &obj)
{
    // mSimWorld->removeRigidBody(obj.GetSimBody().get());
    MIMIC_INFO("RemoveRigidBody hasn't been implemented\n");
}

void cGenWorld::AddCollisionObject(btCollisionObject *col_obj,
                                   int col_filter_group, int col_filter_mask)
{
    col_filter_mask |= cContactManager::gFlagRayTest;
    MIMIC_ERROR("AddCollisionObject hasn't been finished");
    // mSimWorld->addCollisionObject(col_obj, col_filter_group,
    // col_filter_mask);
}

void cGenWorld::RemoveCollisionObject(btCollisionObject *col_obj)
{
    MIMIC_ERROR("RemoveCollisionObject hasn't been finished");
    // mSimWorld->removeCollisionObject(col_obj);
}

void cGenWorld::AddCharacter(cSimCharacterBase *sim_charbase)
{
    cSimCharacterGen *sim_char = dynamic_cast<cSimCharacterGen *>(sim_charbase);
    mbtGenWorld->AddMultibody(sim_char);
    // sim_char.SetLinearDamping(mDefaultLinearDamping);
    // sim_char.SetAngularDamping(mDefaultAngularDamping);
    // mSimWorld->addMultiBody(sim_char.GetMultiBody().get());

    // const auto &constraints = sim_char.GetConstraints();
    // for (int c = 0; c < static_cast<int>(constraints.size()); ++c)
    // {
    //     mSimWorld->addMultiBodyConstraint(constraints[c].get());
    // }
}

void cGenWorld::RemoveCharacter(cSimCharacterBase *sim_char)
{
    // const auto &constraints = sim_char.GetConstraints();
    // for (int c = 0; c < static_cast<int>(constraints.size()); ++c)
    // {
    //     mSimWorld->removeMultiBodyConstraint(constraints[c].get());
    // }

    // mSimWorld->removeMultiBody(sim_char.GetMultiBody().get());
}

void cGenWorld::Constrain(cSimRigidBody &obj)
{
    Constrain(obj, tVector::Ones(), tVector::Ones());
}

void cGenWorld::Constrain(cSimRigidBody &obj, const tVector &linear_factor,
                          const tVector &angular_factor)
{
    // auto &body = obj.GetSimBody();
    // tVector lin_f = tVector::Ones();
    // tVector ang_f = tVector::Ones();
    // // BuildConsFactor(lin_f, ang_f);

    // lin_f = lin_f.cwiseProduct(linear_factor);
    // ang_f = ang_f.cwiseProduct(angular_factor);

    // body->setLinearFactor(btVector3(static_cast<btScalar>(lin_f[0]),
    //                                 static_cast<btScalar>(lin_f[1]),
    //                                 static_cast<btScalar>(lin_f[2])));
    // body->setAngularFactor(btVector3(static_cast<btScalar>(ang_f[0]),
    //                                  static_cast<btScalar>(ang_f[1]),
    //                                  static_cast<btScalar>(ang_f[2])));
}

void cGenWorld::RemoveConstraint(tConstraintHandle &handle)
{
    MIMIC_WARN("RemoveConstraint hasn't been implemented yet");
    // if (handle.IsValid())
    // {
    //     mSimWorld->removeConstraint(handle.mCons);
    // }
    // handle.Clear();
}

void cGenWorld::AddJoint(const cSimJoint &joint)
{
    MIMIC_WARN("AddJoint hasn't been implemented yet");
    // const auto &cons = joint.GetCons();
    // const auto &mult_body_cons = joint.GetMultBodyCons();
    // assert(cons == nullptr || mult_body_cons == nullptr);
    // if (cons != nullptr)
    // {
    //     mSimWorld->addConstraint(cons.get(),
    //     joint.EnableAdjacentCollision());
    // }

    // if (mult_body_cons != nullptr)
    // {
    //     mSimWorld->addMultiBodyConstraint(mult_body_cons.get());
    // }
}

void cGenWorld::RemoveJoint(cSimJoint &joint)
{
    MIMIC_WARN("Remove hasn't been implemented yet");
    // const auto &cons = joint.GetCons();
    // const auto &mult_body_cons = joint.GetMultBodyCons();
    // if (cons != nullptr)
    // {
    //     mSimWorld->removeConstraint(cons.get());
    // }

    // if (mult_body_cons != nullptr)
    // {
    //     mSimWorld->removeMultiBodyConstraint(mult_body_cons.get());
    // }
}

// void cLCPWorld::BuildConsFactor(tVector &out_linear_factor,
//                              tVector &out_angular_factor)
// {
//     out_linear_factor = tVector::Ones();
//     out_angular_factor = tVector::Ones();
// }

void cGenWorld::SetGravity(const tVector &gravity)
{
    double scale = GetScale();
    mbtGenWorld->SetGravity(gravity * scale);
    // mSimWorld->setGravity(btVector3(static_cast<btScalar>(gravity[0] *
    // scale),
    //                                 static_cast<btScalar>(gravity[1] *
    //                                 scale), static_cast<btScalar>(gravity[2]
    //                                 * scale)));
}

cContactManager::tContactHandle cGenWorld::RegisterContact(int contact_flags,
                                                           int filter_flags)
{
    return mContactManager.RegisterContact(contact_flags, filter_flags);
}

void cGenWorld::UpdateContact(const cContactManager::tContactHandle &handle)
{
    mContactManager.UpdateContact(handle);
}

const tEigenArr<cContactManager::tContactPt> &
cGenWorld::GetContactPts(const cContactManager::tContactHandle &handle) const
{
    return mContactManager.GetContactPts(handle);
}

cContactManager &cGenWorld::GetContactManager() { return mContactManager; }

bool cGenWorld::IsInContact(const cContactManager::tContactHandle &handle) const
{
    return mContactManager.IsInContact(handle);
}

void cGenWorld::RayTest(const tVector &beg, const tVector &end,
                        tRayTestResults &results) const
{
    btScalar scale = static_cast<btScalar>(GetScale());
    btVector3 bt_beg = scale * btVector3(static_cast<btScalar>(beg[0]),
                                         static_cast<btScalar>(beg[1]),
                                         static_cast<btScalar>(beg[2]));
    btVector3 bt_end = scale * btVector3(static_cast<btScalar>(end[0]),
                                         static_cast<btScalar>(end[1]),
                                         static_cast<btScalar>(end[2]));
    btCollisionWorld::ClosestRayResultCallback ray_callback(bt_beg, bt_end);

    mbtGenWorld->GetInternalWorld()->rayTest(bt_beg, bt_end, ray_callback);

    results.clear();
    if (ray_callback.hasHit())
    {
        auto &obj = ray_callback.m_collisionObject;
        const auto &hit_pt = ray_callback.m_hitPointWorld;
        tRayTestResult result;
        result.mObj = static_cast<cSimObj *>(obj->getUserPointer());
        result.mHitPos = tVector(hit_pt[0], hit_pt[1], hit_pt[2], 0) / scale;

        results.push_back(result);
    }
}

void cGenWorld::AddPerturb(const tPerturb &perturb)
{
    mPerturbManager.AddPerturb(perturb);
}

const cPerturbManager &cGenWorld::GetPerturbManager() const
{
    return mPerturbManager;
}

tVector cGenWorld::GetGravity() const
{
    double scale = GetScale();
    return mbtGenWorld->GetGravity() / scale;
}

double cGenWorld::GetScale() const { return mParams.mScale; }

double cGenWorld::GetTimeStep() const { return mTimeStep; }

void cGenWorld::SetDefaultLinearDamping(double damping)
{
    mDefaultLinearDamping = damping;
}

double cGenWorld::GetDefaultLinearDamping() const
{
    return mDefaultLinearDamping;
}

void cGenWorld::SetDefaultAngularDamping(double damping)
{
    mDefaultAngularDamping = damping;
}

double cGenWorld::GetDefaultAngularDamping() const
{
    return mDefaultAngularDamping;
}

btBoxShape *cGenWorld::BuildBoxShape(const tVector &box_size) const
{
    btScalar scale = static_cast<btScalar>(GetScale());
    return new btBoxShape(scale *
                          btVector3(static_cast<btScalar>(box_size[0] * 0.5),
                                    static_cast<btScalar>(box_size[1] * 0.5),
                                    static_cast<btScalar>(box_size[2] * 0.5)));
}

btCapsuleShape *cGenWorld::BuildCapsuleShape(double radius, double height) const
{
    btScalar scale = static_cast<btScalar>(GetScale());
    return new btCapsuleShape(static_cast<btScalar>(scale * radius),
                              static_cast<btScalar>(scale * height));
}

btStaticPlaneShape *cGenWorld::BuildPlaneShape(const tVector &normal,
                                               const tVector &origin) const
{
    btVector3 bt_normal = btVector3(static_cast<btScalar>(normal[0]),
                                    static_cast<btScalar>(normal[1]),
                                    static_cast<btScalar>(normal[2]));
    btVector3 bt_origin = btVector3(static_cast<btScalar>(origin[0]),
                                    static_cast<btScalar>(origin[1]),
                                    static_cast<btScalar>(origin[2]));
    bt_normal.normalize();
    double scale = GetScale();
    btScalar w = static_cast<btScalar>(scale * bt_normal.dot(bt_origin));

    return new btStaticPlaneShape(bt_normal, w);
}

btSphereShape *cGenWorld::BuildSphereShape(double radius) const
{
    btScalar scale = static_cast<btScalar>(GetScale());
    return new btSphereShape(static_cast<btScalar>(scale * radius));
}

btCylinderShape *cGenWorld::BuildCylinderShape(double radius,
                                               double height) const
{
    btScalar scale = static_cast<btScalar>(GetScale());
    return new btCylinderShape(
        btVector3(static_cast<btScalar>(scale * radius),
                  static_cast<btScalar>(0.5 * scale * height),
                  static_cast<btScalar>(scale * radius)));
}

tVector cGenWorld::GetSizeBox(const cSimObj &obj) const
{
    assert(obj.GetShape() == cShape::eShapeBox);
    const btBoxShape *box_shape =
        reinterpret_cast<const btBoxShape *>(obj.GetCollisionShape());
    btVector3 half_len = box_shape->getHalfExtentsWithMargin();
    double scale = GetScale();
    return tVector(half_len[0] * 2, half_len[1] * 2, half_len[2] * 2, 0) /
           scale;
}

tVector cGenWorld::GetSizeCapsule(const cSimObj &obj) const
{
    assert(obj.GetShape() == cShape::eShapeCapsule);
    const btCapsuleShape *shape =
        reinterpret_cast<const btCapsuleShape *>(obj.GetCollisionShape());
    double scale = GetScale();
    double r = shape->getRadius();
    double h = shape->getHalfHeight();
    r /= scale;
    h /= scale;
    return tVector(2 * r, 2 * (h + r), 2 * r, 0);
}

tVector cGenWorld::GetSizePlane(const cSimObj &obj) const
{
    assert(obj.GetShape() == cShape::eShapePlane);
    const btStaticPlaneShape *shape =
        reinterpret_cast<const btStaticPlaneShape *>(obj.GetCollisionShape());
    double scale = GetScale();
    btVector3 n = shape->getPlaneNormal();
    btScalar c = shape->getPlaneConstant();
    return tVector(n[0], n[1], n[2], c / scale);
}

tVector cGenWorld::GetSizeSphere(const cSimObj &obj) const
{
    assert(obj.GetShape() == cShape::eShapeSphere);
    const btSphereShape *ball_shape =
        reinterpret_cast<const btSphereShape *>(obj.GetCollisionShape());
    double r = ball_shape->getRadius();
    double scale = GetScale();
    r /= scale;
    double d = 2 * r;
    return tVector(d, d, d, 0);
}

tVector cGenWorld::GetSizeCylinder(const cSimObj &obj) const
{
    assert(obj.GetShape() == cShape::eShapeCylinder);
    const btCylinderShape *shape =
        reinterpret_cast<const btCylinderShape *>(obj.GetCollisionShape());
    double scale = GetScale();
    double r = shape->getRadius();
    double h = 2 * shape->getHalfExtentsWithMargin()[1];
    r /= scale;
    h /= scale;
    return tVector(2 * r, h, 2 * r, 0);
}

btDynamicsWorld *cGenWorld::GetInternalWorld()
{
    return mbtDynamicsWorldInternal;
}

std::vector<btGenContactForce *> cGenWorld::GetContactInfo() const
{
    return mbtGenWorld->GetContactInfo();
}