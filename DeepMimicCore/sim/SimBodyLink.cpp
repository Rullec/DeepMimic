﻿#include "SimBodyLink.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"

cSimBodyLink::tParams::tParams()
{
	mMass = 0;
	mJointID = gInvalidIdx;
}

cSimBodyLink::cSimBodyLink()
{
	mMass = 0;
	mInertia = tVector::Zero();
	mJointID = gInvalidIdx;
	mSize.setZero();
	mObjShape = cShape::eShapeNull;
	mLinVel.setZero();
	mAngVel.setZero();
}

cSimBodyLink::~cSimBodyLink()
{
	RemoveFromWorld();
}

void cSimBodyLink::Init(const std::shared_ptr<cWorld>& world, const std::shared_ptr<cMultiBody>& mult_body, const tParams& params)
{
	RemoveFromWorld();
	
	mJointID = params.mJointID;
	mMass = params.mMass;
	const btVector3 bt_inertia = mult_body->getLinkInertia(mJointID);
	mInertia = tVector(bt_inertia[0], bt_inertia[1], bt_inertia[2], 0);
	
	mWorld = world;
	mMultiBody = mult_body;
	mType = eTypeDynamic;

	mLinVel.setZero();
	mAngVel.setZero();

	mColObj = std::unique_ptr<btMultiBodyLinkCollider>(mMultiBody->getLink(mJointID).m_collider);	//对撞机, 碰撞器?
	mColObj->setUserPointer(this);
	mColShape = std::unique_ptr<btCollisionShape>(mColObj->getCollisionShape());

	mObjShape = FetchObjShape();
	InitSize(mSize);
	this->GetWorldTransform();
}

tVector cSimBodyLink::GetSize() const
{
	return mSize;
}

tVector cSimBodyLink::GetLinearVelocity() const
{
	return mLinVel;
}

tVector cSimBodyLink::GetLinearVelocity(const tVector& local_pos) const
{
	tVector ang_vel = GetAngularVelocity();
	tVector lin_vel = GetLinearVelocity();
	tQuaternion rot = GetRotation();
	tVector world_pos = cMathUtil::QuatRotVec(rot, local_pos);
	tVector vel = lin_vel + ang_vel.cross3(world_pos);
	return vel;
}

void cSimBodyLink::SetLinearVelocity(const tVector& vel)
{
	assert(false); // unsupported
}


tVector cSimBodyLink::GetAngularVelocity() const
{
	// 什么坐标系下的? 是世界坐标系吗?
	return mAngVel;
}

void cSimBodyLink::SetAngularVelocity(const tVector& vel)
{
	assert(false); // unsupported
}

double cSimBodyLink::GetMass() const
{
	return mMass;
}

tVector cSimBodyLink::GetInertia() const
{
	return mInertia;
}

double cSimBodyLink::GetFriction() const
{
	return mColObj->getFriction();
}

void cSimBodyLink::SetFriction(double friction)
{
	mColObj->setFriction(friction);
}

void cSimBodyLink::ApplyForce(const tVector& force)
{
	// bullet中的apply force
	btScalar scale = static_cast<btScalar>(mWorld->GetScale());
	mMultiBody->addLinkForce(mJointID, scale * btVector3(force[0], force[1], force[2]));
}

void cSimBodyLink::ApplyForce(const tVector& force, const tVector& local_pos)
{
	btScalar scale = static_cast<btScalar>(mWorld->GetScale());
	tMatrix world_mat = GetWorldTransform();
	tVector world_pos = world_mat * local_pos;
	tVector torque = world_pos.cross3(force);

	ApplyForce(force);
	ApplyTorque(torque);
}

void cSimBodyLink::ApplyTorque(const tVector& torque)
{
	btScalar scale = static_cast<btScalar>(mWorld->GetScale());
	mMultiBody->addLinkTorque(mJointID, scale * scale * btVector3(torque[0], torque[1], torque[2]));
	
}

void cSimBodyLink::ClearForces()
{
	mMultiBody->clearForcesAndTorques();
}

cShape::eShape cSimBodyLink::GetShape() const
{
	return mObjShape;
}

void cSimBodyLink::UpdateVel(const tVector& lin_vel, const tVector& ang_vel)
{
	mLinVel = lin_vel;
	mAngVel = ang_vel;
}

const std::shared_ptr<cMultiBody>& cSimBodyLink::GetMultBody() const
{
	return mMultiBody;
}

int cSimBodyLink::GetJointID() const
{
	return mJointID;
}

const btCollisionObject* cSimBodyLink::GetCollisionObject() const
{
	return mColObj.get();
}

btCollisionObject* cSimBodyLink::GetCollisionObject()
{
	return mColObj.get();
}

void cSimBodyLink::InitSize(tVector& out_size) const
{
	out_size.setZero();
	switch (mObjShape)
	{
	case cShape::eShapeBox:
		out_size = mWorld->GetSizeBox(*this);
		break;
	case cShape::eShapeCapsule:
		out_size = mWorld->GetSizeCapsule(*this);
		break;
	case cShape::eShapeSphere:
		out_size = mWorld->GetSizeSphere(*this);
		break;
	case cShape::eShapeCylinder:
		out_size = mWorld->GetSizeCylinder(*this);
		break;
	default:
		printf("Unsupported body link shape\n");
		assert(false);
		break;
	}
}

cShape::eShape cSimBodyLink::FetchObjShape() const
{
	cShape::eShape obj_shape = cShape::eShapeNull;
	int shape_type = mColShape->getShapeType();
	switch (shape_type)
	{
	case BOX_SHAPE_PROXYTYPE:
		obj_shape = cShape::eShapeBox;
		break;
	case CAPSULE_SHAPE_PROXYTYPE:
		obj_shape = cShape::eShapeCapsule;
		break;
	case SPHERE_SHAPE_PROXYTYPE:
		obj_shape = cShape::eShapeSphere;
		break;
	case CYLINDER_SHAPE_PROXYTYPE:
		obj_shape = cShape::eShapeCylinder;
		break;
	default:
		printf("Unsupported object shape\n");
		assert(false);
		break;
	}
	return obj_shape;
}

void cSimBodyLink::RemoveFromWorld()
{
	if (mWorld != nullptr && mColObj != nullptr)
	{
		mWorld->RemoveCollisionObject(GetCollisionObject());
		mWorld.reset();
		mColShape.reset();
		mColObj.reset();
	}
}