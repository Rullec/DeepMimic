#pragma once

#include "sim/SimObj.h"

class cSimBodyLink : public cSimObj, public btDefaultMotionState
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	struct tParams
	{
		double mMass;
		int mJointID;

		tParams();
	};

	cSimBodyLink();
	virtual ~cSimBodyLink();

	virtual void Init(const std::shared_ptr<cWorld>& world, const std::shared_ptr<cMultiBody>& mult_body, const tParams& params);
	virtual tVector GetSize() const;
	
	virtual tVector GetLinearVelocity() const;
	virtual tVector GetLinearVelocity(const tVector& local_pos) const;
	virtual void SetLinearVelocity(const tVector& vel);
	virtual tVector GetAngularVelocity() const;
	virtual void SetAngularVelocity(const tVector& vel);

	virtual tVector GetInertia() const;
	virtual double GetMass() const;
	virtual double GetFriction() const;
	virtual void SetFriction(double friction);
	virtual void ApplyForce(const tVector& force);
	virtual void ApplyForce(const tVector& force, const tVector& local_pos);
	virtual void ApplyTorque(const tVector& torque);
	virtual void ClearForces();

	virtual cShape::eShape GetShape() const;
	virtual void UpdateVel(const tVector& lin_vel, const tVector& ang_vel);
	virtual const std::shared_ptr<cMultiBody>& GetMultBody() const;
	virtual int GetJointID() const;
    virtual cSimObj::eObjType GetObjType() const {return eSimBodyLink;}
protected:
	std::shared_ptr<cMultiBody> mMultiBody;
	std::unique_ptr<btMultiBodyLinkCollider> mColObj;	// 从这个地方拿到当前mesh link的真正位置

	int mJointID;
	double mMass;
	tVector mInertia;
	tVector mSize;				// 通过一个4d的vector，4个实数是不能够准确描述的
	cShape::eShape mObjShape;	// 对象的形状，盒子、胶囊、圆柱等有限个, enum。形状具体参数是从json中读到的

	tVector mLinVel;			// 4d的LinVel 线速度  4d = direct + value?
	tVector mAngVel;			// 4d的AngVel　角速度 4d = axis + value?

	virtual void InitSize(tVector& out_size) const;
	virtual cShape::eShape FetchObjShape() const;

	virtual void RemoveFromWorld();

	virtual const btCollisionObject* GetCollisionObject() const;
	virtual btCollisionObject* GetCollisionObject();
};