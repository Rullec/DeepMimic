#pragma once
#include "BallJoint.h"

class UniversalJoint:public BallJoint
{
public:
	UniversalJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d joint_position, int id_);
	~UniversalJoint();

	virtual void updateJointState(VectorXd &q, VectorXd &q_dot, bool updateDynamic = true);
	virtual void clampRotation(VectorXd& _q, VectorXd& _qdot);

protected:

	virtual void updateLocalTransform();

	virtual void updateRotationMatrix();

	virtual void computeLocalTransformDerive();
	virtual void computeLocalTSecondDerive();
	virtual void computeLocalTThirdDerive();

	virtual void getOrientationByq(Matrix3d &m, VectorXd q);

	virtual void initGeneralizedInfo();

};

