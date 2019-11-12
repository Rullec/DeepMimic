#pragma once
#include "BallJoint.h"
class FixedJoint:public BallJoint
{
public:
	FixedJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_, Vector3d joint_position, int id_);
	~FixedJoint();



protected:

	virtual void updateRotationMatrix();

	virtual void computeLocalTransformDerive();
	virtual void computeLocalTSecondDerive();
	virtual void computeLocalTThirdDerive();

	virtual void getOrientationByq(Matrix3d &m, VectorXd q);

	virtual void initGeneralizedInfo();

};

