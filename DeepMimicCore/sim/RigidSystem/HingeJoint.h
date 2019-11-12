#pragma once
#include "BallJoint.h"

class HingeJoint :public BallJoint
{
public:
	HingeJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d joint_position, int id_, int hingeType);
	//HingeJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_, Vector3d joint_position, int id_, int hingeType, Matrix3d hingeOrientation);

	~HingeJoint();
	Matrix4d hingeOrientation_local;
	virtual void updateJacobiByGivenPosition(Vector3d x_position, MatrixXd &JK_v_);
	virtual void updateJacobiByGivenRestPosition(Vector3d x_position, MatrixXd &JK_v_);
protected:

	//error function
	virtual void applyOriOrientation();
	//
	virtual void applyOriOrientationtomTq();
	virtual void applyOriOrientationtomTqq();
	virtual void applyOriOrientationtomTqqq();

	virtual void updateRotationMatrix();

	virtual void computeLocalTransformDerive();
	virtual void computeLocalTSecondDerive();
	virtual void computeLocalTThirdDerive();

	virtual void getOrientationByq(Matrix3d &m, VectorXd q);


	virtual void initGeneralizedInfo();


	//0 x 1 y 2 z
	int hingetype;

	



};

