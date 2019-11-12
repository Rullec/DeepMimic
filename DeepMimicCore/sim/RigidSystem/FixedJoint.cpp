#include "FixedJoint.h"


FixedJoint::FixedJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_, Vector3d joint_position, int id_) :BallJoint(link, ori_position_, ori_position_w_, ori_orientation_, joint_position, id_)
{
	initGeneralizedInfo();
}

FixedJoint::~FixedJoint()
{
}

void FixedJoint::updateRotationMatrix()
{

}

void FixedJoint::computeLocalTransformDerive()
{

}

void FixedJoint::computeLocalTSecondDerive()
{

}

void FixedJoint::computeLocalTThirdDerive()
{

}

void FixedJoint::getOrientationByq(Matrix3d &m, VectorXd q)
{

}

void FixedJoint::initGeneralizedInfo()
{
	r = 0;
	generalized_position.resize(r);
	generalized_velocity.resize(r);

	jw.resize(3, r);
	jv.resize(3, r);

	jv_dot.resize(3, r);
	jw_dot.resize(3, r);

	jv_g.resize(3, r);
	jw_g.resize(3, r);
	jv_dot_g.resize(3, r);
	jw_dot_g.resize(3, r);

	mTq.resize(r);
	mTqq.resize(r);

	for (int i = 0; i < r; i++)
	{
		mTqq[i].resize(r);
	}

	if (getIfComputeThirdDerive())
	{
		mTqqq.resize(r);
		for (int i = 0; i < r; i++)
		{
			mTqqq[i].resize(r);
			for (int j = 0; j < r; j++)
			{
				mTqqq[i][j].resize(r);
			}
		}
	}

	R_m.resize(r);
	R_m_firstDeriv.resize(r);
	R_m_secondDerive.resize(r);
	R_m_thirdDerive.resize(r);
}
