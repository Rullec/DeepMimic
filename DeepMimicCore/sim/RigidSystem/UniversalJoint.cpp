#include "UniversalJoint.h"
#include "Functions/EulerAngelRotationMatrix.h"
#include "Functions/SkewMatrix.h"
#include <iostream>
#include <iomanip> 



UniversalJoint::UniversalJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d joint_position, int id_) :BallJoint(link, ori_position_, ori_position_w_, ori_orientation_w_, joint_position, id_)
{
	this->initGeneralizedInfo();
}

UniversalJoint::~UniversalJoint()
{

}

void UniversalJoint::updateJointState(VectorXd &q, VectorXd &q_dot, bool updateDynamic /*= true*/)
{
	// rx ry rz x y z
	//std::cout << "[debug] set root q = " << q.transpose() << std::endl;
	for (int i = 0; i < r; i++)
	{
		generalized_position.data()[i] = q.data()[generalized_offset + i];
		generalized_velocity.data()[i] = q_dot.data()[generalized_offset + i];
	}

	Matrix3d m;
	
	getOrientationByq(m, generalized_position);
	
	//std::cout << "universal joint orientation: \n" << m << std::endl;
	orientation = m*ori_orientation;
	
	/*std::cout << "univeral q: " << generalized_position.transpose() << std::endl;
	std::cout << "universal joint ori_orientation: \n" << ori_orientation << std::endl;
	std::cout << "universal joint orientation: \n" << orientation << std::endl;*/

	position.data()[0] = ori_position.data()[0] + generalized_position.data()[3];	// 都是delta，而非真实的坐标 x
	position.data()[1] = ori_position.data()[1] + generalized_position.data()[4];	// y 
	position.data()[2] = ori_position.data()[2] + generalized_position.data()[5];	
	updateLocalTransform();

	//std::cout << "local transform " << std::endl;
	//std::cout << local_TransForm << std::endl;

	//update term in world frame from parent
	updatePoseInWorld();

	if (!updateDynamic)
	{
		return;
	}

	computeJacobiMatrix(q_dot);
	//updateAngularVelovity(q, q_dot);
	//compute dot term in world frame
	VectorXd velocity = JK*q_dot;
	linear_velocity.data()[0] = velocity.data()[0];
	linear_velocity.data()[1] = velocity.data()[1];
	linear_velocity.data()[2] = velocity.data()[2];

	angular_velocity.data()[0] = velocity.data()[3];
	angular_velocity.data()[1] = velocity.data()[4];
	angular_velocity.data()[2] = velocity.data()[5];
	//for debug
	//Vector3d angularvolocityerror = angular_velocity - IterativeAngularVelovity;
	////std::cout << "angular_velocity-IterativeAngularVelovity" << (angular_velocity - IterativeAngularVelovity).transpose() << std::endl;
	//if (angularvolocityerror.norm()>1e-8)
	//{
	//	std::cout << "error in Interative angular velovity" << std::endl;
	//}

}

void UniversalJoint::clampRotation(VectorXd& _q, VectorXd& _qdot)
{
	for (int i = 0; i < 3; i++)
	{
		_q.data()[generalized_offset + i];
		_qdot.data()[generalized_offset + i];

		if (_q.data()[generalized_offset + i] > M_PI)
		{
			_q.data()[generalized_offset + i] -= 2 * M_PI;
		}
		else if (_q.data()[generalized_offset + i] < -M_PI)
		{
			_q.data()[generalized_offset + i] += 2 * M_PI;
		}
	}
}

void UniversalJoint::updateLocalTransform()
{
	local_TransForm.topLeftCorner<3, 3>() = orientation;
	local_TransForm.data()[12] = position.data()[0];
	local_TransForm.data()[13] = position.data()[1];
	local_TransForm.data()[14] = position.data()[2];
}

void UniversalJoint::updateRotationMatrix()
{
	xconventionTransform(R_m[0], generalized_position[0]);
	yconventionTransform(R_m[1], generalized_position[1]);
	zconventionTransform(R_m[2], generalized_position[2]);


	xconventionRotation_dx(R_m_firstDeriv[0], generalized_position[0]);
	yconventionRotation_dy(R_m_firstDeriv[1], generalized_position[1]);
	zconventionRotation_dz(R_m_firstDeriv[2], generalized_position[2]);


	xconventionRotation_dxdx(R_m_secondDerive[0], generalized_position[0]);
	yconventionRotation_dydy(R_m_secondDerive[1], generalized_position[1]);
	zconventionRotation_dzdz(R_m_secondDerive[2], generalized_position[2]);
	

	if (getIfComputeThirdDerive())
	{
		xconventionRotation_dxdxdx(R_m_thirdDerive[0], generalized_position[0]);
		yconventionRotation_dydydy(R_m_thirdDerive[1], generalized_position[1]);
		zconventionRotation_dzdzdz(R_m_thirdDerive[2], generalized_position[2]);
		
	}
}

void UniversalJoint::computeLocalTransformDerive()
{
	mTq[0].setZero();
	mTq[1].setZero();
	mTq[2].setZero();

	mTq[0] = R_m_firstDeriv[0] * R_m[1] * R_m[2];
	mTq[1] = R_m[0] * R_m_firstDeriv[1] * R_m[2];
	mTq[2] = R_m[0] * R_m[1] * R_m_firstDeriv[2];

	mTq[3].setZero();
	mTq[3].data()[12] = 1;
	mTq[4].setZero();
	mTq[4].data()[13] = 1;
	mTq[5].setZero();
	mTq[5].data()[14] = 1;

}



void UniversalJoint::computeLocalTSecondDerive()
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mTqq[i][j].setIdentity();
			for (int k = 3 - 1; k >= 0; k--)
			{
				if (k == i&&i != j)
				{
					mTqq[i][j] = R_m_firstDeriv[k] * mTqq[i][j];
				}
				else if (k == i&&i == j)
				{
					mTqq[i][j] = R_m_secondDerive[k] * mTqq[i][j];
				}
				else if (k != i&&k == j)
				{
					mTqq[i][j] = R_m_firstDeriv[k] * mTqq[i][j];
				}
				else
				{
					mTqq[i][j] = R_m[k] * mTqq[i][j];
				}

			}
		}
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 3; j < 6; j++)
		{
			mTqq[i][j].setZero();
		}
	}

	for (int i = 3; i < 6; i++)
	{
		for (int j = 0; j < r; j++)
		{
			mTqq[i][j].setZero();
		}
	}
}
#define countfunction(input,countx,county,countz) \
	switch (input)\
{\
	case 0: countx++;\
		break;\
	case 1: county++;\
		break;\
	case 2: countz++;\
		break;\
	default:\
		break;\
}\

void UniversalJoint::computeLocalTThirdDerive()
{
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < r; j++)
		{
			for (int k = 0; k < r; k++)
			{
				mTqqq[i][j][k].setZero();
			}
		}
	}

	int counts[3];
	counts[0] = 0;
	counts[1] = 0;
	counts[2] = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				counts[0] = 0;
				counts[1] = 0;
				counts[2] = 0;
				countfunction(i, counts[0], counts[1], counts[2]);
				countfunction(j, counts[0], counts[1], counts[2]);
				countfunction(k, counts[0], counts[1], counts[2]);

				mTqqq[i][j][k].setIdentity();
				Matrix4d mx[3];
				for (int n = 0; n < 3; n++)
				{
					/*std::cout << i << " " << j << " " << k << std::endl;
					std::cout << counts[n] << std::endl;*/
					if (counts[n] == 0)
					{
						mx[n] = R_m[n];
					}
					else if (counts[n] == 1)
					{
						mx[n] = R_m_firstDeriv[n];
					}
					else if (counts[n] == 2)
					{
						mx[n] = R_m_secondDerive[n];
					}
					else if (counts[n] == 3)
					{
						mx[n] = R_m_thirdDerive[n];
					}
				}
				mTqqq[i][j][k] = mx[0] * mx[1] * mx[2];
			}
		}
	}

}

void UniversalJoint::getOrientationByq(Matrix3d &m, VectorXd q)
{
	// from zyx to xyz
	//m = xconventionRotation(q.data()[0])*yconventionRotation(q.data()[1])*zconventionRotation(q.data()[2]);
	m = zconventionRotation(q.data()[2])*yconventionRotation(q.data()[1])*xconventionRotation(q.data()[0]);
}

void UniversalJoint::initGeneralizedInfo()
{
	r = 6;


	generalized_position.resize(r);
	generalized_velocity.resize(r);

	generalized_position.setZero();
	generalized_velocity.setZero();

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

	//only 3 angle
	R_m.resize(3);
	R_m_firstDeriv.resize(3);
	R_m_secondDerive.resize(3);
	R_m_thirdDerive.resize(3);

}
