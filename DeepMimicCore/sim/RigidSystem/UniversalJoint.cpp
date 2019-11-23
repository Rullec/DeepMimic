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

/*
	@Function: UniversalJoint::updateJointState
	
	This function is used to set the state of articulated system.
*/
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

/*
	@Function: UniversalJoint::clampRotation
	
	This function is used to normalize the generalized coordinates(euler angles)
*/
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

/*
	@Function: UniversalJoint::updateLocalTransform

	This function is used to combine a 3x3 rotation matrix and a 3x1 translational vector into a 4x4 transformal matrix
*/
void UniversalJoint::updateLocalTransform()
{
	local_TransForm.topLeftCorner<3, 3>() = orientation;
	local_TransForm.data()[12] = position.data()[0];
	local_TransForm.data()[13] = position.data()[1];
	local_TransForm.data()[14] = position.data()[2];
}

/*
	@Function: UniversalJoint::updateRotationMatrix
	This function is used to compute:
		1. 3 individual rotation matrix: Rx, Ry, Rz
		2. 1st order of derivates of these 3 mats: dRx/dx, dRy/dy, dRz/dz
		3. 2nd order of ...: dRx^2/dx^2, etc
	3rd order of derivates are ignored
*/
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

/*
	@Function: UniversalJoint::computeLocalTransformDerive
		
	This function computes the rotation matrix from 3 individual components, according to a given rotaion order
	For example, for order "ZYX" (which means first rotates by Z-axis, then Y-axis and X-axis followed)
	In this circumstance, the ultimate rotation matrix are managed to be R_total = Rx * Ry * Rz.

	mTq[0] = dRdx = dRx/dx * Ry * Rz (the first components)
	mTq[1] = dRdy = Rx * dRy/dy * Rz (the second components)
	mTq[2] = dRdz = Rx * Ry * dRz/dz (the third components)
*/
void UniversalJoint::computeLocalTransformDerive()
{
	mTq[0].setZero();
	mTq[1].setZero();
	mTq[2].setZero();

	if (gRotationOrder == eRotationOrder::ZYX)
	{
		// 旋转顺序z->y->x
		mTq[0] = R_m_firstDeriv[0] * R_m[1] * R_m[2]; // dTdx = (dRx/dx) * Ry * Rz
		mTq[1] = R_m[0] * R_m_firstDeriv[1] * R_m[2];
		mTq[2] = R_m[0] * R_m[1] * R_m_firstDeriv[2];
	}
	else if (gRotationOrder == eRotationOrder::XYZ)
	{
		// 旋转顺序x->y->z, T = Rz * Ry * Rx
		mTq[0] = R_m[2] * R_m[1] * R_m_firstDeriv[0];	// mTq[0] = dT/dx = Rz * Ry * (dRx/dx)
		mTq[1] = R_m[2] * R_m_firstDeriv[1] * R_m[0];	// mTq[1] = Rz * (dRy/dy) * Rx
		mTq[2] = R_m_firstDeriv[2] * R_m[1] * R_m[0];	// mTq[2] = (dRz/dz) *Ry * Rx
	}
	else
	{
		std::cout << "[error] UniversalJoint::computeLocalTransformDerive Unsupported rotation order: " << ROTATION_ORDER_NAME[gRotationOrder] << std::endl;
		exit(1);
	}

	// for final transfromational matrix T(4x4), (dT)/(dtranslation_x) =
	/*
		| 0 0 0 1 |
		| 0 0 0 0 |
		| 0 0 0 0 | 
		| 0 0 0 0 |
		and eigen is default column major
	*/
	mTq[3].setZero();
	mTq[3].data()[12] = 1;
	mTq[4].setZero();
	mTq[4].data()[13] = 1;
	mTq[5].setZero();
	mTq[5].data()[14] = 1;

}


/*
	@Function: UniversalJoint::computeLocalTSecondDerive
		
	It can be used to compute second order derivates for local rotation matrix R = Rz * Ry * Rx(it depends on the rotation order)
	There will be as much as 9 VALID individual components, which can be placed from mTqq[0][0] to mTqq[3][3]


	mTqq[0][0] = dR^2/dx^2 = d(dR/dx)/dx = d(dRx/dx * Ry * Rz)/dx = dRx^2/dx^2 * Ry * Rz
	...
	mTqq[1][2] = dR^2/dydz = d(dR/dy)/dz = d( Rx * dRy/dy * Rz)/dz = Rx * dRy/dy * dRz/dz
	...

	But for Universal Joint(root), there will be up to 6 DOFs = [rx, ry, rz, x, y, z]
	
*/
void UniversalJoint::computeLocalTSecondDerive()
{
	if (6 != r)
	{
		std::cout << "[error] UniversalJoint::computeLocalTSecondDerive: the local dof of this ball joint is not 3 " << std::endl;
		exit(1);
	}

	int rotation_order[3] = {  };
	// this array will be used later, it will decide which rotation matrix will be multiplicated 1st, 2nd, 3rd
	// and control the rotation order in this way
	if (gRotationOrder == eRotationOrder::XYZ)
	{// in this order
		rotation_order[0] = 0; // Rx will be multiplicated first
		rotation_order[1] = 1; // then Ry second
		rotation_order[2] = 2; // then Rz third
	}
	else if (gRotationOrder == eRotationOrder::ZYX)
	{
		rotation_order[0] = 2; // Rz will be multiplicated first
		rotation_order[1] = 1; // then Ry second
		rotation_order[2] = 0; // then Rx third
	}
	else
	{
		std::cout << "[error] UniversalJoint::computeLocalTSecondDerive: Unsupported rotation order" << std::endl;
		exit(1);
	}

	// r = 6 but we only care the first 3 rotation DOF: rx, ry, rz
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mTqq[i][j].setIdentity();
			for (int selected_matrix_order = 0; selected_matrix_order < 3; selected_matrix_order++)
			{
				int k = rotation_order[selected_matrix_order]; // choose which dimension was chosed to be multiplicated
				if (k == i && i != j)
				{
					mTqq[i][j] = R_m_firstDeriv[k] * mTqq[i][j];
				}
				else if (k == i && i == j)
				{
					// 如果计算dx^2 / dy^2 / dz^2
					// 这项只进入一次, 就是
					mTqq[i][j] = R_m_secondDerive[k] * mTqq[i][j];
				}
				else if (k != i && k == j)
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

	// any second order terms where the translational DOF (x, y, z) are involved in should be set to zero
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

/*

	compute 3rd derivate, ignore
*/
void UniversalJoint::computeLocalTThirdDerive()
{
	std::cout << "[error] UniversalJoint::computeLocalTThirdDerive: this function didn't support different rotation order but ZYX" << std::endl;
	exit(1);
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
	// R = Rx * Ry * Rz, zyx旋转顺序
	if (gRotationOrder == eRotationOrder::XYZ)
	{
		m = zconventionRotation(q.data()[2]) * yconventionRotation(q.data()[1]) * xconventionRotation(q.data()[0]);
	}
	else if (gRotationOrder == eRotationOrder::ZYX)
	{
		m = xconventionRotation(q.data()[0]) * yconventionRotation(q.data()[1]) * zconventionRotation(q.data()[2]);
	}
	else
	{
		std::cout << "[error] UniversalJoint::getOrientationByq Unsupported rotation order: " << ROTATION_ORDER_NAME[gRotationOrder];
	}
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
