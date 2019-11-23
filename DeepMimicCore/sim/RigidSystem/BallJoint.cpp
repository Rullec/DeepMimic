#include "BallJoint.h"
#include "Functions/EulerAngelRotationMatrix.h"
#include "Functions/SkewMatrix.h"
#include <iostream>
#include <iomanip> 

BallJoint::BallJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d joint_position_, int id_)
	:LoboJointV2(link, ori_position_, ori_position_w_, ori_orientation_w_, joint_position_, id_)
{
	initGeneralizedInfo();
}

BallJoint::~BallJoint()
{

}

void BallJoint::updateJointState(VectorXd &q, VectorXd &q_dot, bool updateDynamic)
{
	for (int i = 0; i < r; i++)
	{
		generalized_position.data()[i] = q.data()[generalized_offset + i];
		generalized_velocity.data()[i] = q_dot.data()[generalized_offset + i];
	}

	Matrix3d m;
	m.setIdentity();
	getOrientationByq(m, generalized_position);

	orientation = ori_orientation*m;
	/*std::cout << "ball q: " << generalized_position.transpose() << std::endl;
	std::cout << "ball joint ori_orientation: \n" << ori_orientation << std::endl;
	std::cout << "ball joint orientation: \n" << orientation << std::endl;*/

	/*std::cout << "orientation" << std::endl;
	std::cout << xconventionRotation(generalized_position.data()[0]) << std::endl;
	std::cout << yconventionRotation(generalized_position.data()[1]) << std::endl;
	std::cout << zconventionRotation(generalized_position.data()[2]) << std::endl;*/

	//position = (-orientation*joint_position + joint_position) + ori_position;
	/*position = (Matrix3d::Identity() - orientation)*joint_position + ori_position;*/
	//position = orientation*ori_position;

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
	//updateAngularVelovity(q,q_dot);
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

//	std::cout << "node id " << this->getJoint_id() << std::endl;
//	std::cout << linear_velocity.transpose() << std::endl;
}

void BallJoint::clampRotation(VectorXd& _q, VectorXd& _qdot)
{
	for (int i = 0; i < r; i++)
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

void BallJoint::initGlobalTerm()
{
	JK.resize(6, getGlobalR());
	JK.setZero();
	JK_dot.resize(6, getGlobalR());
	JK_dot.setZero();

	JK_v.resize(3, getGlobalR());
	JK_w.resize(3, getGlobalR());

	JK_vdot.resize(3, getGlobalR());
	JK_wdot.resize(3, getGlobalR());

	//all dependent DOFs
	mWq.resize(getDependentDOfs());
	mWqq.resize(getDependentDOfs());
	for (int i = 0; i < getDependentDOfs(); i++)
	{
		mWqq[i].resize(getDependentDOfs());
	}

	if (getIfComputeThirdDerive())
	{
		mWqqq.resize(getDependentDOfs());
		for (int i = 0; i < getDependentDOfs(); i++)
		{
			mWqqq[i].resize(getDependentDOfs());
			for (int j = 0; j < getDependentDOfs(); j++)
			{
				mWqqq[i][j].resize(getDependentDOfs());
			}
		}
	}

	JK_vq.resize(getDependentDOfs());
	JK_wq.resize(getDependentDOfs());

	JK_vdotq.resize(getDependentDOfs());
	JK_wdotq.resize(getDependentDOfs());

	for (int i = 0; i < getDependentDOfs(); i++)
	{
		JK_vq[i].resize(3, getGlobalR());
		JK_wq[i].resize(3, getGlobalR());

		JK_vdotq[i].resize(3, getGlobalR());
		JK_wdotq[i].resize(3, getGlobalR());
	}

}

void BallJoint::applyOriOrientation()
{

	for (int i = 0; i < R_m.size(); i++)
	{
		R_m[i] = R_m[i] * ori_orientation_4d;
		R_m_firstDeriv[i] = R_m_firstDeriv[i] * ori_orientation_4d;
		R_m_secondDerive[i] = R_m_secondDerive[i] * ori_orientation_4d;
		if (getIfComputeThirdDerive())
		{
			R_m_thirdDerive[i] = R_m_thirdDerive[i] * ori_orientation_4d;
		}
	}
}

void BallJoint::applyOriOrientationtomTq()
{
	for (int i = 0; i < mTq.size(); i++)
	{
		mTq[i] = ori_orientation_4d*mTq[i];
	}
}

void BallJoint::applyOriOrientationtomTqq()
{
	for (size_t i = 0; i < mTqq.size(); i++)
	{
		for (size_t j = 0; j < mTqq[i].size(); j++)
		{
			mTqq[i][j] = ori_orientation_4d*mTqq[i][j];
		}
	}
}

void BallJoint::applyOriOrientationtomTqqq()
{
	for (size_t i = 0; i < mTqqq.size(); i++)
	{
		for (size_t j = 0; j < mTqqq[i].size(); j++)
		{
			for (size_t k = 0; k < mTqqq[i][j].size(); k++)
			{
				mTqqq[i][j][k] = ori_orientation_4d*mTqqq[i][j][k];
			}
		}
	}
}

void BallJoint::updateRotationMatrix()
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

void BallJoint::computeTransformFirstDerive()
{
	//compute mTq;
	computeLocalTransformDerive();	// joint rotation and translation derivative with repect to q
	applyOriOrientationtomTq();		//mTq computation finished

	//compute mWq;
	computeGlobalTransformDerive();
}

/* 

	@Function: BallJoint::computeLocalTransformDerive
	@params: void
	@return: void
	
	For	T_{4*4} =| Rot_{3*3}, Trans_{3*1}|
				 |0,   0,   0,	1		 |_{4*4}, which represent the transformation from parent joint to child joint

	This function compute mTq = [mTq[0], mTq[1], mTq[2]] = [dT/dq_x, dT/dq_y, dT/dq_z].
	Obviously, q_x, q_y and q_z are 3 generalized coordinates for this ball joint.
*/
void BallJoint::computeLocalTransformDerive()
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
		std::cout << "[error] BallJoint::computeLocalTransformDerive Unsupported rotation order: " << ROTATION_ORDER_NAME[gRotationOrder] << std::endl;
		exit(1);
	}
	
}

/*
	@Function: BallJoint::computeGlobalTransformDerive
	@params: void
	@return: void

	This function will compute mWq Type std::vector<Eigen::Matrix4d>, mWq is a vector storaged 3 Matrix4d mats, which are:
		mWq[0] = dW/dx, W is the rotation matrix from joint local frame to world frame, x is the first generalized coordinate of this ball joint
		mWq[1] = dW/dy, W is the rotation matrix from joint local frame to world frame
		mWq[2] = dW/dz, W is the rotation matrix from joint local frame to world frame
*/
void BallJoint::computeGlobalTransformDerive()
{
	// 计算dWdq, 是一个有dependentDof项的数组;
	int numParentDOFs = getDependentDOfs() - getR(); // 获取他所依赖的自由度(去除自己的)

	// 对于他的父亲链上的，通过parent link rotation * local rotation获得从当前到
	for (int i = 0; i < numParentDOFs; i++)
	{
		mWq[i] = joint_parent->mWq[i] * local_TransForm;	// 递归乘法，避免重新计算。获取完成的部分之后去乘。
	}

	// 对于我自己的那一部分，就是父亲的世界转换 * 我的局部transform对q求偏导
	for (int i = 0; i < getR(); i++)
	{
		if (joint_parent) // 如果该joint有父亲, mWq中就存储
		{
			mWq[numParentDOFs + i] = joint_parent->global_TransForm*mTq[i];
		}
		else
		{
			mWq[numParentDOFs + i] = mTq[i];
		}
	}
}

// 这个二阶导是需要计算的，在计算jacobian的时候
void BallJoint::computeTransformSecondDerive()
{
	computeLocalTSecondDerive();
	applyOriOrientationtomTqq();
	computeGlobalTSecondDerive();
}

/*
	@Function: BallJoint::computeLocalTSecondDerive
	@params: void
	@return: void

		This function compute the secord order derivate of the local transformation matrix T of this joint.
	for this Eigen::Matrix4d matrix T, its 2nd order derivations have 9 components, which are
	from mTqq[0][0] to mTqq[2][2]. 
		ATTENTION: different rotation order has different formula following. 
		We will use the order "XYZ" as an example, so T_total = Tz * Ty * Tx

		mTqq[0][0] = dT^2/dxdx = d(dT/dx)/dx = d(Tz * Ty * dTx/dx)/dx = Tz * Ty * dTx^2/dx^2
		mTqq[1][2] = dT^2/dydz = d(dT/dy)/dz = d(Tz * dTy/dy * Tx)/dz = dTz/dz * dTy/dy * Tx
		mTqq[2][2] = dT^2/dzdz = ...
*/
void BallJoint::computeLocalTSecondDerive()
{
	// 这里也存在旋转顺序
	if (3 != r)
	{
		std::cout << "[error] BallJoint::computeLocalTSecondDerive: the local dof of this ball joint is not 3 " << std::endl;
		exit(1);
	}

	int rotation_order[3] = { 0, 0, 0 };	
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
		std::cout << "[error] BallJoint::computeLocalTSecondDerive: Unsupported rotation order" << std::endl;
		exit(1);
	}

	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < r; j++)
		{
			mTqq[i][j].setIdentity();
			for (int selected_matrix_order = 0; selected_matrix_order < r; selected_matrix_order++)
			{
				int k = rotation_order[selected_matrix_order]; // choose which dimension was chosed to be multiplicated
				if (k == i&&i != j)
				{
					mTqq[i][j] = R_m_firstDeriv[k] * mTqq[i][j];
				}
				else if (k == i&&i == j)
				{
					// 如果计算dx^2 / dy^2 / dz^2
					// 这项只进入一次, 就是
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

	////dx
	//rotationSecondDerive_dxdx(mTqq[0][0], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dxdy(mTqq[0][1], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dxdz(mTqq[0][2], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);

	////dy
	//rotationSecondDerive_dydx(mTqq[1][0], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dydy(mTqq[1][1], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dydz(mTqq[1][2], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);

	////dz
	//rotationSecondDerive_dzdx(mTqq[2][0], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dzdy(mTqq[2][1], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	//rotationSecondDerive_dzdz(mTqq[2][2], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
}

/*
	@Function: BallJoint::computeGlobalSecondDerive
	@params: void
	@return: void
		This function will compute the 2nd order derivation of the global transformation matrix W
		
*/
void BallJoint::computeGlobalTSecondDerive()
{
	int numParentDOFs = getDependentDOfs() - getR();

	for (int i = 0; i < numParentDOFs; i++)
	{
		for (int j = 0; j < numParentDOFs; j++)
		{
			mWqq[i][j] = joint_parent->mWqq[i][j] * local_TransForm;
		}


		for (int j = 0; j < getR(); j++)
		{
			if (joint_parent)
				mWqq[i][j + numParentDOFs] = joint_parent->mWq[i] * mTq[j];
			else
				mWqq[i][j + numParentDOFs] = mTq[j];
		}
	}

	for (int i = 0; i < getR(); i++)
	{
		for (int j = 0; j < numParentDOFs; j++)
		{
			if (joint_parent)
				mWqq[numParentDOFs + i][j] = joint_parent->mWq[j] * mTq[i];
			else
				mWqq[numParentDOFs + i][j] = mTq[i];


		}

		for (int j = 0; j < getR(); j++)
		{
			if (joint_parent)
				mWqq[numParentDOFs + i][numParentDOFs + j] = joint_parent->global_TransForm * mTqq[i][j];
			else
				mWqq[numParentDOFs + i][numParentDOFs + j] = mTqq[i][j];
		}
	}
}

void BallJoint::computeTransformThirdDerive()
{
	computeLocalTThirdDerive();
	applyOriOrientationtomTqqq();
	computeGlobalTThirdDerive();
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
void BallJoint::computeLocalTThirdDerive()
{
	std::cout << "[error] BallJoint::computeLocalTThirdDerive: this function didn't support different rotation order but ZYX" << std::endl;
	exit(1);
	int counts[3];
	counts[0] = 0;
	counts[1] = 0;
	counts[2] = 0;

	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < r; j++)
		{
			for (int k = 0; k < r; k++)
			{
				counts[0] = 0;
				counts[1] = 0;
				counts[2] = 0;
				countfunction(i, counts[0], counts[1], counts[2]);
				countfunction(j, counts[0], counts[1], counts[2]);
				countfunction(k, counts[0], counts[1], counts[2]);

				mTqqq[i][j][k].setIdentity();
				Matrix4d mx[3];
				for (int n = 0; n < r; n++)
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

// ignore
void BallJoint::computeGlobalTThirdDerive()
{
	std::cout << "[error] BallJoint::computeGlobalTThirdDerive: this function didn't support different rotation order bu ZYX" << std::endl;
	exit(1);
	int numParentDOFs = getDependentDOfs() - getR();

	for (int i = 0; i < numParentDOFs; i++)
	{
		for (int j = 0; j < numParentDOFs; j++)
		{
			for (int k = 0; k < numParentDOFs; k++)
			{
				mWqqq[i][j][k] = joint_parent->mWqqq[i][j][k] * local_TransForm;
			}

			for (int k = 0; k < getR(); k++)
			{
				mWqqq[i][j][numParentDOFs + k] = joint_parent->mWqq[i][j] * mTq[k];
			}
		}

		for (int j = 0; j < getR(); j++)
		{
			for (int k = 0; k < numParentDOFs; k++)
			{
				mWqqq[i][j + numParentDOFs][k] = joint_parent->mWqq[i][k] * mTq[j];
			}

			for (int k = 0; k < getR(); k++)
			{
				mWqqq[i][j + numParentDOFs][k + numParentDOFs] = joint_parent->mWq[i] * mTqq[j][k];
			}
		}
	}

	for (int i = 0; i < getR(); i++)
	{
		for (int j = 0; j < numParentDOFs; j++)
		{
			for (int k = 0; k < numParentDOFs; k++)
			{
				mWqqq[i + numParentDOFs][j][k] = joint_parent->mWqq[j][k] * mTq[i];
			}

			for (int k = 0; k < getR(); k++)
			{
				mWqqq[i + numParentDOFs][j][k + numParentDOFs] = joint_parent->mWq[j] * mTqq[i][k];
			}
		}

		for (int j = 0; j < getR(); j++)
		{
			for (int k = 0; k < numParentDOFs; k++)
			{
				mWqqq[i + numParentDOFs][j + numParentDOFs][k] = joint_parent->mWq[k] * mTqq[i][j];
			}

			for (int k = 0; k < getR(); k++)
			{
				if (joint_parent)
					mWqqq[i + numParentDOFs][j + numParentDOFs][k + numParentDOFs] = joint_parent->global_TransForm*mTqqq[i][j][k];
				else
					mWqqq[i + numParentDOFs][j + numParentDOFs][k + numParentDOFs] = mTqqq[i][j][k];
			}
		}
	}
}

// 计算Jacobian
void BallJoint::computeJacobiMatrix(VectorXd &q_dot)
{
	// 更新旋转矩阵，W, T, W', T', W'', T''
	updateRotationMatrix();
	//applyOriOrientation();

	computeTransformFirstDerive();	// 计算局部、世界变换的对本ball joint 3个广义坐标xyz的一阶导(individually)，R'[3], W'[3]
	computeJacobiW();	// 计算Jw
	computeJacobiV();	// 计算Jv

	//if (ifComputeSecondDerive)
	//{
	computeTransformSecondDerive();
	computeJacobiVdot(q_dot);
	computeJacobiWdot(q_dot);
	//}


	if (getIfComputeThirdDerive())
	{
		computeTransformThirdDerive();
		computeJacobiVdotq(q_dot);
		computeJacobiWdotq(q_dot);
	}

	JK.block(0, 0, 3, getGlobalR()) = JK_v;
	JK.block(3, 0, 3, getGlobalR()) = JK_w;
	JK_dot.block(0, 0, 3, getGlobalR()) = JK_vdot;
	JK_dot.block(3, 0, 3, getGlobalR()) = JK_wdot;

	/*std::cout << JK_dot << std::endl;
	std::cout << "-----------------------------------" << std::endl;*/

	return;
}

/*
	@Func: BallJoint::updateJacobiByGivenPosition
	@params: x_position Type Vector3d, given a specified position
	@params: JK_v Type MatrixXd &, the Jk_v waited to be revised

	Given a position, this function will compute the JK_v matrix in this place.
*/
void BallJoint::updateJacobiByGivenPosition(Vector3d x_position, MatrixXd &JK_v_)
{
	Vector3d target_positionInLocal;
	//target_positionInLocal = ori_orientation_w.transpose()*(x_position - ori_position_w);
	target_positionInLocal = orientation_w.transpose()*(x_position - position_w);	// 目标的局部位置
	//std::cout << "target_positionInLocal: " << target_positionInLocal.transpose() << std::endl;
	int dofs_offset = 0;
	JK_v_.resize(3, getGlobalR());	// 3 * N Jk_v维度
	JK_v_.setZero();

	Vector4d targetpositionlocal_h;	// 齐次坐标形式
	targetpositionlocal_h.data()[0] = target_positionInLocal.data()[0];
	targetpositionlocal_h.data()[1] = target_positionInLocal.data()[1];
	targetpositionlocal_h.data()[2] = target_positionInLocal.data()[2];
	targetpositionlocal_h.data()[3] = 1;

	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		// 从当前joint到root的运动链上，对每一个joint:
		LoboJointV2* joint = chainJointFromRoot[i];
		int localr = joint->getR();
		for (int j = 0; j < localr; j++)
		{
			// 循环3次
			int columnid = joint->getGeneralized_offset() + j;	// 当前坐标在广义坐标中的位置
			int dofsindex = dofs_offset + j;

			Vector4d column = mWq[dofsindex] * targetpositionlocal_h;

			Vector3d temp;
			temp.data()[0] = column.data()[0];
			temp.data()[1] = column.data()[1];
			temp.data()[2] = column.data()[2];


			JK_v_.col(columnid) = temp;
		}
		dofs_offset += localr;
	}
}

void BallJoint::updateJacobiByGivenRestPosition(Vector3d x_position_rest, MatrixXd & JK_v_)
{
	Vector3d target_positionInLocal;
	target_positionInLocal = ori_orientation_w.transpose()*(x_position_rest - ori_position_w);
	//std::cout << "ball target_positionInLocal : " << target_positionInLocal.transpose() << std::endl;
	//std::cout << "=================================\n" << std::endl;
	int dofs_offset = 0;
	JK_v_.resize(3, getGlobalR());
	JK_v_.setZero();
	Vector4d targetpositionlocal_h;
	targetpositionlocal_h.data()[0] = target_positionInLocal.data()[0];
	targetpositionlocal_h.data()[1] = target_positionInLocal.data()[1];
	targetpositionlocal_h.data()[2] = target_positionInLocal.data()[2];
	targetpositionlocal_h.data()[3] = 1;

	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		LoboJointV2* joint = chainJointFromRoot[i];
		int localr = joint->getR();
		for (int j = 0; j < localr; j++)
		{
			int columnid = joint->getGeneralized_offset() + j;
			int dofsindex = dofs_offset + j;

			Vector4d column = mWq[dofsindex] * targetpositionlocal_h;

			Vector3d temp;
			temp.data()[0] = column.data()[0];
			temp.data()[1] = column.data()[1];
			temp.data()[2] = column.data()[2];


			JK_v_.col(columnid) = temp;
		}
		dofs_offset += localr;
	}
}

/*
	@Function: BallJoint::computeJacobiW
	@params: void 
	@return: void

	This function compute Jw
	Jw = 
*/
void BallJoint::computeJacobiW()
{
	int dofs_offset = 0;
	JK_w.setZero();
	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		LoboJointV2* joint = chainJointFromRoot[i];
		int localr = joint->getR();
		for (int j = 0; j < localr; j++)
		{
			int columnid = joint->getGeneralized_offset() + j;
			int dofsindex = dofs_offset + j;
			Matrix3d omegaSkewSymmetric = mWq[dofsindex].topLeftCorner<3, 3>()*global_TransForm.topLeftCorner<3, 3>().transpose();
			JK_w.col(columnid) = fromSkewSysmmtric(omegaSkewSymmetric);
		}
		dofs_offset += localr;
	}

	/*std::cout << "node : ============> " << getJoint_id() << std::endl;
	for (int i = 0; i < mWq.size(); i++)
	{
		std::cout << mWq[i] << std::endl;
	}
	std::cout << "*****************JK***************" << std::endl;
	std::cout << JK_w << std::endl;
	std::cout << "****************global_transofm***************" << std::endl;
	std::cout << mWq[0].topLeftCorner<3, 3>()*global_TransForm.topLeftCorner<3, 3>().transpose() << std::endl;*/
}

void BallJoint::computeJacobiV()
{
	int dofs_offset = 0;
	JK_v.setZero();
	Vector4d masscenterlocal_h;
	masscenterlocal_h.data()[0] = massCenterInLocal.data()[0];
	masscenterlocal_h.data()[1] = massCenterInLocal.data()[1];
	masscenterlocal_h.data()[2] = massCenterInLocal.data()[2];
	masscenterlocal_h.data()[3] = 1;


	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		LoboJointV2* joint = chainJointFromRoot[i];
		int localr = joint->getR();
		for (int j = 0; j < localr; j++)
		{
			int columnid = joint->getGeneralized_offset() + j;
			int dofsindex = dofs_offset + j;

			Vector4d column = mWq[dofsindex] * masscenterlocal_h;

			Vector3d temp;
			temp.data()[0] = column.data()[0];
			temp.data()[1] = column.data()[1];
			temp.data()[2] = column.data()[2];


			JK_v.col(columnid) = temp;
		}
		dofs_offset += localr;
	}
}


void BallJoint::computeJacobiWdot(VectorXd &q_dot)
{
	JK_wdot.setZero();

	for (int i = 0; i < getDependentDOfs(); i++)
	{
		JK_wq[i].setZero();
		for (int j = 0; j < getDependentDOfs(); j++)
		{
			Matrix3d JwqijSkewSymm = mWqq[i][j].topLeftCorner<3, 3>()*global_TransForm.topLeftCorner<3, 3>().transpose() + mWq[j].topLeftCorner<3, 3>()*mWq[i].topLeftCorner<3, 3>().transpose();
			Vector3d Jwqij = fromSkewSysmmtric(JwqijSkewSymm);

			JK_wq[i].col(mapDependentDofs[j]) = Jwqij;
		}
		//need a map
		JK_wdot += JK_wq[i] * q_dot[mapDependentDofs[i]];
	}
}



void BallJoint::computeJacobiVdot(VectorXd &q_dot)
{
	JK_vdot.setZero();

	Vector4d masscenterlocal_h;
	masscenterlocal_h.data()[0] = massCenterInLocal.data()[0];
	masscenterlocal_h.data()[1] = massCenterInLocal.data()[1];
	masscenterlocal_h.data()[2] = massCenterInLocal.data()[2];
	masscenterlocal_h.data()[3] = 1;


	for (int i = 0; i < getDependentDOfs(); i++)
	{
		JK_vq[i].setZero();
		for (int j = 0; j < getDependentDOfs(); j++)
		{
			Vector4d Jvqi = mWqq[i][j] * masscenterlocal_h;
			Vector3d tempv;
			tempv.data()[0] = Jvqi.data()[0];
			tempv.data()[1] = Jvqi.data()[1];
			tempv.data()[2] = Jvqi.data()[2];
			JK_vq[i].col(mapDependentDofs[j]) = tempv;
		}
		JK_vdot += JK_vq[i] * q_dot[mapDependentDofs[i]];
	}
}

void BallJoint::computeJacobiWdotq(VectorXd &q_dot)
{
	std::cout << "[error] BallJoint::computeJacobiWdotq: \
		this function should NOT be called, because it havn't support the rotation order." << std::endl;
	exit(1);
	MatrixXd temp(3, getGlobalR());

	for (int i = 0; i < getDependentDOfs(); i++)
	{
		JK_wdotq[i].setZero();
		for (int j = 0; j < getDependentDOfs(); j++)
		{
			temp.setZero();
			for (int k = 0; k < getDependentDOfs(); k++)
			{
				Matrix3d JwqijkSkewSymm = mWqqq[i][j][k].topLeftCorner<3, 3>()*global_TransForm.topLeftCorner<3, 3>().transpose()
					+ mWqq[i][j].topLeftCorner<3, 3>()*mWq[k].topLeftCorner<3, 3>().transpose()
					+ mWqq[j][k].topLeftCorner<3, 3>()*mWq[i].topLeftCorner<3, 3>().transpose()
					+ mWq[j].topLeftCorner<3, 3>()*mWqq[i][k].topLeftCorner<3, 3>().transpose();
				Vector3d Jwqijk = fromSkewSysmmtric(JwqijkSkewSymm);
				temp.col(mapDependentDofs[k]) = Jwqijk;
			}
			JK_wdotq[i] += temp*q_dot[mapDependentDofs[j]];
		}
	}

}

void BallJoint::computeJacobiVdotq(VectorXd &q_dot)
{
	std::cout << "[error] BallJoint::computeJacobiWdotq: \
		this function should NOT be called, because it havn't support the rotation order." << std::endl;
	exit(1);
	MatrixXd temp(3, getGlobalR());
	Vector4d masscenterlocal_h;
	masscenterlocal_h.data()[0] = massCenterInLocal.data()[0];
	masscenterlocal_h.data()[1] = massCenterInLocal.data()[1];
	masscenterlocal_h.data()[2] = massCenterInLocal.data()[2];
	masscenterlocal_h.data()[3] = 1;


	for (int i = 0; i < getDependentDOfs(); i++)
	{
		JK_vdotq[i].setZero();
		for (int j = 0; j < getDependentDOfs(); j++)
		{
			temp.setZero();
			for (int k = 0; k < getDependentDOfs(); k++)
			{
				Vector4d Jvqi = mWqqq[i][j][k] * masscenterlocal_h;
				Vector3d tempv;
				tempv.data()[0] = Jvqi.data()[0];
				tempv.data()[1] = Jvqi.data()[1];
				tempv.data()[2] = Jvqi.data()[2];
				temp.col(mapDependentDofs[k]) = tempv;
			}
			JK_vdotq[i] += temp*q_dot[mapDependentDofs[j]];
		}
	}
}

void BallJoint::getOrientationByq(Matrix3d &m, VectorXd q)
{
	// R = Rx * Ry * Rz, zyx旋转顺序
	if (gRotationOrder == eRotationOrder::XYZ)
	{
		m = zconventionRotation(q.data()[2]) *yconventionRotation(q.data()[1]) * xconventionRotation(q.data()[0]);
	}
	else if (gRotationOrder == eRotationOrder::ZYX)
	{
		m = xconventionRotation(q.data()[0])*yconventionRotation(q.data()[1])*zconventionRotation(q.data()[2]);
	}
	else
	{
		std::cout << "[error] BallJoint::getOrientationByq Unsupported rotation order: " << ROTATION_ORDER_NAME[gRotationOrder];
	}
}

void BallJoint::initGeneralizedInfo()
{
	r = 3;
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


	R_m.resize(r);
	R_m_firstDeriv.resize(r);
	R_m_secondDerive.resize(r);
	R_m_thirdDerive.resize(r);

}
