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
	computeLocalTransformDerive(); // joint rotation and translation derivative with repect to q
	applyOriOrientationtomTq();  //mTq computation finished
	//compute mWq;
	computeGlobalTransformDerive();
}

void BallJoint::computeLocalTransformDerive()
{
	mTq[0].setZero();
	mTq[1].setZero();
	mTq[2].setZero();

	//绕轴旋转顺序为x y z, R_m和R_m的导求出来之后已经transpose过了，所以这个乘的顺序没有问题
	std::cout << "[error] BallJoint danger " << std::endl;
	exit(1);
	mTq[0] = R_m_firstDeriv[0] * R_m[1] * R_m[2];
	mTq[1] = R_m[0] * R_m_firstDeriv[1] * R_m[2];
	mTq[2] = R_m[0] * R_m[1] * R_m_firstDeriv[2];


	/*for (int i = 0; i < 3; i++)
	{
	std::cout << mTq[i] << std::endl;
	}
	std::cout << "*************compare*********************" << std::endl;
	rotationFirstDerive_dx(mTq[0], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	rotationFirstDerive_dy(mTq[1], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);
	rotationFirstDerive_dz(mTq[2], generalized_position.data()[0], generalized_position.data()[1], generalized_position.data()[2]);

	for (int i = 0; i < 3; i++)
	{
	std::cout << mTq[i] << std::endl;
	}*/

}

void BallJoint::computeGlobalTransformDerive()
{
	int numParentDOFs = getDependentDOfs() - getR();


	for (int i = 0; i < numParentDOFs; i++)
	{
		mWq[i] = joint_parent->mWq[i] * local_TransForm;
	}

	for (int i = 0; i < getR(); i++)
	{
		if (joint_parent)
		{
			mWq[numParentDOFs + i] = joint_parent->global_TransForm*mTq[i];
		}
		else
		{
			mWq[numParentDOFs + i] = mTq[i];
		}
	}
}

void BallJoint::computeTransformSecondDerive()
{
	computeLocalTSecondDerive();
	applyOriOrientationtomTqq();
	computeGlobalTSecondDerive();
}

void BallJoint::computeLocalTSecondDerive()
{
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < r; j++)
		{
			mTqq[i][j].setIdentity();
			for (int k = r - 1; k >= 0; k--)
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
				mWqq[numParentDOFs + i][numParentDOFs + j] = joint_parent->global_TransForm* mTqq[i][j];
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

void BallJoint::computeLocalTThirdDerive()
{
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

void BallJoint::computeGlobalTThirdDerive()
{
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

void BallJoint::computeJacobiMatrix(VectorXd &q_dot)
{
	updateRotationMatrix();
	//applyOriOrientation();

	computeTransformFirstDerive();
	computeJacobiW();
	computeJacobiV();

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

void BallJoint::updateJacobiByGivenPosition(Vector3d x_position, MatrixXd &JK_v_)
{

	Vector3d target_positionInLocal;
	//target_positionInLocal = ori_orientation_w.transpose()*(x_position - ori_position_w);
	target_positionInLocal = orientation_w.transpose()*(x_position - position_w);
	//std::cout << "target_positionInLocal: " << target_positionInLocal.transpose() << std::endl;
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
		std::cout << "[error] Unsupported rotation order: " << ROTATION_ORDER_NAME[gRotationOrder];
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
