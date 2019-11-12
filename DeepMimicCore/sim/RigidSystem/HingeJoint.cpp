#include "HingeJoint.h"
#include "Functions/EulerAngelRotationMatrix.h"
#include "Functions/SkewMatrix.h"
#include <iostream>
#include <iomanip> 

HingeJoint::HingeJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w, Vector3d joint_position, int id_, int hingeType) :BallJoint(link, ori_position_, ori_position_w_, ori_orientation_w, joint_position, id_)
{
	this->hingetype = hingeType;
	this->initGeneralizedInfo();
	hingeOrientation_local.setIdentity(); //local to world frame, not to parent
	hingeOrientation_local.topLeftCorner<3, 3>() = ori_orientation_w;
	this->ori_orientation_w = ori_orientation_w;
	this->ori_orientation_4d.setIdentity();
}

HingeJoint::~HingeJoint()
{

}

void HingeJoint::updateJacobiByGivenPosition(Vector3d x_position, MatrixXd & JK_v_)
{
	Vector3d target_positionInLocal;
	//target_positionInLocal = ori_orientation_w.transpose()*(x_position - ori_position_w);
	target_positionInLocal = orientation_w.transpose()*(x_position - position_w);
	/*std::cout << "orientation_w: " << orientation_w << std::endl;
	std::cout << "position_w: " << position_w.transpose() << std::endl;
	std::cout << "target_positionInLocal: " << target_positionInLocal.transpose() << std::endl;*/
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

void HingeJoint::updateJacobiByGivenRestPosition(Vector3d x_position_rest, MatrixXd & JK_v_)
{
	Vector3d target_positionInLocal;
	target_positionInLocal = ori_orientation_w.transpose()*(x_position_rest - ori_position_w);
	//target_positionInLocal = orientation_w.transpose()*(x_position_rest - position_w);
	//std::cout << "hinge target_positionInLocal : " << target_positionInLocal.transpose() << std::endl;
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

void HingeJoint::applyOriOrientation()
{
	Matrix4d temp = hingeOrientation_local.transpose()*ori_orientation_4d;
	for (int i = 0; i < R_m.size(); i++)
	{
		R_m[i] = hingeOrientation_local*R_m[i] * temp;
		R_m_firstDeriv[i] = hingeOrientation_local*R_m_firstDeriv[i] * temp;
		R_m_secondDerive[i] = hingeOrientation_local*R_m_secondDerive[i] * temp;
		if (getIfComputeThirdDerive())
		{
			R_m_thirdDerive[i] = hingeOrientation_local*R_m_thirdDerive[i] * temp;
		}
	}

}

void HingeJoint::applyOriOrientationtomTq()
{
	mTq[0] = ori_orientation_4d*mTq[0] ;
}

void HingeJoint::applyOriOrientationtomTqq()
{
	mTqq[0][0] = ori_orientation_4d * mTqq[0][0];
}

void HingeJoint::applyOriOrientationtomTqqq()
{
	mTqqq[0][0][0] = ori_orientation_4d * mTqqq[0][0][0];
}


void HingeJoint::updateRotationMatrix()
{
	if (hingetype == 0)
	{
		xconventionTransform(R_m[0], generalized_position[0]);
		xconventionRotation_dx(R_m_firstDeriv[0], generalized_position[0]);
		xconventionRotation_dxdx(R_m_secondDerive[0], generalized_position[0]);
		if (getIfComputeThirdDerive())
		{
			xconventionRotation_dxdxdx(R_m_thirdDerive[0], generalized_position[0]);
		}
	}
	else if (hingetype == 1)
	{
		yconventionTransform(R_m[0], generalized_position[0]);
		yconventionRotation_dy(R_m_firstDeriv[0], generalized_position[0]);
		yconventionRotation_dydy(R_m_secondDerive[0], generalized_position[0]);
		if (getIfComputeThirdDerive())
		{
			yconventionRotation_dydydy(R_m_thirdDerive[0], generalized_position[0]);
		}
	}
	else if (hingetype == 2)
	{
		zconventionTransform(R_m[0], generalized_position[0]);
		zconventionRotation_dz(R_m_firstDeriv[0], generalized_position[0]);
		zconventionRotation_dzdz(R_m_secondDerive[0], generalized_position[0]);
		if (getIfComputeThirdDerive())
		{
			zconventionRotation_dzdzdz(R_m_thirdDerive[0], generalized_position[0]);
		}
	}
}

void HingeJoint::computeLocalTransformDerive()
{
	//mTq[0].setZero();
	mTq[0] = R_m_firstDeriv[0];
}

void HingeJoint::computeLocalTSecondDerive()
{
	mTqq[0][0] = R_m_secondDerive[0];
}

void HingeJoint::computeLocalTThirdDerive()
{
	mTqqq[0][0][0] = R_m_thirdDerive[0];
}

void HingeJoint::getOrientationByq(Matrix3d &m, VectorXd q)
{
	if (hingetype == 0)
	{
		m = xconventionRotation(q.data()[0]);
	}
	else if (hingetype == 1)
	{
		m = yconventionRotation(q.data()[0]);
	}
	else if (hingetype == 2)
	{
		m = zconventionRotation(q.data()[0]);
	}
	
	//Matrix3d hinge = hingeOrientation_local.topLeftCorner<3, 3>();
	/*std::cout << "test1 = >" << (hinge.transpose()*Vector3d::UnitX()).transpose() << std::endl;
	std::cout << "test2 = >" << (m*hinge.transpose()*Vector3d::UnitX()).transpose() << std::endl;
	std::cout << "test3 = >" << (hinge*m*hinge.transpose()*Vector3d::UnitX()).transpose() << std::endl;
	std::cout << hinge << std::endl;*/
}

void HingeJoint::initGeneralizedInfo()
{
	r = 1;

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
