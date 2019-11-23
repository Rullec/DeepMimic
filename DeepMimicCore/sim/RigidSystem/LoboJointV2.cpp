#include "LoboJointV2.h"
#include "LoboLink.h"
#include <iostream>
#include "Functions/SkewMatrix.h"

const enum eRotationOrder gRotationOrder = eRotationOrder::XYZ;// LoboSImulator(after cow) default, deepmimic
//const enum eRotationOrder gRotationOrder = eRotationOrder::ZYX;// LoboSimulator(yifan) default

LoboJointV2::LoboJointV2(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d mass_position_, int id_)
{
	init();
	this->connected_link = link;

	this->ori_position_w = ori_position_w_;
	this->ori_orientation_w = ori_orientation_w_;

	this->orientation.setZero();  //update after we got kinematic chain information
	this->ori_orientation_4d.setZero();
	this->ori_position = ori_position_;

	//ori_orientation_4d.topLeftCorner<3, 3>() = ori_orientation;

	this->position = ori_position;
	this->position_w = ori_position_w;
	this->orientation = ori_orientation;

	this->joint_position = ori_position_;

	this->joint_id = id_;

	//massCenterInLocal = -joint_position;

	massCenterInLocal = mass_position_;  //relative to joint frame

	ifComputeThirdDerive = false;
	ifComputeSecondDerive = false;
	//updateLinkPose();
}


LoboJointV2::LoboJointV2()
{
	init();
}

LoboJointV2::~LoboJointV2()
{

}

void LoboJointV2::resetPose(Vector3d ori_position_/*, Matrix3d ori_orientation_*/)
{
	this->ori_position = ori_position_;
	//this->ori_orientation = ori_orientation_;
	//ori_orientation_4d.setIdentity();
	//ori_orientation_4d.topLeftCorner<3, 3>() = ori_orientation;

	//this->position_w = ori_orientation*ori_position+this->joint_parent->position_w;
	//this->orientation = ori_orientation;

	this->joint_position = ori_position_;
	//jointposition_in_p = ori_position + joint_position;
}

void LoboJointV2::updatePoseInWorld()
{
	//position = (-orientation*joint_position + joint_position) + ori_position;

	if (joint_parent == NULL)
	{
		orientation_w = orientation;
		position_w = position;

		//global_TransForm = local_TransForm;
		global_TransForm = local_TransForm;
	}
	else
	{
		orientation_w = joint_parent->orientation_w*orientation;
		position_w = joint_parent->position_w + joint_parent->orientation_w*ori_position;

		global_TransForm = joint_parent->global_TransForm*local_TransForm;
		orientation_w = global_TransForm.topLeftCorner<3, 3>();
	}

	massCetnerGlobal = position_w + orientation_w*massCenterInLocal;


	/*std::cout << "R" << std::endl;
	std::cout << orientation_w << std::endl;*/
	updateLinkPose();
}

void LoboJointV2::updateAngularVelovity(VectorXd &q, VectorXd &q_dot)
{
	if (joint_parent == nullptr)
	{
		IterativeAngularVelovity.setZero();
		for (size_t i = 0; i < mWq.size(); i++)
		{
			IterativeAngularVelovity += fromSkewSysmmtric(mWq[i].topLeftCorner<3, 3>() *orientation_w.transpose())*q_dot[generalized_offset + i];
		}
	}
	else
	{
		IterativeAngularVelovity = joint_parent->IterativeAngularVelovity;
		for (size_t i = 0; i < mTq.size(); i++)
		{
			IterativeAngularVelovity += joint_parent->orientation_w * fromSkewSysmmtric(mTq[i].topLeftCorner<3, 3>()*orientation.transpose())*q_dot[generalized_offset + i];
		}
	}
}

void LoboJointV2::initOriPositionInWorld()
{
	/*if (!joint_parent)
	{
		ori_position_w = ori_position;
		ori_orientation_w = ori_orientation;
	}
	else
	{
		ori_orientation_w = joint_parent->ori_orientation_w*ori_orientation;
		ori_position_w = joint_parent->ori_position_w + ori_orientation*ori_position;
	}*/
}

void LoboJointV2::updateChainIndex()
{
	std::vector<int> temp;
	LoboJointV2* joint = this;
	std::vector<LoboJointV2*> tempPointer;
	chainJointFromRoot.clear();
	while (true)
	{
		temp.push_back(joint->getJoint_id());
		tempPointer.push_back(joint);
		if (joint->getJoint_parent() != NULL)
		{
			joint = joint->getJoint_parent();
		}
		else
		{
			break;
		}
	}

	chainFromRoot.resize(temp.size());
	for (int i = 0; i < temp.size(); i++)
	{
		chainFromRoot[i] = temp[temp.size() - i - 1];
	}
	chainJointFromRoot.resize(tempPointer.size());
	for (int i = 0; i < tempPointer.size(); i++)
	{
		chainJointFromRoot[i] = tempPointer[tempPointer.size() - i - 1];
	}

	dependentDOfs = 0;
	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		dependentDOfs += chainJointFromRoot[i]->getR();
	}
	mapDependentDofs.resize(dependentDOfs);

	int count = 0;
	for (int i = 0; i < chainJointFromRoot.size(); i++)
	{
		int offset = chainJointFromRoot[i]->getGeneralized_offset();
		int localr = chainJointFromRoot[i]->getR();
		for (int j = 0; j < localr; j++)
		{
			int dofs_In_global = offset + j;
			mapDependentDofs[count] = dofs_In_global;
			count++;
		}
	}


}

void LoboJointV2::updateLinkPose()
{
	if (connected_link == NULL)
	{
		return;
	}
	connected_link->setOrientation_world(orientation_w);
	connected_link->setPosition_world(massCetnerGlobal);
}

void LoboJointV2::computedMassdQ(double mass, Matrix3d inertia, MatrixXd &dmassdq, VectorXd &massQ_)
{
	Matrix3d currentInvertia = global_TransForm.topLeftCorner<3, 3>()*inertia*global_TransForm.topLeftCorner<3, 3>().transpose();
	MatrixXd temp(getGlobalR(), getGlobalR());

	for (int i = 0; i < getDependentDOfs(); i++)
	{
		temp.setZero();
		temp.noalias() += mass*JK_vq[i].transpose()*JK_v;
		temp.noalias() += mass*JK_v.transpose()*JK_vq[i];
		Matrix3d currentInvertia_q = mWq[i].topLeftCorner<3, 3>()*inertia*global_TransForm.topLeftCorner<3, 3>().transpose()
			+ global_TransForm.topLeftCorner<3, 3>()*inertia*mWq[i].topLeftCorner<3, 3>().transpose();

		temp.noalias() += JK_wq[i].transpose()*currentInvertia*JK_w;
		temp.noalias() += JK_w.transpose()*currentInvertia_q*JK_w;
		temp.noalias() += JK_w.transpose()*currentInvertia*JK_wq[i];

		VectorXd tempColumn = temp*massQ_;

		//dmassdq.col(mapDependentDofs[i]) += temp*massQ_;
		int col = mapDependentDofs[i];

		for (int j = 0; j < dmassdq.rows(); j++)
		{
			dmassdq.data()[col*dmassdq.rows() + j] += tempColumn.data()[j];
		}

	}
}

void LoboJointV2::computedCoriolisdQ(double mass, Matrix3d inertia, MatrixXd &dCdq, VectorXd &cQ_, double dqdotdq)
{
	Matrix3d currentInvertia = global_TransForm.topLeftCorner<3, 3>()*inertia*global_TransForm.topLeftCorner<3, 3>().transpose();
	MatrixXd temp(getGlobalR(), getGlobalR());
	Vector3d omega = JK_w*cQ_;
	Matrix3d omega_skew;
	skewMatrix(omega, omega_skew);


	for (int i = 0; i < getDependentDOfs(); i++)
	{
		temp.setZero();
		temp.noalias() += mass*JK_vq[i].transpose()*JK_vdot;
		temp.noalias() += mass*JK_v.transpose()*JK_vdotq[i];

		Matrix3d currentInvertia_q = mWq[i].topLeftCorner<3, 3>()*inertia*global_TransForm.topLeftCorner<3, 3>().transpose()
			+ global_TransForm.topLeftCorner<3, 3>()*inertia*mWq[i].topLeftCorner<3, 3>().transpose();

		temp.noalias() += JK_wq[i].transpose()*currentInvertia*JK_wdot;
		temp.noalias() += JK_w.transpose()*currentInvertia_q*JK_wdot;
		temp.noalias() += JK_w.transpose()*currentInvertia*JK_wdotq[i];

		Vector3d omegaderive = JK_wq[i] * cQ_ /*+ JK_w.col(i)*dqdotdq*/;
		Matrix3d omegaderive_skew;
		skewMatrix(omegaderive, omegaderive_skew);

		temp.noalias() += JK_wq[i].transpose()*omega_skew*currentInvertia*JK_w;
		temp.noalias() += JK_w.transpose()*omegaderive_skew*currentInvertia*JK_w;
		temp.noalias() += JK_w.transpose()*omega_skew*currentInvertia_q*JK_w;
		temp.noalias() += JK_w.transpose()*omega_skew*currentInvertia*JK_wq[i];

		dCdq.col(mapDependentDofs[i]) += temp*cQ_;
	}

}

bool LoboJointV2::isInJoint(Vector3d position)
{
	if (position.x() >= ori_position_w.x() && position.x() <= ori_position_w.x() + 2 * massCenterInLocal.x())
	{
		return true;
	}
	else
		return false;
}

int LoboJointV2::getNumChild()
{
	return joint_children.size();
}

LoboJointV2* LoboJointV2::getChild(int id)
{
	return joint_children[id];
}

void LoboJointV2::addChild(LoboJointV2* child)
{
	joint_children.push_back(child);
}

LoboLink* LoboJointV2::getConnectedLink()
{
	return connected_link;
}

void LoboJointV2::updateLocalTransform()
{
	local_TransForm.topLeftCorner<3, 3>() = orientation;
	local_TransForm.data()[12] = ori_position.data()[0];
	local_TransForm.data()[13] = ori_position.data()[1];
	local_TransForm.data()[14] = ori_position.data()[2];
}

void LoboJointV2::init()
{
	joint_position.setZero();

	position.setZero();
	ori_position.setZero();

	orientation.setZero();
	ori_orientation.setZero();

	position_w.setZero();
	orientation_w.setZero();

	linear_velocity.setZero();
	angular_velocity.setZero();

	joint_parent = NULL;
	connected_link = NULL;


	local_TransForm.setIdentity();
	global_TransForm.setIdentity();

	IterativeAngularVelovity.setZero();
	LimitLower = -1e12;
	LimitUpper = 1e12;
}
