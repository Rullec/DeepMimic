
#include "LoboLink.h"
#include <iostream>
#include "Functions/EulerAngelRotationMatrix.h"

LoboLink::LoboLink(Vector3d position_, Matrix3d orientation_, int id_)
{
	init();
	this->position = position_;
	this->setOrientation(orientation_);
	q_rotation = Eigen::Quaternion<double>(orientation_);

	this->ori_position = position_;
	this->ori_orientation = orientation_;
	this->setLink_id(id_);


}

LoboLink::~LoboLink()
{
	//delete linkMesh;
}

void LoboLink::setParent(LoboLink* parent)
{
	link_parent = parent;
}

void LoboLink::addChildLink(LoboLink* child)
{
	link_children.push_back(child);
}

void LoboLink::updateChainIndex()
{
	std::vector<int> temp;
	LoboLink* link = this;

	while (true)
	{
		temp.push_back(link->getLink_id());
		if (link->getParentLink() != NULL)
		{
			link = link->getParentLink();
		}
		else
		{
			break;
		}
	}

	chainFromRoot.resize(temp.size());
	for (int i = 0; i < temp.size(); i++)
	{
		chainFromRoot[i] = temp[temp.size() - i -1];
	}
}

void LoboLink::updatePoseInWorld()
{
	//if (link_parent == NULL)
	//{
	//	orientation_world = getOrientation();
	//	position = (-getOrientation()*joint_parent->getOri_position() + joint_parent->getOri_position()) + ori_position;
	//	position_world = position;
	//	return;
	//}

	////R^0_p * R_k
	//orientation_world = link_parent->getOrientation_world()*getOrientation();
	//position = (-getOrientation()*joint_parent->getOri_position() + joint_parent->getOri_position())+ori_position;
	//position_world = link_parent->getPosition_world() + link_parent->getOrientation_world()*position;
}

void LoboLink::computeMassMatrix(MatrixXd &massMatrix)
{
	massMatrix.resize(6, 6);
	massMatrix.setZero();
	massMatrix.data()[0 * 6 + 0] = mass;
	massMatrix.data()[1 * 6 + 1] = mass;
	massMatrix.data()[2 * 6 + 2] = mass;
	Matrix3d inertia = orientation_world*InertiaTensor*orientation_world.transpose();
	massMatrix.block(3, 3, 3, 3) = inertia;
}

void LoboLink::resetForce()
{
	force.setZero();
	torque.setZero();
}

void LoboLink::addSpringForce(Vector3d positionInLinkFrame, Vector3d target, double ratio)
{
	Vector3d positionInWorld = orientation_world*positionInLinkFrame + position_world;
	Vector3d forcevalue = (target - positionInWorld)*ratio;
	force += forcevalue;

	Vector3d  r = positionInWorld - position_world;
	torque += r.cross(forcevalue);
}

bool LoboLink::isInLink(Vector3d position)
{
	return false;
}

int LoboLink::getNumChildren()
{
	return link_children.size();
}

LoboLink* LoboLink::getChildLink(int index)
{
	return link_children[index];
}

LoboLink* LoboLink::getParentLink()
{
	return link_parent;
}

void LoboLink::initLinkRender(const char* filename)
{
	//linkMesh = new SphereRender(filename, false,false);
}

void LoboLink::updatePose(VectorXd&q, VectorXd &q_dot, double time_step)
{
	/*Vector3d localV = joint_parent->getJvlocal()*q_dot;
	Vector3d angular = joint_parent->getJwlocal()*q_dot;

	orientation = joint_parent->getOrientationByGeneralizedQ(q)*ori_orientation;

	position += time_step*localV;*/


}

void LoboLink::init()
{
	useMeshRender = false;
	orientation_world.setIdentity();
	position_world.setZero();
	InertiaTensor.setIdentity();
	

	linaer_velocity.setZero();
	angular_velocity.setZero();

	force.setZero();
	torque.setZero();

	visualbox.data()[0] = 0.1;
	visualbox.data()[1] = 0.1;
	visualbox.data()[2] = 0.1;

	link_parent = NULL;
	//linkMesh = NULL;

	ConnectJoint = nullptr;
}
