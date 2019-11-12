#pragma once
#include <Eigen/Dense>
#include <vector>
//#include "Render/SphereRender.h"
#include"LoboJointV2.h"
using namespace Eigen;

// base class of rigid link
class LoboLink
{

public:
	LoboLink(Vector3d position, Matrix3d orientation, int id_);
	~LoboLink();

	virtual void setParent(LoboLink* parent);
	virtual void addChildLink(LoboLink* child);
	virtual void updateChainIndex();

	virtual void updatePoseInWorld();

	virtual void computeMassMatrix(MatrixXd &massMatrix);

	virtual void resetForce();
	virtual void addSpringForce(Vector3d positionInLinkFrame, Vector3d target,double ratio);

	bool isInLink(Vector3d position);

	int getNumChildren();
	LoboLink* getChildLink(int index);
	LoboLink* getParentLink();

	virtual void initLinkRender(const char* filename);


	Vector3d getPosition_world() const { return position_world; }
	void setPosition_world(Vector3d val) { position_world = val; }
	Matrix3d getOrientation_world() const { return orientation_world; }
	void setOrientation_world(Matrix3d val) { orientation_world = val; }


	int getLink_id() const { return link_id; }
	void setLink_id(int val) { link_id = val; }

	Matrix3d getOrientation() const { return orientation; }
	void setOrientation(Matrix3d val) { orientation = val; }
	Matrix3d getInertiaTensor() const { return InertiaTensor; }
	void setInertiaTensor(Matrix3d val) { InertiaTensor = val; }


	std::vector<int> getChainFromRoot() const { return chainFromRoot; }
	void setChainFromRoot(std::vector<int> val) { chainFromRoot = val; }

	Vector3d getAngular_velocity() const { return angular_velocity; }
	void setAngular_velocity(Vector3d val) { angular_velocity = val; }
	Vector3d getLinaer_velocity() const { return linaer_velocity; }
	void setLinaer_velocity(Vector3d val) { linaer_velocity = val; }
	double getMass() const { return mass; }
	void setMass(double val) { mass = val; }

	void updatePose(VectorXd&q, VectorXd &q_dot,double time_step);

	Vector3d getForce() const { return force; }
	void setForce(Vector3d val) { force = val; }
	Vector3d getTorque() const { return torque; }
	void setTorque(Vector3d val) { torque = val; }
	Vector3d getPosition() const { return position; }
	void setPosition(Vector3d val) { position = val; }
	Matrix3d getOri_orientation() const { return ori_orientation; }
	void setOri_orientation(Matrix3d val) { ori_orientation = val; }
	Vector3d getOri_position() const { return ori_position; }
	void setOri_position(Vector3d val) { ori_position = val; }

	Vector3d getVisualbox() const { return visualbox; }
	void setVisualbox(Vector3d val) { visualbox = val; }
	//SphereRender* getLinkMesh() const { return linkMesh; }
	//void setLinkMesh(SphereRender* val) { linkMesh = val; }
	bool getUseMeshRender() const { return useMeshRender; }
	void setUseMeshRender(bool val) { useMeshRender = val; }
	Vector3d getVisual_position() const { return visual_position; }
	void setVisual_position(Vector3d val) { visual_position = val; }
	Matrix3d getVisual_orientation() const { return visual_orientation; }
	void setVisual_orientation(Matrix3d val) { visual_orientation = val; }
	std::string getName() const { return name; }
	void setName(std::string val) { name = val; }
	LoboJointV2*getConnectJoint() { return ConnectJoint; }
	void setConnectJoint(LoboJointV2* ConnectJoint_) { ConnectJoint = ConnectJoint_; }
protected:
	LoboJointV2* ConnectJoint;
	virtual void init();

	int link_id;
	std::string name;

	//pose in parent frame mass center
	Vector3d position;
	Matrix3d orientation;
	
	Vector3d ori_position;
	Matrix3d ori_orientation;

	Vector3d visual_position;
	Matrix3d visual_orientation;

	Vector3d position_world;
	Matrix3d orientation_world;

	Eigen::Quaternion<double> q_rotation;

	Matrix3d InertiaTensor;

	// the tree
	LoboLink* link_parent;
	std::vector<LoboLink*> link_children;
	std::vector<int> chainFromRoot;

	

	double mass;

	//in world frame
	Vector3d angular_velocity;
	Vector3d linaer_velocity;

	Vector3d force;
	Vector3d torque;

	Vector3d visualbox;

	//SphereRender* linkMesh;


	bool useMeshRender;
};

