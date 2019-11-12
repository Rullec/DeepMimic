#pragma once
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class LoboJointV2;
class LoboLink;


struct JointXML
{
	JointXML()
	{
		parentid = -1;
		childid = -1;
		name = "";
		type = 1;
		axis = Vector3d::Zero();
		lower = -1e12;
		upper = 1e12;

	}
	int parentid;
	int childid;
	std::string name;
	//0 universal 1 ball 2 hinge 3 fix
	int type;

	Vector3d axis;
	double lower, upper;
	double pose[6];
};

struct LinkXML
{
	double pose[6];
	double visualpose[6];
	double masspose[6];
	double box[3];
	Matrix3d InertialTensor;
	double mass;
	const char* linkname;

	int link_parent;
	std::vector<int> link_child;
	int jointid;

	const char* meshFilePath;

	bool useTriMeshRender;
	bool useBoxRender;

};

class MultiRigidBodyModel
{
public:
	MultiRigidBodyModel();
	~MultiRigidBodyModel();

	//call this function when the building of the tree is done:
	virtual void initGeneralizedInfo();

	virtual void loadXML(const char* filename,double massscale=1.0,double meshscale=1.0);

	//void loadXML2(const char * filename);

	void convertRestpos2Worldpos(Vector3d &pos_w, const Vector3d &pos_r, int jointid);

	virtual void addLink(Matrix3d Inertia_tensor, double mass);
	virtual void addLink(Matrix3d Inertia_tensor, double mass, const std::string link_name);
	virtual void addLink(Matrix3d Inertia_tensor, double mass, Vector3d box);
	virtual void addLink(Matrix3d Inertia_tensor, double mass, const char* meshfilepath);
	virtual void addLink(Matrix3d Inertia_tensor, double mass, const char* meshfilepath, Vector3d box);

	virtual void addJoint(LoboJointV2* parent, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, LoboLink* link);

	virtual void addJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid);
	virtual void addHingeJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid, int hingeType);
	virtual void addUniversalJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid);

	//after every thing is ok, we can compute matrix
	virtual void setModelState(VectorXd &q, VectorXd &q_dot, bool updateDynamic);

	//
	Vector3d getNodeVelocity(int jointid, const Vector3d& nodepos_w,const VectorXd&qvel);

	//mass matrix = JMJ
	//C*q_dot is Coriolis and centrifugal force
	virtual void getMatrix(VectorXd&q, VectorXd &q_dot, MatrixXd &massMatrix, MatrixXd &C, VectorXd &Cq);

	virtual void getMatrix(VectorXd&q, VectorXd &q_dot, VectorXd &massQ_, VectorXd &cQ_, MatrixXd &dMassdq_q, MatrixXd& dCdq_q, MatrixXd &mass, MatrixXd &C, double dqdotdq);

	virtual void getMatrix(VectorXd&q, VectorXd&qvel, VectorXd&qaccel, MatrixXd &dMdq_qaccel, MatrixXd&dCdq_qvel, MatrixXd &M, MatrixXd &C);

	virtual void getJacobiVByGivenPosition(MatrixXd &jacobi_q, Vector3d x_position, int jointid);

	virtual void getJacobiVByGivenRestPosition(MatrixXd &jacobi_q, Vector3d x_position_rest, int jointid);

	//f_cartesian is a 6*num_of_links Vector, it stores the force and toquer for each link in original order
	virtual void convertExternalForce(VectorXd f_cartesian, VectorXd &Q, bool ifgravity);

	virtual void clampRotation(VectorXd& _q, VectorXd& _qdot);

	void UpdateRigidMesh();

	void setIfComputeThirdDerive(bool b);

	void setIfComputeSecondDerive(bool b);

	int getNumofDOFs() const { return numofDOFs; }
	void setNumofDOFs(int val) { numofDOFs = val; }

	int getnNumJoints();
	LoboJointV2* getJoint(int id);
	LoboJointV2* getJoint(std::string jointname);
	LoboLink* getLink(int id);
	LoboLink* getLink(std::string linkname);
	virtual void printMultiBodyInfo();

	bool getUseGravity() const { return useGravity; }
	void setUseGravity(bool val) { useGravity = val; }
	Vector3d getGravity_force() const { return gravity_force; }
	void setGravity_force(Vector3d val) { gravity_force = val; }

	virtual void deleteModel();
	
	std::string getname() { return name; }
protected:
	std::string name;
	virtual void buildJointTreeIndex();

	Vector3d gravity_force;
	bool useGravity;

	std::vector<LoboLink*> links_list;
	std::vector<LoboJointV2*> joints_list;

	std::vector<int> joint_tree_index;
	std::vector<int> generalized_offset; //will tell where is the link k's position in q;

	LoboJointV2* root_joint;

	int numofDOFs;
};

