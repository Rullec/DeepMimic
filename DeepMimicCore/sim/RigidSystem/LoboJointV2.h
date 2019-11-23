#pragma once
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;
class LoboLink;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EIGEN_V_MAT4D std::vector< Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >
#define EIGEN_V_MATXD std::vector< Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd> >

#define EIGEN_VV_MAT4D std::vector< std::vector< Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > >
#define EIGEN_VVV_MAT4D std::vector<std::vector<std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > >>


enum eRotationOrder {
	XYZ = 0,	// first X, then Y, then Z. X->Y->Z. R_{total} = Rz * Ry * Rx;
	XZY,
	XYX,
	XZX,	// x end
	YXZ,
	YZX,
	YXY,
	YZY,	// y end
	ZXY,
	ZYX,
	ZYZ,
	ZXZ,	// z end
};


extern const enum eRotationOrder gRotationOrder;// rotation order. declared here and defined in LoboJointV2.cpp 
const std::string ROTATION_ORDER_NAME[] = {
	"XYZ",
	"XZY",
	"XYX",
	"XZX",
	"YXZ",
	"YZX",
	"YXY",
	"YZY",
	"ZXY",
	"ZYX",
	"ZYZ",
	"ZXZ",
};

class LoboJointV2
{
public:
	// the link will rotate around the point of joint_position
	LoboJointV2(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d mass_position_,int id_);
	LoboJointV2();
	~LoboJointV2();


	void resetPose(Vector3d ori_position_/*, Matrix3d ori_orientation*/);
	//call this first this function will update all matrix and poses of the link and joint
	virtual void updateJointState(VectorXd &q, VectorXd &q_dot,bool updateDynamic = true) = 0;
	virtual void clampRotation(VectorXd& _q, VectorXd& _qdot) = 0;

	virtual void updatePoseInWorld();

	virtual void updateAngularVelovity(VectorXd &q, VectorXd &q_dot);

	virtual void initOriPositionInWorld();

	virtual void updateChainIndex();

	virtual void updateLinkPose();

	virtual void initGlobalTerm() = 0 ;

	virtual void computedMassdQ(double mass, Matrix3d indertia, MatrixXd &dmassdq, VectorXd &massQ_);
	
	virtual void computedCoriolisdQ(double mass,Matrix3d inertia,MatrixXd &dCdq, VectorXd &cQ_,double dqdotdq=0);

	virtual void updateJacobiByGivenPosition(Vector3d x_position, MatrixXd &JK_v_){};

	virtual void updateJacobiByGivenRelativePosition(Vector3d x_position, MatrixXd &JK_v_) {};

	virtual void updateJacobiByGivenRestPosition(Vector3d x_position, MatrixXd &JK_v_) {};

	typedef enum { UNIVERSALJOINT, BALLJOINT, HINGEJOINT } JointType;

	bool isInJoint(Vector3d position);

	int getJoint_id() const { return joint_id; }
	void setJoint_id(int val) { joint_id = val; }

	int getNumChild();
	LoboJointV2* getChild(int id);

	//set and get the pointer to parent
	LoboJointV2* getJoint_parent() const { return joint_parent; }
	void setJoint_parent(LoboJointV2* val) { joint_parent = val; }
	void addChild(LoboJointV2* child);

	LoboLink* getConnectedLink();


	int getGeneralized_offset() const { return generalized_offset; }
	void setGeneralized_offset(int val) { generalized_offset = val; }
	int getR() const { return r; }
	void setR(int val) { r = val; }
	int getGlobalR() const { return globalR; }
	void setGlobalR(int val) { globalR = val; }
	int getDependentDOfs() const { return dependentDOfs; }
	void setDependentDOfs(int val) { dependentDOfs = val; }
	bool getIfComputeThirdDerive() const { return ifComputeThirdDerive; }

	void setIfComputeThirdDerive(bool val) { ifComputeThirdDerive = val; }
	void setIfComputeSecondDerive(bool val) { ifComputeSecondDerive = val; }

	Vector3d getOri_position() const { return ori_position; }
	void setOri_position(Vector3d val) { ori_position = val; }
	std::vector<LoboJointV2*>& getChainJointFromRoot() { return chainJointFromRoot; }

protected:



	//************************************
	// Method:    updateLocalTransform
	// FullName:  BallJoint::updateLocalTransform
	// Access:    virtual protected 
	// Returns:   void
	// Qualifier: 
	// call this after orientation updated
	//************************************
	virtual void updateLocalTransform();

	virtual void init();

	int joint_id;

	LoboLink* connected_link; //after this joint
	LoboJointV2* joint_parent; //before this joint
	std::vector<LoboJointV2*> joint_children;
	std::vector<int> chainFromRoot;
	std::vector<LoboJointV2*> chainJointFromRoot;
	std::vector<int> mapDependentDofs;


	JointType jointType;

	int generalized_offset;
	int r;
	int dependentDOfs;
	int globalR;

	bool ifComputeThirdDerive;
	bool ifComputeSecondDerive;

public:
	std::string name;
	//joint limits
	double LimitUpper, LimitLower;

	//pose in link frame
	Vector3d joint_position;

	//
	Vector3d position;
	Matrix3d orientation;

	//in parent frame
	Vector3d ori_position;	
	Matrix3d ori_orientation;
	Matrix4d ori_orientation_4d;

	//in world frame after rotations at joints
	Vector3d position_w;
	Matrix3d orientation_w;

	//joint rest pose in world frame
	Vector3d ori_position_w;  
	Matrix3d ori_orientation_w;

	//in world frame
	Vector3d linear_velocity;
	Vector3d angular_velocity;

	//link local to joint local  
	//Matrix3d ori_link_orientation;
	
	//term in homogeneous transformation
	//4X4
	//position_in_parent=local_TransForm*position_in_jointframe
	Matrix4d local_TransForm;
	Matrix4d global_TransForm;

	
	
	Vector3d massCenterInLocal;
	Vector3d massCetnerGlobal;
	//ÕâÁ½¸ö
	Vector3d jointposition_in_p; // joint position in parent coordinate same as ori_position;
	Vector3d jointposition_in_pj; // joint position in parent coordinate but the 0 is parent joint.

	EIGEN_V_MAT4D mTq; //partial local transform partial q 
	EIGEN_V_MAT4D mWq; //partial global transform partial q

	EIGEN_VV_MAT4D mTqq;
	EIGEN_VV_MAT4D mWqq;

	EIGEN_VVV_MAT4D mTqqq;
	EIGEN_VVV_MAT4D mWqqq;

	// generalized coordinate;
	VectorXd generalized_position;
	VectorXd generalized_velocity;

	//jacobi matrix
	MatrixXd jv, jw;
	MatrixXd jv_dot, jw_dot;

	MatrixXd jv_g, jw_g, jv_dot_g, jw_dot_g;

	MatrixXd JK, JK_dot;
	MatrixXd JK_v, JK_w;

	EIGEN_V_MATXD JK_vq;
	EIGEN_V_MATXD JK_wq;

	MatrixXd JK_vdot;
	MatrixXd JK_wdot;

	EIGEN_V_MATXD JK_vdotq;
	EIGEN_V_MATXD JK_wdotq;

	//for Jw debug
	Vector3d IterativeAngularVelovity;
};

