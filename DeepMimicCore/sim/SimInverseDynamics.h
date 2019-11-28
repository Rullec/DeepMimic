#pragma once
#include "SimCharacter.h"

/*
	@class cInverseDyanmicsInfo

	This class is used to storage / compute / utilize the essential info in InverseDynamics solving procedure.
*/
class cInverseDynamicsInfo {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	struct tLinkCOMInfo {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		// pos, vel, accel in COM of each link for every frame, all of them are in Cartesian space
		Eigen::MatrixXd mLinkPos, mLinkVel, mLinkAccel;

		// rotation, angular vel, angular accel; 
		// for rotaion the value storaged in mLinkRot is Quaternion().coeff() = [x, y, z, w] 4*1
		// for angular velocity it is axis-angle = [axis_x, axis_y, axis_z, d(theta)/dt] 4*1
		// for angular accel it is axis-angle = [axis_x, axis_y, axis_z, d(theta^2)/dt^2] 4*1
		Eigen::MatrixXd mLinkRot, mLinkAngularVel, mLinkAngularAccel;
		
		Eigen::VectorXd mTimesteps;
	};

	cInverseDynamicsInfo(std::shared_ptr<cSimCharacter> &);

	int GetNumOfFrames();
	double GetMaxTime();
	Eigen::VectorXd GetPose(double time);
	Eigen::Vector3d GetLinkPos(int frame, int body_id);
	Eigen::Vector3d GetLinkVel(int frame, int body_id);
	Eigen::Vector3d GetLinkAccel(int frame, int body_id);
	tVector GetLinkRotation(int frame, int body_id);	// get Quaternion coeff 4*1 [x, y, z, w]
	tVector GetLinkAngularVel(int frame, int body_id);	// get link angular vel 4*1 [wx, wy, wz, 0]
	tVector GetLinkAngularAccel(int frame, int body_id);	// get link angular accel 4*1 [ax, ay, az, 0]
	
	void AddNewFrame(const Eigen::VectorXd & state_, const Eigen::VectorXd & pose_, const Eigen::VectorXd & action_, const Eigen::VectorXd & _contact_info);
	void SolveInverseDynamics();

private:
	enum eIDStatus {
		INVALID = 0,
		PREPARED,
		SOLVED,
	};

	std::shared_ptr<cSimCharacter> mSimChar;
	eIDStatus mIDStatus;

	// read ID essential info from file
	int mNumOfFrames;
	Eigen::MatrixXd mState, mPose, mAction, mContact_info;		
	int mStateSize, mPoseSize, mActionSize, mContact_infoSize;
	
	// info for solving ID (link status)
	std::shared_ptr<struct tLinkCOMInfo> mLinkInfo;
	
	void ComputeLinkInfo();
	void ComputeLinkInfo0(int, int);
	void ComputeLinkInfo1(int, int) ;
	void ComputeLinkInfo2(int, int) ;
	void ComputeJointTorque(const std::shared_ptr<struct tLinkCOMInfo> &, Eigen::VectorXd &) const ;
	void ComputeSingleJointTorque(const std::shared_ptr<struct tLinkCOMInfo> &, Eigen::VectorXd &) const;
	void ComputePDTarget(const Eigen::VectorXd &, Eigen::VectorXd &) const ;

	void PrintLinkInfo();
};

