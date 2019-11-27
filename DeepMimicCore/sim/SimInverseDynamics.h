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
		// pos, vel, accel in COM of each link for every frame
		Eigen::MatrixXd mLinkPos, mLinkVel, mLinkAccel;

		struct tAngularQuaternionInfo {
			Eigen::MatrixXd mLinkJointAngle, mLinkAngularVel, mLinkAngularAccel;
		} mAngularQuaternionInfo;	// quaternion storaged format info
		
		struct tAngularEulerInfo {
			Eigen::MatrixXd mLinkJointAngle, mLinkAngularVel, mLinkAngularAccel;
		} mAngularEulerInfo;		// euler-angle storaged format info 

		Eigen::VectorXd mTimesteps;
	};

	cInverseDynamicsInfo(std::shared_ptr<cSimCharacter> &);

	int GetNumOfFrames();
	double GetMaxTime();
	Eigen::VectorXd GetPose(double time);
	Eigen::Vector3d GetLinkPos(int frame, int body_id);
	Eigen::Vector3d GetLinkVel(int frame, int body_id);
	Eigen::Vector3d GetLinkAccel(int frame, int body_id);
	tQuaternion GetLinkJointAngle_quaternion(int frame, int body_id);
	tQuaternion GetLinkAngularVel_quaternion(int frame, int body_id);
	tQuaternion GetLinkAngularAccel_quaternion(int frame, int body_id);
	Eigen::Vector3d GetLinkJointAngle_euler(int frame, int body_id);
	Eigen::Vector3d GetLinkAngularVel_euler(int frame, int body_id);
	Eigen::Vector3d GetLinkAngularAccel_euler(int frame, int body_id);

	void AddNewFrame(const Eigen::VectorXd & state_, const Eigen::VectorXd & pose_, const Eigen::VectorXd & action_, const Eigen::VectorXd & _contact_info);
	void ComputeLinkInfo();

private:
	std::shared_ptr<cSimCharacter> mSimChar;

	// number of frames
	int mNumOfFrames;

	// infos from ID recording file
	Eigen::MatrixXd mState, mPose, mAction, mContact_info;
	int mStateSize, mPoseSize, mActionSize, mContact_infoSize;
	
	// pos/vel/accel of COM for each link in every frame
	std::shared_ptr<struct tLinkCOMInfo> mLinkInfo;

	void ComputeLinkInfo0(int, int);
	void ComputeLinkInfo1(int, int);
	void ComputeLinkInfo2(int, int);
};

