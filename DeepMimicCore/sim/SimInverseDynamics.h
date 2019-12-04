#pragma once
#include <BulletInverseDynamics/details/IDLinearMathInterface.hpp>
#include <BulletInverseDynamics/MultiBodyTree.hpp>
#include "BulletInverseDynamics/IDConfig.hpp"
#include <BulletDynamics/Featherstone/btMultiBodyLink.h>
#include <util/MathUtil.h>


#define MAX_FRAME_NUM 10000
class btMultiBody;
class btMultiBodyDynamicsWorld;
class cSimCharacter;
struct tForceInfo {
	int mId;
	tVector mPos, mForce;
	tForceInfo()
	{
		mId = -1;
		mPos = mForce = tVector::Zero();
	}
};
class cIDSolver {
public:
	enum eSolvingMode {
		VEL = 0,
		POS,
	};
	cIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world);
	void ClearID();
	void SetTimestep(double deltaTime);
	void PreSim();
	void PostSim();

private:
	// ID vars
	cSimCharacter * mSimChar;
	btMultiBody* mMultibody;
	btMultiBodyDynamicsWorld * mWorld;
	btInverseDynamicsBullet3::MultiBodyTree * mInverseModel;
	bool mEnableExternalForce;
	bool mEnableExternalTorque;
	bool mEnableSolveID;
	bool mFloatingBase;
	int mDof;
	int mNumLinks;		// including root
	double mCurTimestep;
	std::map<int, int> mWorldId2InverseId;	// map the world array index to id in inverse dynamics
	std::map<int, int> mInverseId2WorldId;	// reverse map above
	int mFrameId;
	eSolvingMode mSolvingMode;


	// ID buffer vars
	std::vector<tForceInfo> mContactForces;
	std::vector<tVector> mJointForces;	// reference joint torque(except root) in link frame
	std::vector<tVector> mSolvedJointForces;	// solved joint torques from
	std::vector<tVector> mExternalForces;// for each link, external forces in COM
	std::vector<tVector> mExternalTorques;	// for each link, external torques
	std::vector<tMatrix> mLinkRot;	// local to world rotation mats
	std::vector<tVector> mLinkPos;	// link COM pos in world frame

	//// permanent memory
	tVectorXd mBuffer_q[MAX_FRAME_NUM];		// generalized coordinate "q" buffer, storaged for each frame
	tVectorXd mBuffer_u[MAX_FRAME_NUM];		// q_dot = u buffer, storaged for each frame
	tVectorXd mBuffer_u_dot[MAX_FRAME_NUM];	//

	// tools 
	void AddJointForces();
	void AddExternalForces();
	void RecordContactForces();
	void RecordJointForces();
	void RecordMultibodyInfo(std::vector<tMatrix> & local_to_world_rot, std::vector<tVector> & link_pos_world) const;
	void RecordGeneralizedInfo(btInverseDynamicsBullet3::vecx & q, btInverseDynamicsBullet3::vecx & q_dot) const;
	void RecordGeneralizedInfo(tVectorXd & q, tVectorXd & q_dot) const;
	void SolveID();
	void ApplyContactForcesToID();
	void ApplyExternalForcesToID();
	tVectorXd CalculateGeneralizedVel(const tVectorXd & q_before, const tVectorXd & q_after, double timestep) const;
};