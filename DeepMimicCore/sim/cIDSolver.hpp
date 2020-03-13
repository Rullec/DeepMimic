#pragma once
#include <BulletInverseDynamics/details/IDLinearMathInterface.hpp>
#include <BulletInverseDynamics/MultiBodyTree.hpp>
#include "BulletInverseDynamics/IDConfig.hpp"
#include <BulletDynamics/Featherstone/btMultiBodyLink.h>
#include <util/MathUtil.h>
#include <map>
#define MAX_FRAME_NUM 10000

enum eIDSolverType{
	Online,
	Offline,
	SOLVER_TYPE_NUM
};

struct tForceInfo {
	int mId;
	tVector mPos, mForce;
	tForceInfo()
	{
		mId = -1;
		mPos = mForce = tVector::Zero();
	}
};

class btMultiBody;
class btMultiBodyDynamicsWorld;
class cSimCharacter;

class cIDSolver{
public:
	cIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, eIDSolverType type);
	eIDSolverType GetType();
	virtual void Reset() = 0;
	virtual void PreSim() = 0;
	virtual void PostSim() = 0;
	virtual void SetTimestep(double) = 0;

protected:
	eIDSolverType mType;

	// skeleton profile
	cSimCharacter * mSimChar;
	btMultiBody* mMultibody;
	btMultiBodyDynamicsWorld * mWorld;
	btInverseDynamicsBullet3::MultiBodyTree * mInverseModel;
	bool mFloatingBase;
	int mDof;
	int mNumLinks;		// including root
	std::map<int, int> mWorldId2InverseId;	// map the world array index to id in inverse dynamics
	std::map<int, int> mInverseId2WorldId;	// reverse map above

	// record functions
	void RecordMultibodyInfo(std::vector<tMatrix> & local_to_world_rot, std::vector<tVector> & link_pos_world) const;
	void RecordGeneralizedInfo(tVectorXd & q, tVectorXd & q_dot) const;
	void RecordJointForces(std::vector<tVector> & mJointForces) const;
	void RecordContactForces(std::vector<tForceInfo> &mContactForces, double mCurTimestep, std::map<int, int> &mWorldId2InverseId) const;
	void ApplyContactForcesToID(const std::vector<tForceInfo> &mContactForces, const std::vector<tVector> & mLinkPos, const std::vector<tMatrix> & mLinkRot) const;
	void ApplyExternalForcesToID(const std::vector<tVector> & link_poses, const std::vector<tMatrix> & link_rot, const std::vector<tVector> & ext_forces, const std::vector<tVector> & ext_torques) const;

	// calculation funcs
	tVectorXd CalculateGeneralizedVel(const tVectorXd & q_before, const tVectorXd & q_after, double timestep) const;
	
	// solving single step
	virtual void SolveIDSingleStep(std::vector<tVector> & solved_joint_forces,
		const std::vector<tForceInfo> & contact_forces,
		const std::vector<tVector> link_pos, 
		const std::vector<tMatrix> link_rot, 
		const tVectorXd * mBuffer_q,
		const tVectorXd * mBuffer_u,
		const tVectorXd * mBuffer_u_dot,
		int frame_id,
		const std::vector<tVector> &mExternalForces,
		const std::vector<tVector> &mExternalTorques)const;
};
