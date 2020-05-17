#pragma once

#include "scenes/RLSceneSimChar.h"
#include "anim/KinCharacter.h"

class cSceneImitate : virtual public cRLSceneSimChar
{
	struct RewardParams{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		// reward weight for 5 terms
		double pose_w;
		double vel_w;
		double end_eff_w;
		double root_w;
		double com_w;

		// scale params
		double pose_scale;
		double vel_scale;
		double end_eff_scale;
		double root_scale;
		double com_scale;
		double err_scale;

		// root sub reward weight (under the jurisdiction of root_w)
		double root_pos_w;
		double root_rot_w;
		double root_vel_w;
		double root_angle_vel_w;

		RewardParams();
	};
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cSceneImitate();
	virtual ~cSceneImitate();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();

	virtual const std::shared_ptr<cKinCharacter>& GetKinChar() const;
	virtual void EnableRandRotReset(bool enable);
	virtual bool EnabledRandRotReset() const;

	virtual double CalcReward(int agent_id) const;
	virtual eTerminate CheckTerminate(int agent_id) const;

	virtual std::string GetName() const;
	virtual void SyncKinCharNewCycleInverseDynamic(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const;
protected:

	std::string mMotionFile;
	std::string mAngleDiffDir;
	std::string mRewardFile;
	std::shared_ptr<cKinCharacter> mKinChar;

	struct RewardParams RewParams;
	Eigen::VectorXd mJointWeights;
	bool mEnableRandRotReset;
	bool mSyncCharRootPos;
	bool mSyncCharRootRot;
	bool mEnableRootRotFail;
	bool mEnableAngleDiffLog;
	double mHoldEndFrame;

	virtual bool BuildCharacters();

	virtual void CalcJointWeights(const std::shared_ptr<cSimCharacter>& character, Eigen::VectorXd& out_weights) const;
	virtual bool BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl);
	virtual void BuildKinChar();
	virtual bool BuildKinCharacter(int id, std::shared_ptr<cKinCharacter>& out_char) const;
	virtual void UpdateCharacters(double timestep);
	virtual void UpdateKinChar(double timestep);

	virtual void ResetCharacters();
	virtual void ResetKinChar();
	virtual void SyncCharacters();
	virtual bool EnableSyncChar() const;
	virtual void InitCharacterPosFixed(const std::shared_ptr<cSimCharacter>& out_char);

	virtual void InitRewardWeights();
	virtual void SetRewardParams(Json::Value & root);
	virtual void InitJointWeights();
	virtual void ResolveCharGroundIntersect();
	virtual void ResolveCharGroundIntersect(const std::shared_ptr<cSimCharacter>& out_char) const;
	virtual void SyncKinCharRoot();
	virtual void SyncKinCharNewCycle(const cSimCharacter& sim_char, cKinCharacter& out_kin_char) const;

	virtual double GetKinTime() const;
	virtual bool CheckKinNewCycle(double timestep) const;
	virtual bool HasFallen(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char, const cKinCharacter& kin_char) const;
	
	virtual double CalcRandKinResetTime();
	virtual double CalcRewardImitate(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
	virtual void DiffLogOutput(const cSimCharacter& sim_char, const cKinCharacter& ref_char) const;
};