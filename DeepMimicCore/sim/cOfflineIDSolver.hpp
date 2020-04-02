#include "cIDSolver.hpp"
#include <string>

class btMultiBodyDynamicsWorld;
class cSimCharacter;
class cMotion;

enum eOfflineSolverMode{
    INVALID,
    Save,
    Display,
    Solve
};

namespace Json{
    class Value;
};

// Offline Inverse Dynamics Solver, differ from the online way...
class cOfflineIDSolver:public cIDSolver{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cOfflineIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, const std::string & config);
    virtual void Reset() override final;
    eOfflineSolverMode GetOfflineSolverMode();
    
    // "save" mode APIs
    // "save" mode means that, this solver is focus on recording the running trajectories and save it to a specified file.
    // So the resulted file can be used as an input of the ID solving procedure, also can be verified in "display" mode.
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void SetTimestep(double deltaTime) override final;

    // "solve" mode
    void OfflineSolve();

    // "display mode"
    void DisplaySet();

protected:
    eOfflineSolverMode mMode;

	struct {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		std::string mSaveTrajRoot = "";
        std::string mSaveMotionRoot = "";
        int mCurEpoch = 0;
		int mCurFrameId = 0;
		std::vector<tVector> mTruthJointForces[MAX_FRAME_NUM];
        std::vector<tVector> mSolvedJointForces[MAX_FRAME_NUM];
        tVectorXd mBuffer_q[MAX_FRAME_NUM], mBuffer_u[MAX_FRAME_NUM], mBuffer_u_dot[MAX_FRAME_NUM];
        std::vector<tMatrix> mLinkRot[MAX_FRAME_NUM];	// local to world rotation mats
	    std::vector<tVector> mLinkPos[MAX_FRAME_NUM];	// link COM pos in world frame
        double mTimesteps[MAX_FRAME_NUM];   // timesteps
        cMotion * mMotion;
        std::vector<tForceInfo> mContactForces[MAX_FRAME_NUM];
        std::vector<tVector> mExternalForces[MAX_FRAME_NUM], mExternalTorques[MAX_FRAME_NUM];
        tVector mLinearMomentum[MAX_FRAME_NUM], mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
        tVectorXd mCharPoses[MAX_FRAME_NUM];
	} mSaveInfo;

    enum eLoadMode{
        INVALID,
        LOAD_MOTION,
        LOAD_TRAJ
    };
    struct {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        eLoadMode mLoadMode = eLoadMode::INVALID;
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat;
        cMotion * mMotion = nullptr;
        tVectorXd mTimesteps;
        std::vector<std::vector<tForceInfo>> mContactForces;
        std::vector<std::vector<tMatrix>> mLinkRot;	// local to world rotation mats
        std::vector<std::vector<tVector>> mLinkPos;	// link COM pos in world frame
        std::vector<std::vector<tVector>> mExternalForces, mExternalTorques;
        std::vector<std::vector<tVector>> mTruthJointForces;
        int mTotalFrame = 0;
        int mCurFrame = 0;
    } mLoadInfo;    // work for display and solve

    // ways
    void ParseConfig(const std::string & path);
    void ParseConfigSave(const Json::Value & save_value);
    void ParseConfigSolve(const Json::Value & solve_value);
    void ParseConfigDisplay(const Json::Value & display_value);

    void LoadTraj(const std::string & path);
    void LoadMotion(const std::string & path, cMotion * motion) const;
    void SaveTraj(const std::string & path);
    void SaveMotion(const std::string & path, cMotion * motion) const;

    void VerifyMomentum();
};