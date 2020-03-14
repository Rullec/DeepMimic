#include "cIDSolver.hpp"
#include <string>

class btMultiBodyDynamicsWorld;
class cSimCharacter;

enum eOfflineSolverMode{
    INVALID,
    Save,
    Display,
    Solve
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
    void SaveToFile(const std::string & path);
    void LoadFromFile(const std::string & path);
    virtual void SetTimestep(double deltaTime) override final;

    // "solve" mode
    void OfflineSolve();

    // "display mode"
    void DisplaySet();

protected:
    eOfflineSolverMode mMode;

	struct {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		std::string mSavePath = "";
        int mCurEpoch = 0;
		int mCurFrameId = 0;
		std::vector<tVector> mTruthJointForces[MAX_FRAME_NUM];
        std::vector<tVector> mSolvedJointForces[MAX_FRAME_NUM];
        tVectorXd mBuffer_q[MAX_FRAME_NUM], mBuffer_u[MAX_FRAME_NUM], mBuffer_u_dot[MAX_FRAME_NUM];
        std::vector<tMatrix> mLinkRot[MAX_FRAME_NUM];	// local to world rotation mats
	    std::vector<tVector> mLinkPos[MAX_FRAME_NUM];	// link COM pos in world frame
        double mTimesteps[MAX_FRAME_NUM];   // timesteps
        std::vector<tForceInfo> mContactForces[MAX_FRAME_NUM];
        std::vector<tVector> mExternalForces[MAX_FRAME_NUM], mExternalTorques[MAX_FRAME_NUM];
	} mSaveInfo;

    struct {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat;
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
};