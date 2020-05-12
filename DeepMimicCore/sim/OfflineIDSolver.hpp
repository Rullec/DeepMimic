#include "IDSolver.hpp"
#include <string>

class btMultiBodyDynamicsWorld;
class cSimCharacter;
class cMotion;

enum eOfflineSolverMode{
    INVALID,
    Save,       // run the simulation and save a single trajectory for this character.
    Display,    // display a motion or trajectory kinematically.
    Solve,      // given a single trajectory, solve the Inverse Dynamic Offline. Output the result.
    Sample      // sample a batch of trajectories and save all of them on the disk.
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
        std::vector<tVector> mLinkVel[MAX_FRAME_NUM];   // link COM vel in world frame
        std::vector<tVector> mLinkOmega[MAX_FRAME_NUM]; // link angular momentum in world frame
        std::vector<tVector> mLinkDiscretVel[MAX_FRAME_NUM];   // link COM vel in world frame calculated from differential of link positions
        std::vector<tVector> mLinkDiscretOmega[MAX_FRAME_NUM]; // link angular momentum in world frame  from differential of link rotation
        tVectorXd mTruthAction[MAX_FRAME_NUM];        // the current action recorded from the controller of this char
        tVectorXd mTruthPDTarget[MAX_FRAME_NUM];        // the current action recorded from the controller of this char

        double mTimesteps[MAX_FRAME_NUM];   // timesteps
        cMotion * mMotion;
        std::vector<tForceInfo> mContactForces[MAX_FRAME_NUM];
        std::vector<tVector> mExternalForces[MAX_FRAME_NUM], mExternalTorques[MAX_FRAME_NUM];
        tVector mLinearMomentum[MAX_FRAME_NUM], mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
        tVectorXd mCharPoses[MAX_FRAME_NUM];
	} mSaveInfo;    // work for save mode

    enum eLoadMode{
        INVALID,
        LOAD_MOTION,
        LOAD_TRAJ
    };
    struct {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        eLoadMode mLoadMode = eLoadMode::INVALID;
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat;
        cMotion * mMotion = nullptr;
        tVectorXd mTimesteps;
        std::vector<std::vector<tForceInfo>> mContactForces;
        std::vector<std::vector<tMatrix>> mLinkRot;	// local to world rotation mats
        std::vector<std::vector<tVector>> mLinkPos;	// link COM pos in world frame
        std::vector<std::vector<tVector>> mExternalForces, mExternalTorques;
        std::vector<std::vector<tVector>> mTruthJointForces;
        int mTotalFrame = 0;
        int mCurFrame = 0;
        bool mEnableOutputMotionInfo = false;
        std::string mOutputMotionInfoPath = "";
    } mLoadInfo;    // work for display and solve mode

    struct {
        int mSampleEpoches;         // the epoche number of trajectoris we want.
        std::string mSampleTrajsDir;    // the saving dir of trajectoris, for example "data/walk/"
        std::string mSampleTrajsRootName;   // the root of trajectories' filename, for example "traj_walk.json". Then "traj_walk_xxx.json" will be genearated and saved
        std::string mSummaryTableFilename;  // These recorded trajs' info will be recorded in this file. It can be "summary_table.json" 
    } mSampleInfo;  // work for sample mode

    struct tSummaryTable{
        std::string mSampleCharFile, mSampleControllerFile; // the character filepath and the controller filepath. They ares recorded in the summary table for clearity
        std::string mTimeStamp;         // the year-month-date timestamp recorded in the summary table
        int mTotalEpochNum;             // The number of individual trajectory files this summary table managed
        double mTotalLengthTime;        // The total time length for trajs recorded in this file. the unit is second.
        int mTotalLengthFrame;          // The total frame number for trajs recorded in this file. the unit is second.

        struct tSingleEpochInfo{        // This struct records info about a single traj
            int frame_num;              // the frame number it contains
            double length_second;       // the length of this trajs, recorded in seconds
            std::string traj_filename;   // the filepath of this traj files
        };
        std::vector<tSingleEpochInfo> mEpochInfos;
        void WriteToDisk(const std::string & path);
    };
    tSummaryTable mSummaryTable;

    // tools
    void ParseConfig(const std::string & path);
    void ParseConfigSave(const Json::Value & save_value);
    void ParseConfigSolve(const Json::Value & solve_value);
    void ParseConfigSample(const Json::Value & sample_value);
    void ParseConfigDisplay(const Json::Value & display_value);

    void LoadTraj(const std::string & path);
    void LoadMotion(const std::string & path, cMotion * motion) const;
    std::string SaveTraj(const std::string & path);
    void SaveMotion(const std::string & path, cMotion * motion) const;
    void InitSampleSummaryTable();
    void UpdateSummaryTable();

    void VerifyMomentum();
    void PrintLoadInfo(const std::string &, bool disable_root = true);
    void PrintSampleInfo();
};