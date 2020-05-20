#pragma once
#include "IDSolver.hpp"

/*  Interactive Inverse Dynamic Solver is inherited from the functional IDSolver
    It offers IO operation, such as loading / exporting trajectories or train data,
    and create an uniform data storage for its subclass.
*/

namespace Json{
    class Value;
};
class cMotion;
class cSceneImitate;
class cInteractiveIDSolver : public cIDSolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cInteractiveIDSolver(cSceneImitate * imitate_scene, eIDSolverType type);
    ~cInteractiveIDSolver();
    
protected:

    // this struct are used to storage "trajectory" infos when we are sampling the controller.
    // joint force ground truth, contact info... Nearly everything is included.
    // this struct can be exported to "*.traj" or normal DeepMimic motion data "*.txt"
	struct tSaveInfo{
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
        double mRewards[MAX_FRAME_NUM];    // rewards
        double mRefTime[MAX_FRAME_NUM];     // current time in kinchar reference motion
        cMotion * mMotion;
        std::vector<tContactForceInfo> mContactForces[MAX_FRAME_NUM];
        std::vector<tVector> mExternalForces[MAX_FRAME_NUM], mExternalTorques[MAX_FRAME_NUM];
        tVector mLinearMomentum[MAX_FRAME_NUM], mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
        tVectorXd mCharPoses[MAX_FRAME_NUM];
	};
    tSaveInfo mSaveInfo;        // instantiated


    // load mode: Set up different flag when we load different data. 
    // It takes an effect on the behavior of our ID Solver
    enum eLoadMode{
        INVALID,
        LOAD_MOTION,
        LOAD_TRAJ
    };

    // load info struct. Namely it is used for storaging the loaded info from fril.
    struct tLoadInfo{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        eLoadMode mLoadMode = eLoadMode::INVALID;
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat;
        cMotion * mMotion = nullptr;
        tVectorXd mTimesteps, mRewards, mMotionRefTime;
        std::vector<std::vector<tContactForceInfo>> mContactForces;
        std::vector<std::vector<tMatrix>> mLinkRot;	// local to world rotation mats
        std::vector<std::vector<tVector>> mLinkPos;	// link COM pos in world frame
        std::vector<std::vector<tVector>> mExternalForces, mExternalTorques;
        std::vector<std::vector<tVector>> mTruthJointForces;
        int mTotalFrame = 0;
        int mCurFrame = 0;
        bool mEnableOutputMotionInfo = false;
        std::string mOutputMotionInfoPath = "";
    };
    struct tLoadInfo mLoadInfo;    // work for display and solve mode

    // This struct are used in the derived cSampleIDSolver.
    // when we are workng in "sample" mode, usually tons of trajectories will be sampled and storaged
    // we need to specify the storage dir... for it.
    struct {
        int mSampleEpoches;         // the epoche number of trajectoris we want.
        std::string mSampleTrajsDir;    // the saving dir of trajectoris, for example "data/walk/"
        std::string mSampleTrajsRootName;   // the root of trajectories' filename, for example "traj_walk.json". Then "traj_walk_xxx.json" will be genearated and saved
        std::string mSummaryTableFilename;  // These recorded trajs' info will be recorded in this file. It can be "summary_table.json" 
    } mSampleInfo;  // work for sample mode

    // Summary table, It will be used in cSampleIDSolver and cOfflineIDSolver
    struct tSummaryTable{
        tSummaryTable();
        std::string mSampleCharFile, mSampleControllerFile; // the character filepath and the controller filepath. They ares recorded in the summary table for clearity
        std::string mTimeStamp;         // the year-month-date timestamp recorded in the summary table
        int mTotalEpochNum;             // The number of individual trajectory files this summary table managed
        double mTotalLengthTime;        // The total time length for trajs recorded in this file. the unit is second.
        int mTotalLengthFrame;          // The total frame number for trajs recorded in this file. the unit is second.

        struct tSingleEpochInfo{        // This struct records info about a single traj
            int frame_num;              // the frame number it contains
            double length_second;       // the length of this trajs, recorded in seconds
            std::string traj_filename;   // the filepath of this traj files
            std::string train_data_filename;    // the filepath of train data that generated by cOfflineSolver.
            // Attention: this key "train_data_filename" should always be set to null in cSampleSolver.
            // Only when we are generating the train data in cOfflineSolver can this value being set.
            tSingleEpochInfo();
        };
        std::vector<tSingleEpochInfo> mEpochInfos;
        void WriteToDisk(const std::string & path);
        void LoadFromDisk(const std::string & path);
    };
    tSummaryTable mSummaryTable;

    // used for storaged Inverse Dynamic result. instantiated in OfflineSolveIDSolver.hpp
    struct tSingleFrameIDResult{    // ID result of single frame
        tVectorXd state, action;    // state & action, used in DeepMimic Neural Network training
        double reward;              // reward, calculated by cSceneImitate::CalcRewardImitate()
    };

    // IO tools
    // load methods
    void LoadTraj(tLoadInfo & load_info, const std::string & path);     // load our custon trajectroy "*.traj"
    std::string SaveTraj(tSaveInfo & mSaveInfo, const std::string & path) const;    // save it
    void PrintLoadInfo(tLoadInfo & load_info, const std::string &, bool disable_root = true) const;
    void LoadMotion(const std::string & path, cMotion * motion) const;  // load deepmimic motion
    void SaveMotion(const std::string & path, cMotion * motion) const;  // save action
    void SaveTrainData(const std::string & path, std::vector<tSingleFrameIDResult> &) const;      // save train data "*.train", only "state, action, reward " trio pair will be storaged in it.
};