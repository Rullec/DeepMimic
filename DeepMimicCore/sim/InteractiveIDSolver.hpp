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
        std::vector<tForceInfo> mContactForces[MAX_FRAME_NUM];
        std::vector<tVector> mExternalForces[MAX_FRAME_NUM], mExternalTorques[MAX_FRAME_NUM];
        tVector mLinearMomentum[MAX_FRAME_NUM], mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
        tVectorXd mCharPoses[MAX_FRAME_NUM];
	};
    tSaveInfo mSaveInfo;    // work for save mode


    enum eLoadMode{
        INVALID,
        LOAD_MOTION,
        LOAD_TRAJ
    };

    struct tLoadInfo{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        eLoadMode mLoadMode = eLoadMode::INVALID;
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat;
        cMotion * mMotion = nullptr;
        tVectorXd mTimesteps, mRewards, mMotionRefTime;
        std::vector<std::vector<tForceInfo>> mContactForces;
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

    // IO tools
    // load methods
    void LoadTraj(tLoadInfo & load_info, const std::string & path);
    std::string SaveTraj(tSaveInfo & mSaveInfo, const std::string & path) const;
    void PrintLoadInfo(tLoadInfo & load_info, const std::string &, bool disable_root = true) const;
    void LoadMotion(const std::string & path, cMotion * motion) const;
    void SaveMotion(const std::string & path, cMotion * motion) const;
};