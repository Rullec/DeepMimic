#pragma once
#include "util/MathUtil.h"
#define MAX_FRAME_NUM 10000

class cSimCharacterBase;
class cMotion;
class cSceneImitate;

enum eTrajFileVersion
{
    UNSET,
    V1,
    V2,
    V3
};

struct tContactForceInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int mId; // applied link id in Inverse Dynamics order but not deepmimic
             // order
    tVector mPos, mForce;
    bool mIsSelfCollision; // Does this contact belong to the self collision in
                           // this character?
    tContactForceInfo()
    {
        mIsSelfCollision = false;
        mId = -1;
        mPos = mForce = tVector::Zero();
    }
};

struct tSaveInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::string mSaveTrajRoot = "";
    std::string mSaveMotionRoot = "";
    int mCurEpoch = 0;
    int mCurFrameId = 0;
    tEigenArr<tVector> mTruthJointForces[MAX_FRAME_NUM];
    tEigenArr<tVector> mSolvedJointForces[MAX_FRAME_NUM];
    tVectorXd mBuffer_q[MAX_FRAME_NUM], mBuffer_u[MAX_FRAME_NUM],
        mBuffer_u_dot[MAX_FRAME_NUM];
    tEigenArr<tMatrix> mLinkRot[MAX_FRAME_NUM]; // local to world rotation mats
    tEigenArr<tVector> mLinkPos[MAX_FRAME_NUM]; // link COM pos in world frame
    tEigenArr<tVector> mLinkVel[MAX_FRAME_NUM]; // link COM vel in world frame
    tEigenArr<tVector>
        mLinkOmega[MAX_FRAME_NUM]; // link angular momentum in world frame
    tEigenArr<tVector>
        mLinkDiscretVel[MAX_FRAME_NUM]; // link COM vel in world frame
                                        // calculated from differential of
                                        // link positions
    tEigenArr<tVector>
        mLinkDiscretOmega[MAX_FRAME_NUM];    // link angular momentum in world
                                             // frame  from differential of
                                             // link rotation
    tVectorXd mTruthAction[MAX_FRAME_NUM];   // the current action recorded from
                                             // the controller of this char
    tVectorXd mTruthPDTarget[MAX_FRAME_NUM]; // the current action recorded from
                                             // the controller of this char

    double mTimesteps[MAX_FRAME_NUM]; // timesteps
    double mRewards[MAX_FRAME_NUM];   // rewards
    double mRefTime[MAX_FRAME_NUM]; // current time in kinchar reference motion
    cMotion *mMotion;
    tEigenArr<tContactForceInfo> mContactForces[MAX_FRAME_NUM];
    tEigenArr<tVector> mExternalForces[MAX_FRAME_NUM],
        mExternalTorques[MAX_FRAME_NUM];
    tVector mLinearMomentum[MAX_FRAME_NUM],
        mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
    tVectorXd mCharPoses[MAX_FRAME_NUM];
};

// load mode: Set up different flag when we load different data.
// It takes an effect on the behavior of our ID Solver
enum eLoadMode
{
    INVALID,
    LOAD_MOTION,
    LOAD_TRAJ
};

// load info struct. Namely it is used for storaging the loaded info from
// fril.
class cCtPDController;
struct tLoadInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tLoadInfo();
    std::string mLoadPath;
    eLoadMode mLoadMode;
    Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat,
        mCharPoseMat;
    cMotion *mMotion;
    tVectorXd mTimesteps, mRewards, mMotionRefTime;
    std::vector<tEigenArr<tContactForceInfo>> mContactForces;
    tEigenArr<tEigenArr<tMatrix>> mLinkRot; // local to world rotation mats
    tEigenArr<tEigenArr<tVector>> mLinkPos; // link COM pos in world frame
    tEigenArr<tEigenArr<tVector>> mExternalForces,
        mExternalTorques; // external forces applied on each link
    tEigenArr<tEigenArr<tVector>>
        mTruthJointForces; // The ground truth joint torques loaded from
                           // some .traj files will be storaged here. mostly
                           // for debug purpose
    int mTotalFrame;
    int mCurFrame;
    bool mEnableOutputMotionInfo; // if this option is set to true, when the
                                  // tLoadInfo is loaded from the disk, the
                                  // Loadfunc will export a summary report about
                                  // the loaded info
    std::string mOutputMotionInfoPath; // used accompany with the last one, it
                                       // points out the location of report file
                                       // that will be overwritted.

    // methods
    void LoadTrajV1(cSimCharacterBase *sim_char,
                    const std::string &path); // load our custon trajectroy
                                              // "*.traj" v1 for full format
    void
    LoadTrajV2(cSimCharacterBase *sim_char,
               const std::string &path); // load our custon trajectroy "*.traj"
};

/**
 * \brief                   Trajectory recorder
 */
class cTrajRecorder
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cTrajRecorder(cSceneImitate *scene, const std::string &conf);
    virtual ~cTrajRecorder();
    virtual void PreSim();
    virtual void PostSim();
    virtual void Reset();
    virtual void SetTimestep(double);

protected:
    cSimCharacterBase *mSimChar;
    cSceneImitate *mScene;
    tSaveInfo mSaveInfo;
    std::string mTrajSavePath;
    int mRecordMaxFrame;
    virtual void ParseConfig(const std::string &conf);
    virtual void RecordActiveForceGen();
    virtual void ReadContactInfo();
    virtual void ReadContactInfoGen();
    virtual void ReadContactInfoRaw();
    virtual void VerifyDynamicsEquation();
};