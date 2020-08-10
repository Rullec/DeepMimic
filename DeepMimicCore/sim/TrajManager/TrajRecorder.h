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
    std::vector<tVector> mTruthJointForces[MAX_FRAME_NUM];
    std::vector<tVector> mSolvedJointForces[MAX_FRAME_NUM];
    tVectorXd mBuffer_q[MAX_FRAME_NUM], mBuffer_u[MAX_FRAME_NUM],
        mBuffer_u_dot[MAX_FRAME_NUM];
    std::vector<tMatrix>
        mLinkRot[MAX_FRAME_NUM]; // local to world rotation mats
    std::vector<tVector> mLinkPos[MAX_FRAME_NUM]; // link COM pos in world frame
    std::vector<tVector> mLinkVel[MAX_FRAME_NUM]; // link COM vel in world frame
    std::vector<tVector>
        mLinkOmega[MAX_FRAME_NUM]; // link angular momentum in world frame
    std::vector<tVector>
        mLinkDiscretVel[MAX_FRAME_NUM]; // link COM vel in world frame
                                        // calculated from differential of
                                        // link positions
    std::vector<tVector>
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
    std::vector<tContactForceInfo> mContactForces[MAX_FRAME_NUM];
    std::vector<tVector> mExternalForces[MAX_FRAME_NUM],
        mExternalTorques[MAX_FRAME_NUM];
    tVector mLinearMomentum[MAX_FRAME_NUM],
        mAngularMomentum[MAX_FRAME_NUM]; // linear, ang momentum for each frame
    tVectorXd mCharPoses[MAX_FRAME_NUM];
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
    virtual void ReadContactInfo();
    virtual void ReadContactInfoGen();
    virtual void ReadContactInfoRaw();
};