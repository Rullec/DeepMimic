#pragma once
#include "util/MathUtil.h"

#define MAX_FRAME_NUM 10000
namespace Json
{
class Value;
};

enum eTrajFileVersion
{
    UNSET,
    V1,
    V2,
    V3,
    NUM_TRAJ_VERSION
};
eTrajFileVersion CalcTrajVersion(int version);
enum eLoadMode
{
    INVALID,
    LOAD_MOTION,
    LOAD_TRAJ
};

struct tContactForceInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int mId; // applied link id in deepmimic order, we assume the base link in
             // bullet featherstone is invisible in the collision response
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

class cMotion;
void LoadMotion(const std::string &path,
                cMotion *motion); // load deepmimic motion
struct tSaveInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::string mIntegrationScheme = ""; // the integration scheme of simulation
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

    std::string SaveTraj(const std::string &traj_dir,
                         const std::string &traj_rootname,
                         eTrajFileVersion mTrajFileVersion);
    void SaveTrajV1(Json::Value &root);
    void SaveTrajV2(Json::Value &root);
    void SaveTrajV3(Json::Value &root);

protected:
    void SaveCommonInfo(Json::Value &value, int frame_id) const;
    void SavePosVelAccel(Json::Value &value, int frame_id) const;
    void SaveContactInfo(Json::Value &value, int frame_id) const;
    void SaveExtForce(Json::Value &value, int frame_id) const;
    void SaveTruthJointForce(Json::Value &value, int frame_id) const;
    void SaveTruthAction(Json::Value &value, int frame_id) const;
    void SaveTruthPDTarget(Json::Value &value, int frame_id) const;
    void SaveCharPose(Json::Value &value, int frame_id) const;
};

class cSimCharacterBase;

struct tLoadInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    tLoadInfo();
    std::string mLoadPath;
    eLoadMode mLoadMode;
    eTrajFileVersion mVersion;
    /*
        mPosMat:   for featherstone backend the angles of each joint in bullet
                    for generalized backend, the gen coordinate
        mVelMat:    for featherstone backend, the rotate vel of join
                    for gen backend, the generalized velocity
    */

    //
    Eigen::MatrixXd mPosMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat,
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
    std::string mIntegrationScheme;
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
    void LoadTraj(cSimCharacterBase *sim_char, const std::string &path);
    void LoadTrajV1(cSimCharacterBase *sim_char,
                    const Json::Value &value); // load our custon trajectroy
                                               // "*.traj" v1 for full format
    void
    LoadTrajV2(cSimCharacterBase *sim_char,
               const Json::Value &value); // load our custon trajectroy "*.traj"

    void PrintLoadInfo(cSimCharacterBase *sim_char, const std::string &,
                       bool disable_root = true) const;

protected:
    // eTrajFileVersion CalcTrajVersion(const std::string &confg) const;
    void LoadCommonInfo(const Json::Value &value, double &timestep,
                        double &motion_ref_time, double &reward) const;
    void LoadContactInfo(const Json::Value &cur_contact_info,
                         tEigenArr<tContactForceInfo> &contact_array) const;
    void LoadPosVelAndAccel(const Json::Value &cur_frame, int frame_id,
                            tMatrixXd &pos_mat, tMatrixXd &vel_mat,
                            tMatrixXd &accel_mat) const;
    void LoadCharPose(const Json::Value &cur_frame, tVectorXd &pose) const;
    void LoadTruthAction(const Json::Value &cur_frame, tVectorXd &action) const;
    void LoadJointTorques(const Json::Value &cur_frame,
                          tEigenArr<tVector> &joint_torques);
    void LoadExternalForceTorque(const Json::Value &cur_frame,
                                 tEigenArr<tVector> &forces,
                                 tEigenArr<tVector> &torques) const;
    void RecordLinkPosRot(tEigenArr<tVector> link_pos,
                          tEigenArr<tMatrix> &link_rot) const;
};

// Summary table, It will be used in cSampleIDSolver and cOfflineIDSolver
struct tSummaryTable
{
    tSummaryTable();
    std::string mSampleCharFile,
        mSampleControllerFile; // the character filepath and the controller
                               // filepath. They ares recorded in the
                               // summary table for clearity
    std::string mTimeStamp;    // the year-month-date timestamp recorded in the
                               // summary table
    // std::string mActionThetaDistFile; // storage of target files with symbol
    //                                   // of angular values in the axis angle
    std::string
        mSampleTrajDir; // the root dir for storaging sample trajectories
    std::string mIDTraindataDir; // the root dir for storaging the results
                                 // of Inverse Dynamic
    std::string mMrTrajDir;      // the root dir for storaging the results after
                                 // motion retargeing
    int mTotalEpochNum; // The number of individual trajectory files this
                        // summary table managed

    struct tSingleEpochInfo
    {                         // This struct records info about a single traj
        int frame_num;        // the frame number it contains
        double length_second; // the length of this trajs, recorded in seconds
        std::string sample_traj_filename; // the filepath of this traj files
        std::string mr_traj_filename;     // the filepath of traj filester after
                                          // being motion retargeted
        std::string train_filename;       // the filepath of train data that
                                          // generated by cOfflineSolver.
        // Attention: this key "train_filename" should always be set to null
        // in cSampleSolver. Only when we are generating the train data in
        // cOfflineSolver can this value being set.
        tSingleEpochInfo();
    };
    std::vector<tSingleEpochInfo> mEpochInfos;
    void WriteToDisk(const std::string &path, bool append = true);
    void LoadFromDisk(const std::string &path);
};