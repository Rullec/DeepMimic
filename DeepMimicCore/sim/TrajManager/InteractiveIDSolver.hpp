#pragma once
#include "IDSolver.hpp"
/*  Interactive Inverse Dynamic Solver is inherited from the functional IDSolver
    It offers IO operation, such as loading / exporting trajectories or train
   data, and create an uniform data storage for its subclass.
*/
namespace Json
{
class Value;
};
class cMotion;
class cSceneImitate;
class cInteractiveIDSolver : public cIDSolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cInteractiveIDSolver(cSceneImitate *imitate_scene,
                                  eIDSolverType type, const std::string &conf);

    virtual ~cInteractiveIDSolver();
    static void
    SaveTrajV1(tSaveInfo &mSaveInfo,
               Json::Value &root); // save trajectories for full version
    static void
    SaveTrajV2(tSaveInfo &mSaveInfo,
               Json::Value &root); // save trajectories for simplified version
protected:
    eTrajFileVersion mTrajFileVersion;
    // this struct are used to storage "trajectory" infos when we are sampling
    // the controller. joint force ground truth, contact info... Nearly
    // everything is included. this struct can be exported to "*.traj" or normal
    // DeepMimic motion data "*.txt"

    tSaveInfo mSaveInfo; // instantiated

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
    struct tLoadInfo
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::string mLoadPath = "";
        eLoadMode mLoadMode = eLoadMode::INVALID;
        Eigen::MatrixXd mPoseMat, mVelMat, mAccelMat, mActionMat, mPDTargetMat,
            mCharPoseMat;
        cMotion *mMotion = nullptr;
        tVectorXd mTimesteps, mRewards, mMotionRefTime;
        std::vector<std::vector<tContactForceInfo>> mContactForces;
        std::vector<std::vector<tMatrix>>
            mLinkRot; // local to world rotation mats
        std::vector<std::vector<tVector>>
            mLinkPos; // link COM pos in world frame
        std::vector<std::vector<tVector>> mExternalForces,
            mExternalTorques; // external forces applied on each link
        std::vector<std::vector<tVector>>
            mTruthJointForces; // The ground truth joint torques loaded from
                               // some .traj files will be storaged here. mostly
                               // for debug purpose
        int mTotalFrame = 0;
        int mCurFrame = 0;
        bool mEnableOutputMotionInfo =
            false; // if this option is set to true, when the tLoadInfo is
                   // loaded from the disk, the Loadfunc will export a summary
                   // report about the loaded info
        std::string mOutputMotionInfoPath =
            ""; // used accompany with the last one, it points out the location
                // of report file that will be overwritted.
    };
    struct tLoadInfo mLoadInfo; // work for display and solve mode

    // This struct are used in the derived cSampleIDSolver.
    // when we are workng in "sample" mode, usually tons of trajectories will be
    // sampled and storaged we need to specify the storage dir... for it.
    struct
    {
        int mSampleEpoches; // the epoche number of trajectoris we want.
        std::string mSampleTrajsDir;      // the saving dir of trajectoris, for
                                          // example "data/walk/"
        std::string mSampleTrajsRootName; // the root of trajectories' filename,
                                          // for example "traj_walk.json". Then
                                          // "traj_walk_xxx.json" will be
                                          // genearated and saved
        std::string mSummaryTableFilename; // These recorded trajs' info will be
                                           // recorded in this file. It can be
                                           // "summary_table.json"
        std::string
            mActionThetaDistFilename; // This file will record action theta
                                      // distribution for spherical joints
    } mSampleInfo;                    // work for sample mode

    // Summary table, It will be used in cSampleIDSolver and cOfflineIDSolver
    struct tSummaryTable
    {
        tSummaryTable();
        std::string mSampleCharFile,
            mSampleControllerFile; // the character filepath and the controller
                                   // filepath. They ares recorded in the
                                   // summary table for clearity
        std::string mTimeStamp; // the year-month-date timestamp recorded in the
                                // summary table
        std::string mActionThetaDistFile; // storage of target files with symbol
                                          // of angular values in the axis angle
        std::string
            mSampleTrajDir; // the root dir for storaging sample trajectories
        std::string mIDTraindataDir; // the root dir for storaging the results
                                     // of Inverse Dynamic
        std::string mMrTrajDir; // the root dir for storaging the results after
                                // motion retargeing
        int mTotalEpochNum; // The number of individual trajectory files this
                            // summary table managed

        struct tSingleEpochInfo
        {                  // This struct records info about a single traj
            int frame_num; // the frame number it contains
            double
                length_second; // the length of this trajs, recorded in seconds
            std::string sample_traj_filename; // the filepath of this traj files
            std::string mr_traj_filename; // the filepath of traj filester after
                                          // being motion retargeted
            std::string train_filename;   // the filepath of train data that
                                          // generated by cOfflineSolver.
            // Attention: this key "train_filename" should always be set to null
            // in cSampleSolver. Only when we are generating the train data in
            // cOfflineSolver can this value being set.
            tSingleEpochInfo();
        };
        std::vector<tSingleEpochInfo> mEpochInfos;
        void WriteToDisk(const std::string &path, bool append = true);
        void LoadFromDisk(const std::string &path);

    private:
        tLogger mLogger;
    };
    tSummaryTable mSummaryTable;

    // test functionality:
    const int mActionThetaGranularity = 200;
    tMatrixXd mActionThetaDist; // (optional) this matrix is used to storage the
                                // symbol of each spherical joints' actions'
                                // theta value. each row i represents a
                                // spherical joint each col j has 100 blanks,
                                // represent the symbol for joint i  when the
                                // character is running in j% phase of motion

    // used for storaged Inverse Dynamic result. instantiated in
    // OfflineSolveIDSolver.hpp
    struct tSingleFrameIDResult
    { // ID result of single frame
        tVectorXd state,
            action; // state & action, used in DeepMimic Neural Network training
        double
            reward; // reward, calculated by cSceneImitate::CalcRewardImitate()
    };

    // IO tools
    // load and save methods
    // Note that which will be called among V1 and V2 is fully determined by the
    // variable mTrajFileVersion DO NOT mannually call V1 and V2 functions
    void LoadTraj(tLoadInfo &load_info,
                  const std::string
                      &path); // load our custom trajectroy "*.traj" generally
    void LoadTrajV1(tLoadInfo &load_info,
                    const std::string &path); // load our custon trajectroy
                                              // "*.traj" v1 for full format
    void
    LoadTrajV2(tLoadInfo &load_info,
               const std::string &path); // load our custon trajectroy "*.traj"
                                         // v2 for simplified format
    std::string
    SaveTraj(tSaveInfo &mSaveInfo, const std::string &traj_dir,
             const std::string &traj_rootname) const; // save trajectories
    void PrintLoadInfo(tLoadInfo &load_info, const std::string &,
                       bool disable_root = true) const;
    void LoadMotion(const std::string &path,
                    cMotion *motion) const; // load deepmimic motion
    void SaveMotion(const std::string &path,
                    cMotion *motion) const; // save action
    void SaveTrainData(const std::string &dir, const std::string &filename,
                       std::vector<tSingleFrameIDResult> &)
        const; // save train data "*.train", only "state, action, reward " trio
               // pair will be storaged in it.

    void InitActionThetaDist(cSimCharacter *sim_char, tMatrixXd &mat) const;
    void LoadActionThetaDist(const std::string &path, tMatrixXd &mat) const;
    void SaveActionThetaDist(const std::string &path, tMatrixXd &mat) const;
    void ParseConfig(const std::string &path);
};