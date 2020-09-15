#pragma once
#include "IDSolver.h"
#include "Trajectory.h"
class cOfflineIDSolver : public cIDSolver
{
public:
    explicit cOfflineIDSolver(cSceneImitate *imitate,
                              const std::string &config);
    virtual ~cOfflineIDSolver() = default;
    // virtual void PreSim() = 0;
    // virtual void PostSim() = 0;
    // virtual void Reset() = 0;
    // virtual void SetTimestep(double) = 0;

protected:
    enum eSolveMode
    {
        INVALID_SOLVEMODE,
        SingleTrajSolveMode,
        BatchTrajSolveMode,
        SolveModeNum
    };

    const std::string SolveModeStr[eSolveMode::SolveModeNum] = {
        "INVALID_SOLVEMODE", "SingleTraj", "BatchTraj"};

    enum eSolveTarget
    {
        INVALID_SOLVETARGET,
        SampledTraj,
        MRedTraj, // motion retargeted trajectory
        SolveTargetNum
    };

    const std::string SolveTargetstr[eSolveTarget::SolveTargetNum] = {
        "INVALID_SOLVETARGET", "SampledTraj", "MRedTraj"};

    eSolveMode mSolveMode; // solve mode: single_solve or batch_solve?
                           // config for SingleTrajSolveMode
    struct
    {
        std::string mSolveTrajPath; // which trajectory do you want to use for
                                    // solving Inverse Dynamic?
        std::string
            mExportDataPath; // The result of Inverse Dynamic is a sequence of
                             // "state, action, reward" triplet, named
                             // XXX.train. Where do you want to save it?
    } mSingleTrajSolveConfig;

    // config for BatchTrajSolveMode
    struct
    {
        std::string
            mOriSummaryTableFile; // You need to specify a summary table which
                                  // records the details info of batch trajs. It
                                  // is the output of cSampleIDSolver
        std::string
            mDestSummaryTableFile;  // You need to specify a summary table which
                                    // records the details info of batch trajs.
                                    // It is the output of cSampleIDSolver
        std::string mExportDataDir; // The same as mExportDataPath
        eSolveTarget mSolveTarget;
    } mBatchTrajSolveConfig;

    // used for storaged Inverse Dynamic result. instantiated in
    // OfflineSolveIDSolver.h
    struct tSingleFrameIDResult
    { // ID result of single frame
        tVectorXd state,
            action; // state & action, used in DeepMimic Neural Network training
        double
            reward; // reward, calculated by cSceneImitate::CalcRewardImitate()
    };

    tLoadInfo mLoadInfo;
    tSummaryTable mSummaryTable;
    bool mEnableActionVerfied; // .traj files sometimes include the resulting
                               // actions (for debugging), do you want to verify
                               // the ID result with this ground truth?
    bool mEnableTorqueVerified;   // verified torque
    bool mEnableDiscreteVerified; // verified discrete vel and accel
    bool mEnableRewardRecalc;     // .traj files usually include the old reward
                                  // value, do you want to calculate it again?
                                  // performance cost.
    std::string mRefMotionPath;   // You must specify the reference motion when
                                // OfflineSolve() try to recalculate the reward
                                // for each frame.
    std::string mRetargetCharPath; // The character skeleton file which belongs
                                   // to this trajectory
    // bool mEnableRestoreThetaByActionDist; // if open, ID result will be
    // revised
    //                                       // by an external theta
    //                                       distribution
    //                                       // file. It's a way to remove the
    //                                       // ambiguity of axis angle repre.
    // bool mEnableRestoreThetaByGT;         // restore theta by ground truth

    // Here are 2 types of trajectories in summary table that We can solve. The
    // first is the raw trajectory coming from sampling directly The second are
    // trajectoris generated by RobotControl after being retargeted to a
    // (possible) new skeleton. Which should we select to solve? It is
    // controlled by enum mSolveTarget

    // methods
    void ParseConfig(const std::string &conf);
    void ParseSingleTrajConfig(const Json::Value &single_traj_config);
    void ParseBatchTrajConfig(const Json::Value &batch_traj_config);

    eSolveTarget ParseSolveTargetInBatchMode(const std::string &name) const;
    eSolveMode ParseSolvemode(const std::string &name) const;

    void SaveTrainData(const std::string &dir, const std::string &filename,
                       const std::vector<tSingleFrameIDResult> &)
        const; // save train data "*.train", only "state, action, reward " trio
               // pair will be storaged in it.

    virtual void
    SingleTrajSolve(std::vector<tSingleFrameIDResult> &IDResult) = 0;
    virtual void BatchTrajsSolve(const std::string &path) = 0;

    void LoadBatchInfoMPI(const std::string &table_path,
                          std::vector<int> &solved_traj_ids,
                          std::vector<std::string> &solved_traj_names);
    void AddBatchInfoMPI(
        int global_traj_id, const std::string target_traj_filename_full,
        const std::vector<tSingleFrameIDResult> &mResult,
        const std::vector<tSummaryTable::tSingleEpochInfo> &old_epoch_info,
        double total_time);
};