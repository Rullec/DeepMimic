#include "InteractiveIDSolver.hpp"

class cSceneImitate;
class cOfflineIDSolver : public cInteractiveIDSolver
{
public:
    explicit cOfflineIDSolver(cSceneImitate * imitate, const std::string & config);
    ~cOfflineIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;
protected:
    enum eOfflineSolveMode 
    {
        INVALID,
        SingleTrajSolveMode,
        BatchTrajSolveMode,
        OfflineSolveModeNum
    };
    
    const std::string mConfPath[eOfflineSolveMode::OfflineSolveModeNum] = {
        "INVALID",
        "SingleTraj",
        "BatchTraj"
    };

    eOfflineSolveMode mOfflineSolveMode;

    // config for SingleTrajSolveMode
    struct {
        std::string mSolveTrajPath;     // which trajectory do you want to use for solving Inverse Dynamic?
        std::string mExportDataPath;    // The result of Inverse Dynamic is a sequence of "state, action, reward" triplet, named XXX.train. Where do you want to save it?
    } mSingleTrajSolveConfig;

    // config for BatchTrajSolveMode
    struct {
        std::string mSummaryTableFile;  // You need to specify a summary table which records the details info of batch trajs. It is the output of cSampleIDSolver
        std::string mExportDataDir;     // The same as mExportDataPath
        bool mEnableRestoreThetaByActionDist;   // if open, ID result will be revised by an external theta distribution file. It's a way to remove the ambiguity of axis angle repre.
        bool mEnableRestoreThetaByGT;   // restore theta by ground truth
    } mBatchTrajSolveConfig;
    
    bool        mEnableActionVerfied;     // .traj files sometimes include the resulting actions (for debugging), do you want to verify the ID result with this ground truth? 
    bool        mEnableRewardRecalc; // .traj files usually include the old reward value, do you want to calculate it again? performance cost.
    std::string mRefMotionPath;     // You must specify the reference motion when OfflineSolve() try to recalculate the reward for each frame.
    std::string mRetargetCharPath;  // The character skeleton file which belongs to this trajectory

    // methods
    void ParseConfig(const std::string & conf);
    void ParseSingleTrajConfig(const Json::Value & single_traj_config);
    void ParseBatchTrajConfig(const Json::Value & batch_traj_config);

    void SingleTrajSolve(std::vector<tSingleFrameIDResult> & IDResult);
    void BatchTrajsSolve(const std::string & path);
    void RestoreActionByThetaDist(std::vector<tSingleFrameIDResult> & IDResult);
    void RestoreActionByGroundTruth(std::vector<tSingleFrameIDResult> & IDResult);
};