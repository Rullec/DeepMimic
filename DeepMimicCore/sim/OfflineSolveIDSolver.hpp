#include "InteractiveIDSolver.hpp"

class cSceneImitate;
class cOfflineSolveIDSolver : public cInteractiveIDSolver
{
public:
    explicit cOfflineSolveIDSolver(cSceneImitate * imitate, const std::string & config);
    ~cOfflineSolveIDSolver();
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
    } mBatchTrajSolveConfig;
    
    bool        mVerfiedAction;     // .traj files sometimes include the resulting actions (for debugging), do you want to verify the ID result with this ground truth? 
    bool        mRecalculateReward; // .traj files usually include the old reward value, do you want to calculate it again? performance cost.
    std::string mRefMotionPath;     // You must specify the reference motion when OfflineSolve() try to recalculate the reward for each frame.
    std::string mRetargetCharPath;  // The character skeleton file which belongs to this trajectory

    // methods
    void ParseConfig(const std::string & conf);
    void ParseSingleTrajConfig(const Json::Value & single_traj_config);
    void ParseBatchTrajConfig(const Json::Value & batch_traj_config);

    void SingleTrajSolve(std::vector<tSingleFrameIDResult> & IDResult);
    void BatchTrajsSolve(const std::string & path);
};