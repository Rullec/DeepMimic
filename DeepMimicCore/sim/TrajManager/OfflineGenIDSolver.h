#pragma once
#include "OfflineIDSolver.h"
#include "sim/TrajManager/Trajectory.h"

class cSceneImitate;
class tSingleFrameIDResult;
class btGenContactAwareController;
class cOfflineGenIDSolver : public cOfflineIDSolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cOfflineGenIDSolver(cSceneImitate *imitate,
                                 const std::string &config);
    virtual ~cOfflineGenIDSolver() = default;
    virtual void Reset() override final;
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void SetTimestep(double) override final;

protected:
    void Init();
    virtual void
    SingleTrajSolve(std::vector<tSingleFrameIDResult> &IDResult) override final;
    virtual void BatchTrajsSolve(const std::string &path) override final;

    btGenContactAwareController *mAdviser;
    double mCurTimestep;
    bool mInited;
    std::string mCurrentTrajPath;
    std::string mCurrentOutputPath;
    std::vector<tSingleFrameIDResult> mIDResult;
    tVectorXd mPosePre, mVelPre;
    tLoadInfo *mRefTraj;
    // batch solve essentials
    std::vector<tSummaryTable::tSingleEpochInfo> mOldEpochInfos;
    std::vector<int> mBatchTrajIdArray;
    std::vector<std::string> mBatchNameArray;
    int mBatchCurLocalTrajId;
};