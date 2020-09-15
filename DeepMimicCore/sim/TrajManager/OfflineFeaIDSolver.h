#include "OfflineIDSolver.h"
#include "sim/TrajManager/Trajectory.h"

class cSceneImitate;
class cOfflineFeaIDSolver : public cOfflineIDSolver
{
public:
    explicit cOfflineFeaIDSolver(cSceneImitate *imitate,
                                 const std::string &config);
    virtual ~cOfflineFeaIDSolver() = default;
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;

protected:
    void
    SingleTrajSolve(std::vector<tSingleFrameIDResult> &IDResult) override final;
    void BatchTrajsSolve(const std::string &path) override final;
    // void RestoreActionByThetaDist(std::vector<tSingleFrameIDResult>
    // &IDResult); void
    // RestoreActionByGroundTruth(std::vector<tSingleFrameIDResult> &IDResult);
};