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

    // vars
    struct tSingleFrameIDResult{    // ID result of single frame
        tVectorXd state, action;
        double reward;
    };

    std::string mSolveTrajPath;     // which trajectory do you want to use for solving Inverse Dynamic?
    std::string mExportDataPath;    // The result of Inverse Dynamic is a sequence of "state, action, reward" triplet, named XXX.train. Where do you want to save it?
    std::string mRefMotionPath;     // You must specify the reference motion when OfflineSolve() try to recalculate the reward for each frame.
    std::string mRetargetCharPath;  // The character skeleton file which belongs to this trajectory

    // methods
    void Parseconfig(const std::string & conf);
    void OfflineSolve(std::vector<tSingleFrameIDResult> & IDResult);
};