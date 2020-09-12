#include "IDSolver.h"
#include "Trajectory.h"

class cSceneImitate;
class cDisplayIDSolver : public cIDSolver
{
public:
    explicit cDisplayIDSolver(cSceneImitate *sim_char,
                              const std::string &config);
    virtual ~cDisplayIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;

protected:
    void Parseconfig(const std::string &conf);
    tLoadInfo mLoadInfo;
    int mDisplayEpoch;
};