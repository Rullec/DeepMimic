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
    void Parseconfig(const std::string & conf);
    void OfflineSolve();
};