#include "InteractiveIDSolver.hpp"

class cSceneImitate;
class cSampleIDSolver : public cInteractiveIDSolver
{
public:
    explicit cSampleIDSolver(cSceneImitate * imitate_scene, const std::string & config);
    ~cSampleIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;
protected:
    void Parseconfig(const std::string & conf);
    void InitSampleSummaryTable();
    void PrintSampleInfo();

    // MPI UTILS
    bool mEnableIDTest;
};