#include "InteractiveIDSolver.hpp"

class cSampleIDSolver : public cInteractiveIDSolver
{
public:
    explicit cSampleIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, const std::string & config);
    ~cSampleIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;
protected:
    void Parseconfig(const std::string & conf);
    void InitSampleSummaryTable();
    void PrintSampleInfo();
};