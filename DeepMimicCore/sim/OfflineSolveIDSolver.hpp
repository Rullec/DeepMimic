#include "InteractiveIDSolver.hpp"

class cOfflineSolverIDSolver : public cInteractiveIDSolver
{
public:
    explicit cOfflineSolverIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, const std::string & config);
    ~cOfflineSolverIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;
protected:
    void Parseconfig(const std::string & conf);
    void OfflineSolve();
};