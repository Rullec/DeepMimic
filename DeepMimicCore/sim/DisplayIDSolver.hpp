#include "InteractiveIDSolver.hpp"

class cDisplayIDSolver : public cInteractiveIDSolver
{
public:
    explicit cDisplayIDSolver(cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world, const std::string & config);
    ~cDisplayIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;
protected:
    void Parseconfig(const std::string & conf);
};