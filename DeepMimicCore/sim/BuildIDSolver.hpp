#include <iostream>
#include "cIDSolver.hpp"
#include <memory>

class cSimCharacter;
class btMultiBodyDynamicsWorld;
std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf, cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world);
