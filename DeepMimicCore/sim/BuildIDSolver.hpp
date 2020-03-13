#include <iostream>
#include "cIDSolver.hpp"

class cSimCharacter;
class btMultiBodyDynamicsWorld;
std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf, cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world);