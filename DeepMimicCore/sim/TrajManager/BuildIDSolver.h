#include "IDSolver.h"
#include <iostream>
#include <memory>

class cSimCharacter;
class btMultiBodyDynamicsWorld;
class cKinCharacter;
// std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf,
// cSimCharacter * sim_char, cKinCharacter * kin_char, btMultiBodyDynamicsWorld
// * world);
std::shared_ptr<cIDSolver> BuildIDSolver(const std::string &conf,
                                         cSceneImitate *imitate_scene);