#include "BuildIDSolver.hpp"
#include "../scenes/SceneImitate.h"
#include "OnlineIDSolver.hpp"
#include "DisplayIDSolver.hpp"
#include "SampleIDSolver.hpp"
#include "OfflineSolveIDSolver.hpp"
#include <util/JsonUtil.h>
#include <memory>

// std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf, cSimCharacter * sim_char, cKinCharacter * kin_char, btMultiBodyDynamicsWorld * world)
std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf, cSceneImitate * imitate_scene)
{
    Json::Value ID_JSON_VALUE;
    if(false == cJsonUtil::LoadJson(conf, ID_JSON_VALUE))
    {
        std::cout <<"[error] parse id config json failed = " << conf << std::endl;
        exit(1);
    }

    // 1. check type
    const std::string mode_str = ID_JSON_VALUE["IDMode"].asString();
    eIDSolverType type = eIDSolverType::INVALID;
    for(int i=0; i<eIDSolverType::SOLVER_TYPE_NUM; i++)
    {
        // std::cout << gIDSolverTypeStr[i] <<" " << mode_str <<" " << (gIDSolverTypeStr[i] == mode_str) << std::endl;
        if(gIDSolverTypeStr[i] == mode_str)
        {
            type = static_cast<eIDSolverType>(i);
            break;
        }
    }
    if(type == eIDSolverType::INVALID) 
    {
        std::cout <<"[error] BuildIDSolver failed: invalid solver type " << mode_str << std::endl;
        exit(1);
    }

    // 2. create solver accordly
    std::shared_ptr<cIDSolver> solver = nullptr;
    switch (type)
    {
    case eIDSolverType::Display: solver = std::shared_ptr<cIDSolver>(new cDisplayIDSolver(imitate_scene, conf)); break;
    case eIDSolverType::OfflineSolve: solver = std::shared_ptr<cIDSolver>(new cOfflineIDSolver(imitate_scene, conf)); break;
    case eIDSolverType::Online: solver = std::shared_ptr<cIDSolver>(new cOnlineIDSolver(imitate_scene)); break;
    case eIDSolverType::Sample: solver = std::shared_ptr<cIDSolver>(new cSampleIDSolver(imitate_scene, conf)); break;
    default:
        {
            std::cout <<"[error] BuildIDSolver mode illegal = " << mode_str << std::endl;
            exit(1);
            break;
        }
    }
    
    return solver;
}
