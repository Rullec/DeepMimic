#include "BuildIDSolver.hpp"
#include "cOfflineIDSolver.hpp"
#include "cOnlineIDSolver.hpp"
#include <util/JsonUtil.h>
#include <memory>

std::shared_ptr<cIDSolver> BuildIDSolver(const std::string & conf, cSimCharacter * sim_char, btMultiBodyDynamicsWorld * world)
{
    Json::Value ID_JSON_VALUE;
    if(false == cJsonUtil::ParseJson(conf, ID_JSON_VALUE))
    {
        std::cout <<"[error] parse id config json failed = " << conf << std::endl;
        exit(1);
    }

    const std::string mode_str = ID_JSON_VALUE["IDMode"].asString();
    std::shared_ptr<cIDSolver> solver = nullptr;
    if("Online" == mode_str)
    {
        solver = std::shared_ptr<cIDSolver>(new cOnlineIDSolver(sim_char, world));
    }
    else if("Offline" == mode_str)
    {
        solver = std::shared_ptr<cIDSolver>(new cOfflineIDSolver(sim_char, world, conf));
    }
    else
    {
        std::cout <<"[error] BuildIDSolver mode illegal = " << mode_str << std::endl;
        exit(1);
    }
    // std::cout <<"solver type = " << solver->GetType() << std::endl;
    // exit(1);
    return solver;
}
