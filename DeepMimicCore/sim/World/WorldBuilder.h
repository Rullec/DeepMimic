#include "sim/World/LCPWorld.h"
#include "sim/World/World.h"
#include "sim/World/WorldBase.h"

class cWorldBuilder
{
public:
    static void BuildWorld(std::shared_ptr<cWorldBase> &world,
                           const cWorldBase::tParams &params);
};