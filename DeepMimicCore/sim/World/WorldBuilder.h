#include "sim/World/GenWorld.h"
#include "sim/World/FeaWorld.h"
#include "sim/World/WorldBase.h"

class cWorldBuilder
{
public:
    static void BuildWorld(std::shared_ptr<cWorldBase> &world,
                           const cWorldBase::tParams &params);
};