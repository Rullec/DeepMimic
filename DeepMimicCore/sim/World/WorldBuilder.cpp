#include "sim/World/WorldBuilder.h"
#include "util/LogUtil.h"

void cWorldBuilder::BuildWorld(std::shared_ptr<cWorldBase> &world,
                               const cWorldBase::tParams &params)
{
    MIMIC_DEBUG("begin to build world, type {}", params.mWorldType);
    eWorldType type;
    for (int i = 0; i < eWorldType::NUM_WORLD_TYPE; i++)
    {
        if (params.mWorldType == gWorldType[i])
        {
            type = static_cast<eWorldType>(i);
        }
    }

    switch (type)
    {
    case eWorldType::FEATHERSTONE_WORLD:
        world = std::shared_ptr<cWorldBase>(
            dynamic_cast<cWorldBase *>(new cFeaWorld()));
        break;
    case eWorldType::GENERALIZED_WORLD:
        world = std::shared_ptr<cWorldBase>(
            dynamic_cast<cWorldBase *>(new cGenWorld()));
        break;
    default:
        MIMIC_ERROR("Unsupported world type: {} ", gWorldType[type]);
        break;
    }
    MIMIC_DEBUG("build world succ, type = {}", params.mWorldType);
}