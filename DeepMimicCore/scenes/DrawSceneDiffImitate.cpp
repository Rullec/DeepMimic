#include "DrawSceneDiffImitate.h"
#include "SceneDiffImitate.h"

cDrawSceneDiffImitate::cDrawSceneDiffImitate() {}
cDrawSceneDiffImitate::~cDrawSceneDiffImitate() {}

void cDrawSceneDiffImitate::BuildScene(
    std::shared_ptr<cSceneSimChar> &out_scene) const
{
    out_scene = std::shared_ptr<cSceneDiffImitate>(new cSceneDiffImitate());
}