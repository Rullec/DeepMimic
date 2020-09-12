#include "CtPDController.h"
cCtPDController::cCtPDController() { mEnableSolvePDTargetTest = false; }

void cCtPDController::UpdateTimeOnly(double timestep) { mTime += timestep; }

void cCtPDController::SetEnableSolvePDTargetTest(bool value)
{
    mEnableSolvePDTargetTest = value;
}