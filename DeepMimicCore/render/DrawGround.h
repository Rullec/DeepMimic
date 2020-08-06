#pragma once

#include "render/DrawMesh.h"
#include "sim/World/Ground.h"

class cDrawGround
{
public:
    static void BuildMesh(const cGround *ground, cDrawMesh *out_mesh);

protected:
    static void BuildMeshPlane(const cGround *ground, cDrawMesh *out_mesh);
};