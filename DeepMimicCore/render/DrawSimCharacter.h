#pragma once

#include "render/Camera.h"
#include "sim/Controller/DeepMimicCharController.h"
#include "sim/SimItems/SimCharacter.h"
#include "sim/World/Ground.h"
#include "util/CircularBuffer.h"

class cDrawSimCharacter
{
    // 绘制角色的文件
public:
    static void Draw(const cSimCharacterBase &character, const tVector &fill_tint,
                     const tVector &line_col, bool enable_draw_shape = false);
    static void DrawCoM(const cSimCharacterBase &character, double marker_size,
                        double vel_scale, const tVector &col,
                        const tVector &offset);
    static void DrawTorque(const cSimCharacterBase &character,
                           const tVector &offset);
    static void DrawBodyVel(const cSimCharacterBase &character,
                            double lin_vel_scale, double ang_vel_scale,
                            const tVector &offset);
    static void DrawInfoValLog(const cCircularBuffer<double> &val_log,
                               const cCamera &cam);

protected:
    static void DrawSimBody(const cSimCharacterBase &character,
                            const tVector &fill_tint, const tVector &line_col);
};
