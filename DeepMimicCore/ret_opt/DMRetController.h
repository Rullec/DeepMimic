//
// Created by ljf on 2020/6/19.
//

#ifndef DEEPMIMICCORE_DMRETCONTROLLER_H
#define DEEPMIMICCORE_DMRETCONTROLLER_H
#include "Controller.h"

struct DeepMimicData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    const mat* joint_mat;
    mat*    motion;
    int     n_joints;
    int     n_frames;

    std::vector<std::string>* joint_names;
    std::vector<std::string>* link_names;

};


class DMRetController : public Controller {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    DMRetController();
    DMRetController(const char* file);

    DMRetController(BaseRender *render);
    virtual void Test() override ;
    virtual void ClearController();
    void TestDeepMimicShapeVarRetargeting();
    bool RunDeepMimicShapeVarRetargeting(DeepMimicData& data);
protected:
    bool ConvertJointMatToRobotModel(mat& joint_mat, RobotModel* model);
    bool ConvertDMMotionToEulerAngle(mat& motion_mat, RobotModel* model, std::vector<double>& ans);
};
#endif //DEEPMIMICCORE_DMRETCONTROLLER_H
