//
// Created by ljf on 2020/7/3.
//

#ifndef DEEPMIMICCORE_CSHAPEMOTIONNODE_H
#define DEEPMIMICCORE_CSHAPEMOTIONNODE_H
#include "util/MathUtil.h"

class cShapeMotionNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Eigen::VectorXd body_shape_param;
    Eigen::MatrixXd motion_mat;
};


#endif //DEEPMIMICCORE_CSHAPEMOTIONNODE_H
