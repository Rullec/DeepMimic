//
// Created by ljf on 2020/6/19.
//

#ifndef DEEPMIMICCORE_RETOPTIMPL_H
#define DEEPMIMICCORE_RETOPTIMPL_H
#include "../util/MathUtil.h"


class cRetOptImpl {
public:
    struct tParam {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        const Eigen::MatrixXd* joint_mat;
        Eigen::MatrixXd* motion_mat;
        std::vector<std::string>* joint_names;
        std::vector<std::string>* link_names;
    };

    cRetOptImpl() {}
    virtual ~cRetOptImpl() {}

    virtual void Run(tParam);
    void SaveJointMat(const char *file, const Eigen::MatrixXd& joint_mat);
    void SaveMotionMat(const char *file, const Eigen::MatrixXd &motion_mat);
};


#endif //DEEPMIMICCORE_RETOPTIMPL_H
