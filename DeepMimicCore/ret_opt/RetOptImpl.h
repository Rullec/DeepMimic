//
// Created by ljf on 2020/6/19.
//

#ifndef DEEPMIMICCORE_RETOPTIMPL_H
#define DEEPMIMICCORE_RETOPTIMPL_H
#include "../util/MathUtil.h"
#include "ShapeMotionMemPool.h"


class DMRetController;
class cRetOptImpl {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    struct tParam {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        const Eigen::MatrixXd*      joint_mat;
        Eigen::MatrixXd*            motion_mat;
        std::vector<std::string>*   joint_names;
        std::vector<std::string>*   link_names;
        Eigen::VectorXd             body_shape_param;
    };

    cRetOptImpl();
    virtual ~cRetOptImpl();

    virtual void Run(tParam&);
    void SaveJointMat(const char *file, const Eigen::MatrixXd& joint_mat);
    void SaveMotionMat(const char *file, const Eigen::MatrixXd &motion_mat);
    void SaveShapeParam(const char * file, const Eigen::VectorXd&);
    void DumpMotionPool(const char* dir);
    void SetStdJointMat(const tMatrixXd& m) {this->std_joint_mat = m; std_joint_mat_set=true;}
    bool IsStdJointMatSet() const {return std_joint_mat_set;}

    ShapeMotionMemPool* GetShapeMotionPool() const {return shape_motion_pool;}
protected:
    DMRetController* controller;
    ShapeMotionMemPool* shape_motion_pool;
    tMatrixXd std_joint_mat;
    bool std_joint_mat_set;
};


#endif //DEEPMIMICCORE_RETOPTIMPL_H
