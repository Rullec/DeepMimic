//
// Created by ljf on 2020/7/3.
//

#ifndef DEEPMIMICCORE_NORMALSHAPEMOTIONMEMPOOL_H
#define DEEPMIMICCORE_NORMALSHAPEMOTIONMEMPOOL_H

#include "ShapeMotionMemPool.h"

class NormalShapeMotionMemPool: public ShapeMotionMemPool {
public:
    NormalShapeMotionMemPool(unsigned int length, double thresh);
    ~NormalShapeMotionMemPool() override;

    void Insert(Eigen::VectorXd &body_shape_param, Eigen::MatrixXd *motion_mat) override;

    void                Insert(cShapeMotionNode &node) override;
    cShapeMotionNode *  FindNearestOne(Eigen::VectorXd& param) override;

    virtual unsigned int GetLength() override;

    cShapeMotionNode *GetShapeMotionNode(int id) override;

protected:
    double                      thresh;
    unsigned int                curr_idx;
    bool                        is_full;

    tEigenArr<cShapeMotionNode> pool;

};


#endif //DEEPMIMICCORE_NORMALSHAPEMOTIONMEMPOOL_H
