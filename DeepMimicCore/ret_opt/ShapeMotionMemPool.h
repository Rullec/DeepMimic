//
// Created by ljf on 2020/7/3.
//

#ifndef DEEPMIMICCORE_SHAPEMOTIONMEMPOOL_H
#define DEEPMIMICCORE_SHAPEMOTIONMEMPOOL_H
#include "cShapeMotionNode.h"

class ShapeMotionMemPool {
public:
    explicit ShapeMotionMemPool(unsigned int length);
    virtual ~ShapeMotionMemPool() {}
    virtual void Insert(Eigen::VectorXd& body_shape_param, Eigen::MatrixXd* motion_mat) = 0;
    virtual void Insert(cShapeMotionNode& node) = 0;
    virtual cShapeMotionNode* FindNearestOne(Eigen::VectorXd& param) = 0;
    virtual unsigned int GetLength() {return length;}
protected:
    unsigned int length;
};


#endif //DEEPMIMICCORE_SHAPEMOTIONMEMPOOL_H
