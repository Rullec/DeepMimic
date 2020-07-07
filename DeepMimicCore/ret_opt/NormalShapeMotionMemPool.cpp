//
// Created by ljf on 2020/7/3.
//

#include "NormalShapeMotionMemPool.h"
#include <iostream>

NormalShapeMotionMemPool::NormalShapeMotionMemPool(unsigned int length, double thresh) :
        ShapeMotionMemPool(length),
        thresh(thresh),
        curr_idx(0),
        is_full(false) {
    pool.resize(length);
}

NormalShapeMotionMemPool::~NormalShapeMotionMemPool() {
    pool.clear();
}

void NormalShapeMotionMemPool::Insert(cShapeMotionNode &node) {
    pool[curr_idx] = node;
    if (curr_idx + 1 == length) is_full = true;
    curr_idx = (curr_idx + length + 1) % length;
}

cShapeMotionNode *NormalShapeMotionMemPool::FindNearestOne(Eigen::VectorXd& param) {
    unsigned int n = is_full ? length : curr_idx;

    for(unsigned int i = 0; i < n; ++i) {
        // compute distance
        double dist = (pool[i].body_shape_param - param).norm();
        if (dist < thresh) {
            return &pool[i];
        }
    }
    return nullptr;
}

void NormalShapeMotionMemPool::Insert(Eigen::VectorXd &body_shape_param, Eigen::MatrixXd *motion_mat) {
    cShapeMotionNode node;
    node.body_shape_param.noalias() = body_shape_param;
    node.motion_mat.noalias() = *motion_mat;
    Insert(node);
}

unsigned int NormalShapeMotionMemPool::GetLength() {
    return is_full ? length : curr_idx+1;
}


