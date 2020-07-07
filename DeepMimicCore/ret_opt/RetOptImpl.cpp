//
// Created by ljf on 2020/6/19.
//

#include "RetOptImpl.h"
#include "DMRetController.h"
#include "NormalShapeMotionMemPool.h"
#include <iostream>
#include <fstream>
#include <cmath>

cRetOptImpl::cRetOptImpl() {
    controller = new DMRetController("data/0704/tp/110602.tp");
    shape_motion_pool = new NormalShapeMotionMemPool(100, 1.8e-3);
}

cRetOptImpl::~cRetOptImpl() {
    delete controller;
}


void cRetOptImpl::Run(cRetOptImpl::tParam& param) {
    // 1. search in the pool
    cShapeMotionNode* node = shape_motion_pool->FindNearestOne(param.body_shape_param);
    if (node) {
        std::cout << "[log] Find valid closest motion\n";
        std::cout << "pool.size(): " << shape_motion_pool->GetLength() << std::endl;
        *(param.motion_mat) = node->motion_mat;
        return;
    }

    DeepMimicData data;
    data.joint_mat  = param.joint_mat;
    data.motion     = param.motion_mat;
    data.n_joints   = param.joint_mat->rows();
    data.n_frames   = param.motion_mat->rows();
    data.joint_names = param.joint_names;
    data.link_names  = param.link_names;


    controller->RunDeepMimicShapeVarRetargeting(data);
    controller->ClearController();

    shape_motion_pool->Insert(param.body_shape_param, param.motion_mat);
//    SaveJointMat("/home/ljf/playground/project/RobotControl/Data/joint_mat/0620/062001.txt", *(data.joint_mat));
//    SaveMotionMat("/home/ljf/playground/project/RobotControl/Data/motion_mat/0620/062001.txt", *(data.motion));
//    exit(-1);
}

void cRetOptImpl::SaveJointMat(const char *file, const Eigen::MatrixXd& joint_mat) {
    std::fstream fout(file, std::ios::out);
    if (!fout.is_open()) {
        std::cerr << "[Error] cannot open file: " << file << std::endl;
        exit(-1);
    }
    fout << joint_mat.rows() << " " << joint_mat.cols() << std::endl;
//    fout << joint_mat << std::endl;

    for(int i = 0; i < joint_mat.rows(); ++i) {
        for(int j = 0; j < joint_mat.cols(); ++j) {
            if (!std::isinf(joint_mat.coeffRef(i, j)))
                fout << joint_mat.coeffRef(i, j);
            else fout << 0;
            fout << " ";
        }
        fout << "\n";
    }
    fout.close();
}

void cRetOptImpl::SaveMotionMat(const char *file, const Eigen::MatrixXd &motion_mat) {
    std::fstream fout(file, std::ios::out);
    if (!fout.is_open()) {
        std::cerr << "[Error] cannot open file: " << file << std::endl;
        exit(-1);
    }
    fout << motion_mat.rows() << " " << motion_mat.cols() << std::endl;
    fout << motion_mat << std::endl;
    fout.close();
}

