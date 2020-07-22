//
// Created by ljf on 2020/6/19.
//

#include "RetOptImpl.h"
#include "DMRetController.h"
#include "NormalShapeMotionMemPool.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <dirent.h>

cRetOptImpl::cRetOptImpl() {
    controller = new DMRetController("data/0707/tp/070701.tp");
//    shape_motion_pool = new NormalShapeMotionMemPool(100, 1.8e-1);
    shape_motion_pool = new NormalShapeMotionMemPool(100, 1.8e-3);

    std_joint_mat_set = false;
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

void cRetOptImpl::DumpMotionPool(const char *dir) {
    // 1. judge dir existing
    auto dir_result = opendir(dir);
    if (dir_result == nullptr) {
        std::cout << "[Error] please set correct motion logging dir" << std::endl;
        return;
    }
    // 2. save std joint mat
    std::string std_joint_mat_dir(dir);
    std_joint_mat_dir.append("/std_joint.txt");
    SaveJointMat(std_joint_mat_dir.data(), std_joint_mat);
    // 3. loop over motion pool and log skeleton and motion
    auto pool_size = shape_motion_pool->GetLength();
    for(auto i = 0; i < pool_size; ++i) {
        auto* p = shape_motion_pool->GetShapeMotionNode(i);
        std::string motion_path(dir);// + "/motion_i.txt");
        motion_path.append("/motion_");
        motion_path.append(std::to_string(i));
        motion_path.append(".txt");
        SaveMotionMat(motion_path.data(), p->motion_mat);

        std::string skeleton_path(dir);
        skeleton_path.append("/joint_");
        skeleton_path.append(std::to_string(i));
        skeleton_path.append(".txt");
        SaveShapeParam(skeleton_path.data(), p->body_shape_param);
    }
}

void cRetOptImpl::SaveShapeParam(const char *file, const Eigen::VectorXd& param) {
    std::fstream fout(file, std::ios::out);
    if (!fout.is_open()) {
        std::cerr << "[Error] cannot open file: " << file << std::endl;
        exit(-1);
    }
    fout << param.rows() << " " << param.cols() << std::endl;
    fout << param << std::endl;
    fout.close();
}
