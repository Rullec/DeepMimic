#include "CtCtrlUtil.h"
#include "anim/KinTree.h"
#include <iostream>
#include <limits>
using namespace std;

const double gDefaultOffsetPDBound = 10;
const double gDefaultRotatePDBound = M_PI;

void cCtCtrlUtil::BuildBoundsTorque(const Eigen::MatrixXd &joint_mat,
                                    int joint_id, Eigen::VectorXd &out_min,
                                    Eigen::VectorXd &out_max)
{
    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    out_min = std::numeric_limits<double>::quiet_NaN() *
              Eigen::VectorXd::Ones(joint_dim);
    out_max = std::numeric_limits<double>::quiet_NaN() *
              Eigen::VectorXd::Ones(joint_dim);

    double torque_lim = cKinTree::GetTorqueLimit(joint_mat, joint_id);
    double force_lim = cKinTree::GetForceLimit(joint_mat, joint_id);

    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        out_min.fill(-torque_lim);
        out_max.fill(torque_lim);
        break;
    case cKinTree::eJointTypePrismatic:
        out_min.fill(-force_lim);
        out_max.fill(force_lim);
        break;
    case cKinTree::eJointTypePlanar:
        out_min.fill(-force_lim);
        out_max.fill(force_lim);
        out_min[joint_dim - 1] = -torque_lim;
        out_max[joint_dim - 1] = torque_lim;
        break;
    case cKinTree::eJointTypeFixed:
        break;
    case cKinTree::eJointTypeSpherical:
        out_min.segment(0, joint_dim - 1).fill(-torque_lim);
        out_max.segment(0, joint_dim - 1).fill(torque_lim);
        out_min[joint_dim - 1] = 0;
        out_max[joint_dim - 1] = 0;
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildBoundsVel(const Eigen::MatrixXd &joint_mat, int joint_id,
                                 Eigen::VectorXd &out_min,
                                 Eigen::VectorXd &out_max)
{
    const double max_ang_vel = 20 * M_PI;
    const double max_lin_vel = 5;

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);

    out_min = std::numeric_limits<double>::quiet_NaN() *
              Eigen::VectorXd::Ones(joint_dim);
    out_max = std::numeric_limits<double>::quiet_NaN() *
              Eigen::VectorXd::Ones(joint_dim);

    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        out_min.fill(-max_ang_vel);
        out_max.fill(max_ang_vel);
        break;
    case cKinTree::eJointTypePrismatic:
        out_min.fill(-max_lin_vel);
        out_max.fill(max_lin_vel);
        break;
    case cKinTree::eJointTypePlanar:
        out_min.fill(-max_lin_vel);
        out_max.fill(max_lin_vel);
        out_min[joint_dim - 1] = -max_ang_vel;
        out_max[joint_dim - 1] = max_ang_vel;
        break;
    case cKinTree::eJointTypeFixed:
        break;
    case cKinTree::eJointTypeSpherical:
        out_min.segment(0, joint_dim - 1).fill(-max_ang_vel);
        out_max.segment(0, joint_dim - 1).fill(max_ang_vel);
        out_min[joint_dim - 1] = 0;
        out_max[joint_dim - 1] = 0;
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildBoundsPD(const Eigen::MatrixXd &joint_mat, int joint_id,
                                Eigen::VectorXd &out_min,
                                Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        BuildBoundsPDRevolute(joint_mat, joint_id, out_min, out_max);
        break;
    case cKinTree::eJointTypePrismatic:
        BuildBoundsPDPrismatic(joint_mat, joint_id, out_min, out_max);
        break;
    case cKinTree::eJointTypePlanar:
        BuildBoundsPDPlanar(joint_mat, joint_id, out_min, out_max);
        break;
    case cKinTree::eJointTypeFixed:
        BuildBoundsPDFixed(joint_mat, joint_id, out_min, out_max);
        break;
    case cKinTree::eJointTypeSpherical:
        BuildBoundsPDSpherical(joint_mat, joint_id, out_min, out_max);
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildOffsetScaleTorque(const Eigen::MatrixXd &joint_mat,
                                         int joint_id,
                                         Eigen::VectorXd &out_offset,
                                         Eigen::VectorXd &out_scale)
{
    std::cout << "void cCtCtrlUtil::BuildOffsetScaleTorque(const "
                 "Eigen::MatrixXd& joint_mat, int joint_id, Eigen::VectorXd& "
                 "out_offset, Eigen::VectorXd& out_scale)"
              << std::endl;
    // 从joint_mat这个矩阵中，获取action的均值和标准差
    const double default_torque_lim = 300;
    const double default_force_lim = 3000;

    int joint_dim =
        cKinTree::GetParamSize(joint_mat, joint_id); // joint有多少维度?
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id); // joint是什么类别?
    out_offset = Eigen::VectorXd::Zero(joint_dim);   // mean为0
    out_scale = Eigen::VectorXd::Ones(joint_dim);    // scale为１

    // 获取torque的Lim
    double torque_lim =
        cKinTree::GetTorqueLimit(joint_mat, joint_id); // torque的lim
    double force_lim =
        cKinTree::GetForceLimit(joint_mat, joint_id); // torque的lim

    std::cout << "joint " << joint_id << " torque lim = " << torque_lim
              << std::endl;
    std::cout << "joint " << joint_id << " force lim = " << force_lim
              << std::endl;

    if (!std::isfinite(
            torque_lim)) //如果torque lim是无穷的话，就设置为缺省的300
    {
        torque_lim = default_torque_lim;
    }

    if (!std::isfinite(
            force_lim)) // 力也是这样，但是我并没有找到哪里有力的信息...
    {
        force_lim = default_force_lim;
    }
    std::cout << "[scale torque lim check] joint " << joint_id
              << " torque lim = " << torque_lim << std::endl;
    std::cout << "[scale torque lim check] joint " << joint_id
              << " force lim = " << force_lim << std::endl;
    // 在这里不会设置None类型，也就是root...
    // 由此可知torque lim不能是0
    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        out_scale.fill(1 / torque_lim);
        break;
    case cKinTree::eJointTypePrismatic:
        out_scale.fill(1 / force_lim);
        break;
    case cKinTree::eJointTypePlanar:
        out_scale.fill(1 / force_lim);
        out_scale[joint_dim - 1] =
            1 / torque_lim; // 对于那些具有滑动副的系统，我们才需要force lim
        break;
    case cKinTree::eJointTypeFixed:
        break;
    case cKinTree::eJointTypeSpherical:
        out_scale.fill(1 / torque_lim);
        out_scale[joint_dim - 1] =
            0; // 而对于球而言，out_scale 的最后一个数字是0，
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildOffsetScaleVel(const Eigen::MatrixXd &joint_mat,
                                      int joint_id, Eigen::VectorXd &out_offset,
                                      Eigen::VectorXd &out_scale)
{
    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);

    const double ang_vel_scale = 1 / (10 * M_PI);
    const double lin_vel_scale = 1 / 2.5;

    out_offset = Eigen::VectorXd::Zero(joint_dim);
    out_scale = Eigen::VectorXd::Ones(joint_dim);

    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        out_scale.fill(ang_vel_scale);
        break;
    case cKinTree::eJointTypePrismatic:
        out_scale.fill(lin_vel_scale);
        break;
    case cKinTree::eJointTypePlanar:
        out_scale.fill(lin_vel_scale);
        out_scale[joint_dim - 1] = ang_vel_scale;
        break;
    case cKinTree::eJointTypeFixed:
        break;
    case cKinTree::eJointTypeSpherical:
        out_scale.segment(0, joint_dim - 1).fill(ang_vel_scale);
        out_scale[joint_dim - 1] = 1;
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildOffsetScalePD(const Eigen::MatrixXd &joint_mat,
                                     int joint_id, Eigen::VectorXd &out_offset,
                                     Eigen::VectorXd &out_scale)
{
    // std::cout <<"void cCtCtrlUtil::BuildOffsetScalePD" <<std::endl;
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    switch (joint_type)
    {
    case cKinTree::eJointTypeRevolute:
        BuildOffsetScalePDRevolute(joint_mat, joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointTypePrismatic:
        BuildOffsetScalePDPrismatic(joint_mat, joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointTypePlanar:
        BuildOffsetScalePDPlanar(joint_mat, joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointTypeFixed:
        BuildOffsetScalePDFixed(joint_mat, joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointTypeSpherical:
        BuildOffsetScalePDSpherical(joint_mat, joint_id, out_offset, out_scale);
        break;
    default:
        assert(false); // unsupported joint type
        break;
    }
}

void cCtCtrlUtil::BuildBoundsPDRevolute(const Eigen::MatrixXd &joint_mat,
                                        int joint_id, Eigen::VectorXd &out_min,
                                        Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeRevolute);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_min = Eigen::VectorXd::Zero(joint_dim);
    out_max = Eigen::VectorXd::Zero(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultRotatePDBound;
            val_high = gDefaultRotatePDBound;
        }

        double mean_val = 0.5 * (val_high + val_low);
        double delta = val_high - val_low;
        val_low = mean_val - 2 * delta;
        val_high = mean_val + 2 * delta;

        out_min[i] = val_low;
        out_max[i] = val_high;
    }
}

void cCtCtrlUtil::BuildBoundsPDPrismatic(const Eigen::MatrixXd &joint_mat,
                                         int joint_id, Eigen::VectorXd &out_min,
                                         Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypePrismatic);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_min = Eigen::VectorXd::Zero(joint_dim);
    out_max = Eigen::VectorXd::Zero(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultOffsetPDBound;
            val_high = gDefaultOffsetPDBound;
        }

        double mean_val = 0.5 * (val_high + val_low);
        double delta = val_high - val_low;
        val_low = mean_val - delta;
        val_high = mean_val + delta;
        out_min[i] = val_low;
        out_max[i] = val_high;
    }
}

void cCtCtrlUtil::BuildBoundsPDPlanar(const Eigen::MatrixXd &joint_mat,
                                      int joint_id, Eigen::VectorXd &out_min,
                                      Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypePlanar);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_min = Eigen::VectorXd::Zero(joint_dim);
    out_max = Eigen::VectorXd::Zero(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultOffsetPDBound;
            val_high = gDefaultOffsetPDBound;
        }
        out_min[i] = val_low;
        out_max[i] = val_high;
    }
}

void cCtCtrlUtil::BuildBoundsPDFixed(const Eigen::MatrixXd &joint_mat,
                                     int joint_id, Eigen::VectorXd &out_min,
                                     Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeFixed);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_min = Eigen::VectorXd::Zero(joint_dim);
    out_max = Eigen::VectorXd::Zero(joint_dim);
}

void cCtCtrlUtil::BuildBoundsPDSpherical(const Eigen::MatrixXd &joint_mat,
                                         int joint_id, Eigen::VectorXd &out_min,
                                         Eigen::VectorXd &out_max)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeSpherical);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_min = -Eigen::VectorXd::Ones(joint_dim);
    out_max = Eigen::VectorXd::Ones(joint_dim);

#if defined(ENABLE_PD_SPHERE_AXIS)
    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    double val_low = lim_low.minCoeff();
    double val_high = lim_high.maxCoeff();
    bool valid_lim = val_high >= val_low;
    if (!valid_lim)
    {
        val_low = -gDefaultRotatePDBound;
        val_high = gDefaultRotatePDBound;
    }

    double mean_val = 0.5 * (val_high + val_low);
    double delta = val_high - val_low;
    val_low = mean_val - 2 * delta;
    val_high = mean_val + 2 * delta;

    out_min[0] = val_low;
    out_max[0] = val_high;
#endif
}

void cCtCtrlUtil::BuildOffsetScalePDRevolute(const Eigen::MatrixXd &joint_mat,
                                             int joint_id,
                                             Eigen::VectorXd &out_offset,
                                             Eigen::VectorXd &out_scale)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeRevolute);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_offset = Eigen::VectorXd::Zero(joint_dim);
    out_scale = Eigen::VectorXd::Ones(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultRotatePDBound;
            val_high = gDefaultRotatePDBound;
        }

        double curr_offset = -0.5 * (val_high + val_low);
        double curr_scale = 1 / (val_high - val_low);
        curr_scale *= 0.5;

        out_offset[i] = curr_offset;
        out_scale[i] = curr_scale;
        //std::cout << "[joint offset and scale compute] joint revolute " << joint_id << \
		//"val_high, val_low = (" << val_high << ", " << val_low<<"), curr_offset = " << curr_offset <<", curr_scale = " << curr_scale << std::endl;
    }
    //std::cout <<"[joint offset and scale compute] joint revolute " << joint_id <<" offset = "\
	// << out_offset.transpose() <<", scale = " << out_scale.transpose() << std::endl;
}

void cCtCtrlUtil::BuildOffsetScalePDPrismatic(const Eigen::MatrixXd &joint_mat,
                                              int joint_id,
                                              Eigen::VectorXd &out_offset,
                                              Eigen::VectorXd &out_scale)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypePrismatic);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_offset = Eigen::VectorXd::Zero(joint_dim);
    out_scale = Eigen::VectorXd::Ones(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultOffsetPDBound;
            val_high = gDefaultOffsetPDBound;
        }
        out_offset[i] = -0.5 * (val_high + val_low);
        out_scale[i] = 2 / (val_high - val_low);
    }
}

void cCtCtrlUtil::BuildOffsetScalePDPlanar(const Eigen::MatrixXd &joint_mat,
                                           int joint_id,
                                           Eigen::VectorXd &out_offset,
                                           Eigen::VectorXd &out_scale)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypePlanar);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_offset = Eigen::VectorXd::Zero(joint_dim);
    out_scale = Eigen::VectorXd::Ones(joint_dim);

    tVector lim_low = cKinTree::GetJointLimLow(joint_mat, joint_id);
    tVector lim_high = cKinTree::GetJointLimHigh(joint_mat, joint_id);

    for (int i = 0; i < joint_dim; ++i)
    {
        double val_low = lim_low[i];
        double val_high = lim_high[i];
        bool valid_lim = val_high >= val_low;
        if (!valid_lim)
        {
            val_low = -gDefaultOffsetPDBound;
            val_high = gDefaultOffsetPDBound;
        }
        out_offset[i] = -0.5 * (val_high + val_low);
        out_scale[i] = 2 / (val_high - val_low);
    }
}

void cCtCtrlUtil::BuildOffsetScalePDFixed(const Eigen::MatrixXd &joint_mat,
                                          int joint_id,
                                          Eigen::VectorXd &out_offset,
                                          Eigen::VectorXd &out_scale)
{
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeFixed);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_offset = Eigen::VectorXd::Zero(joint_dim);
    out_scale = Eigen::VectorXd::Ones(joint_dim);
    // std::cout <<"[joint offset and scale compute] joint fixed " << joint_id
    // <<" offset = " << out_offset.transpose() <<", scale = " <<
    // out_scale.transpose() << std::endl;
}

void cCtCtrlUtil::BuildOffsetScalePDSpherical(const Eigen::MatrixXd &joint_mat,
                                              int joint_id,
                                              Eigen::VectorXd &out_offset,
                                              Eigen::VectorXd &out_scale)
{
    // 对于球类铰链，我们有:
    // std::cout <<"void cCtCtrlUtil::BuildOffsetScalePDSpherical
    // 这个可能才是真的" << std::endl;
    cKinTree::eJointType joint_type =
        cKinTree::GetJointType(joint_mat, joint_id);
    assert(joint_type == cKinTree::eJointTypeSpherical);

    int joint_dim = cKinTree::GetParamSize(joint_mat, joint_id);
    out_offset = Eigen::VectorXd::Zero(joint_dim); // mean = 0
    out_scale = Eigen::VectorXd::Ones(joint_dim);  // scale = 1

#if defined(ENABLE_PD_SPHERE_AXIS)
    tVector lim_low = cKinTree::GetJointLimLow(
        joint_mat, joint_id); // low lim(torque), 3个方向上的low
    tVector lim_high = cKinTree::GetJointLimHigh(
        joint_mat, joint_id); // hight lim(torque) high vector, 3个方向上的hight

    double val_low = lim_low.minCoeff();   // 挑３个low里面最小的
    double val_high = lim_high.maxCoeff(); // 3个high里面最大的
    bool valid_lim = val_high >= val_low;
    if (!valid_lim) // 如果被判定为无效的话，就会设置成默认的val(也就是high <
                    // low，被称之为无效)
    {
        val_low = -gDefaultRotatePDBound;
        val_high = gDefaultRotatePDBound;
    }

    double curr_offset = 0;
    double curr_scale =
        1 /
        (val_high - val_low); // 如果high和low是一样的，那么scale 就会变成inf
    if (false == std::isfinite(curr_scale) || abs(val_high - val_low) < 1e-3)
    {
        std::cout << "[error] cCtCtrlUtil::BuildOffsetScalePDSpherical joint "
                  << joint_id << " val_high = " << val_high << " val_low"
                  << val_low << std::endl;
        std::cout << "it will make training Nan failed(norm_mean)\n";
        exit(1);
    }

    curr_scale *= 0.5;

    out_offset(0) = curr_offset;
    out_scale(0) = curr_scale;
    out_offset(3) = -0.2;
#else
    out_offset(0) = -0.2;
#endif

    // std::cout <<"[joint offset and scale compute] joint spherical " <<
    // joint_id << "val_high, val_low = (" << val_high << ", " << val_low<<")"<<
    // ",  offset = " << out_offset.transpose() <<", scale = " <<
    // out_scale.transpose() << std::endl;
}