#include "KinTree.h"
#include "util/LogUtil.h"
#include <iostream>
/**
 * \brief               Calculate the derivative of root rot err w.r.t pose0
 * 
 *  let current root rotation quaternion = q0, target quaternion = q1
 *      t = (q1 * q0.conj()).w()
 *      root_rot_err = [2 * std::acos(t)]^2
 * 
 *  so, the final result : d(root_rot_err)/d pose0 = d(root_rot_err)/dt * dt/dw * dw/dpose0
 *      
 *  part1: d(root_rot_err)/dt = -8 * arccos(t) / (q - t^2)^{0.5}
 *  part2: dt/dw =  [ d(q1q0conj)/dq0 ].row(0)
 *  part3: dw/dpose0 = depends on the definition of "pose"
 *          for normal none joint, it's identity
 *          for bipedal none joint, it's part of d(quaternion)/daxis
*/
tVectorXd cKinTree::CalcDRootRotErrDPose0(const Eigen::MatrixXd &joint_mat,
                                          const Eigen::VectorXd &pose0,
                                          const Eigen::VectorXd &pose1)
{
    int root_id = GetRoot(joint_mat);

    // for any non root joint, run for different types
    int dof = cKinTree::GetParamSize(joint_mat, root_id);
    auto type = cKinTree::GetJointType(joint_mat, root_id);
    tQuaternion rot0 = GetRootRot(joint_mat, pose0),
                rot1 = GetRootRot(joint_mat, pose1);
    int offset = GetParamOffset(joint_mat, root_id);
    int size = GetParamSize(joint_mat, root_id);
    // 1. part 1
    tQuaternion q1q0conj = cKinTree::CalcRootRotDiff(joint_mat, pose0, pose1);
    double t = q1q0conj.w(); // t < 1

    // add 1e-10 on the denominator for stability
    double d_root_rot_err_dt = 0;
    if (std::fabs(std::fabs(t) - 1) < 1e-9)
    {
        if (t > 0)
            d_root_rot_err_dt = -8;
        else
            d_root_rot_err_dt = 8;
    }
    else
    {
        d_root_rot_err_dt = -8 * std::acos(t) / std::sqrt(1 - std::pow(t, 2));
    }

    // 2. part 2
    tVector dtdw = cMathUtil::Calc_Dq1q0conj_Dq0(rot0, rot1).row(0);

    // 3. part 3
    tVector d_root_rot_err_dq0 = d_root_rot_err_dt * dtdw;
    tVectorXd dRootRotErr_dpose0 = tVectorXd::Zero(dof);
    switch (type)
    {
    case eJointType::eJointTypeNone:
    {
        dRootRotErr_dpose0.segment(gPosDim, gRotDim) = d_root_rot_err_dq0;
        break;
    }

    case eJointType::eJointTypeBipedalNone:
    { // pose format [translate_y, translate_z, rot_x]
        tMatrix dquaternion_daa = cMathUtil::Calc_DQuaternion_DAxisAngle(
            tVector(1, 0, 0, 0) * pose0[offset + 2]);
        dRootRotErr_dpose0[2] =
            (d_root_rot_err_dq0.transpose() * dquaternion_daa)[0];
        break;
    }
    default:
        MIMIC_ERROR("unsupported joint type {}", type);
        break;
    }

    // fill into the total length vector
    tVectorXd dRootRotErr_dpose0_total = tVectorXd::Zero(pose0.size());
    dRootRotErr_dpose0_total.segment(offset, size) = dRootRotErr_dpose0;
    return dRootRotErr_dpose0_total;
}

/**
 * \brief           Calculate d(joint_pose_err) / d pose0
*/
tVectorXd cKinTree::CalcDPoseErrDPose0(const Eigen::MatrixXd &joint_mat,
                                       int joint_id,
                                       const Eigen::VectorXd &pose0,
                                       const Eigen::VectorXd &pose1)
{
    int root_id = GetRoot(joint_mat);
    int offset = cKinTree::GetParamOffset(joint_mat, joint_id);
    int size = cKinTree::GetParamSize(joint_mat, joint_id);
    tVectorXd dErrdPose0_total = tVectorXd::Zero(pose0.size());
    if (root_id == joint_id)
    {
        MIMIC_ERROR("DRootPoseErrDPose0 hasn't been implemented");
        CalcDRootRotErrDPose0(joint_mat, pose0, pose1);
    }
    else
    {
        // for any non root joint, run for different types
        // the definition of pose err are same for all joints, except spherical
        tVectorXd dErrdPose0 = tVectorXd::Zero(size);
        auto type = cKinTree::GetJointType(joint_mat, joint_id);
        switch (type)
        {

        case eJointType::eJointTypeSpherical:
        {
            /*
                w = q1 * q0.diff().w
                t = 2 * arccos(w)
                err = t^2

                derr/dq0 = derr/dt * dt / dw * dw / dq0
            */

            tVector q0_coef = pose0.segment(offset, size),
                    q1_coef = pose1.segment(offset, size);
            tQuaternion q0 = cMathUtil::VecToQuat(q0_coef),
                        q1 = cMathUtil::VecToQuat(q0_coef);
            double w = cMathUtil::QuatDiff(q0, q1).w();
            double t = 2 * std::acos(w);
            double derr_dt = 2 * t;
            double dt_dw = -2 / std::sqrt(1 - std::pow(w, 2));
            tVector dw_dq0 = cMathUtil::Calc_Dq1q0conj_Dq0(q0, q1).row(0);
            dErrdPose0 = derr_dt * dt_dw * dw_dq0;
            break;
        }
        default:

        {
            // err = (pose0 - pose1).dot(pose0 - pose1)
            dErrdPose0 = 2 * (pose0 - pose1).segment(offset, size);
            break;
        }
        }
        dErrdPose0_total.segment(offset, size) = dErrdPose0;
    }
    return dErrdPose0_total;
}

/**
 * \brief           Calc d(root_rotvel_err)/dvel0
*/
tVectorXd cKinTree::CalcDRootRotVelErrDVel0(const Eigen::MatrixXd &joint_mat,
                                            const Eigen::VectorXd &vel0,
                                            const Eigen::VectorXd &vel1)
{
    int root_id = cKinTree::GetRoot(joint_mat);
    auto root_type = cKinTree::GetJointType(joint_mat, root_id);
    int st = cKinTree::GetParamOffset(joint_mat, root_id),
        size = cKinTree::GetParamSize(joint_mat, root_id);
    tVectorXd total_deriv = -2 * (vel1 - vel0);
    tVectorXd dRootVelErrdvel0 = tVectorXd::Zero(vel0.size());
    switch (root_type)
    {
    case eJointType::eJointTypeNone:
    {
        dRootVelErrdvel0.segment(st + gPosDim, gRotDim) =
            total_deriv.segment(st + gPosDim, gRotDim);
        break;
    }
    case eJointType::eJointTypeBipedalNone:
    {
        // only the third freedom of bipedal none is rotation
        dRootVelErrdvel0[st + 2] = total_deriv[st + 2];
        break;
    }
    default:
        MIMIC_ERROR("unsupported joint type {}", root_type);
        break;
    }
    return dRootVelErrdvel0;
}