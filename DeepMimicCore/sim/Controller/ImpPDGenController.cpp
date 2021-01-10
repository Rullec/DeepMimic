#include "ImpPDGenController.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "util/LogUtil.h"
#include <iostream>

typedef cImpPDGenController cPDCtrl;

cPDCtrl::cImpPDGenController()
{
    MIMIC_INFO("cImpPDGenController constructed");
    Clear();
}
cPDCtrl::~cImpPDGenController() {}
/**
 * \brief           Given Kp and Kd parameters, init this stable PD controller
 */
void cPDCtrl::Init(cSimCharacterGen *gen_char, const tVectorXd &kp,
                   const tVectorXd &kd)
{
    mChar = gen_char;
    InitGains(kp, kd);
    VerifyController();
}

void cPDCtrl::Clear()
{
    mKp.resize(0);
    mKd.resize(0);
    mTarget_q.resize(0);
    mTarget_qdot.resize(0);
    mChar = nullptr;
}

void cPDCtrl::Reset()
{
    mTarget_q.resize(0);
    mTarget_qdot.resize(0);
}
/**
 * \brief               Get the Generalized coordinate PD target
 */
void cPDCtrl::GetPDTarget_q(tVectorXd &q, tVectorXd &qdot) const
{
    q = this->mTarget_q;
    qdot = this->mTarget_qdot;
}

/**
 * \brief               Set the generalized coordinate PD target
 */
void cPDCtrl::SetPDTarget_q(const tVectorXd &q, const tVectorXd &qdot)
{
    MIMIC_ASSERT(q.size() == GetPDTargetSize());
    MIMIC_ASSERT(qdot.size() == GetPDTargetSize());
    MIMIC_ASSERT(q.hasNaN() == false);
    MIMIC_ASSERT(qdot.hasNaN() == false);
    mTarget_q = q;
    // mTarget_q.setZero();
    mTarget_qdot = qdot;
}

/**
 * \brief           Verify the PD coefficient, make sure that their size is
 * correct, their value are non-negative
 */
void cPDCtrl::VerifyController()
{
    MIMIC_INFO("begin to verify cImpPDGenController");

    int target_size = GetPDTargetSize();

    if (mKp.size() != target_size || mKd.size() != target_size)
    {
        MIMIC_ERROR("cImpPDGenController Kp/Kd invalid size {}/{} != {}",
                    mKp.size(), mKd.size(), target_size);
    }

    MIMIC_ASSERT(mKp.minCoeff() >= 0 && mKd.minCoeff() >= 0);
}

/**
 * \brief                   The size of given PD target should be the same as q
 * without root freedom
 */
int cPDCtrl::GetPDTargetSize()
{
    MIMIC_ASSERT(mChar != nullptr);

    int size = mChar->GetNumOfFreedom();
    // int size = mChar->GetNumOfFreedom() -
    // mChar->GetRoot()->GetNumOfFreedom();
    // MIMIC_INFO("PD target size {}", size);
    return size;
}

/**
 * \brief           Calculate the control force
 */
#define SPD
void cPDCtrl::UpdateControlForce(double dt, tVectorXd &out_tau)
{
    // std::cout << "[cPDCtrl] cur target = " << mTarget_q.transpose()
    //   << std::endl;
    // 1. check the PD target "q" and "qdot" should the same as the dof of this
    // character. they are in generalized coordinate
    MIMIC_ASSERT(mTarget_qdot.size() == GetPDTargetSize() &&
                 mTarget_q.size() == GetPDTargetSize() &&
                 "the PD target has been set up well");
    CheckVelExplode();
#ifdef SPD
    UpdateControlForceSPD(dt, out_tau);
#else
    UpdateControlForceNative(dt, out_tau);
    std::cout << "native control force = " << out_tau.transpose() << std::endl;
#endif

    // for underactutated system, the first 6 root doms should be ignored now
    PostProcessControlForce(out_tau);

    // out_tau.setZero();
    // MIMIC_INFO("out tau = {}", out_tau.transpose());
    // MIMIC_INFO("q = {}", q.transpose());
    // exit(1);
}

/**
 * \brief               Extract & init the mKp and mKd used in the computation
 * by given a joint-size-long kp & kd
 */
void cPDCtrl::InitGains(const tVectorXd &kp, const tVectorXd &kd)
{
    // 1. judge input
    int num_of_joints = mChar->GetNumOfJoint(); // except roots
    MIMIC_ASSERT(kp.size() == num_of_joints && kd.size() == num_of_joints);

    // 2. expand the joint-size-long kp & kd -> dof-long kp & kd
    int num_of_freedom = mChar->GetNumOfFreedom();
    mKp = tVectorXd::Zero(num_of_freedom);
    mKd = tVectorXd::Zero(num_of_freedom);

    int st_pos = 0;
    for (int i = 0; i < num_of_joints; i++)
    {
        double joint_kp = kp[i], joint_kd = kd[i];
        auto joint = mChar->GetJointById(i);
        int joint_dof = joint->GetNumOfFreedom();
        mKp.segment(st_pos, joint_dof).fill(joint_kp);
        mKd.segment(st_pos, joint_dof).fill(joint_kd);
        st_pos += joint_dof;
    }
    MIMIC_INFO("Init gains: kp = {}, kd = {}", mKp.transpose(),
               mKd.transpose());
}

/**
 * \brief                   Check and warn the velocity explode in qdot
 * (generalized velocity) when we try to calculate the control forces
 */
void cPDCtrl::CheckVelExplode() const
{
    tVectorXd qdot = mChar->Getqdot();
    for (int i = 0; i < mChar->GetNumOfFreedom(); i++)
    {
        if ((std::fabs(qdot[i])) >= mChar->GetMaxVel() - 1e-5)
        {
            MIMIC_WARN(
                "qdot {} = {} exceed the max_vel {}, belongs to joint {}", i,
                qdot[i], mChar->GetMaxVel(),
                mChar->GetJointByDofId(i)->GetName());
        }
    }
}

/**
 * \brief                   Update control forces natively PD
 */
void cPDCtrl::UpdateControlForceNative(double dt, tVectorXd &out_tau)
{
    tVectorXd q = mChar->Getq(), qdot = mChar->Getqdot();
    out_tau =
        mKp.cwiseProduct(mTarget_q - q) + mKd.cwiseProduct(mTarget_qdot - qdot);
}

/**
 * \brief                   Stable PD
 */
void cPDCtrl::UpdateControlForceSPD(double dt, tVectorXd &out_tau) const
{
    tVectorXd q_cur = mChar->Getq(), qdot_cur = mChar->Getqdot();
    tVectorXd q_next_err = mTarget_q - (q_cur + dt * qdot_cur);
    tVectorXd qdot_next_err = mTarget_qdot - qdot_cur;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat = mKp.asDiagonal();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kd_mat = mKd.asDiagonal();

    // tVectorXd Q =
    //     mKp.cwiseProduct(q_next_err) + mKd.cwiseProduct(qdot_next_err);
    // std::cout << "Q = " << Q.transpose() << std::endl;
    tMatrixXd M = mChar->GetMassMatrix();
    M += dt * Kd_mat;

    tVectorXd qddot_pred =
        M.inverse() * (Kp_mat * q_next_err + Kd_mat * qdot_next_err -
                       mChar->GetCoriolisMatrix() * qdot_cur);
    // std::cout << "qddot pred = " << qddot_pred.transpose() << std::endl;
    out_tau = Kp_mat * q_next_err + Kd_mat * (qdot_next_err - dt * qddot_pred);
    MIMIC_ASSERT(out_tau.hasNaN() == false);
    // std::cout << "out tau SPD = " << out_tau.transpose() << std::endl;
    // exit(1);
}

void cPDCtrl::PostProcessControlForce(tVectorXd &out_tau) const
{
    // set the torque of root joint to zero
    auto root_joint = mChar->GetRoot();
    if (root_joint->GetJointType() == JointType::NONE_JOINT)
    {
        // MIMIC_INFO("set root joint(dof {}) tau to zero",
        //            root_joint->GetNumOfFreedom());
        out_tau.segment(0, root_joint->GetNumOfFreedom()).setZero();
    }

    // clamp the joint torque
    int st_pos = 0;
    int num_of_joints = mChar->GetNumOfJoint();
    for (int i = 0; i < num_of_joints; i++)
    {
        auto &joint = mChar->GetJoint(i);
        int joint_dof = mChar->GetJointById(i)->GetNumOfFreedom();

        double torque_lim = joint.GetTorqueLimit();
        // std::cout << "joint " << i << " torque lim = " << torque_lim
        //           << std::endl;
        double current_norm = out_tau.segment(st_pos, joint_dof).norm();
        if (current_norm > torque_lim)
        {
            out_tau.segment(st_pos, joint_dof) *= torque_lim / current_norm;
        }
        st_pos += joint_dof;
    }
    // std::cout << "[debug] out_tau = " << out_tau.transpose() << std::endl;
    MIMIC_ASSERT(out_tau.hasNaN() == false);
}

tVectorXd cPDCtrl::CalcPDTargetByControlForce(double dt, const tVectorXd &pose,
                                              const tVectorXd &vel,
                                              const tVectorXd &ctrl_force) const
{
    // 1. push state
    auto gen_char = dynamic_cast<cSimCharacterGen *>(mChar);
    tVectorXd before_pose = gen_char->GetPose(),
              before_vel = gen_char->GetVel();
    // gen_char->PushState("CalcPDTarget");

    // 2. set current state
    gen_char->SetPose(pose);
    gen_char->SetVel(vel);

    // 3. Given torque, calculate the PD target
    int num_of_freedom = gen_char->GetNumOfFreedom();
    MIMIC_ASSERT(ctrl_force.size() == num_of_freedom);
    tVectorXd target;
    {
        tMatrixXd mat_Kp = mKp.asDiagonal(), mat_Kd = mKd.asDiagonal();
        double dt2 = dt * dt;
        tMatrixXd M_tilde_inv =
            (mChar->GetMassMatrix() + dt * mat_Kd).inverse();
        tVectorXd q_cur = gen_char->Getq(), qdot_cur = gen_char->Getqdot();
        tMatrixXd C = mChar->GetCoriolisMatrix();
        tMatrixXd I = tMatrixXd::Identity(num_of_freedom, num_of_freedom);

        tMatrixXd A = (I - dt * mat_Kd * M_tilde_inv) * mat_Kp;
        tVectorXd b = (dt * mat_Kd * M_tilde_inv - I) * mat_Kp * q_cur;
        tVectorXd c = -1 *
                      (dt * mat_Kp + mat_Kd - dt * mat_Kd * M_tilde_inv * C -
                       dt * mat_Kd * M_tilde_inv * mat_Kd -
                       dt2 * mat_Kd * M_tilde_inv * mat_Kp) *
                      qdot_cur;
        target = A.inverse() * (ctrl_force - b - c);
        target.segment(0, 6).setZero();
        // std::cout << "[debug] solved q = " << target.transpose() <<
        // std::endl;
        target = gen_char->ConvertqToPose(target);
        target = target.segment(7, target.size() - 7);
    }
    // 4. restore and return
    // MIMIC_ERROR("hasn't been implemented yet");
    gen_char->SetPose(before_pose);
    gen_char->SetVel(before_vel);
    return target;
}

/**
 * \brief           calculate jacobian d(gen_ctrl_force)/d(target_q) based on SPD
 * the gen_ctrl force is full-length generalized force
 * the target_q is the full length generalized force
 *  For more details, please check the note "20201121 重新思考SPD"
 * 
 *      d(\tau)/d(q_bar) = Kp - dt * Kd * (M + dt * Kd).inv() * Kp
*/
tMatrixXd cPDCtrl::CalcDCtrlForceDTargetq(double dt)
{
    auto gen_char = dynamic_cast<cSimCharacterGen *>(mChar);
    tMatrixXd mat_Kp = mKp.asDiagonal(), mat_Kd = mKd.asDiagonal();
    tMatrixXd M_tilde_inv = (mChar->GetMassMatrix() + dt * mat_Kd).inverse();
    return mat_Kp - dt * mat_Kd * M_tilde_inv * mat_Kp;
}

/**
 * \brief           test jacobian d(gen_ctrl_force)/d(target_q) based on SPD
*/
void cPDCtrl::TestDCtrlForceDTargetq(double dt)
{
    // 1. get old control force and the derivatives
    auto gen_char = dynamic_cast<cSimCharacterGen *>(mChar);
    int dof = gen_char->GetNumOfFreedom();
    tVectorXd old_tar_q = mTarget_q, old_tar_qdot = mTarget_qdot;
    tVectorXd old_tau = tVectorXd::Zero(dof);
    UpdateControlForce(dt, old_tau);
    tMatrixXd DctrlforceDtargetq = CalcDCtrlForceDTargetq(dt);

    // 2. check the numerical derivatives
    double eps = 1e-5;
    for (int i = 0; i < dof; i++)
    {
        mTarget_q[i] += eps;
        tVectorXd new_tau;
        UpdateControlForce(dt, new_tau);
        tVectorXd num_deriv = (new_tau - old_tau) / eps;
        tVectorXd ideal_deriv = DctrlforceDtargetq.col(i);
        tVectorXd diff = ideal_deriv - num_deriv;
        BTGEN_ASSERT(diff.norm() < 10 * eps);
        mTarget_q[i] -= eps;
    }

    // 3. restore
    mTarget_q = old_tar_q;
    mTarget_qdot = old_tar_qdot;
}