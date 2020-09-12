#include "CtPDFeaController.h"
#include "sim/SimItems/SimCharacter.h"
#include <iostream>
using namespace std;

const std::string gPDControllersKey = "PDControllers";

cCtPDFeaController::cCtPDFeaController() : cCtController()
{
    mGravity = gGravity;
}

cCtPDFeaController::~cCtPDFeaController() {}

void cCtPDFeaController::Reset()
{
    cCtController::Reset();
    mPDCtrl.Reset();
}

void cCtPDFeaController::Clear()
{
    cCtController::Clear();
    mPDCtrl.Clear();
}

void cCtPDFeaController::SetEnableSolvePDTargetTest(bool v)
{
    cCtPDController::SetEnableSolvePDTargetTest(v);
    mPDCtrl.SetEnableSolvePDTargetTest(v);
}
void cCtPDFeaController::SetGravity(const tVector &gravity)
{
    mGravity = gravity;
}

std::string cCtPDFeaController::GetName() const { return "ct_pd"; }

void cCtPDFeaController::SetupPDControllers(const Json::Value &json,
                                            const tVector &gravity)
{

    Eigen::MatrixXd pd_params;
    bool succ = false;
    if (!json[gPDControllersKey].isNull())
    {
        succ = cPDController::LoadParams(json[gPDControllersKey], pd_params);
    }

    if (succ)
    {
        mPDCtrl.Init(mChar, pd_params, gravity);
    }

    mValid = succ;
    if (!mValid)
    {
        printf("Failed to initialize Ct-PD controller\n");
        mValid = false;
    }
}

bool cCtPDFeaController::ParseParams(const Json::Value &json)
{
    /*
            在这里读入了pd controller的参数　并且进行了解析
     */
    bool succ = cCtController::ParseParams(json);
    SetupPDControllers(json, mGravity);
    return succ;
}

void cCtPDFeaController::UpdateBuildTau(double time_step,
                                        Eigen::VectorXd &out_tau)
{
    // 在这里更新PD controller
    UpdatePDCtrls(time_step, out_tau);
}

void cCtPDFeaController::UpdatePDCtrls(double time_step,
                                       Eigen::VectorXd &out_tau)
{
    int num_dof = mChar->GetNumDof();
    out_tau = Eigen::VectorXd::Zero(num_dof); // 假设输出的tau是这个
    mPDCtrl.UpdateControlForce(time_step, out_tau);
}

void cCtPDFeaController::ApplyAction(const Eigen::VectorXd &action)
{
    // std::cout <<"void cCtPDController::ApplyAction " << std::endl;
    cCtController::ApplyAction(action);
    SetPDTargets(action);
}

void cCtPDFeaController::BuildJointActionBounds(int joint_id,
                                                Eigen::VectorXd &out_min,
                                                Eigen::VectorXd &out_max) const
{
    const Eigen::MatrixXd &joint_mat = mChar->GetJointMat();
    cCtCtrlUtil::BuildBoundsPD(joint_mat, joint_id, out_min, out_max);
}

void cCtPDFeaController::BuildJointActionOffsetScale(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    // std::cout <<"void cCtPDController::BuildJointActionOffsetScale(int
    // joint_id, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const"
    // <<std::endl;
    const Eigen::MatrixXd &joint_mat = mChar->GetJointMat();
    cCtCtrlUtil::BuildOffsetScalePD(joint_mat, joint_id, out_offset, out_scale);
}

/**
 * \brief				convert PD target pose to action which
 * is truly used in training
 * \param				the ref of given PD target. It will be
 * revised in this function!
 *
 * The only difference between action and PD target is their representation.
 * For ball joints, their pd targets are quaternion; but their actions are axis
 * angle We need to convert it in Inverse Dynamics
 */
void cCtPDFeaController::CalcActionByTargetPose(tVectorXd &pd_target)
{
    assert(pd_target.size() == this->GetActionSize());
    int root_id = mChar->GetRootID();
    int root_size = mChar->GetParamSize(root_id);
    int num_joints = mChar->GetNumJoints();
    int ctrl_offset = GetActionCtrlOffset();

    for (int j = root_id + 1; j < num_joints; ++j)
    {
        if (mPDCtrl.IsValidPDCtrl(j))
        {
            int retarget_joint = RetargetJointID(j);
            // for this joint
            int param_offset = mChar->GetParamOffset(retarget_joint);
            int param_size = mChar->GetParamSize(retarget_joint);

            param_offset -= root_size;
            param_offset += ctrl_offset;

            // convert quaternion to axis angle
            Eigen::VectorXd theta = pd_target.segment(param_offset, param_size);
            // std::cout <<"for joint " << j << " tar pose = " <<
            // theta.transpose() << std::endl;
            ConvertTargetPoseToAction(j, theta);
            // std::cout <<"for joint " << j << " action = " <<
            // theta.transpose() << std::endl;
            pd_target.segment(param_offset, param_size) = theta;
        }
    }
}

void cCtPDFeaController::ConvertActionToTargetPose(
    int joint_id, Eigen::VectorXd &out_theta) const
{
#if defined(ENABLE_PD_SPHERE_AXIS)
    cKinTree::eJointType joint_type = GetJointType(joint_id);
    if (joint_type == cKinTree::eJointTypeSpherical)
    {
        // for ball joint, out_theta = [angle(rad), ax, ay, az], it is
        // action, 轴角
        double rot_theta = out_theta[0];
        tVector axis = tVector(out_theta[1], out_theta[2], out_theta[3], 0);
        if (axis.squaredNorm() == 0)
        {
            axis[2] = 1;
        }

        axis.normalize();
        tQuaternion quat =
            cMathUtil::AxisAngleToQuaternion(axis, rot_theta); // 四元数

        if (FlipStance())
        {
            cKinTree::eJointType joint_type = GetJointType(joint_id);
            if (joint_type == cKinTree::eJointTypeSpherical)
            {
                quat = cMathUtil::MirrorQuaternion(quat, cMathUtil::eAxisZ);
            }
        }
        out_theta = cMathUtil::QuatToVec(quat);
    }
#endif
}

void cCtPDFeaController::ConvertTargetPoseToAction(
    int joint_id, Eigen::VectorXd &out_theta) const
{
#if defined(ENABLE_PD_SPHERE_AXIS)
    cKinTree::eJointType joint_type = GetJointType(joint_id);
    if (joint_type == cKinTree::eJointTypeSpherical)
    {
        // raw input quaternion = [x, y, z, w]
        // tQuaternion quater = tQuaternion(out_theta[3], out_theta[0],
        // out_theta[1], out_theta[2]);

        // 2020/05/12 revised by Xudong: now input quaternion should be [w,
        // x, y, z]
        tQuaternion quater = cMathUtil::VecToQuat(out_theta);

        quater.normalize();
        tVector axis_angle =
            cMathUtil::QuaternionToAxisAngle(quater); //[theta, ax, ay, az]
        out_theta[0] = axis_angle.norm();
        axis_angle.normalize();
        out_theta[1] = axis_angle[0];
        out_theta[2] = axis_angle[1];
        out_theta[3] = axis_angle[2];
    }
#endif
}

cKinTree::eJointType cCtPDFeaController::GetJointType(int joint_id) const
{
    const cPDController &ctrl = mPDCtrl.GetPDCtrl(joint_id);
    const cSimBodyJoint &joint = ctrl.GetJoint();
    cKinTree::eJointType joint_type = joint.GetType();
    return joint_type;
}

const tVectorXd &cCtPDFeaController::GetCurAction() const { return mCurAction; }

const tVectorXd &cCtPDFeaController::GetCurPDTargetPose() const
{
    return mCurPDTargetPose;
}

void cCtPDFeaController::SetPDTargets(const Eigen::VectorXd &targets)
{
    // std::cout << "[log] cCtPDController::SetPDTargets cur action = " <<
    // targets.transpose() << std::endl;
    mCurAction = targets;
    int root_id = mChar->GetRootID();
    int root_size = mChar->GetParamSize(root_id);
    int num_joints = mChar->GetNumJoints();
    int ctrl_offset = GetActionCtrlOffset();

    mCurPDTargetPose = tVectorXd::Zero(GetActionSize());
    for (int j = root_id + 1; j < num_joints; ++j)
    {
        //　对于后面的每一个joint，都来逐个设置其theta到真正的controller
        //(ImplicitPD::mPDCtrl中)
        // 这个cCtPDController又只是一层包装而已
        // std::cout << "void cCtPDController::SetPDTargets(const
        // Eigen::VectorXd& targets) set target theta" << std::endl;
        if (mPDCtrl.IsValidPDCtrl(j))
        {
            int retarget_joint = RetargetJointID(j);
            // 对于当前这个joint
            int param_offset = mChar->GetParamOffset(retarget_joint);
            int param_size = mChar->GetParamSize(retarget_joint);

            param_offset -= root_size;
            param_offset += ctrl_offset;
            // 这里调用了Eigen的segment方法，把他切开了：得到offset起始的segment个元素
            // 这里输入的targets其实是action; action中spherical
            // joint是axis angle，我们现在要把所有aa转化为四元数
            // 四元数才是真正的PD target
            Eigen::VectorXd theta = targets.segment(param_offset, param_size);
            ConvertActionToTargetPose(j, theta);
            mCurPDTargetPose.segment(param_offset, param_size) = theta;
            mPDCtrl.SetTargetTheta(j, theta); // 设置targettheta
        }
    }
    // std::cout <<"[log] cCtPDController::SetPDTargets cur pd target pose = "
    // << mCurPDTarget.transpose() << std::endl;
    // CalcActionByTargetPose(mCurPDTarget);
    // std::cout <<"[log] cCtPDController::SetPDTargets restored action = " <<
    // mCurPDTarget.transpose() << std::endl; exit(1);
}

// void cCtPDFeaController::CalcPDTarget(const Eigen::VectorXd &torque_,
//                                       Eigen::VectorXd out_pd_target)
// {
//     // ApplyAction(out_pd_target);
//     int pd_target_size = GetActionSize();
//     out_pd_target = Eigen::VectorXd::Zero(pd_target_size);
//     /*
//             in PD target:
//             spherical - 4 (axis angle)
//             revolute - 1 (joint angle)

//             target = Kp
//     */

//     for (int i = 0; i < mChar->GetNumJoints(); i++)
//     {
//         // 1. get torque
//         tVector tau = tVector::Zero();
//         tau = torque_.segment(i * 4, 4);

//         //// 2.
//         // auto x = mChar->GetJoint(i).GetType();
//         // switch (x)
//         //{
//         //	case cKinTree::eJointType::eJointTypeRevolute: tau =
//         //}
//     }
// }

void cCtPDFeaController::CalcPDTargetByTorque(double dt, const tVectorXd &pose,
                                              const tVectorXd &vel,
                                              const tVectorXd &torque,
                                              tVectorXd &pd_target)
{
    mPDCtrl.SolvePDTargetByTorque(dt, pose, vel, torque, pd_target);
    if (mChar->HasFloatingBase())
        pd_target = pd_target.segment(7, pd_target.size() - 7);

}