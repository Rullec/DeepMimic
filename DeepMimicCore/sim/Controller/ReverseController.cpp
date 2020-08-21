#include "ReverseController.h"
#include "util/LogUtil.h"
#include <fstream>
#include <iostream>
#include <util/TimeUtil.hpp>
// #include <windows.h>
// #define OUTPUT_LOG
void removeRow(Eigen::MatrixXd &matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd &matrix, unsigned int colToRemove);
void removeRow(tVectorXd &vec, unsigned int rowToRemove);

extern std::string controller_details_path;

cReverseController::cReverseController(cSimCharacterBase *sim_char)
{
    mChar = dynamic_cast<cSimCharacter *>(sim_char);
    MIMIC_ASSERT(mChar != nullptr);
    // std::ofstream fout("logs/controller_logs/pd_target_debug.log");
    // fout << std::endl;
    I.resize(mChar->GetNumDof(), mChar->GetNumDof());
    I.setIdentity();
    mEnableSolving = false;
    mEnableFastSolving = true;

    if (mEnableFastSolving)
    {
        BuildTopo();
    }
}

void cReverseController::CalcPDTarget(const tVectorXd &input_torque,
                                      const tVectorXd &input_pose,
                                      const tVectorXd &input_cur_vel,
                                      tVectorXd &output_pd_target)
{

    // 1. calculate pose_next = input_pose + input_cur_vel
    tVectorXd pose_next;
    {
        tVectorXd pose_vel_quaternion;
        cKinTree::VelToPoseDiff(mChar->GetJointMat(), input_pose, input_cur_vel,
                                pose_vel_quaternion);
        pose_next = input_pose + mTimestep * pose_vel_quaternion;
        cKinTree::PostProcessPose(mChar->GetJointMat(), pose_next);
        // std::cout << "input pose = " << input_pose.transpose() << std::endl;
        // std::cout << "next pose = " << pose_next.transpose() << std::endl;
    }

    // std::cout <<"cReverseController::CalcPDTarget pose next = " <<
    // pose_next.transpose() << std::endl;
    // 2. calculate pos diff = PD_target - pose_next
    tVectorXd pose_diff;
    // get all joints' pose diff from torque, except root joint.
    // there is no active for ce on root joint, so resolve the root joint's
    // pose_diff is impossible
    CalcPoseDiffFromTorque(input_torque, input_pose, input_cur_vel, pose_diff);
#ifdef OUTPUT_LOG
    std::ofstream fout(controller_details_path, std::ios::app);
#endif
    // std::cout << "pose diff = " << pose_diff.transpose() << std::endl;
    // 3. integrate the pos diff then get the PD target, PD target = pose_next +
    // diff
    {
        int num_joints = mChar->GetNumJoints();
        int num_dof = mChar->GetNumDof();
        const Eigen::MatrixXd &joint_mat = mChar->GetJointMat();
        assert(pose_diff.size() == num_dof);
        assert(pose_next.size() == num_dof);

        // clear all.
        output_pd_target.resize(num_dof);
        output_pd_target.setZero();

        for (int id = 1; id < num_joints; id++)
        {
            // for each joint except root, calculate pose_next = pose_cur +
            // pose_diff
            int param_offset = cKinTree::GetParamOffset(joint_mat, id);
            int param_size = cKinTree::GetParamSize(joint_mat, id);

            cKinTree::eJointType type = cKinTree::GetJointType(joint_mat, id);
            switch (type)
            {
            case cKinTree::eJointType::eJointTypeSpherical:
            {
                tQuaternion q1 = cMathUtil::VecToQuat(
                    pose_next.segment(param_offset, param_size));
                tQuaternion q_diff = cMathUtil::AxisAngleToQuaternion(
                    pose_diff.segment(param_offset, param_size));
                tQuaternion q2 = q1 * q_diff;

                if (q2.coeffs()[3] < 0)
                    q2 = cMathUtil::MinusQuaternion(q2);
                assert(q2.coeffs()[3] > 0);
                output_pd_target.segment(param_offset, param_size) =
                    cMathUtil::QuatToVec(q2);
                break;
            }
            default:
                output_pd_target.segment(param_offset, param_size) =
                    pose_next.segment(param_offset, param_size) +
                    pose_diff.segment(param_offset, param_size);
                break;
            }
#ifdef OUTPUT_LOG
            fout << " joint " << id << " pd component = "
                 << output_pd_target.segment(param_offset, param_size)
                        .transpose()
                 << std::endl;
#endif
        }
    }
    // std::cout <<"now pd target = " << output_pd_target.transpose() <<
    // std::endl;
}

/*
        @Function: CalcPoseDiffFromTorque
                this function will calculate the "pose_err" in CalcControlForces
   by the control torque
        @params: input_torque, the torque we applied to this character, as the
   output of CalcControlForces
        @params: input_pose, the pose when we try to calculate torque in
   CalcControlForces
        @params: input_vel, the pose velocity we get the same as above
        @params: output_pos_diff, the pose diff between PD target and the
   reference pose, NOT THE INPUT POSE! ATTENTION FOR THE FINAL OUTPUT

*/
void cReverseController::CalcPoseDiffFromTorque(const tVectorXd &input_torque,
                                                const tVectorXd &input_pose,
                                                const tVectorXd &input_vel,
                                                tVectorXd &output_pos_diff)
{
    if (false == mEnableSolving)
    {
        std::cout << "[error] cReverseController::CalcPDTarget didn't "
                     "prepare well, can not solve PD\n";
        exit(1);
    }

    // \tau = kp * (q_target - q_ref) + kd * (-timestep * q_accel)
    // 1. calculate q_ref = q_cur + timestep * q_vel
    const Eigen::VectorXd &cur_pose =
        input_pose; // get current character's pose, quaternions for each joint
    const Eigen::VectorXd &cur_vel =
        input_vel; // get char's vel: vel = d(pose)/ dt, the differential
                   // quaternions for each joints

    tVectorXd pose_ref;
    {
        Eigen::VectorXd quadternion_dot;
        const Eigen::MatrixXd &joint_mat = mChar->GetJointMat();
        // use pose & vel to calculate "pose_inc"
        cKinTree::VelToPoseDiff(joint_mat, cur_pose, cur_vel,
                                quadternion_dot); // pose_inc = dqdt
        pose_ref = cur_pose +
                   mTimestep * quadternion_dot; // pose_cur + timestep * dqdt =
                                                // pose_next (predicted)
        cKinTree::PostProcessPose(joint_mat, pose_ref);
    }

    // 2. calulte A and b

    // M_s_inv = (M + mTimestep * Kd_dense).ldlt().solve(I);
    // std::ofstream fout_("tmp.txt");
    // fout_ <<"Ms = \n" << (M + mTimestep * Kd_dense).inverse() << std::endl;
    // M_s_inv = (M + mTimestep * Kd_dense).ldlt().solve(I);
    M_s_inv = (M + mTimestep * Kd_dense).inverse();
    A = M_s_inv * Kp_mat;
    b = -M_s_inv * (Kd_mat * cur_vel + C);

    // 3. calculate E and f
    E = Kp_dense - mTimestep * Kd_mat * A;
    // std::cout <<"input torque size = " << input_torque.size() << std::endl;
    // std::cout <<"res size = " << (Kd_mat * (cur_vel + mTimestep * b)).size()
    // << std::endl;
    f = input_torque + Kd_mat * (cur_vel + mTimestep * b);

    // 4. arrange the matrix and solve final target
    // it can be more efficient if we delete some "zero" dofs, for example the
    // first 7 nums  and each 4th number for spherial joints cTimeUtil::Begin();
    // 	output_pos_diff = E.ldlt().solve(f);
    // cTimeUtil::End();
    // cTimeUtil::Begin();
    // 	auto output_pos_diff_new = FastSolve(E, f);
    // cTimeUtil::End();
    output_pos_diff = FastSolve(E, f);
    // std::cout <<"solve diff = " << (output_pos_diff -
    // output_pos_diff_new).norm() << std::endl; std::cout <<"new res
    // compatibility = " << (E * output_pos_diff_new - f).norm() << std::endl;
    // std::cout <<"old res compatibility = " << (E * output_pos_diff -
    // f).norm() << std::endl; exit(1);
    tVectorXd err = E * output_pos_diff - f;
    if (err.norm() > 1e-6)
    {
        std::cout << "[error] cReverseController::CalcPoseDiffFromTorque "
                     "solved err norm = "
                  << err.norm() << " = " << err.transpose();
        exit(1);
    }

    bool solved_error = false;
    auto diff = E * output_pos_diff - f;
    if (diff.norm() > 1e-6)
    {
        std::cout << "cReverseController::CalcPoseDiffFromTorque solved "
                     "output pose diff error\n";
        std::cout << "E size = " << E.rows() << " " << E.cols() << std::endl;
        std::cout << "f size = " << f.rows() << " " << f.cols() << std::endl;
        std::cout << "diff = " << diff.transpose() << std::endl;
        std::cout << "E = \n" << E << std::endl;
        solved_error = true;
    }

    // for debug purpose, calculate "acc"
    // Eigen::VectorXd acc = Kp_mat * pose_err + Kd_mat * vel_err - C;
    auto acc = Kp_mat * output_pos_diff + Kd_mat * (-cur_vel) - C;

#ifdef OUTPUT_LOG
    std::cout << "verbose log attention! to " << controller_details_path
              << std::endl;
    std::ofstream fout(controller_details_path, std::ios::app);
    fout << "-----------------------------\n";
    fout << "Kp = \n" << Kp_mat.toDenseMatrix() << std::endl;
    fout << "Kd = \n" << Kd_mat.toDenseMatrix() << std::endl;
    fout << "cur pose = " << cur_pose.transpose() << std::endl;
    fout << "next pose = " << pose_ref.transpose() << std::endl;
    fout << "cur vel = " << cur_vel.transpose() << std::endl;
    fout << "input torque = " << input_torque.transpose() << std::endl;
    fout << "input torque norm = " << input_torque.norm() << std::endl;
    fout << "Q = " << acc.transpose() << std::endl;
    fout << "A = \n" << A << std::endl;
    fout << "b = " << b.transpose() << std::endl;
    fout << "E = \n" << E << std::endl;
    fout << "f = " << f.transpose() << std::endl;
    fout << "E_sub = \n" << E_sub << std::endl;
    fout << "f_sub = " << f_sub.transpose() << std::endl;
    fout << "Err = " << output_pos_diff.transpose() << std::endl;
    fout << "M = \n" << M << std::endl;
    fout << "C = \n" << C << std::endl;
    if (solved_error)
        fout << "solve error = " << diff.transpose() << std::endl;
        // exit(1);
#endif

    mEnableSolving = false;
}

void cReverseController::CalcAction(const tVectorXd &input_torque,
                                    const tVector &input_cur_pose,
                                    const tVectorXd &input_cur_vel,
                                    tVector &output_action)
{
    tVectorXd pd_target;
    CalcPDTarget(input_torque, input_cur_pose, input_cur_vel, pd_target);
}

void cReverseController::SetParams(double timestep_, const Eigen::MatrixXd &M_,
                                   const Eigen::MatrixXd &C_,
                                   const Eigen::VectorXd &kp_,
                                   const Eigen::VectorXd &kd_)
{
    mTimestep = timestep_;
    M = M_;
    C = C_;

    Kp_mat = kp_.asDiagonal();
    Kp_dense = Kp_mat.toDenseMatrix();
    Kd_mat = kd_.asDiagonal();
    Kd_dense = Kd_mat.toDenseMatrix();
    mEnableSolving = true;
}

/*
        @Function: cReverseController::BuildTopo
        These equations "acc = M.ldlt().solve(acc)" are sparse according to the
   structure of multibody, We can extract the non-zero rows and cols for fasting
   solve. And this function will get the Topo of these sparse matrix.
*/
void cReverseController::BuildTopo()
{

    /* 1. RawIndexList and NewIndexList are 2 list of int pairs.
            They are 'corosponding to' each other.
            For example, RawIndexList = {<7, 9>, <10, 10>}
            and NewINdexList = {<0, 2>, <3, 3>}
            means that ,the [0-2, 0-2] 3x3 block in new mat
                            should be set to
                                            [7-9, 7-9] 3x3 block in raw mat
    */
    assert(mChar != nullptr);
    mRawIndexLst.clear(), mNewIndexLst.clear();

    int raw_index_offset = 0, new_index_offset = 0;
    int new_mat_size = 0;
    std::shared_ptr<cMultiBody> multibody = mChar->GetMultiBody();
    if (multibody->hasFixedBase() == false)
        raw_index_offset += 7;
    for (int id = 0; id < multibody->getNumLinks(); id++)
    {
        auto cur_type = multibody->getLink(id).m_jointType;
        switch (cur_type)
        {
        case btMultibodyLink::eFeatherstoneJointType::eSpherical:
        {
            mRawIndexLst.push_back(
                idx_pair(raw_index_offset, raw_index_offset + 2));
            mNewIndexLst.push_back(
                idx_pair(new_index_offset, new_index_offset + 2));
            new_index_offset += 3, raw_index_offset += 4;
            new_mat_size += 3;
            break;
        }
        case btMultibodyLink::eFeatherstoneJointType::eRevolute:
        {
            mRawIndexLst.push_back(
                idx_pair(raw_index_offset, raw_index_offset));
            mNewIndexLst.push_back(
                idx_pair(new_index_offset, new_index_offset));
            new_index_offset += 1, raw_index_offset += 1;
            new_mat_size += 1;
        }
        case btMultibodyLink::eFeatherstoneJointType::eFixed:
        {
            continue;
        }
        default:
            std::cout << "cReverseController::BuildTopo unsupported "
                         "joint type "
                      << cur_type << std::endl;
            exit(1);
        }
    }

    // for(int i=0; i<mRawIndexLst.size(); i++)
    // {
    // 	auto raw_idx = mRawIndexLst[i], new_idx = mNewIndexLst[i];
    // 	std::cout <<"[log] topo " << i <<" from raw idx " << raw_idx.first <<" "
    // << raw_idx.second
    // 		<<" to new idx " << new_idx.first <<" " << new_idx.second <<
    // std::endl;
    // }
    // exit(1);

    // std::cout << "char dofs = " << mChar->GetNumDof() << std::endl;
    mRawMatSize = mChar->GetNumDof();
    mNewMatSize = new_mat_size; // The size of new sys mat and raw sys mat
}

enum eTransferMode
{
    squeeze = 0, // from raw_mat to new_mat
    expand,      // from new_mat to raw_mat
    MAX_MODE_NUM
} transfer_mode;

tVectorXd cReverseController::FastSolve(Eigen::MatrixXd &A, tVectorXd &b) const
{
    // std::cout << "[log] cReverseController::FastSolve \n";
    assert(A.rows() == mRawMatSize && A.cols() == mRawMatSize);
    assert(b.size() == mRawMatSize);
    if (mNewMatSize == 0)
        return tVectorXd::Zero(mRawMatSize);

    Eigen::MatrixXd new_A;
    tVectorXd new_b;
    MatrixTransfer(A, new_A, eTransferMode::squeeze);
    VectorTransfer(b, new_b, eTransferMode::squeeze);
    tVectorXd res_squeeze = new_A.inverse() * new_b, res_expand;
    VectorTransfer(res_expand, res_squeeze, eTransferMode::expand);
    // for(int i=0; i<res_expand.size(); i++)
    // {
    // 	if(res_expand.segment(i, 1).hasNaN() == true) res_expand[i] = 0;
    // }
    // assert(res_expand.hasNaN() == false);
    // std::cout <<" A = " << A << std::endl;
    // std::cout <<" res = " << res_expand.transpose() << std::endl;
    // std::cout <<" b = " << b.transpose() << std::endl;
    // std::cout << (A * res_expand - b).norm() << std::endl;
    tVectorXd solve_err = A * res_expand - b;
    if (solve_err.norm() > 1e-6)
    {
        std::cout << "[error] cReverseController::FastSolve solve err = "
                  << solve_err.norm() << " " << solve_err.transpose()
                  << std::endl;
        exit(1);
    }

    return res_expand;
    // std::ofstream fout("tmp.txt");
    // // fout <<"raw A = \n" << A << std::endl;

    // MatrixTransfer(A, new_A, eTransferMode::squeeze);	// squeeze from
    // sparse A to compact new_A
    // // fout <<"new A = \b" << new_A << std::endl;
    // tVectorXd new_b;
    // fout <<"raw b = " << b.transpose() << std::endl;
    // VectorTransfer(b, new_b, eTransferMode::squeeze);
    // fout <<"new b = " << new_b.transpose() << std::endl;
    // tVectorXd restore_b;
    // VectorTransfer(restore_b, new_b, eTransferMode::expand);
    // std::cout <<"restore b diff = " << (restore_b -  b).norm() << std::endl;
    // exit(1);
    // return tVector::Zero();
}

void cReverseController::MatrixTransfer(Eigen::MatrixXd &raw_mat,
                                        Eigen::MatrixXd &new_mat,
                                        int mode) const
{
    assert(mode < MAX_MODE_NUM && mode >= 0);

    // squeeze, from sparse raw mat to compact new mat
    // expand, from compant new mat to sparse raw mat
    if (mode == eTransferMode::squeeze)
    {
        assert(raw_mat.rows() == mRawMatSize && raw_mat.cols() == mRawMatSize);
        new_mat.resize(mNewMatSize, mNewMatSize);
        new_mat.setZero();
    }
    else
    {
        assert(new_mat.rows() == mNewMatSize && new_mat.cols() == mNewMatSize);
        raw_mat.resize(mRawMatSize, mRawMatSize);
        raw_mat.setZero();
    }

    assert(mRawIndexLst.size() == mNewIndexLst.size());
    int pair_num = mRawIndexLst.size();
    for (int row_idx = 0; row_idx < pair_num; row_idx++)
    {
        int raw_row_st = mRawIndexLst[row_idx].first,
            new_row_st = mNewIndexLst[row_idx].first;
        int row_size =
            mRawIndexLst[row_idx].second - mRawIndexLst[row_idx].first + 1;
        assert(row_size == (mNewIndexLst[row_idx].second -
                            mNewIndexLst[row_idx].first + 1));

        for (int col_idx = 0; col_idx < pair_num; col_idx++)
        {
            int raw_col_st = mRawIndexLst[col_idx].first,
                new_col_st = mNewIndexLst[col_idx].first;
            int col_size =
                mRawIndexLst[col_idx].second - mRawIndexLst[col_idx].first + 1;
            assert(col_size == (mNewIndexLst[col_idx].second -
                                mNewIndexLst[col_idx].first + 1));

            if (mode == eTransferMode::squeeze)
            {
                new_mat.block(new_row_st, new_col_st, row_size, col_size) =
                    raw_mat.block(raw_row_st, raw_col_st, row_size, col_size);
            }
            else
            {
                raw_mat.block(raw_row_st, raw_col_st, row_size, col_size) =
                    new_mat.block(new_row_st, new_col_st, row_size, col_size);
            }
        }
    }
}

void cReverseController::VectorTransfer(tVectorXd &raw_vec, tVectorXd &new_vec,
                                        int mode) const
{
    assert(mode < MAX_MODE_NUM && mode >= 0);

    // squeeze, from sparse raw mat to compact new mat
    // expand, from compant new mat to sparse raw mat
    if (mode == eTransferMode::squeeze)
    {
        assert(raw_vec.size() == mRawMatSize);
        new_vec.resize(mNewMatSize, 1);
        new_vec.setZero();
    }
    else
    {
        assert(new_vec.size() == mNewMatSize);
        raw_vec.resize(mRawMatSize, 1);
        raw_vec.setZero();
    }

    assert(mRawIndexLst.size() == mNewIndexLst.size());
    int pair_num = mRawIndexLst.size();
    for (int idx = 0; idx < pair_num; idx++)
    {
        int raw_st = mRawIndexLst[idx].first, new_st = mNewIndexLst[idx].first;
        int size = mRawIndexLst[idx].second - mRawIndexLst[idx].first + 1;
        assert(size ==
               (mNewIndexLst[idx].second - mNewIndexLst[idx].first + 1));

        if (mode == eTransferMode::squeeze)
        {
            new_vec.segment(new_st, size) = raw_vec.segment(raw_st, size);
        }
        else
        {
            raw_vec.segment(raw_st, size) = new_vec.segment(new_st, size);
        }
    }
}