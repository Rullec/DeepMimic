#include "SceneDiffImitate.h"
#include "sim/World/GenWorld.h"
#include "util/LogUtil.h"

cSceneDiffImitate::cSceneDiffImitate()
{
    // MIMIC_INFO("cSceneDiffImitate created");
    mEnableTestDRewardDAction = false;
}
cSceneDiffImitate::~cSceneDiffImitate() {}

void cSceneDiffImitate::ParseArgs(const std::shared_ptr<cArgParser> &parser)
{
    cSceneImitate::ParseArgs(parser);
    parser->ParseBoolCritic("enable_test_reward_action_derivative",
                            mEnableTestDRewardDAction);
}

/**
 * \brief           Initialize this diff imitate scene
*/
void cSceneDiffImitate::Init()
{
    cSceneImitate::Init();

    // check that all characters are gen type
    for (int i = 0; i < GetNumChars(); i++)
    {
        MIMIC_ASSERT(eSimCharacterType::Generalized ==
                     GetCharacter(i)->GetCharType());
    }
}

/**
 * \brief       Get D(Reward)/D(Action)
 * 
 * dr/da = 
 *      dr/d x_{cur} 
 *      * 
 *      d (x_cur) / d_u
 *      *
 *      d u / da
*/
tVectorXd cSceneDiffImitate::CalcDRewardDAction()
{
    if (mEnableTestDRewardDAction == true)
    {
        Test();
    }
    return CalcDrDa();
}

/**
 * \brief           Test the derivatives
*/
void cSceneDiffImitate::Test()
{
    // 1. test pose error item
    {
        // test dq1 * q0.conj / dq0
        // cMathUtil::TestCalc_Dq1q0conj_Dq0();
        // cMathUtil::TestCalc_DQuaterniontDAxisAngle();
        TestDRootRotErrDpose0();
        TestDJointPoseErrDpose0();
        TestDPoseRewardDpose0();
        TestDPoseRewardDq();
    }

    // 2. test the vel error
    {
        TestDVelRewardDvel0();
        TestDVelRewardDqdot();
    }

    // 3. test dposedq and dveldqdot
    {
        auto gen_char = GetDefaultGenChar();
        gen_char->TestCalcDposedq();
        gen_char->TestCalcDveldqdot();
    }

    // 5. test controller (inside the controller)
    // 4. total test method
    {
        TestDrDxcur();
        TestDrDa();
    }
}

/**
 * \brief           Test D(root_rot_err)/Dpose0
 * 
*/
void cSceneDiffImitate::TestDRootRotErrDpose0()
{
    auto sim_char = GetDefaultGenChar();
    sim_char->PushState("TestDRootRotErrDpose0");
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    tVectorXd pose0 = sim_char->GetPose();
    // pose0.setRandom();
    tVectorXd pose1 = kin_char->GetPose();

    double old_err = cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
    double eps = 1e-5;
    int root_id = cKinTree::GetRoot(joint_mat);
    int st = cKinTree::GetParamOffset(joint_mat, root_id),
        size = cKinTree::GetParamSize(joint_mat, root_id);
    tVectorXd ideal_dErrdpose0 =
        cKinTree::CalcDRootRotErrDPose0(joint_mat, pose0, pose1)
            .segment(cKinTree::GetParamOffset(joint_mat, root_id),
                     cKinTree::GetParamSize(joint_mat, root_id));
    tVectorXd num_dErrdpose0 = tVectorXd::Zero(size);
    for (int i = st; i < st + size; i++)
    {
        pose0[i] += eps;
        double new_err = cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
        num_dErrdpose0[i - st] = (new_err - old_err) / eps;
        pose0[i] -= eps;
    }
    tVectorXd diff = ideal_dErrdpose0 - num_dErrdpose0;
    if (diff.norm() > 10 * eps)
    {
        std::cout << "[error] TestDRootRotErrDpose0 failed\n";
        std::cout << "ideal = " << ideal_dErrdpose0.transpose() << std::endl;
        std::cout << "num = " << num_dErrdpose0.transpose() << std::endl;
        std::cout << "diff = " << diff.transpose() << std::endl;
        std::cout << "pose0 = " << pose0.transpose() << std::endl;
        std::cout << "pose1 = " << pose1.transpose() << std::endl;
        exit(0);
    }
    else
    {
        std::cout << "[log] TestDRootRotErrDpose0 succ\n";
    }
    sim_char->PopState("TestDRootRotErrDpose0");
}

/**
 * \brief           Test D(joint_pose_err)/Dpose0
*/
void cSceneDiffImitate::TestDJointPoseErrDpose0()
{
    auto sim_char = GetDefaultGenChar();
    sim_char->PushState("TestDJointPoseErrDpose0");
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    tVectorXd pose0 = sim_char->GetPose();
    // pose0.setRandom();
    tVectorXd pose1 = kin_char->GetPose();
    double eps = 1e-5;

    for (int joint_id = 1; joint_id < cKinTree::GetNumJoints(joint_mat);
         joint_id++)
    {

        double old_err =
            cKinTree::CalcPoseErr(joint_mat, joint_id, pose0, pose1);
        int st = cKinTree::GetParamOffset(joint_mat, joint_id),
            size = cKinTree::GetParamSize(joint_mat, joint_id);
        tVectorXd ideal_dErrdpose0 =
            cKinTree::CalcDPoseErrDPose0(joint_mat, joint_id, pose0, pose1)
                .segment(st, size);
        tVectorXd num_dErrdpose0 = tVectorXd::Zero(size);
        for (int i = st; i < st + size; i++)
        {
            pose0[i] += eps;
            double new_err =
                cKinTree::CalcPoseErr(joint_mat, joint_id, pose0, pose1);
            num_dErrdpose0[i - st] = (new_err - old_err) / eps;
            pose0[i] -= eps;
        }
        tVectorXd diff = ideal_dErrdpose0 - num_dErrdpose0;
        if (diff.norm() > 10 * eps)
        {
            std::cout << "[error] TestDJointPoseErrDpose0 failed\n";
            std::cout << "ideal = " << ideal_dErrdpose0.transpose()
                      << std::endl;
            std::cout << "num = " << num_dErrdpose0.transpose() << std::endl;
            std::cout << "diff = " << diff.transpose() << std::endl;
        }
        else
        {
            std::cout << "[log] TestDJointPoseErrDpose0 succ\n";
        }
    }
    sim_char->PopState("TestDJointPoseErrDpose0");
}

void cSceneDiffImitate::TestDrDxcur()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("TestDrDxcur");
    int num_of_freedom = gen_char->GetNumOfFreedom();
    gen_char->Setq(tVectorXd::Random(num_of_freedom));
    gen_char->Setqdot(tVectorXd::Random(num_of_freedom));

    tVectorXd ana_drdx = CalcDrDxcur();
    // std::cout << "ana drdxcur = " << ana_drdx.transpose() << std::endl;
    double old_r = CalcReward(0);

    // check for drdq
    tVectorXd q = gen_char->Getq();
    tVectorXd qdot = gen_char->Getqdot();
    double eps = 1e-5;
    {
        for (int i = 0; i < num_of_freedom; i++)
        {
            q[i] += eps;
            gen_char->Setq(q);
            double new_r = CalcReward(0);
            double num_deri = (new_r - old_r) / eps;
            double ana_deri = ana_drdx[i];
            if (std::fabs(num_deri - ana_deri) > 10 * eps)
            {
                std::cout << "[error] test drdxur q failed for idx " << i
                          << std::endl;
                std::cout << "num = " << num_deri << std::endl;
                std::cout << "ideal = " << ana_deri << std::endl;
                exit(0);
            }
            q[i] -= eps;
        }
    }
    gen_char->Setq(q);

    // check for drdqdot
    {
        for (int i = 0; i < num_of_freedom; i++)
        {
            qdot[i] += eps;
            gen_char->Setqdot(qdot);
            double new_r = CalcReward(0);
            double num_deri = (new_r - old_r) / eps;
            double ana_deri = ana_drdx[i + num_of_freedom];
            if (std::fabs(num_deri - ana_deri) > 10 * eps)
            {
                std::cout << "[error] test drdxur qdot failed for idx " << i
                          << std::endl;
                std::cout << "num = " << num_deri << std::endl;
                std::cout << "ideal = " << ana_deri << std::endl;
                exit(0);
            }
            qdot[i] -= eps;
        }
    }
    gen_char->Setqdot(qdot);
    gen_char->PopState("TestDrDxcur");
    std::cout << "[log] TestDrDxcur succ = " << ana_drdx.transpose()
              << std::endl;
}

/**
 * \brief       Get D(reward) / D x_cur
*/
tVectorXd cSceneDiffImitate::CalcDrDxcur()
{
    auto gen_char = GetDefaultGenChar();
    int dof = gen_char->GetNumOfFreedom();
    tVectorXd drdx = tVectorXd::Zero(dof * 2);

    // if the char doesn't fall, we calc the derivatives
    // otherwise the deriv is zero
    if (HasFallen(*(gen_char.get())) == false)
    {
        tVectorXd drdq = CalcDPoseRewardDq(), drdqdot = CalcDVelRewardDqdot();
        MIMIC_ASSERT(drdq.size() == dof && drdqdot.size() == dof);
        drdx.segment(0, dof) = drdq;
        drdx.segment(dof, dof) = drdqdot;
    }
    return drdx;
}

std::shared_ptr<cSimCharacterGen> cSceneDiffImitate::GetDefaultGenChar()
{
    return std::dynamic_pointer_cast<cSimCharacterGen>(GetCharacter(0));
}
/**
 * \brief           Calc D(pose_reward)/Dpose0
 * pose_err = w1 * root_err + \sum_j wj * joint_pose_err
 * pose_reward = e^{-err_scale * pose_scale * pose_err}
*/
tVectorXd cSceneDiffImitate::CalcDPoseRewardDpose0()
{
    // 1. get the weight & scale
    double err_scale = RewParams.err_scale, pose_scale = RewParams.pose_scale;
    double pose_weight = RewParams.pose_w;

    auto gen_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const tMatrixXd joint_mat = gen_char->GetJointMat();
    const tVectorXd &pose0 = gen_char->GetPose();
    const tVectorXd &pose1 = kin_char->GetPose();

    double total_err = 0;
    tVectorXd DerrDPose0 = tVectorXd::Zero(gen_char->GetPose().size());
    // 2. handle  joints
    for (int j_id = 0; j_id < cKinTree::GetNumJoints(joint_mat); j_id++)
    {
        double weight = mJointWeights[j_id];
        if (cKinTree::IsRoot(joint_mat, j_id))
        {
            total_err +=
                weight * cKinTree::CalcRootRotErr(joint_mat, pose0, pose1);
            DerrDPose0 += weight * cKinTree::CalcDRootRotErrDPose0(
                                       joint_mat, pose0, pose1);
        }
        else
        {
            total_err +=
                weight * cKinTree::CalcPoseErr(joint_mat, j_id, pose0, pose1);
            DerrDPose0 += weight * cKinTree::CalcDPoseErrDPose0(joint_mat, j_id,
                                                                pose0, pose1);
        }
    }
    // std::cout << "total err = " << total_err << " total rew = "
    //           << pose_weight * std::exp(-err_scale * pose_scale * total_err)
    //           << std::endl;
    // 3. get DrewardDerr
    double DrewardDerr = -pose_weight * err_scale * pose_scale *
                         std::exp(-err_scale * pose_scale * total_err);
    tVectorXd DrewardDpose0 = DrewardDerr * DerrDPose0;
    BTGEN_ASSERT(DrewardDpose0.hasNaN() == false);
    return DrewardDpose0;
}

/**
 * \brief               Test d(Reward)/dpose
*/
void cSceneDiffImitate::TestDPoseRewardDpose0()
{
    auto &sim_char = *GetDefaultGenChar().get();
    sim_char.PushState("TestDPoseRewardDpose0");
    auto &kin_char = *GetKinChar().get();
    tVectorXd pose0 = sim_char.GetPose(), pose1 = kin_char.GetPose();
    tVectorXd old_pose = pose0;
    // pose0.setRandom();
    cKinTree::PostProcessPose(sim_char.GetJointMat(), pose0);
    sim_char.SetPose(pose0);

    // old reward
    double old_rew = CalcPoseReward(sim_char, kin_char);
    // std::cout << "legacy old rew = " << old_rew << std::endl;
    tVectorXd ideal_dRdpose0 = CalcDPoseRewardDpose0();
    double eps = 1e-5;
    for (int i = 0; i < pose0.size(); i++)
    {
        pose0[i] += eps;
        sim_char.SetPose(pose0);
        double new_rew = CalcPoseReward(sim_char, kin_char);
        double num_dRdpose0 = (new_rew - old_rew) / eps;
        double diff = std::fabs(num_dRdpose0 - ideal_dRdpose0[i]);
        // MIMIC_INFO("num {}, ideal {}, diff {}", num_dRdpose0, ideal_dRdpose0[i],
        //            diff);
        if (diff > 10 * eps)
        {
            MIMIC_ERROR("num {}, ideal {}, diff {}", num_dRdpose0,
                        ideal_dRdpose0[i], diff);
        }
        pose0[i] -= eps;
    }

    // restore
    sim_char.SetPose(old_pose);
    std::cout << "[log] TestDPoseRewardDpose0 succ\n";
    sim_char.PopState("TestDPoseRewardDpose0");
}

/**
 * \brief           Test d(VelReward) / vel0
*/
void cSceneDiffImitate::TestDVelRewardDvel0()
{
    auto &sim_char = *GetDefaultGenChar().get();
    sim_char.PushState("TestDVelRewardDvel0");
    auto &kin_char = *GetKinChar().get();
    tVectorXd vel0 = sim_char.GetVel(), vel1 = kin_char.GetVel();
    tVectorXd old_vel = vel0;
    vel0.setRandom();
    sim_char.SetVel(vel0);

    // old reward
    double old_rew = CalcVelReward(sim_char, kin_char);
    // std::cout << "legacy old rew = " << old_rew << std::endl;
    tVectorXd ideal_dRdvel0 = CalcDVelRewardDvel0();
    double eps = 1e-5;
    for (int i = 0; i < vel0.size(); i++)
    {
        vel0[i] += eps;
        sim_char.SetVel(vel0);
        double new_rew = CalcVelReward(sim_char, kin_char);
        double num_dRdvel0 = (new_rew - old_rew) / eps;
        double diff = std::fabs(num_dRdvel0 - ideal_dRdvel0[i]);
        // MIMIC_INFO("num {}, ideal {}, diff {}", num_dRdpose0, ideal_dRdpose0[i],
        //            diff);
        if (diff > 10 * eps)
        {
            MIMIC_ERROR("idx {}, num {}, ideal {}, diff {}", i, num_dRdvel0,
                        ideal_dRdvel0[i], diff);
        }
        vel0[i] -= eps;
    }

    // restore
    sim_char.SetVel(old_vel);
    std::cout << "[log] TestDVelRewardDvel0 succ\n";
    sim_char.PopState("TestDVelRewardDvel0");
}

/**
 * \brief           Calc d(Vel Reward) / dvel0
 * 
 * drdvel0 = dr/dvel_err * dvel_err / dvel0
*/
tVectorXd cSceneDiffImitate::CalcDVelRewardDvel0()
{
    // 1. calc dr/dvel_err
    double err_scale = RewParams.err_scale, vel_scale = RewParams.vel_scale;
    double vel_weight = RewParams.vel_w;

    auto gen_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const tMatrixXd joint_mat = gen_char->GetJointMat();
    const tVectorXd &vel0 = gen_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();

    double total_err = 0;
    tVectorXd DerrDvel0 = tVectorXd::Zero(gen_char->GetVel().size());
    // 2. handle  joints
    for (int j_id = 0; j_id < cKinTree::GetNumJoints(joint_mat); j_id++)
    {
        double weight = mJointWeights[j_id];
        int st = cKinTree::GetParamOffset(joint_mat, j_id),
            size = cKinTree::GetParamSize(joint_mat, j_id);
        if (cKinTree::IsRoot(joint_mat, j_id))
        {
            total_err +=
                weight * cKinTree::CalcRootAngVelErr(joint_mat, vel0, vel1);
            DerrDvel0.segment(st, size) +=
                weight *
                cKinTree::CalcDRootRotVelErrDVel0(joint_mat, vel0, vel1)
                    .segment(st, size);
        }
        else
        {
            total_err +=
                weight * cKinTree::CalcVelErr(joint_mat, j_id, vel0, vel1);
            DerrDvel0.segment(st, size) +=
                -weight * 2 * (vel1 - vel0).segment(st, size);
        }
    }
    // std::cout << "total err = " << total_err << " total rew = "
    //           << vel_weight * std::exp(-err_scale * vel_scale * total_err)
    //           << std::endl;
    // 3. get DrewardDerr
    double DrewardDerr = -vel_weight * err_scale * vel_scale *
                         std::exp(-err_scale * vel_scale * total_err);
    tVectorXd DrewardDvel0 = DrewardDerr * DerrDvel0;
    BTGEN_ASSERT(DrewardDvel0.hasNaN() == false);
    return DrewardDvel0;
}

/**
 * \brief               only calculate the pose reward and vel reward
*/
double cSceneDiffImitate::CalcRewardImitate(cSimCharacterBase &sim_char,
                                            cKinCharacter &ref_char) const
{
    // return cSceneImitate::CalcRewardImitate(sim_char, ref_char);
    auto &gen_char = *dynamic_cast<cSimCharacterGen *>(&sim_char);
    double pose_rew = CalcPoseReward(gen_char, ref_char),
           vel_rew = CalcVelReward(gen_char, ref_char);
    // std::cout << "pose rew = " << pose_rew << std::endl;
    // std::cout << "vel rew = " << vel_rew << std::endl;
    // cMathUtil::TestCalc_DQuaterion_DEulerAngles();

    // exit(0);
    double total_rew = pose_rew + vel_rew;
    // printf("[debug] pose rew %.5f, vel rew %.5f, total rew %.5f\n", pose_rew,
    //        vel_rew, total_rew);
    return total_rew;
}

/**
 * \brief               Calculate d(pose_reward)/dq
*/
tMatrixXd cSceneDiffImitate::CalcDPoseRewardDq()
{
    tVectorXd drdq = CalcDPoseRewardDpose0();
    auto gen_char = GetDefaultGenChar();
    drdq = drdq.transpose() * gen_char->CalcDposedq(gen_char->Getq());
    // std::cout << "drdq = " << drdq.transpose() << "q norm = " << gen_char->Getq().norm();
    return drdq;
}

/**
 * \brief               Test d(pose_reward)/dq
*/
void cSceneDiffImitate::TestDPoseRewardDq()
{
    auto &sim_char = *GetDefaultGenChar().get();
    sim_char.PushState("TestDPoseRewardDq");
    auto &kin_char = *GetKinChar().get();
    tVectorXd q = sim_char.Getq();
    tVectorXd old_q = q;
    // q.setRandom();
    sim_char.Setq(q);

    // old reward
    double old_rew = CalcPoseReward(sim_char, kin_char);
    // std::cout << "legacy old rew = " << old_rew << std::endl;
    tVectorXd ideal_dRdq = CalcDPoseRewardDq();
    double eps = 1e-5;
    for (int i = 0; i < q.size(); i++)
    {
        q[i] += eps;
        sim_char.Setq(q);
        double new_rew = CalcPoseReward(sim_char, kin_char);
        double num_dRdq = (new_rew - old_rew) / eps;
        double diff = std::fabs(num_dRdq - ideal_dRdq[i]);
        // MIMIC_INFO("num {}, ideal {}, diff {}", num_dRdpose0, ideal_dRdpose0[i],
        //            diff);
        if (diff > 10 * eps)
        {
            MIMIC_ERROR("num {}, ideal {}, diff {}", num_dRdq, ideal_dRdq[i],
                        diff);
        }
        q[i] -= eps;
    }

    // restore
    sim_char.Setq(old_q);
    std::cout << "[log] TestDPoseRewardDq succ = " << ideal_dRdq.transpose()
              << std::endl;
    sim_char.PopState("TestDPoseRewardDq");
}

/**
 * \brief           Calculate d(vel_reward)/dqdot
*/
tVectorXd cSceneDiffImitate::CalcDVelRewardDqdot()
{
    auto gen_char = GetDefaultGenChar();
    tVectorXd dVelReward_dqdot = CalcDVelRewardDvel0();
    dVelReward_dqdot = dVelReward_dqdot.transpose() *
                       gen_char->CalcDveldqdot(gen_char->Getqdot());
    return dVelReward_dqdot;
}

/**
 * \brief           test d(vel_reward)/dqdot
*/
void cSceneDiffImitate::TestDVelRewardDqdot()
{
    auto &sim_char = *GetDefaultGenChar().get();
    sim_char.PushState("TestDVelRewardDqdot");
    auto &kin_char = *GetKinChar().get();
    tVectorXd qdot = sim_char.Getqdot();
    tVectorXd old_qdot = qdot;
    qdot.setRandom();
    sim_char.Setqdot(qdot);

    // old reward
    double old_rew = CalcVelReward(sim_char, kin_char);
    // std::cout << "legacy old rew = " << old_rew << std::endl;
    tVectorXd ideal_dRdqdot = CalcDVelRewardDqdot();
    double eps = 1e-5;
    for (int i = 0; i < qdot.size(); i++)
    {
        qdot[i] += eps;
        sim_char.Setqdot(qdot);
        double new_rew = CalcVelReward(sim_char, kin_char);
        double num_dRdqdot = (new_rew - old_rew) / eps;
        double diff = std::fabs(num_dRdqdot - ideal_dRdqdot[i]);
        // MIMIC_INFO("num {}, ideal {}, diff {}", num_dRdpose0, ideal_dRdpose0[i],
        //            diff);
        if (diff > 10 * eps)
        {
            MIMIC_ERROR("num {}, ideal {}, diff {}", num_dRdqdot,
                        ideal_dRdqdot[i], diff);
        }
        qdot[i] -= eps;
    }

    // restore
    sim_char.Setqdot(old_qdot);
    std::cout << "[log] TestDVelRewardDqdot succ = "
              << ideal_dRdqdot.transpose() << std::endl;
    sim_char.PopState("TestDVelRewardDqdot");
}

/**
 * \brief           calc d(reward)/d(action)
 *  it is the ultimate gradient function
 *      drda = dr/dxcur * dxcur/dctrlforce * dctrlforce/da
*/
#include "sim/Controller/CtPDGenController.h"
tVectorXd cSceneDiffImitate::CalcDrDa()
{
    // tMatrixXd CalcDxurDctrlforce();
    // tMatrixXd CalcDctrlforceDa();
    // std::cout << "[debug] calc drda timestep = " << mTimestep << std::endl;
    tVectorXd DrDx = CalcDrDxcur();
    auto bt_gen_world =
        std::dynamic_pointer_cast<cGenWorld>(mWorldBase)->GetInternalGenWorld();
    tMatrixXd DxDCtrlForce = bt_gen_world->GetDxnextDCtrlForce();
    auto gen_char = GetDefaultGenChar();
    auto gen_ctrl = std::dynamic_pointer_cast<cCtPDGenController>(
        gen_char->GetController());
    tMatrixXd DctrlforceDaction = gen_ctrl->CalcDCtrlForceDAction(mTimestep);
    tVectorXd DrDa = DrDx.transpose() * DxDCtrlForce * DctrlforceDaction;
    // std::cout << "DrDa = " << DrDa.transpose() << std::endl;
    return DrDa;
}

/**
 * \brief           calc d(reward)/d(action)
*/
void cSceneDiffImitate::TestDrDa()
{
    tVectorXd ideal_drda = CalcDrDa();

    // 1. get the previous model result, get the old action
    auto bt_gen_world =
        std::dynamic_pointer_cast<cGenWorld>(mWorldBase)->GetInternalGenWorld();
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("TestDrDa");
    auto gen_ctrl = std::dynamic_pointer_cast<cCtPDGenController>(
        gen_char->GetController());
    tVectorXd q_old, qdot_old;
    bt_gen_world->GetLastFrameGenCharqAndqdot(q_old, qdot_old);
    tVectorXd q_cur = gen_char->Getq(), qdot_cur = gen_char->Getqdot();

    // std::cout << "qold = " << q_old.transpose() << std::endl;
    // std::cout << "qnew = " << q_cur.transpose() << std::endl;
    // std::cout << "qdot_old = " << qdot_old.transpose() << std::endl;
    // std::cout << "qdot_cur = " << qdot_cur.transpose() << std::endl;
    tVectorXd action_old = gen_ctrl->GetCurAction();
    double old_reward = CalcReward(0);
    // 2. change the action, calculate the reward, and get the result
    double eps = 1e-5;
    double dt = 1.0 / 600;
    bool err = false;
    for (int i = 0; i < gen_ctrl->GetActionSize(); i++)
    {
        gen_char->SetqAndqdot(q_old, qdot_old);
        action_old[i] += eps;
        gen_ctrl->ApplyAction(action_old);

        cSceneSimChar::Update(dt);

        double new_reward = CalcReward(0);
        double num_drdai = (new_reward - old_reward) / eps;
        double ideal_drdai = ideal_drda[i];
        double diff = std::fabs(ideal_drdai - num_drdai);
        if (diff > 10 * eps || std::isinf(diff) || std::isnan(diff))
        {
            std::cout << "[error] test drda failed for " << i << std::endl;
            printf("ideal = %.5f, num = %.5f\n", ideal_drdai, num_drdai);
            printf("old rew = %.5f, new rew = %.5f\n", old_reward, new_reward);
            err = true;
        }
        action_old[i] -= eps;
    }
    gen_char->SetqAndqdot(q_cur, qdot_cur);
    gen_ctrl->ApplyAction(action_old);
    gen_char->PopState("TestDrDa");
    if (err)
        std::cout << "[error] Test DrDa failed\n";
    else
        std::cout << "[log] Test DrDa succ\n";
}