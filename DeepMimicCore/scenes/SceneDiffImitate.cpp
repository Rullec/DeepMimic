#include "SceneDiffImitate.h"
#include "sim/Controller/CtPDGenController.h"
#include "sim/World/GenWorld.h"
#include "util/LogUtil.h"
const std::string
    cSceneDiffImitate::gDerivModeStr[cSceneDiffImitate::NUM_DERIV_MODE] = {
        "single_step", "single_step_sum", "multi_steps"};

cSceneDiffImitate::cSceneDiffImitate()
{
    // MIMIC_INFO("cSceneDiffImitate created");
    mEnableTestDRewardDAction = false;
    mPBuffer.clear();
    mQBuffer.clear();
    mDrdaSingleBuffer.clear();
    mDebugOutput = false;
}
cSceneDiffImitate::~cSceneDiffImitate() {}

void cSceneDiffImitate::ParseArgs(const std::shared_ptr<cArgParser> &parser)
{
    cSceneImitate::ParseArgs(parser);
    parser->ParseBoolCritic("enable_test_reward_action_derivative",
                            mEnableTestDRewardDAction);

    std::string diffmode_str;
    parser->ParseStringCritic("diff_scene_mode", diffmode_str);
    mDerivMode = cSceneDiffImitate::ParseDerivMode(diffmode_str);
    MIMIC_INFO("diff scene mode {}", gDerivModeStr[mDerivMode]);
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

    // set ctrl to calculate the derivatives
    GetDefaultGenCtrl()->SetEnableCalcDeriv(true);
    MIMIC_ASSERT(GetDefaultGenCtrl()->GetEnableCalcDeriv());

    // {
    //     auto gen_char = GetDefaultGenChar();
    //     tVectorXd x = gen_char->Getx();
    //     x += tVectorXd::Ones(x.size()) * 1e-2;
    //     gen_char->Setx(x);

    //     TestDRootPosErrDx();
    //     TestDRootRotErrDx();
    //     TestDRootLinVelErrDx();
    //     TestDRootAngVelErrDx();
    //     TestDRootRewardDx();
    //     // TestEndEffectorRewardByGivenErr();
    //     // TestDEndEffectorRewardDq();
    //     // TestDRootRewardDqDqdot();

    //     // for (int i = 0; i < gen_char->GetNumOfLinks(); i++)
    //     // {
    //     //     // TestDJointPosRel0Dq(i);
    //     //     TestDEndEffectorErrDq(i);
    //     // }
    // }
    // exit(0);
}

/**
 * \brief       Get D(Reward)/D(Action)
 *  it is the ultimate gradient function
 *  dr/da = 
 *      dr/d x_{cur} * d (x_cur) / d_u * d u / da
 * 
 *  This function will be calculated before the new action is applied
*/
tVectorXd cSceneDiffImitate::CalcDRewardDAction()
{
    if (mEnableTestDRewardDAction == true)
    {
        Test();
    }

    //

    // d(r_t)/d(x_{t+1}^1) * d(x_{t+1}^1)/d(a)
    tVectorXd DrDa;
    if (mDerivMode == eDerivMode::DERIV_SINGLE_STEP)
    {

        DrDa = CalcDrDxcur().transpose() * CalcDxurDa();
    }
    else if (mDerivMode == eDerivMode::DERIV_SINGLE_STEP_SUM)
    {
        // std::cout << "[warn] calcdrda in sum mode, the buffer will be cleared "
        //              "after this\n";
        DrDa = tVectorXd::Zero(GetActionSize(0));
        for (auto &x : mDrdaSingleBuffer)
            DrDa += x;
        // std::cout << "sum drda = " << DrDa.transpose() << std::endl;
        mDrdaSingleBuffer.clear();
    }
    else
    {
        MIMIC_ERROR("unsupported mode {}", mDerivMode);
    }
    // {
    //     // test
    //     tVectorXd Drda_single =
    //         CalcDrDxcur().transpose() * CalcDxurDa_SingleStep();
    //     tVectorXd Drda_multi =
    //         CalcDrDxcur().transpose() * CalcDxurDa_MultiStep();
    //     // std::cout << "[single] drda = " << Drda_single.transpose() << std::endl;
    //     // std::cout << "[multi] drda = " << Drda_multi.transpose() << std::endl;
    // }
    return DrDa;
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

    // 3. test the end effector reward
    {
        TestDEndEffectorRewardDq();
    }
    // 3. test dposedq and dveldqdot
    {
        auto gen_char = GetDefaultGenChar();
        gen_char->TestCalcDposedq();
        gen_char->TestCalcDveldqdot();
    }

    // 4. test root reward
    {
        TestDRootPosErrDx();
        TestDRootRotErrDx();
        TestDRootLinVelErrDx();
        TestDRootAngVelErrDx();
        TestDRootRewardDx();
    }

    // 4. total test method
    {
        TestDrDxcur();
        TestDRewardDAction();
        // TestP();
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
 *      Calculate the derivative of current reward w.r.t current x
 *      In the train cycle, it is d(r_t)/d(x_{t+1}^1)
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
        tVectorXd dPoseRewarddq = CalcDPoseRewardDq(),
                  dVelRewarddqdot = CalcDVelRewardDqdot(),
                  dEndEffectordq = CalcDEndEffectorRewardDq();
        tVectorXd dRootRewdx = CalcDRootRewardDx();
        MIMIC_ASSERT(dPoseRewarddq.size() == dof &&
                     dVelRewarddqdot.size() == dof);
        drdx.segment(0, dof) = dPoseRewarddq + dEndEffectordq;
        drdx.segment(dof, dof) = dVelRewarddqdot;
        drdx += dRootRewdx;
    }
    return drdx;
}

/**
 * \brief           Calcudlate d(x_cur)/d(action)
 *      
 *      d(x_{t+1}^1)/da
*/
tMatrixXd cSceneDiffImitate::CalcDxurDa()
{
    tMatrixXd deriv;
    switch (mDerivMode)
    {
    case eDerivMode::DERIV_SINGLE_STEP:
        deriv = CalcDxurDa_SingleStep();
        break;
    case eDerivMode::DERIV_SINGLE_STEP_SUM:
        deriv = CalcDxurDa_SingleStep();
        break;
    case eDerivMode::DERIV_MULTI_STEPS:
        deriv = CalcDxurDa_MultiStep();
        break;
    default:
        MIMIC_ASSERT(false);
        break;
    }
    return deriv;
}

std::shared_ptr<cSimCharacterGen> cSceneDiffImitate::GetDefaultGenChar() const
{
    return std::dynamic_pointer_cast<cSimCharacterGen>(GetCharacter(0));
}

std::shared_ptr<cCtPDGenController> cSceneDiffImitate::GetDefaultGenCtrl()
{
    return std::dynamic_pointer_cast<cCtPDGenController>(
        GetDefaultGenChar()->GetController());
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
           vel_rew = CalcVelReward(gen_char, ref_char),
           ee_rew = CalcEndEffectorReward(gen_char, ref_char),
           root_rew = CalcRootReward(gen_char, ref_char);
    // std::cout << "pose rew = " << pose_rew << std::endl;
    // std::cout << "vel rew = " << vel_rew << std::endl;
    // cMathUtil::TestCalc_DQuaterion_DEulerAngles();

    // exit(0);
    double total_rew = pose_rew + vel_rew + ee_rew + root_rew;
    // printf("[debug] pose rew %.5f, vel rew %.5f, ee_rew %.5f, root_rew %.5f, "
    //        "total rew %.5f\n",
    //        pose_rew, vel_rew, ee_rew, root_rew, total_rew);
    return total_rew;
}

/**
 * \brief               Calculate d(pose_reward)/dq
*/
tVectorXd cSceneDiffImitate::CalcDPoseRewardDq()
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
*/
void cSceneDiffImitate::TestDRewardDAction()
{
    mEnableTestDRewardDAction = false;
    tVectorXd ideal_drda = CalcDRewardDAction();
    mEnableTestDRewardDAction = true;

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

/**
 * \brief           given a string Parse the derivative mode
*/
cSceneDiffImitate::eDerivMode cSceneDiffImitate::ParseDerivMode(std::string str)
{
    for (int i = 0; i < cSceneDiffImitate::eDerivMode::NUM_DERIV_MODE; i++)
    {
        if (cSceneDiffImitate::gDerivModeStr[i] == str)
            return static_cast<cSceneDiffImitate::eDerivMode>(i);
    }
    MIMIC_ERROR("invalid deriv mode str {}", str);
    return cSceneDiffImitate::NUM_DERIV_MODE;
}

/**
 * \brief           Calc DxurDa only use single step information
 *      we assume: d(xur)/da = d(xur)/d(u_cur) * d(u_cur)/da
 * 
 *      it should be: 
 *      d(x_{t+1}^1)/da = 
 *          d(x_{t+1})/d(u_t) * d(u_t)/da
 *                  
*/
tMatrixXd cSceneDiffImitate::CalcDxurDa_SingleStep() { return CalcQ(); }

/**
 * \brief           Calculate the d(x_cur)/da by multi substeps info
 *  Basically, the formula is:
 *          d(x_k)da  = d(x_k)/d(x_{k-1}) * d(x_{k-1})da + d(x_k)/d(u_{k-1}) * d(u_{k-1})/da
 * 
 *      let Dk = d(x_k)da, 
 *          Pk = d(x_k)/d(x_{k-1}), 
 *          D_{k-1} = d(x_{k-1})da, 
 *          Q_k = d(x_k)/d(u_{k-1}) * d(u_{k-1})/da
 * 
 *      we have 
 *          Dk = Pk * D_{k-1} + Qk
 * For more details, please check the note "20210112 diffMBRL对多个substep求导"
*/
tVectorXd cSceneDiffImitate::CalcDxurDa_MultiStep()
{
    // printf("[debug] P buffer size %d, Q buffer size %d\n", mPBuffer.size(),
    //        mQBuffer.size());
    MIMIC_ASSERT(mPBuffer.size() == mQBuffer.size());
    if (mPBuffer.size() != 20)
    {
        // if current P buffer is not 20, it must be zero.
    }
    int size = mPBuffer.size();

    /*
        Dk = Pk * Dk-1 + Qk
        ...
        D1 = P1 * D0 + Q1
        D0 = Q0
    */

    tMatrixXd DxDa = mQBuffer[0];

    for (int i = std::max(1, size - 3); i < size; i++)
    {
        DxDa = mPBuffer[i] * DxDa + mQBuffer[i];
    }
    // std::cout << "[debug] calc dxda done, size = " << size << std::endl;
    return DxDa;
}

/**
 * \brief           Calculate the P Matrix at this moment
 *      
 *      the P Matrix is d(x_cur)/d(x_prev), approximately
 * 
 *      P = A * \frac{d u}{d x} 
 *          + [dt*I; I] * \tilde{M}^{-1} * M * dqdot/dx
 * 
 *      A is the state transition matrix in semi-implicit scheme
 *      A = dt * [dt*I; I] * \tilde{M}^{-1}
 * 
 *      \tilde{M} = (M + dt * C)
*/
tMatrixXd cSceneDiffImitate::CalcP()
{
    // 1. some prerequistes
    MIMIC_ASSERT(mTimestep > 0);
    auto gen_char = GetDefaultGenChar();
    auto gen_ctrl = GetDefaultGenCtrl();
    const tMatrixXd &M = gen_char->GetMassMatrix(),
                    &C = gen_char->GetCoriolisMatrix();

    int dof = gen_char->GetNumOfFreedom();
    int state_size = 2 * dof;
    tMatrixXd tilde_M_inv = (M + mTimestep * C).inverse();
    tMatrixXd dtI_I = tMatrixXd::Zero(state_size, dof);
    dtI_I.block(0, 0, dof, dof).setIdentity();
    dtI_I.block(dof, 0, dof, dof).setIdentity();
    dtI_I.block(0, 0, dof, dof) *= mTimestep;

    // 2. form P part1: A * dudx
    tMatrixXd P = mTimestep * dtI_I * tilde_M_inv *
                  gen_ctrl->CalcDCtrlForceDx_Approx(mTimestep);

    // 3. form P part2:
    // [dt*I; I] * \tilde{M}^{-1} * M * dqdot/dx
    tMatrixXd dqdot_dx = tMatrixXd::Zero(dof, state_size);
    dqdot_dx.block(0, dof, dof, dof).setIdentity();
    P += dtI_I * tilde_M_inv * M * dqdot_dx;

    return P;
}

/**
 * \brief           Calculate the Q vector at this moment
 *      the Q vector is d(x_cur)/d(u_prev) * d(u_prev)/da, approximately
*/
tMatrixXd cSceneDiffImitate::CalcQ()
{
    auto bt_gen_world =
        std::dynamic_pointer_cast<cGenWorld>(mWorldBase)->GetInternalGenWorld();

    // d(x_{t+1}^1)/dut
    tMatrixXd DxDCtrlForce = bt_gen_world->GetDxnextDCtrlForce();
    auto gen_char = GetDefaultGenChar();
    auto gen_ctrl = GetDefaultGenCtrl();
    if (mDebugOutput)
        std::cout << "[debug] DxDCtrlForce = \n" << DxDCtrlForce << std::endl;
    // it is in current timestep, wrong result
    tMatrixXd DctrlforceDaction_old = gen_ctrl->GetDCtrlForceDAction();
    if (mDebugOutput)
    {

        std::cout << "DctrlforceDaction = \n"
                  << DctrlforceDaction_old << std::endl;

        std::cout << "DxDaction = \n"
                  << DxDCtrlForce * DctrlforceDaction_old << std::endl;
    }
    return DxDCtrlForce * DctrlforceDaction_old;
}

void cSceneDiffImitate::Reset()
{
    cSceneImitate::Reset();
    ClearPQBuffer();
    mDrdaSingleBuffer.clear();
}

void cSceneDiffImitate::ClearPQBuffer()
{
    mPBuffer.clear();
    mQBuffer.clear();
    // std::cout << "[debug] clear PQ buffer!\n";
}

/**
 * \brief           update the current scene
*/
void cSceneDiffImitate::Update(double dt)
{
    mTimestep = dt;
    // 0. if it's in the multistep mode, consider to calcualte the P and Q
    if (mDerivMode == eDerivMode::DERIV_MULTI_STEPS)
    {
        // 0.1 calcualte P (confirmed, it's and it should be calculated in the old timestep, here)
        mPBuffer.push_back(CalcP());
    }

    // 1. update the imitate scene
    cSceneImitate::Update(dt);

    // 0.2 calcualte Q, I think is should be caclulated after the update
    if (mDerivMode == eDerivMode::DERIV_MULTI_STEPS)
        mQBuffer.push_back(CalcQ());

    if (mDerivMode == eDerivMode::DERIV_SINGLE_STEP_SUM)
    {
        mDrdaSingleBuffer.push_back(CalcDrDxcur().transpose() * CalcDxurDa());
    }
    // std::cout << "drda = " << CalcDRewardDAction().transpose() << std::endl;
    if (mDebugOutput)
    {

        tVectorXd drdx = CalcDrDxcur();
        tMatrixXd dxda = CalcDxurDa_SingleStep();
        std::cout << "[debug] cur drdx = " << CalcDrDxcur().transpose()
                  << std::endl;
        std::cout << "[debug] cur drda = " << drdx.transpose() * dxda
                  << std::endl;
    }
}

/**
 * \brief       Test the d(x_{t+1})/d(x_t) numerically
 * 
*/
void cSceneDiffImitate::TestP()
{
    auto Getx = [](cSimCharacterGen *gen_char) -> tVectorXd {
        int dof = gen_char->GetNumOfFreedom();
        int state_size = dof * 2;
        tVectorXd x = tVectorXd::Zero(state_size);
        x.segment(0, dof) = gen_char->Getq();
        x.segment(dof, dof) = gen_char->Getqdot();
        return x;
    };

    // 0. push the current state
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_p");
    // 1. get current P, then {get current x_{t+1} by simulation}
    tMatrixXd P = CalcP();
    printf("P size %d %d\n", P.rows(), P.cols());
    gen_char->PushState("update");
    Update(mTimestep);
    tVectorXd xnext_old = Getx(gen_char.get());
    gen_char->PopState("update");

    // 2. set eps, set the state vector, add & minus eps on each element then {do forward simulation}, get new x_{t+1}
    // then calc the nuermcally grad
    double eps = 1e-5;
    int dof = gen_char->GetNumOfFreedom(), state_size = 2 * dof;
    tVectorXd q = gen_char->Getq(), qdot = gen_char->Getqdot();
    for (int i = 0; i < state_size; i++)
    {
        // add eps
        if (i < dof)
        {
            // q size
            q[i] += eps;
        }
        else
        {
            // qdot size
            qdot[i - dof] += eps;
        }

        gen_char->SetqAndqdot(q, qdot);
        gen_char->PushState("update");
        Update(mTimestep);
        tVectorXd xnext_new = Getx(gen_char.get());
        gen_char->PopState("update");

        tVectorXd num_dxnext_dxold = (xnext_new - xnext_old) / eps;
        tVectorXd approx_dxnext_dxold = P.col(i);
        tVectorXd diff = approx_dxnext_dxold - num_dxnext_dxold;
        printf("------------idx %d-------------\n", i);
        std::cout << "approx_dxnext_dxold = " << approx_dxnext_dxold.transpose()
                  << std::endl;
        std::cout << "num_dxnext_dxold = " << num_dxnext_dxold.transpose()
                  << std::endl;
        std::cout << "diff = " << diff.transpose() << std::endl;
        printf("diff percentage %.5f%%\n",
               diff.norm() / num_dxnext_dxold.norm() * 100);
        // minus eps
        if (i < dof)
        {
            // q size
            q[i] -= eps;
        }
        else
        {
            // qdot size
            qdot[i - dof] -= eps;
        }
    }

    // 3. pop the old state
    gen_char->PopState("test_p");
    // exit(0);
}

void cSceneDiffImitate::SetAction(int agent_id, const Eigen::VectorXd &action)
{
    // std::cout << "[debug] set aciton\n";
    ClearPQBuffer();
    cSceneImitate::SetAction(agent_id, action);
}

/**
 * \brief           Calc the derivative of end effector reward w.r.t q
 * 
 *  let r_{ee} is the end effector reward
 *      err_{ee} is the end effector error
 *      s_{err} is the error scale
 *      s_{ee} is the end effector scale
 *      
 *          
 *  r_{ee} = w_ee * exp(-s_{err} * s_{ee} * err_{ee})
 * 
 *  err_{ee} = \sum_i 
 *              \Vert p_{rel1}^i - p_{rel0}^i \Vert^2
 * 
 *  here the p_{rel1}^i is the refernece relative pos of link i w.r.t root,
 *  and the p_{rel0}^i is the realy relative pos of link i w.r.t root
 *      
 *      p_{rel0}^i = T_{ori} * (pi - p0 - g)
 * 
 *  pi is the real pos of link i
 *  p0 is the world pos of link 0 (root)
 *  g is the ground height
 * 
 *  In summary, the gradient of this reward is composed by the following steps:
 * 
 *  d(r_{ee})/dq = d(r_{ee}) / d err_{ee}
 *                 \* 
 *                 d err_{ee} / d q
 * 
*/
tVectorXd cSceneDiffImitate::CalcDEndEffectorRewardDq() const
{
    // 0. prepare

    auto sim_char = GetDefaultGenChar();
    int num_of_links = sim_char->GetNumOfLinks();
    // 1. calculate d err_{ee}/dq
    int num_of_ee = sim_char->CalcNumEndEffectors();
    int num_of_freedom = sim_char->GetNumOfFreedom();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();

    const Eigen::VectorXd &pose0 = sim_char->GetPose();
    const Eigen::VectorXd &pose1 = kin_char->GetPose();
    tMatrix origin_trans = sim_char->BuildOriginTrans();
    tMatrix kin_origin_trans = kin_char->BuildOriginTrans();
    tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
    tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
    tEigenArr<tMatrix> DOriginTransDq;
    sim_char->CalcDOriginTransDq(DOriginTransDq);

    tMatrixXd Jv_root = sim_char->GetRoot()->GetJKv();
    double end_eff_err = 0;
    tVectorXd derree_dq = tVectorXd::Zero(num_of_freedom);
    for (int i = 0; i < num_of_links; i++)
    {
        if (true == sim_char->IsEndEffector(i))
        {
            /*
                1.1 calcualte the d(err_{ee}^i)/dq
                d(err_{ee}^i)/dq 
                    = d(err_{ee}^i)/d(p_{rel0}^i) 
                      \* 
                      d(p_{rel0}^i)/dq
            */

            /*
                1.1 calculate 
                    d(err_{ee}^i)/d(p_{rel0}^i)
                    =
                    2 * (p_{rel0}^i - p_{rel1}^i)^T

           */

            tVector pos0 = sim_char->CalcJointPos(i);
            tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, i);
            double ground_h0 = mGround->SampleHeight(pos0);
            double ground_h1 = kin_char->GetOriginPos()[1];
            tVector pos_rel0 = pos0 - root_pos0;
            tVector pos_rel1 = pos1 - root_pos1;
            pos_rel0[1] = pos0[1] - ground_h0;
            pos_rel1[1] = pos1[1] - ground_h1;
            pos_rel0 = origin_trans * pos_rel0;
            pos_rel1 = kin_origin_trans * pos_rel1;

            tVectorXd derree_dprel0i = 2 * (pos_rel0 - pos_rel1).transpose();

            /*
                1.2 calcualte d(p_{rel0}^i)/dq
                    =
                    dT_ori/dq * p_{rel0}^i + T_ori * (dpidq - dp0/dq)
            */
            // tMatrixXd dprel0i_dq = tMatrixXd::Zero(4, num_of_freedom);
            // tMatrixXd Jv_diff = tMatrixXd::Ones(4, num_of_freedom);
            // Jv_diff.block(0, 0, 3, num_of_freedom).noalias() =
            //     sim_char->GetJointById(i)->GetJKv() - Jv_root;
            // tMatrixXd second_term = origin_trans * Jv_diff;
            // for (int i = 0; i < num_of_freedom; i++)
            // {
            //     dprel0i_dq.col(i) = (DOriginTransDq[i] * pos_rel0);
            // }
            // dprel0i_dq += second_term;

            // 1.3 calcualte d(err_{ee}^i)/d(q)
            derree_dq += CalcDEndEffectorErrDq(i) / num_of_ee;
            // 1.4 add err
            double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
            end_eff_err += curr_end_err;
        }
    }

    /*
        2. calcualte the first part, d(r_{ee})/derr_{ee}
            d(r_{ee})/derr_{ee}
            = -w_ee * s_err * s_ee * exp(-s_err * s_ee * err_ee)
    */
    end_eff_err /= num_of_ee;
    // std::cout << "total end eff err = " << end_eff_err << std::endl;

    // std::cout << "derrdq = " << derree_dq.transpose() << std::endl;
    double err_scale = RewParams.err_scale,
           end_eff_scale = RewParams.end_eff_scale,
           end_eff_w = RewParams.end_eff_w;
    double dree_derree = -end_eff_w * err_scale * end_eff_scale *
                         std::exp(-err_scale * end_eff_scale * end_eff_err);

    tVectorXd dree_dq = dree_derree * derree_dq;
    return dree_dq;
}

/**
 * \brief           Test the deriv of ee reward
*/
void cSceneDiffImitate::TestDEndEffectorRewardDq()
{
    auto &gen_char = *(GetDefaultGenChar().get());
    gen_char.PushState("test_ee");
    auto &kin_char = *(GetKinChar().get());
    double old_r = CalcEndEffectorReward(gen_char, kin_char);
    tVectorXd drdq = CalcDEndEffectorRewardDq();
    double eps = 1e-5;

    tVectorXd q = gen_char.Getq();
    // std::cout << "[debug] dend_effector_reward/dq = " << drdq.transpose()
    //           << std::endl;
    for (int i = 0; i < gen_char.GetNumOfFreedom(); i++)
    {
        q[i] += eps;

        gen_char.Setq(q);
        double new_r = CalcEndEffectorReward(gen_char, kin_char);
        double num_drdqi = (new_r - old_r) / eps;
        double ana_drdqi = drdq[i];
        double diff = ana_drdqi - num_drdqi;
        // printf("idx %d old r %.5f new r %.5f\n", i, old_r, new_r);
        if (std::fabs(diff) > 10 * eps)
        {
            std::cout << "[error] TestDEndEffectorRewardDq failed for idx " << i
                      << std::endl;
            std::cout << "ana = " << ana_drdqi << std::endl;
            std::cout << "num = " << num_drdqi << std::endl;
            std::cout << "diff = " << diff << std::endl;
            exit(0);
        }

        q[i] -= eps;
    }
    gen_char.PopState("test_ee");
    std::cout << "[log] TestDEndEffectorRewardDq succ = " << drdq.transpose()
              << std::endl;
}

/**
 * \brief           Calculate the pos_{rel1}, the realtive position in MOCAP data
*/
tVector cSceneDiffImitate::CalcJointPosRel1(int id) const
{
    // auto gen_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const Eigen::VectorXd &pose1 = kin_char->GetPose();
    auto gen_char = GetDefaultGenChar();
    tMatrixXd joint_mat = gen_char->GetJointMat();
    tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
    tVector pos1 = cKinTree::CalcJointWorldPos(joint_mat, pose1, id);
    double ground_h1 = kin_char->GetOriginPos()[1];
    tVector pos_rel1 = pos1 - root_pos1;
    pos_rel1[1] = pos1[1] - ground_h1;
    tMatrix kin_origin_trans = kin_char->BuildOriginTrans();
    pos_rel1 = kin_origin_trans * pos_rel1;
    return pos_rel1;
    // const auto &joint_mat = gen_char->GetJointMat();

    // const Eigen::VectorXd &pose0 = gen_char->GetPose();
    // tVector joint_pos = gen_char->CalcJointPos(id);
    // tVector root_pos = cKinTree::GetRootPos(joint_mat, pose0);

    // double ground_h0 = mGround->SampleHeight(joint_pos);
    // joint_pos -= root_pos;
    // joint_pos[1] -= ground_h0;
    // joint_pos = gen_char->BuildOriginTrans() * joint_pos;
    // return joint_pos;
}

/**
 * \brief       pos_{rel0} for given joint "id" is:
 *          the relative pos between joint id and root joint in XZ axis
 *          + 
 *          the relative height from ground to joint pos in Y axis
 *          in sim world (true)
*/
tVector cSceneDiffImitate::CalcJointPosRel0(int id) const
{
    auto gen_char = GetDefaultGenChar();
    tVector pos0 = gen_char->CalcJointPos(id);

    tVector root_pos0 =
        cKinTree::GetRootPos(gen_char->GetJointMat(), gen_char->GetPose());

    tVector pos_rel0 = pos0 - root_pos0;
    double ground_h0 = mGround->SampleHeight(pos0);

    // here, we give the real height of this joint to pos_rel0
    pos_rel0[1] = pos0[1] - ground_h0;
    tMatrix origin_trans = gen_char->BuildOriginTrans();
    pos_rel0 = origin_trans * pos_rel0;
    return pos_rel0;
}
/**
 * \brief           Calculate the derivative of joint pos_{rel0}  w.r.t q
 *      the pos_{rel0} is the relative pos in XZ axis, but height in Y axis in sim world
 *      the pos_{rel1} is the relative pos in XZ axis, but height in Y axis in MOCAP
*/
tMatrixXd cSceneDiffImitate::CalcDJointPosRel0Dq(int id) const
{
    auto gen_char = GetDefaultGenChar();
    int dof = gen_char->GetNumOfFreedom();
    tMatrix origin_trans = gen_char->BuildOriginTrans();
    tMatrixXd jac_diff = tMatrixXd::Zero(4, dof);
    jac_diff.block(0, 0, 3, dof) = gen_char->GetJointById(id)->GetJKv() -
                                   gen_char->GetJointById(0)->GetJKv();
    jac_diff.row(1) = gen_char->GetJointById(id)->GetJKv().row(1);

    tMatrixXd part2 = origin_trans * jac_diff;
    tVector joint_pos = gen_char->CalcJointPos(id);
    tVector root_pos = gen_char->CalcJointPos(0);
    tVector prel = joint_pos - root_pos;
    double ground_height = mGround->SampleHeight(joint_pos);

    // here we give the real height of joint pos to prel[1]
    prel[1] = joint_pos[1] - ground_height;
    tEigenArr<tMatrix> dorigin_trans_dq;
    gen_char->CalcDOriginTransDq(dorigin_trans_dq);
    tMatrixXd part1 = tMatrixXd::Zero(4, dof);
    for (int i = 0; i < dof; i++)
    {
        part1.col(i) = dorigin_trans_dq[i] * prel;
    }
    return part1 + part2;
}

/**
 * \brief           Test CalcDJointPosRel0Dq, 
 * Note that, 
 *      pos_rel0 is the relative position in sim world
 *      pos_rel1 is the relative pos in MOCAP
*/
void cSceneDiffImitate::TestDJointPosRel0Dq(int id)
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_djointpos");
    tVector old_pos = CalcJointPosRel0(id);
    tMatrixXd ana_deriv = CalcDJointPosRel0Dq(id);
    // std::cout << "[debug] test joint " << id << " pos ana deriv = \n"
    //           << ana_deriv << std::endl;
    tVectorXd q = gen_char->Getq();
    double eps = 1e-5;
    for (int i = 0; i < gen_char->GetNumOfFreedom(); i++)
    {
        q[i] += eps;
        gen_char->Setq(q);
        tVector new_pos = CalcJointPosRel0(id);
        tVector num_derivi = (new_pos - old_pos) / eps;
        tVector ana_derivi = cMathUtil::Expand(ana_deriv.col(i), 0);
        tVector diff = ana_derivi - num_derivi;

        if (diff.norm() > eps)
        {
            std::cout << "[error] test_djointpos failed for joint " << id
                      << " idx " << i << std::endl;
            std::cout << "ana = " << ana_derivi.transpose() << std::endl;
            std::cout << "num = " << num_derivi.transpose() << std::endl;
            std::cout << "diff = " << diff.transpose() << std::endl;

            exit(0);
        }

        q[i] -= eps;
    }
    gen_char->PopState("test_djointpos");
    std::cout << "test_djointpos test succ for joint " << id << std::endl;
}

double cSceneDiffImitate::CalcEndEffectorErr(int id) const
{
    tVector pos_rel0 = CalcJointPosRel0(id); // sim relative pos
    tVector pos_rel1 = CalcJointPosRel1(id); // mocap relative pos
    double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
    return curr_end_err;
}

tVectorXd cSceneDiffImitate::CalcDEndEffectorErrDq(int id) const
{
    /*
        err = |pos_rel1 - pos_rel0|^2

        d(err)/dq = d(err)/d(pos_rel0) * d(pos_rel0)/dq

                  = 2 * (pos_rel0 - pos_rel1).T * d(pos_rel1)/dq
    */
    tMatrixXd dpos_rel0_dq = CalcDJointPosRel0Dq(id);
    tVector pos_rel0 = CalcJointPosRel0(id);
    tVector pos_rel1 = CalcJointPosRel1(id);
    // std::cout << "joint " << id << " pos_rel0 = " << pos_rel0.transpose()
    //           << " pos_rel1 = " << pos_rel1.transpose()
    //           << " diff = " << (pos_rel1 - pos_rel0).transpose() << std::endl;

    // std::cout << "dpos_rel0_dq = \n" << dpos_rel0_dq << std::endl;
    return 2 * (pos_rel0 - pos_rel1).transpose() * dpos_rel0_dq;
}

void cSceneDiffImitate::TestDEndEffectorErrDq(int id)
{
    // std::cout << "----------begin to test dEndEffectorErrDq for joint " << id
    //           << " -------------\n";
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_derr_dq");

    tVectorXd derrdq = CalcDEndEffectorErrDq(id);
    double err_old = CalcEndEffectorErr(id);
    int dof = gen_char->GetNumOfFreedom();
    tVectorXd q = gen_char->Getq();
    double eps = 1e-5;
    // std::cout << "derrdq = " << derrdq.transpose() << std::endl;
    for (int i = 0; i < dof; i++)
    {
        q[i] += eps;
        gen_char->Setq(q);

        double err_new = CalcEndEffectorErr(id);
        double num_derrdqi = (err_new - err_old) / eps;
        double ana_derrdqi = derrdq[i];

        double diff = ana_derrdqi - num_derrdqi;

        if (std::fabs(diff) > eps)
        {
            std::cout << "[error] test d err_ee dq idx " << i << " failed\n";
            std::cout << "ana = " << ana_derrdqi << std::endl;
            std::cout << "num = " << num_derrdqi << std::endl;
            std::cout << "old = " << err_old << std::endl;
            std::cout << "new = " << err_new << std::endl;
            exit(0);
        }
        // else
        // {
        //     std::cout << "joint " << id << " dof " << i << " diff = " << diff
        //               << std::endl;
        // }
        q[i] -= eps;
    }
    gen_char->PopState("test_derr_dq");
    std::cout << "[log] TestDEndEffectorErrDq for joint " << id << " succ\n";
}

/**
 * \brief
*/
double cSceneDiffImitate::CalcDEndEffectorRewardDErr(double err)
{
    return -RewParams.end_eff_w * RewParams.err_scale *
           RewParams.end_eff_scale *
           std::exp(-RewParams.err_scale * RewParams.end_eff_scale * err);
}

void cSceneDiffImitate::TestEndEffectorRewardByGivenErr()
{
    auto gen_char = GetDefaultGenChar();
    int ee_id = gen_char->GetNumOfLinks() - 1;
    double old_err = CalcEndEffectorErr(ee_id);
    double ana_deriv = CalcDEndEffectorRewardDErr(old_err);
    double old_rew =
        RewParams.end_eff_w *
        std::exp(-RewParams.err_scale * RewParams.end_eff_scale * old_err);

    double eps = 1e-5;
    double new_rew = RewParams.end_eff_w *
                     std::exp(-RewParams.err_scale * RewParams.end_eff_scale *
                              (old_err + eps));
    double num_deriv = (new_rew - old_rew) / eps;
    double diff = num_deriv - ana_deriv;
    std::cout << "diff = " << diff << std::endl;
    std::cout << "ana = " << ana_deriv << std::endl;
    std::cout << "num = " << num_deriv << std::endl;
    std::cout << "dbdc = " << CalcDEndEffectorErrDq(ee_id).transpose()
              << std::endl;
    std::cout << "dadb = " << ana_deriv << std::endl;
    exit(0);
}

/**
 * \brief               Calculate the deriv of root reward w.r.t x = [q, qdot]
 *  For more details, please check the note "root reward对q和qdot求导"
*/
#include "BulletGenDynamics/btGenModel/Joint.h"
tVectorXd cSceneDiffImitate::CalcDRootRewardDx()
{
    // A is the prefix
    double A =
        -RewParams.err_scale * RewParams.root_scale * RewParams.root_w *
        std::exp(-RewParams.err_scale * RewParams.root_scale * CalcRootErr());
    // other termis
    tVectorXd derrdx = RewParams.root_pos_w * CalcDRootPosErrDx() +
                       RewParams.root_rot_w * CalcDRootRotErrDx() +
                       RewParams.root_vel_w * CalcDRootLinVelErrDx() +
                       RewParams.root_angle_vel_w * CalcDRootAngVelErrDx();

    return A * derrdx;
}

double cSceneDiffImitate::CalcRootErr() const
{
    return RewParams.root_pos_w * CalcRootPosErr() +
           RewParams.root_rot_w * CalcRootRotErr() +
           RewParams.root_vel_w * CalcRootLinVelErr() +
           RewParams.root_angle_vel_w * CalcRootAngVelErr();
}
/**
 * \brief               Test the deriv of root reward w.r.t x = [q & qdot]
*/
void cSceneDiffImitate::TestDRootRewardDx()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_drootrewdx");
    int dof = gen_char->GetNumOfFreedom();
    int x_size = 2 * dof;
    tVectorXd x = gen_char->Getx();
    double old_r = CalcRootReward(*(gen_char.get()), *(GetKinChar().get()));
    tVectorXd ana_deriv = CalcDRootRewardDx();
    double eps = 1e-5;
    for (int i = 0; i < x_size; i++)
    {
        x[i] += eps;
        gen_char->Setx(x);

        double new_r = CalcRootReward(*(gen_char.get()), *(GetKinChar().get()));
        double num_di = (new_r - old_r) / eps;
        double ana_di = ana_deriv[i];
        double diff = ana_di - num_di;
        if (std::fabs(diff) > 100 * eps)
        {
            std::cout << "[error] test d root rew dx failed for idx " << i
                      << std::endl;
            std::cout << "diff = " << diff << std::endl;
            std::cout << "ana = " << ana_di << std::endl;
            std::cout << "num = " << num_di << std::endl;
            std::cout << "deriv = " << ana_deriv.transpose() << std::endl;

            exit(0);
        }
        x[i] -= eps;
    }
    gen_char->PopState("test_drootrewdx");
    std::cout << "[log] TestDRootRewardDx succ = " << x.transpose()
              << std::endl;
}

// 6.1 root pos
/**
 * \brief           
*/
double cSceneDiffImitate::CalcRootPosErr() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();

    // sim_char: simulation character
    // kin_char: the representation of motion data
    const Eigen::VectorXd &pose0 = sim_char->GetPose();
    const Eigen::VectorXd &pose1 = kin_char->GetPose();
    tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
    tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
    double root_ground_h0 = mGround->SampleHeight(sim_char->GetRootPos());
    double root_ground_h1 = kin_char->GetOriginPos()[1];
    root_pos0[1] -= root_ground_h0;
    root_pos1[1] -= root_ground_h1;
    double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
    return root_pos_err;
}
/**
 * \brief           calculate derivative d(root_pos_err)/dx
*/
tVectorXd cSceneDiffImitate::CalcDRootPosErrDx() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();

    // sim_char: simulation character
    // kin_char: the representation of motion data
    const Eigen::VectorXd &pose0 = sim_char->GetPose();
    const Eigen::VectorXd &pose1 = kin_char->GetPose();
    const tVectorXd &vel0 = sim_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();
    tVector root_pos0 = cKinTree::GetRootPos(joint_mat, pose0);
    tVector root_pos1 = cKinTree::GetRootPos(joint_mat, pose1);
    double root_ground_h0 = mGround->SampleHeight(sim_char->GetRootPos());
    double root_ground_h1 = kin_char->GetOriginPos()[1];
    root_pos0[1] -= root_ground_h0;
    root_pos1[1] -= root_ground_h1;

    auto root_joint = sim_char->GetJointById(0);
    int dof = sim_char->GetNumOfFreedom();
    tVectorXd dedx = tVectorXd::Zero(2 * dof);
    dedx.segment(0, dof).noalias() =
        2 * (root_pos0 - root_pos1).transpose().segment(0, 3) *
        root_joint->GetJKv();
    return dedx;
}
/**
 * \brief           
*/
void cSceneDiffImitate::TestDRootPosErrDx()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_root_pos_err");
    double old_err = CalcRootPosErr();
    tVectorXd drootposerr_dx = CalcDRootPosErrDx();
    tVectorXd x = gen_char->Getx();
    int dof = gen_char->GetNumOfFreedom();
    double eps = 1e-5;
    for (int i = 0; i < 2 * dof; i++)
    {
        x[i] += eps;
        gen_char->Setx(x);
        double new_err = CalcRootPosErr();
        double num_di = (new_err - old_err) / eps;
        double ana_di = drootposerr_dx[i];
        double diff = ana_di - num_di;

        if (std::fabs(diff) > 10 * eps)
        {
            std::cout << "num = " << num_di << std::endl;
            std::cout << "ana = " << ana_di << std::endl;
            std::cout << "diff = " << diff << std::endl;
            MIMIC_ERROR("test_root_pos_err failed for idx {}", i);
            exit(0);
        }
        x[i] -= eps;
    }

    gen_char->PopState("test_root_pos_err");
    std::cout << "[log] TestDRootPosErrDx succ = " << drootposerr_dx.transpose()
              << std::endl;
}
// 6.2 root rot
/**
 * \brief           Calculate the root rotation error
*/
double cSceneDiffImitate::CalcRootRotErr() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();

    // sim_char: simulation character
    // kin_char: the representation of motion data
    const Eigen::VectorXd &pose0 = sim_char->GetPose();
    const Eigen::VectorXd &pose1 = kin_char->GetPose();
    tQuaternion root_rot0 = cKinTree::GetRootRot(joint_mat, pose0);
    tQuaternion root_rot1 = cKinTree::GetRootRot(joint_mat, pose1);
    double root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1);
    root_rot_err *= root_rot_err;
    return root_rot_err;
}
/**
 * \brief           Calculate d(root_rot_err)/dx NUMERICALLY
*/
tVectorXd cSceneDiffImitate::CalcDRootRotErrDx()
{
    // this derivative should only relates to the gen coords root freedoms
    auto gen_char = GetDefaultGenChar();
    double old_err = CalcRootRotErr();
    int dof = gen_char->GetNumOfFreedom();
    tVectorXd dedx = tVectorXd::Zero(2 * dof);
    int rot_freedom_st = -1, rot_freedom_ed = -1;
    {
        auto root_joint = gen_char->GetJointById(gen_char->GetRootID());
        auto joint_type = root_joint->GetJointType();
        switch (joint_type)
        {
        case JointType::NONE_JOINT:
        {
            rot_freedom_st = 3;
            rot_freedom_ed = rot_freedom_st + 3;
            break;
        }
        case JointType::BIPEDAL_NONE_JOINT:
        {
            rot_freedom_st = 2;
            rot_freedom_ed = rot_freedom_st + 1;
            break;
        }
        default:
            MIMIC_ERROR("DRootRotErrDx unsupported {}", joint_type);
            break;
        }
    }

    double eps = 1e-8;
    tVectorXd q = gen_char->Getq();
    for (int id = rot_freedom_st; id < rot_freedom_ed; id++)
    {
        q[id] += eps;
        gen_char->Setq(q);
        double new_err = CalcRootRotErr();
        dedx[id] = (new_err - old_err) / eps;
        q[id] -= eps;
    }
    gen_char->Setq(q);
    return dedx;
}

/**
 * \brief           
*/
void cSceneDiffImitate::TestDRootRotErrDx()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_root_rot_err");
    double old_err = CalcRootRotErr();
    tVectorXd drootroterr_dx = CalcDRootRotErrDx();
    tVectorXd x = gen_char->Getx();
    int dof = gen_char->GetNumOfFreedom();
    double eps = 1e-5;
    for (int i = 0; i < 2 * dof; i++)
    {
        x[i] += eps;
        gen_char->Setx(x);
        double new_err = CalcRootRotErr();
        double num_di = (new_err - old_err) / eps;
        double ana_di = drootroterr_dx[i];
        double diff = ana_di - num_di;

        if (std::fabs(diff) > 10 * eps)
        {
            printf("[error] test_root_rot_err failed for idx %d\n", i);
            std::cout << "ana = " << ana_di << std::endl;
            std::cout << "num = " << num_di << std::endl;
            std::cout << "diff = " << diff << std::endl;
            exit(0);
        }
        x[i] -= eps;
    }

    gen_char->PopState("test_root_rot_err");
    std::cout << "[log] TestDRootRotErrDx succ = " << drootroterr_dx.transpose()
              << std::endl;
}
// 6.3 root lin vel
/**
 * \brief           Calculate root lin vel error
*/
double cSceneDiffImitate::CalcRootLinVelErr() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    const tVectorXd &vel0 = sim_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();
    tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
    tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
    double root_vel_err = (root_vel1 - root_vel0).squaredNorm();
    return root_vel_err;
}
/**
 * \brief           Calculate d(roto_linvel_err)/dx
*/
tVectorXd cSceneDiffImitate::CalcDRootLinVelErrDx() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    const tVectorXd &vel0 = sim_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();
    tVector root_vel0 = cKinTree::GetRootVel(joint_mat, vel0);
    tVector root_vel1 = cKinTree::GetRootVel(joint_mat, vel1);
    int dof = sim_char->GetNumOfFreedom();
    tVectorXd dedx = tVectorXd::Zero(2 * dof);
    dedx.segment(dof, dof).noalias() =
        2 * (root_vel0 - root_vel1).transpose().segment(0, 3) *
        sim_char->GetJointById(sim_char->GetRootID())->GetJKv();
    return dedx;
}
/**
 * \brief           
*/
void cSceneDiffImitate::TestDRootLinVelErrDx()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_root_linvel_err");
    double old_err = CalcRootLinVelErr();
    tVectorXd drootLinVelerr_dx = CalcDRootLinVelErrDx();
    tVectorXd x = gen_char->Getx();
    int dof = gen_char->GetNumOfFreedom();
    double eps = 1e-5;
    for (int i = 0; i < 2 * dof; i++)
    {
        x[i] += eps;
        gen_char->Setx(x);
        double new_err = CalcRootLinVelErr();
        double num_di = (new_err - old_err) / eps;
        double ana_di = drootLinVelerr_dx[i];
        double diff = ana_di - num_di;

        if (std::fabs(diff) > 10 * eps)
        {
            printf("[error] test_root_linvel_err failed for idx %d\n", i);
            std::cout << "ana = " << ana_di << std::endl;
            std::cout << "num = " << num_di << std::endl;
            std::cout << "diff = " << diff << std::endl;
            exit(0);
        }
        x[i] -= eps;
    }

    gen_char->PopState("test_root_linvel_err");
    std::cout << "[log] TestDRootLinVelErrDx succ = "
              << drootLinVelerr_dx.transpose() << std::endl;
}
// 6.4 root ang vel
/**
 * \brief           Calc root ang vel error
*/
double cSceneDiffImitate::CalcRootAngVelErr() const
{
    auto sim_char = GetDefaultGenChar();
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    const tVectorXd &vel0 = sim_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();
    tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
    tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);
    double root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm();
    return root_ang_vel_err;
}
/**
 * \brief           Calculate d(root_angvel_err)/dx
 *      1. (root_angvel_err)/dq = prefix * dJwdq * qdot
 *      2. (root_angvel_err)/dqdot = prefix * Jw
*/
tVectorXd cSceneDiffImitate::CalcDRootAngVelErrDx() const
{
    auto sim_char = GetDefaultGenChar();
    MIMIC_ASSERT(sim_char->GetComputeSecondDerive() == true);
    auto kin_char = GetKinChar();
    const auto &joint_mat = sim_char->GetJointMat();
    const tVectorXd &vel0 = sim_char->GetVel();
    const tVectorXd &vel1 = kin_char->GetVel();
    tVector root_ang_vel0 = cKinTree::GetRootAngVel(joint_mat, vel0);
    tVector root_ang_vel1 = cKinTree::GetRootAngVel(joint_mat, vel1);

    tVector3d prefix = 2 * (root_ang_vel0 - root_ang_vel1).segment(0, 3);
    auto root_joint =
        dynamic_cast<Joint *>(sim_char->GetJointById(sim_char->GetRootID()));
    const tMatrixXd Jw = root_joint->GetJKw();
    int dof = sim_char->GetNumOfFreedom();
    tMatrixXd dJwdq_qdot = tMatrixXd::Zero(3, dof);
    tVectorXd qdot = sim_char->Getqdot();

    // calculate dJkwdq automatically
    root_joint->ComputeDJkwdq();
    for (int i = 0; i < root_joint->GetNumOfFreedom(); i++)
    {
        tMatrixXd dd = root_joint->GetdJKwdq_3xnversion(i);
        // std::cout << "idx " << i << " size = " << dd.rows() << " " << dd.cols()
        //           << std::endl;
        dJwdq_qdot.col(i).noalias() =
            dd * qdot.segment(0, root_joint->GetNumOfFreedom());
    }

    // part1 + part2
    tVectorXd dedx = tVectorXd::Zero(2 * dof);
    dedx.segment(0, dof) = prefix.transpose() * dJwdq_qdot;
    dedx.segment(dof, dof) = prefix.transpose() * Jw;
    return dedx;
}
/**
 * \brief           
*/
void cSceneDiffImitate::TestDRootAngVelErrDx()
{
    auto gen_char = GetDefaultGenChar();
    gen_char->PushState("test_root_angvel_err");
    double old_err = CalcRootAngVelErr();
    tVectorXd drootAngVelerr_dx = CalcDRootAngVelErrDx();
    tVectorXd x = gen_char->Getx();
    int dof = gen_char->GetNumOfFreedom();
    double eps = 1e-5;
    for (int i = 0; i < 2 * dof; i++)
    {
        x[i] += eps;
        gen_char->Setx(x);
        double new_err = CalcRootAngVelErr();
        double num_di = (new_err - old_err) / eps;
        double ana_di = drootAngVelerr_dx[i];
        double diff = ana_di - num_di;

        if (std::fabs(diff) > 10 * eps)
        {
            printf("[error] test_root_angvel_err failed for idx %d\n", i);
            std::cout << "ana = " << ana_di << std::endl;
            std::cout << "num = " << num_di << std::endl;
            std::cout << "diff = " << diff << std::endl;
            std::cout << "d = " << drootAngVelerr_dx.transpose() << std::endl;
            exit(0);
        }
        x[i] -= eps;
    }

    gen_char->PopState("test_root_angvel_err");
    std::cout << "[log] TestDRootAngVelErrDx succ = "
              << drootAngVelerr_dx.transpose() << std::endl;
}