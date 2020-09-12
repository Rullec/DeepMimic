#include "TrajRecorder.h"
#include "IDSolver.h"
#include "scenes/SceneImitate.h"
#include "sim/Controller/CtPDFeaController.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/World/FeaWorld.h"
#include "sim/World/GenWorld.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include <iostream>

cTrajRecorder::cTrajRecorder(cSceneImitate *scene, const std::string &conf)
{
    mScene = scene;
    mSimChar = scene->GetCharacter().get();
    ParseConfig(conf);
    MIMIC_TRACE("build traj recoder by {}", conf);
}

cTrajRecorder::~cTrajRecorder() {}

/**
 * \brief               Record info before simulation
 */
void cTrajRecorder::PreSim()
{
    mSaveInfo.mCharPoses[mSaveInfo.mCurFrameId] = mSimChar->GetPose();

    if (eSimCharacterType::Generalized == mSimChar->GetCharType())
    {
        RecordActiveForceGen();
    }

    auto gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
    if (gen_char != nullptr)
    {
        std::cout << "-----------------------------frame "
                  << mSaveInfo.mCurFrameId << "----------------------\n";
        std::cout << "q = " << gen_char->Getq().transpose() << std::endl;
        std::cout << "qdot = " << gen_char->Getqdot().transpose() << std::endl;
        // std::endl; std::cout << "M = \n" << gen_char->GetMassMatrix() <<
        // std::endl; std::cout << "C = \n" << gen_char->GetCoriolisMatrix() <<
        // std::endl;
    }
}

/**
 * \brief               Record info after simulation
 */
void cTrajRecorder::PostSim()
{
    ReadContactInfo();

    mSaveInfo.mCurFrameId++;
    if (mRecordMaxFrame == mSaveInfo.mCurFrameId)
    {
        MIMIC_INFO("record frames exceed the upper bound {}, reset",
                   mSaveInfo.mCurFrameId);
        Reset();
    }
}

/**
 * \brief               Reset & save current trajectory per episode
 */
void cTrajRecorder::Reset()
{
    Json::Value root;
    mSaveInfo.SaveTrajV2(root);

    cJsonUtil::WriteJson(mTrajSavePath, root, true);
    MIMIC_INFO("cTrajRecoder::Reset finished, save traj to {}, exit",
               mTrajSavePath);
    VerifyDynamicsEquation();
    exit(1);
}

/**
 *
 */
void cTrajRecorder::SetTimestep(double t)
{
    mSaveInfo.mTimesteps[mSaveInfo.mCurFrameId] = t;
}

/**
 * \brief               Parse config
 *          This function will parse the trajectory recoder's config to control
 * its behavior.
 *          1. save place?
 *          2. save version?
 */
void cTrajRecorder::ParseConfig(const std::string &conf)
{
    Json::Value root;
    cJsonUtil::LoadJson(conf, root);
    mTrajSavePath = cJsonUtil::ParseAsString("traj_save_filename", root);
    mRecordMaxFrame = cJsonUtil::ParseAsInt("record_max_frame", root);

    MIMIC_DEBUG("trajectory recoder parse config: get traj save path{}",
                mTrajSavePath);
}

void cTrajRecorder::ReadContactInfoGen()
{
    int num_of_links = mSimChar->GetNumBodyParts();
    auto &cur_contact_info = mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    cur_contact_info.clear();

    tContactForceInfo pt_info;
    for (int i = 0; i < num_of_links; i++)
    {
        for (auto &pt : mSimChar->GetBodyPart(i)->GetContactPts())
        {
            pt_info.mId = i;
            pt_info.mIsSelfCollision = pt.mIsSelfCollision;
            pt_info.mForce = pt.mForce;
            pt_info.mPos = pt.mPos;
            cur_contact_info.push_back(pt_info);
        }
    }
    MIMIC_INFO("record {} contact points in traj recorder",
               cur_contact_info.size());

    MIMIC_WARN("link id += 1 in order to get matched with MIMICControl");
    for (auto &x : cur_contact_info)
    {
        x.mId += 1;
    }
}
void cTrajRecorder::ReadContactInfoRaw()
{

    MIMIC_ERROR("hasn't been implemented")
}
void cTrajRecorder::ReadContactInfo()
{
    switch (mSimChar->GetCharType())
    {
    case eSimCharacterType::Featherstone:
        ReadContactInfoRaw();
        break;
    case eSimCharacterType::Generalized:
        ReadContactInfoGen();
        break;
    default:
        break;
    }

    const auto &current_forces =
        mSaveInfo.mContactForces[mSaveInfo.mCurFrameId];
    int frame = mSaveInfo.mCurFrameId;
    if (current_forces.size())
    {
        MIMIC_DEBUG("frame {} num of contacts {}", frame,
                    current_forces.size());
        for (int i = 0; i < current_forces.size(); i++)
        {
            MIMIC_DEBUG("contact {} force = {}", i,
                        current_forces[i].mForce.transpose());
        }
    }
}

/**
 * \brief           Record the active force of gen
 */
void cTrajRecorder::RecordActiveForceGen()
{
    auto gen = dynamic_cast<cSimCharacterGen *>(mSimChar);
    mSaveInfo.mTruthAction[mSaveInfo.mCurFrameId] = gen->GetGeneralizedForce();
}

void cTrajRecorder::VerifyDynamicsEquation()
{
    tVectorXd qbefore, qcur, qnext, qdot, qddot;
    cSimCharacterGen *gen_char = dynamic_cast<cSimCharacterGen *>(mSimChar);
    double dt = mSaveInfo.mTimesteps[0];
    int dof = gen_char->GetNumOfFreedom();
    int num_of_links = gen_char->GetNumOfLinks();
    for (int frame_id = 1; frame_id < mSaveInfo.mCurFrameId - 1; frame_id++)
    {
        // MIMIC_DEBUG("frame id {}", frame_id);
        tVectorXd cur_pose = mSaveInfo.mCharPoses[frame_id];
        qcur = gen_char->ConvertPoseToq(cur_pose);
        qbefore = gen_char->ConvertPoseToq(mSaveInfo.mCharPoses[frame_id - 1]);
        qnext = gen_char->ConvertPoseToq(mSaveInfo.mCharPoses[frame_id + 1]);
        // gen_char->SetPose(cur_pose);

        qdot = (qcur - qbefore) / dt;
        qddot = (qbefore + qnext - 2 * qcur) / (std::pow(dt, 2));
        tVectorXd qddot_num = (qbefore + qnext - 2 * qcur);
        gen_char->SetqAndqdot(qcur, qdot);
        tMatrixXd M = gen_char->GetMassMatrix(),
                  C = gen_char->GetCoriolisMatrix();

        tVectorXd LHS = M * qddot + C * qdot;

        // get the contact gen force
        const auto &cur_contact_forces = mSaveInfo.mContactForces[frame_id];
        int num_of_contacts = cur_contact_forces.size();
        tVectorXd Q_contact = tVectorXd::Zero(dof);
        for (int c_id = 0; c_id < num_of_contacts; c_id++)
        {
            int link_id = cur_contact_forces[c_id].mId;
            tVector force = cur_contact_forces[c_id].mForce;
            tVector pos = cur_contact_forces[c_id].mPos;
            // MIMIC_DEBUG("contact {} link {} force {} pos {} ", c_id, link_id,
            //             force.transpose(), pos.transpose());
            tMatrixXd jac;
            gen_char->ComputeJacobiByGivenPointTotalDOFWorldFrame(
                link_id - 1, pos.segment(0, 3), jac);
            Q_contact += jac.transpose() * force.segment(0, 3);
        }
        tVectorXd Q_G = tVectorXd::Zero(dof);
        for (int link_id = 0; link_id < num_of_links; link_id++)
        {
            Q_G += gen_char->GetLinkById(link_id)->GetJKv().transpose() *
                   gGravity.segment(0, 3) *
                   gen_char->GetBodyPart(link_id)->GetMass();
        }
        tVectorXd active_force = LHS - Q_G - Q_contact;
        // tVectorXd active_diff = mSaveInfo.mTruthAction[frame_id] -
        // active_force;

        // if (frame_id >= 322)
        // {
        //     // MIMIC_DEBUG("active diff {}", active_diff.transpose());
        //     // MIMIC_OUTPUT("frame {} M norm = {}", frame_id, M.norm());
        //     // MIMIC_OUTPUT("frame {} C norm = {}", frame_id, C.norm());
        //     // MIMIC_OUTPUT("frame {} q = {}", frame_id, qcur.norm());
        //     // MIMIC_OUTPUT("frame {} qdot = {}", frame_id, qdot.norm());
        //     MIMIC_OUTPUT("frame {} qddot norm = {}", frame_id, qddot.norm());
        //     MIMIC_OUTPUT("frame {} qddot = {}", frame_id, qddot.transpose());
        //     MIMIC_OUTPUT("frame {} qddot num = {}", frame_id,
        //                  qddot_num.transpose());
        //     MIMIC_OUTPUT("frame {} qbefore {} qcur {} qnext {} ", frame_id,
        //                  qbefore[4], qcur[4], qnext[4]);
        //     // MIMIC_OUTPUT("frame {} Mqddot = {}", frame_id, (M *
        //     // qddot).norm()); MIMIC_OUTPUT("frame {} dt = {}", frame_id,
        //     dt);

        //     // MIMIC_OUTPUT("frame {} LHS = {}", frame_id, LHS.transpose());
        //     // MIMIC_OUTPUT("frame {} QG {}", frame_id, Q_G.transpose());
        //     // MIMIC_OUTPUT("frame {} active force {}", frame_id,
        //     //              active_force.transpose());
        // }
    }
    // MIMIC_ERROR("Verify Dynamics equ done");
    exit(1);
}
