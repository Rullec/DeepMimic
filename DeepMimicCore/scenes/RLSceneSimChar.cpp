#include "RLSceneSimChar.h"
#include "sim/Controller/CtController.h"
#include <iostream>

cRLSceneSimChar::cRLSceneSimChar()
{
    mEnableFallEnd = true;
    mAnnealSamples = gInvalidIdx;
}

cRLSceneSimChar::~cRLSceneSimChar() {}

void cRLSceneSimChar::ParseArgs(const std::shared_ptr<cArgParser> &parser)
{
    cRLScene::ParseArgs(parser);
    cSceneSimChar::ParseArgs(parser);

    parser->ParseBool("enable_fall_end", mEnableFallEnd);
    parser->ParseInt("anneal_samples", mAnnealSamples);

    mTimerParamsEnd = mTimerParams;
    mArgParser->ParseDouble("time_end_lim_min", mTimerParamsEnd.mTimeMin);
    mArgParser->ParseDouble("time_end_lim_max", mTimerParamsEnd.mTimeMax);
    mArgParser->ParseDouble("time_end_lim_exp", mTimerParamsEnd.mTimeExp);
}

void cRLSceneSimChar::Init()
{
    cRLScene::Init();
    cSceneSimChar::Init();

    mAgentReg.Clear();
    RegisterAgents();
    mAgentReg.PrintInfo();

    SetupTimerAnnealer(mTimerAnnealer);
}

void cRLSceneSimChar::Clear()
{
    cRLScene::Clear();
    cSceneSimChar::Clear();

    mAgentReg.Clear();
}

int cRLSceneSimChar::GetNumAgents() const { return mAgentReg.GetNumAgents(); }

bool cRLSceneSimChar::NeedNewAction(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->NeedNewAction();
}

void cRLSceneSimChar::RecordState(int agent_id,
                                  Eigen::VectorXd &out_state) const
{
    const auto &ctrl = GetController(agent_id);
    // std::cout <<"record state in void cRLSceneSimChar::RecordState(int
    // agent_id, Eigen::VectorXd& out_state) const" << std::endl;
    ctrl->RecordState(out_state);
}

void cRLSceneSimChar::RecordPose(int agent_id, Eigen::VectorXd &out_state) const
{
    const cSimCharacterBase *sim_char = GetAgentChar(agent_id);
    const Eigen::VectorXd &char_pose = sim_char->GetPose();
    const int char_pose_size = char_pose.size();
    cCtController *ctrl =
        dynamic_cast<cCtController *>(sim_char->GetController().get());
    if (nullptr == ctrl)
    {
        out_state.resize(0);
        std::cout << "[error] get controller failed when RecordPose"
                  << std::endl;
    }
    else
    {
        out_state.resize(1 + char_pose_size);
        out_state[0] = 1.0 / ctrl->GetUpdateRate(); // 1 / update_frenquency
        out_state.block(1, 0, char_pose_size, 1) = char_pose;
    }
}

void cRLSceneSimChar::RecordGoal(int agent_id, Eigen::VectorXd &out_goal) const
{
    class cCtController;
    // std::cout << "get goal from controller " << std::endl;
    const auto &ctrl = GetController(agent_id);
    ctrl->RecordGoal(out_goal);
}

void cRLSceneSimChar::RecordContactInfo(int agent_id,
                                        Eigen::VectorXd &out_goal) const
{
    std::shared_ptr<cWorldBase> world = GetWorld();
    if (nullptr != world)
    {
        // 1. contact_manager is supposed to manage all the contact infos.
        const cContactManager &contact_manager = world->GetContactManager();
        out_goal.resize(contact_manager.GetNumTotalContactPts() * 7);

        // 2. for body i, its contact info is located in contact_managed[i]
        for (int i = 0, cur_index = 0; i < contact_manager.GetNumEntries(); i++)
        {
            const tEigenArr<cContactManager::tContactPt> &p =
                contact_manager.GetContactPts(i);
            for (auto &cur_pt : p)
            {
                // format: body_id(1) + pt_pos(3) + pt_force(3) = 7
                out_goal[cur_index + 0] = i;
                out_goal.block(cur_index + 1, 0, 3, 1) =
                    cur_pt.mPos.block(0, 0, 3, 1);
                out_goal.block(cur_index + 3 + 1, 0, 3, 1) =
                    cur_pt.mForce.block(0, 0, 3, 1);
                cur_index += 7;
            }
        }
    }
    else
    {
        out_goal.resize(0);
        printf("[warn] world ptr is null in "
               "cRLSceneSimChar::RecordContactInfo\n");
    }
}

void cRLSceneSimChar::SetAction(int agent_id, const Eigen::VectorXd &action)
{
    // std::cout <<"void cRLSceneSimChar::SetAction" << action.transpose( ) <<
    // std::endl;
    const auto &ctrl = GetController(agent_id);
    ctrl->ApplyAction(action);
}

eActionSpace cRLSceneSimChar::GetActionSpace(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->GetActionSpace();
}

int cRLSceneSimChar::GetStateSize(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->GetStateSize();
}

int cRLSceneSimChar::GetGoalSize(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->GetGoalSize();
}

int cRLSceneSimChar::GetActionSize(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->GetActionSize();
}

int cRLSceneSimChar::GetNumActions(int agent_id) const
{
    const auto &ctrl = GetController(agent_id);
    return ctrl->GetNumActions();
}

void cRLSceneSimChar::BuildStateOffsetScale(int agent_id,
                                            Eigen::VectorXd &out_offset,
                                            Eigen::VectorXd &out_scale) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildStateOffsetScale(out_offset, out_scale);
}

void cRLSceneSimChar::BuildGoalOffsetScale(int agent_id,
                                           Eigen::VectorXd &out_offset,
                                           Eigen::VectorXd &out_scale) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildGoalOffsetScale(out_offset, out_scale);
}

void cRLSceneSimChar::BuildActionOffsetScale(int agent_id,
                                             Eigen::VectorXd &out_offset,
                                             Eigen::VectorXd &out_scale) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildActionOffsetScale(out_offset, out_scale);
}

void cRLSceneSimChar::BuildActionBounds(int agent_id, Eigen::VectorXd &out_min,
                                        Eigen::VectorXd &out_max) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildActionBounds(out_min, out_max);
}

void cRLSceneSimChar::BuildStateNormGroups(int agent_id,
                                           Eigen::VectorXi &out_groups) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildStateNormGroups(out_groups);
}

void cRLSceneSimChar::BuildGoalNormGroups(int agent_id,
                                          Eigen::VectorXi &out_groups) const
{
    const auto &ctrl = GetController(agent_id);
    ctrl->BuildGoalNormGroups(out_groups);
}

double cRLSceneSimChar::GetRewardMin(int agent_id) const
{
    return mAgentReg.GetAgent(agent_id)->GetRewardMin();
}

double cRLSceneSimChar::GetRewardMax(int agent_id) const
{
    return mAgentReg.GetAgent(agent_id)->GetRewardMax();
}

cRLSceneSimChar::eTerminate cRLSceneSimChar::CheckTerminate(int agent_id) const
{
    // std::cout <<"cRLSceneSimChar::eTerminate
    // cRLSceneSimChar::CheckTerminate(int agent_id) const" << std::endl;
    bool fail = false;
    if (EnableFallEnd()) // 是否打开掉落end?
    {
        const auto &character = GetAgentChar(agent_id); //获取角色
        fail |= HasFallen(*character); // 按位取或，这是个bool值。
    }
    // 如果坠落，那么就是eTerminateFail, 表示因为Motion失败才停止
    // 否则的话，就是暂时无法判断
    eTerminate terminated = (fail) ? eTerminateFail : eTerminateNull;
    return terminated;
}

bool cRLSceneSimChar::CheckValidEpisode() const
{
    // 检查有效的episode - 功能就是检查速度爆炸
    for (int i = 0; i < GetNumChars(); ++i)
    {
        // 对于每个character
        double max_vel_threshold = 100.0;
        const auto &sim_char = GetCharacter(i);
        bool exp = sim_char->HasVelExploded(max_vel_threshold); // 速度爆炸的
        if (exp)
        {
            // 如果速度速度爆炸，表明失效
            return false;
        }
    }
    return true;
}

void cRLSceneSimChar::LogVal(int agent_id, double val)
{
    const auto &ctrl = GetController(agent_id);
    cDeepMimicCharController *trl_ctrl =
        dynamic_cast<cDeepMimicCharController *>(ctrl.get());
    if (trl_ctrl != nullptr)
    {
        trl_ctrl->LogVal(val);
    }
}

void cRLSceneSimChar::SetSampleCount(int count)
{
    cRLScene::SetSampleCount(count);
    UpdateTimerParams();
}

std::string cRLSceneSimChar::GetName() const { return "RL Sim Character"; }

void cRLSceneSimChar::ResetParams()
{
    cRLScene::ResetParams();
    cSceneSimChar::ResetParams();
}

void cRLSceneSimChar::ResetScene()
{
    cRLScene::ResetScene();
    cSceneSimChar::ResetScene();
}

const std::shared_ptr<cCharController> &cRLSceneSimChar::GetController() const
{
    const auto &character = GetCharacter();
    return character->GetController();
}

const std::shared_ptr<cCharController> &
cRLSceneSimChar::GetController(int agent_id) const
{
    return mAgentReg.GetAgent(agent_id);
}

cSimCharacterBase *cRLSceneSimChar::GetAgentChar(int agent_id) const
{
    return mAgentReg.GetChar(agent_id);
}

void cRLSceneSimChar::PreUpdate(double timestep)
{
    cSceneSimChar::PreUpdate(timestep);

    for (int a = 0; a < GetNumAgents(); ++a)
    {
        const auto &ctrl = mAgentReg.GetAgent(a);
        bool new_action = ctrl->NeedNewAction();
        if (new_action)
        {
            NewActionUpdate(a);
        }
    }
}

void cRLSceneSimChar::ResetTimers()
{
    cSceneSimChar::ResetTimers();
    if (mMode == eModeTest)
    {
        mTimer.SetMaxTime(mTimerParamsEnd.mTimeMax);
    }
}

void cRLSceneSimChar::NewActionUpdate(int agent_id) {}

bool cRLSceneSimChar::EnableFallEnd() const { return mEnableFallEnd; }

void cRLSceneSimChar::RegisterAgents()
{
    int num_chars = GetNumChars();
    for (int i = 0; i < num_chars; ++i)
    {
        const auto &character = GetCharacter(i);
        const auto &ctrl = character->GetController();
        RegisterAgent(ctrl, character);
    }
}

void cRLSceneSimChar::RegisterAgent(
    const std::shared_ptr<cCharController> &ctrl,
    const std::shared_ptr<cSimCharacterBase> &character)
{
    std::vector<int> ids;
    RegisterAgent(ctrl, character, ids);
}

void cRLSceneSimChar::RegisterAgent(
    const std::shared_ptr<cCharController> &ctrl,
    const std::shared_ptr<cSimCharacterBase> &character, std::vector<int> &out_ids)
{
    if (ctrl != nullptr)
    {
        int id = mAgentReg.AddAgent(ctrl, character.get());
        out_ids.push_back(id);

        int num_ctrls = ctrl->NumChildren();
        for (int i = 0; i < num_ctrls; ++i)
        {
            const auto &child_ctrl = ctrl->GetChild(i);
            RegisterAgent(child_ctrl, character, out_ids);
        }
    }
}

void cRLSceneSimChar::SetupTimerAnnealer(cAnnealer &out_annealer) const
{
    cAnnealer::tParams params;
    params.mType = cAnnealer::eTypePow;
    params.mPow = 4.0;
    out_annealer.Init(params);
}

void cRLSceneSimChar::UpdateTimerParams()
{
    if (mAnnealSamples > 0)
    {
        // 随着采样数的增长，逐渐的增大时间限制。
        // 最开始t是0, 然后t会逐渐增大到1。
        // 用得到的这个t去做shift mean，逐渐的逼近mean的方法。
        double t = static_cast<double>(mSampleCount) / mAnnealSamples;
        double lerp = mTimerAnnealer.Eval(t);
        cTimer::tParams blend_params =
            mTimerParams.Blend(mTimerParamsEnd, lerp);
        mTimer.SetParams(blend_params);
    }
}