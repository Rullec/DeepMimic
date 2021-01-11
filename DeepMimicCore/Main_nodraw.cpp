#include "DeepMimicCore.h"
#include "util/LogUtil.h"
#include <string>
#include <vector>

std::vector<std::string> gArgs;

void FormatArgs(int argc, char **argv, std::vector<std::string> &out_args)
{
    using namespace std;
    out_args.resize(argc);
    for (int i = 0; i < argc; ++i)
    {
        // 这个函数中把参数作为字符串全部保存下来
        out_args[i] = std::string(argv[i]);
    }
}
std::unique_ptr<cDeepMimicCore> gCore;

void SetupDeepMimicCore()
{
    bool enable_draw = false;
    gCore = std::unique_ptr<cDeepMimicCore>(new cDeepMimicCore(enable_draw));
    gCore->ParseArgs(gArgs); // 参数解析ok
    gCore->Init();           // 建立了imitate scene

    // get agent 个数
    int num_agents = gCore->GetNumAgents();
    for (int id = 0; id < num_agents; ++id)
    {
        int action_space = gCore->GetActionSpace(id);
        int state_size = gCore->GetStateSize(id);
        int goal_size = gCore->GetGoalSize(id);
        int action_size = gCore->GetActionSize(id);
        int num_actions = gCore->GetNumActions(id);

        auto s_offset = gCore->BuildStateOffset(id);
        auto s_scale = gCore->BuildStateScale(id);
        auto g_offset = gCore->BuildGoalOffset(id);
        auto g_scale = gCore->BuildGoalScale(id);
        auto a_offset = gCore->BuildActionOffset(id);
        auto a_scale = gCore->BuildActionScale(id);

        auto action_min = gCore->BuildActionBoundMin(id);
        auto action_max = gCore->BuildActionBoundMax(id);

        auto state_norm_groups = gCore->BuildStateNormGroups(id);
        auto goal_norm_groups = gCore->BuildGoalNormGroups(id);

        double reward_min = gCore->GetRewardMin(id);
        double reward_max = gCore->GetRewardMax(id);
        double reward_fail = gCore->GetRewardFail(id);
        double reward_succ = gCore->GetRewardSucc(id);

        int xx = 0;
        ++xx;
    }
}
extern tVectorXd global_action;
int gSampleCount = 0;
auto convert_to_vector_double = [](const std::vector<double> &vec) {
    const Eigen::VectorXd ret =
        Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(vec.data(),
                                                            vec.size());
    return ret;
};
auto convert_to_vector_int = [](const std::vector<int> &vec) {
    const Eigen::VectorXi ret =
        Eigen::Map<const Eigen::VectorXi, Eigen::Unaligned>(vec.data(),
                                                            vec.size());
    return ret;
};

void SimMainLoop(double dt = 1.0 / 600)
{
    for (int id = 0; id < gCore->GetNumAgents(); ++id)
    {
        if (gCore->NeedNewAction(id))
        {
            auto s = gCore->RecordState(id);
            auto g = gCore->RecordGoal(id);
            double r = gCore->CalcReward(id);
            MIMIC_INFO("current reward {}", r);
            if (global_action.size() != 0)
            {
                std::vector<double> drda = gCore->CalcDRewardDAction();
                std::cout << "[debug] drda = "
                          << convert_to_vector_double(drda).transpose()
                          << std::endl;
            }
            ++gSampleCount;

            if (global_action.size() == 0)
            {
                global_action =
                    -0.1 * tVectorXd::Ones(gCore->GetActionSize(id));
                global_action[0] = 0.5;
                global_action[1] = -1.2;
            }
            MIMIC_ASSERT(global_action.size() == gCore->GetActionSize(id));
            std::vector<double> action(0);
            for (int num = 0; num < gCore->GetActionSize(id); num++)
            {
                // action.push_back((std::rand() % 100) / 100.0);
                action.push_back(global_action[num]);
                // global_action[num] += 0.1;
            }
            MIMIC_INFO("main: set ref motion as action {}",
                       global_action.transpose());
            // action[action.size() - 1] = 1.3;
            gCore->SetAction(id, action);
        }
    }
    // std::cout << "main core update time = " << timestep << std::endl;
    gCore->Update(dt);

    if (gCore->IsRLScene())
    {
        bool end_episode = gCore->IsEpisodeEnd();
        bool valid_episode = gCore->CheckValidEpisode();
        if (end_episode || !valid_episode)
        {
            for (int id = 0; id < gCore->GetNumAgents(); ++id)
            {
                int terminated = gCore->CheckTerminate(id);
                if (terminated)
                {
                    MIMIC_INFO("Agent {} check terminated", id);
                }
            }
            gCore->SetSampleCount(gSampleCount);
            gCore->Reset();
            // exit(1);
        }
    }
}
int main(int argc, char **argv)
{
    srand(0);
    FormatArgs(argc, argv, gArgs);

    SetupDeepMimicCore();

    while (true)
        SimMainLoop();

    return EXIT_SUCCESS;
}
