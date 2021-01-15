#include <iostream>

#include "DeepMimicCore.h"

#include "util/FileUtil.h"

#include "render/DrawUtil.h"
#include "render/TextureDesc.h"
#include "util/LogUtil.h"

// Dimensions of the window we are drawing into.
int gWinWidth = 800;
int gWinHeight = static_cast<int>(gWinWidth * 9.0 / 16.0);
// int gWinWidth = 1055;
// int gWinHeight = 450;
bool gReshaping = false;

// intermediate frame buffers
std::unique_ptr<cTextureDesc> gDefaultFrameBuffer;

// anim
const double gFPS = 60.0;
const double gAnimStep = 1.0 / gFPS;
const int gDisplayAnimTime = static_cast<int>(1000 * gAnimStep);
extern bool gAnimating;

int gSampleCount = 0;

double gPlaybackSpeed = 1;
const double gPlaybackDelta = 0.05;

// FPS counter
int gPrevTime = 0;
double gUpdatesPerSec = 0;

std::vector<std::string> gArgs;
std::unique_ptr<cDeepMimicCore> gCore;

auto convert_to_vector_double = [](const std::vector<double> &vec) {
    // const tVectorXd ret = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
    //     vec.data(), vec.size());
    // std::vector<double> a = {1, 2, 3, 4};
    const Eigen::VectorXd ret =
        Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(vec.data(),
                                                            vec.size());
    return ret;
};
auto convert_to_vector_int = [](const std::vector<int> &vec) {
    // Eigen::VectorXi ret(vec.data(), vec.size());
    const Eigen::VectorXi ret =
        Eigen::Map<const Eigen::VectorXi, Eigen::Unaligned>(vec.data(),
                                                            vec.size());
    return ret;
};
void SetupDeepMimicCore()
{
    bool enable_draw = true;
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
        // std::cout << "action space = " << action_space << std::endl;
        // std::cout << "state size = " << action_space << std::endl;
        // std::cout << "goal size = " << action_space << std::endl;
        // std::cout << "action size = " << action_space << std::endl;
        // std::cout << "action space = " << action_space << std::endl;
        // std::cout << "num actions = " << action_space << std::endl;
        // std::cout << "--------------------\n";
        // std::cout << "state offset = "
        //           << convert_to_vector_double(s_offset).transpose()
        //           << std::endl;
        // std::cout << "state scale = "
        //           << convert_to_vector_double(s_scale).transpose() <<
        //           std::endl;
        // std::cout << "g offset = "
        //           << convert_to_vector_double(g_offset).transpose()
        //           << std::endl;
        // std::cout << "g scale = "
        //           << convert_to_vector_double(g_scale).transpose() <<
        //           std::endl;
        // std::cout << "action offset = "
        //           << convert_to_vector_double(a_offset).transpose()
        //           << std::endl;
        // std::cout << "action scale = "
        //           << convert_to_vector_double(a_scale).transpose() <<
        //           std::endl;

        // std::cout << "action min = "
        //           << convert_to_vector_double(action_min).transpose()
        //           << std::endl;
        // std::cout << "action max = "
        //           << convert_to_vector_double(action_max).transpose()
        //           << std::endl;
        // std::cout << "state norm groups = "
        //           << convert_to_vector_int(state_norm_groups).transpose()
        //           << std::endl;
        // std::cout << "goal norm groups = "
        //           << convert_to_vector_int(goal_norm_groups).transpose()
        //           << std::endl;

        // exit(0);
    }
}

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

void UpdateFrameBuffer()
{
    if (!gReshaping)
    {
        if (gWinWidth != gCore->GetWinWidth() ||
            gWinHeight != gCore->GetWinHeight())
        {
            gCore->Reshape(gWinWidth, gWinHeight);
        }
    }
}

extern tVectorXd global_action;
void Update(double time_elapsed)
{
    int num_substeps = gCore->GetNumUpdateSubsteps(); // the simulation substeps
                                                      // between 2 frames
    double timestep = time_elapsed / num_substeps;
    num_substeps = (time_elapsed == 0) ? 1 : num_substeps;
    // std::cout << "main elasped time = " << time_elapsed << std::endl;
    for (int i = 0; i < num_substeps; ++i)
    {
        for (int id = 0; id < gCore->GetNumAgents(); ++id)
        {
            if (gCore->NeedNewAction(id))
            {
                auto s = gCore->RecordState(id);
                // std::cout << "cur state = "
                //           << convert_to_vector_double(s).transpose()
                //           << std::endl;
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
                // exit(1);
                // std::cout << "main get reward = " << r << std::endl;
                // std::cout <<"state = ";
                // for(auto x : s) std::cout << x <<" ";
                // std::cout << std::endl;

                ++gSampleCount;

                if (global_action.size() == 0)
                {
                    global_action =
                        -0.1 * tVectorXd::Ones(gCore->GetActionSize(id));
                    global_action[0] = 0.5;
                    // global_action[1] = -1.2;
                    // global_action <dRootRotErr_dpose0_total< 1, 0.1, 0.1, 0.1, 0, 1, 0.1, 0.1, 0.1,
                    // 1,
                    //     0.1, 0.1, 0.1, 0, 1, 0.1, 0.1, 0.1;
                    // global_action << 1, 0.2, 0.2, 0.2, 0, 1, 0.2, 0.2, 0.2,
                    // 1,
                    //     0.2, 0.2, 0.2, 0, 1, 0.2, 0.2, 0.2;
                    // global_action.segment(global_action.size() - 4, 4)
                    //     .setRandom();
                    // global_action =
                    // tVectorXd::Random(gCore->GetActionSize(id));
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
        gCore->Update(timestep);

        // we can only get the drda after update
        gCore->CalcDRewardDAction();

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
}

void Draw(void)
{
    UpdateFrameBuffer();
    gCore->Draw();

    glutSwapBuffers();
    gReshaping = false;
}

void Reshape(int w, int h)
{
    gReshaping = true;

    gWinWidth = w;
    gWinHeight = h;

    gDefaultFrameBuffer->Reshape(w, h);
    glViewport(0, 0, gWinWidth, gWinHeight);
    glutPostRedisplay();
}

void StepAnim(double time_step)
{
    Update(time_step);
    gAnimating = true;
    glutPostRedisplay();
}

void Reload() { SetupDeepMimicCore(); }

void Reset() { gCore->Reset(); }

int GetNumTimeSteps()
{
    int num_steps = static_cast<int>(gPlaybackSpeed);
    if (num_steps == 0)
    {
        num_steps = 1;
    }
    num_steps = std::abs(num_steps);
    return num_steps;
}

int CalcDisplayAnimTime(int num_timesteps)
{
    int anim_time =
        static_cast<int>(gDisplayAnimTime * num_timesteps / gPlaybackSpeed);
    anim_time = std::abs(anim_time);
    return anim_time;
}

void Shutdown()
{
    gCore->Shutdown();
    exit(0);
}

int GetCurrTime() { return glutGet(GLUT_ELAPSED_TIME); }

void InitTime()
{
    gPrevTime = GetCurrTime();
    gUpdatesPerSec = 0;
}

void Animate(int callback_val)
{
    const double counter_decay = 0;

    if (gAnimating)
    {
        int num_steps = GetNumTimeSteps();
        int curr_time = GetCurrTime();
        int time_elapsed = curr_time - gPrevTime;
        gPrevTime = curr_time;

        double timestep = (gPlaybackSpeed < 0) ? -gAnimStep : gAnimStep;
        for (int i = 0; i < num_steps; ++i)
        {
            Update(timestep);
        }

        // FPS counting
        double update_count = num_steps / (0.001 * time_elapsed);
        if (std::isfinite(update_count))
        {
            gUpdatesPerSec = counter_decay * gUpdatesPerSec +
                             (1 - counter_decay) * update_count;
            gCore->SetUpdatesPerSec(gUpdatesPerSec);
        }

        int timer_step = CalcDisplayAnimTime(num_steps);
        int update_dur = GetCurrTime() - curr_time;
        timer_step -= update_dur;
        timer_step = std::max(timer_step, 0);

        glutTimerFunc(timer_step, Animate, 0);
        glutPostRedisplay();
    }

    if (gCore->IsDone())
    {
        Shutdown();
    }
}

void ToggleAnimate()
{

    gAnimating = !gAnimating;
    MIMIC_DEBUG("toggle animated {}", gAnimating);
    if (gAnimating)
    {
        glutTimerFunc(gDisplayAnimTime, Animate, 0);
    }
}

void ChangePlaybackSpeed(double delta)
{
    double prev_playback = gPlaybackSpeed;
    gPlaybackSpeed += delta;
    gCore->SetPlaybackSpeed(gPlaybackSpeed);

    if (std::abs(prev_playback) < 0.0001 && std::abs(gPlaybackSpeed) > 0.0001)
    {
        glutTimerFunc(gDisplayAnimTime, Animate, 0);
    }
}

void Keyboard(unsigned char key, int x, int y)
{
    gCore->Keyboard(key, x, y);
    switch (key)
    {
    case 27: // escape
        Shutdown();
        break;
    case ' ':
        ToggleAnimate();
        break;
    case '>':
        StepAnim(gAnimStep);
        break;
    case '<':
        StepAnim(-gAnimStep);
        break;
    case ',':
        ChangePlaybackSpeed(-gPlaybackDelta);
        break;
    case '.':
        ChangePlaybackSpeed(gPlaybackDelta);
        break;
    case '/':
        ChangePlaybackSpeed(-gPlaybackSpeed + 1);
        break;
    case 'l':
        Reload();
        break;
    case 'r':
        Reset();
        break;
    default:
        break;
    }

    glutPostRedisplay();
}

void MouseClick(int button, int state, int x, int y)
{
    gCore->MouseClick(button, state, x, y);
    glutPostRedisplay();
}

void MouseMove(int x, int y)
{
    gCore->MouseMove(x, y);
    glutPostRedisplay();
}

void InitFrameBuffers(void)
{
    gDefaultFrameBuffer = std::unique_ptr<cTextureDesc>(
        new cTextureDesc(0, 0, 0, gWinWidth, gWinHeight, 1, GL_RGBA, GL_RGBA));
}

void InitDraw(int argc, char **argv)
{
    glutInit(&argc, argv);
    // std::cout << "[debug] Init Draw begin 2" << std::endl;
#ifdef __APPLE__
    glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_RGBA | GLUT_DOUBLE |
                        GLUT_DEPTH);
#else
    glutInitContextVersion(3, 2);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif

    glutInitWindowSize(gWinWidth, gWinHeight);
    glutCreateWindow("DeepMimic");
    // std::cout << "Init Draw succ" << std::endl;
}

void SetupDraw()
{
    glutDisplayFunc(Draw);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(MouseClick);
    glutMotionFunc(MouseMove);
    glutTimerFunc(gDisplayAnimTime, Animate, 0);

    InitFrameBuffers();
    Reshape(gWinWidth, gWinHeight);
    gCore->Reshape(gWinWidth, gWinHeight);
}

void DrawMainLoop()
{
    InitTime();
    glutMainLoop();
}

int main(int argc, char **argv)
{
    srand(0);
    FormatArgs(argc, argv, gArgs);

    InitDraw(argc, argv);
    SetupDeepMimicCore();
    SetupDraw();

    DrawMainLoop();

    return EXIT_SUCCESS;
}
