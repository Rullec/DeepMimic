#include "DeepMimicCore.h"

#include "render/DrawUtil.h"
#include "scenes/DrawSceneImitate.h"
#include "scenes/SceneBuilder.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "util/LogUtil.h"

cDeepMimicCore::cDeepMimicCore(bool enable_draw)
{
    mArgParser = std::shared_ptr<cArgParser>(new cArgParser());
    cDrawUtil::EnableDraw(enable_draw);

    mNumUpdateSubsteps = 1;
    mPlaybackSpeed = 1;
    mUpdatesPerSec = 0;
}

cDeepMimicCore::~cDeepMimicCore() {}

void cDeepMimicCore::SeedRand(int seed) { cMathUtil::SeedRand(seed); }

void cDeepMimicCore::ParseArgs(const std::vector<std::string> &args)
{
    mArgParser->LoadArgs(args);

    std::string arg_file = "";
    mArgParser->ParseString("arg_file", arg_file);
    if (arg_file != "")
    {
        // append the args from the file to the ones from the commandline
        // this allows the cmd args to overwrite the file args
        bool succ = mArgParser->LoadFile(arg_file);
        if (!succ)
        {
            printf("Failed to load args from: %s\n", arg_file.c_str());
            assert(false);
        }
    }
    else
    {
        MIMIC_ERROR("please offer arg_file as a parameter");
    }

    mArgParser->ParseInt("num_update_substeps", mNumUpdateSubsteps);
    std::string logging_level;
    mArgParser->ParseString("logging_level", logging_level);
    cLogUtil::SetLoggingLevel(logging_level);
}

void cDeepMimicCore::Init()
{
    // 在这个函数之前, 所有参数都被读取到map中了，但之后什么都没做
    if (EnableDraw())
    {
        cDrawUtil::InitDrawUtil();
        InitFrameBuffer();
    }
    SetupScene();
}

void cDeepMimicCore::Update(double timestep) { mScene->Update(timestep); }

void cDeepMimicCore::Reset()
{
    mScene->Reset();
    mUpdatesPerSec = 0;
}

double cDeepMimicCore::GetTime() const { return mScene->GetTime(); }

std::string cDeepMimicCore::GetName() const { return mScene->GetName(); }

bool cDeepMimicCore::EnableDraw() const { return cDrawUtil::EnableDraw(); }

void cDeepMimicCore::Draw()
{
    if (EnableDraw())
    {
        mDefaultFrameBuffer->BindBuffer();
        mScene->Draw();
        mDefaultFrameBuffer->UnbindBuffer();
    }
}

void cDeepMimicCore::Keyboard(int key, int x, int y)
{
    char c = static_cast<char>(key);
    double device_x = 0;
    double device_y = 0;
    CalcDeviceCoord(x, y, device_x, device_y);
    mScene->Keyboard(c, device_x, device_y);
}

void cDeepMimicCore::MouseClick(int button, int state, int x, int y)
{
    double device_x = 0;
    double device_y = 0;
    CalcDeviceCoord(x, y, device_x, device_y);
    mScene->MouseClick(button, state, device_x, device_y);
}

void cDeepMimicCore::MouseMove(int x, int y)
{
    double device_x = 0;
    double device_y = 0;
    CalcDeviceCoord(x, y, device_x, device_y);
    mScene->MouseMove(device_x, device_y);
}

void cDeepMimicCore::Reshape(int w, int h)
{
    mScene->Reshape(w, h);
    mDefaultFrameBuffer->Reshape(w, h);
    glViewport(0, 0, w, h);
    glutPostRedisplay();
}

void cDeepMimicCore::Shutdown() { mScene->Shutdown(); }

bool cDeepMimicCore::IsDone() const { return mScene->IsDone(); }

cDrawScene *cDeepMimicCore::GetDrawScene() const
{
    return dynamic_cast<cDrawScene *>(mScene.get());
}

void cDeepMimicCore::SetPlaybackSpeed(double speed) { mPlaybackSpeed = speed; }

void cDeepMimicCore::SetUpdatesPerSec(double updates_per_sec)
{
    mUpdatesPerSec = updates_per_sec;
}

int cDeepMimicCore::GetWinWidth() const
{
    return mDefaultFrameBuffer->GetWidth();
}

int cDeepMimicCore::GetWinHeight() const
{
    return mDefaultFrameBuffer->GetHeight();
}

int cDeepMimicCore::GetNumUpdateSubsteps() const { return mNumUpdateSubsteps; }

bool cDeepMimicCore::IsRLScene() const
{
    const auto &rl_scene = GetRLScene();
    return rl_scene != nullptr;
}

int cDeepMimicCore::GetNumAgents() const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetNumAgents();
    }
    return 0;
}

bool cDeepMimicCore::NeedNewAction(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        bool need = rl_scene->NeedNewAction(agent_id);
        // if(need) std::cout <<"[DeepMimicCore] need new action = " << need <<
        // std::endl;
        return need;
    }
    return false;
}

std::vector<double> cDeepMimicCore::RecordState(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd state;
        rl_scene->RecordState(agent_id, state);
        // std::cout << "record state = " << state.transpose() << std::endl;

        std::vector<double> out_state;
        ConvertVector(state, out_state);
        return out_state;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::RecordPose(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd pose;
        rl_scene->RecordPose(agent_id, pose);
        // std::cout << "record pose = " << pose.transpose() << std::endl;

        std::vector<double> out_pose;
        ConvertVector(pose, out_pose);
        return out_pose;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::RecordGoal(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd goal;
        rl_scene->RecordGoal(agent_id, goal);

        std::vector<double> out_goal;
        ConvertVector(goal, out_goal);
        return out_goal;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::RecordContactInfo(int agent_id) const
{
    // align the size of out_goal to 6*7 = 42
    const int contact_size = 84;
    const int INVALID_ID = -1;
    Eigen::VectorXd contact(contact_size);
    contact.setConstant(INVALID_ID);

    const auto &rl_scene = GetRLScene();
    if (nullptr != rl_scene)
    {
        Eigen::VectorXd contact_tmp;
        rl_scene->RecordContactInfo(agent_id, contact_tmp);
        if (contact_tmp.size() > contact_size)
        {
            std::cout << "[error] the size of contact info exceed "
                      << contact_size << std::endl;
            abort();
        }
        if (contact_tmp.hasNaN() == true)
        {
            std::cout << "[error] cDeepMimicCore::RecordContactInfo: contact "
                         "has Nan = "
                      << contact_tmp.transpose() << std::endl;
            exit(1);
        }
        contact.block(0, 0, contact_tmp.size(), 1) = contact_tmp;

        std::vector<double> out_goal;
        ConvertVector(contact, out_goal);
        return out_goal;
    }
    return std::vector<double>(0);
}

void cDeepMimicCore::RestoreContactInfo(
    int agent_id, const std::vector<double> &contact_info) const
{
    std::cout << "the func void cDeepMimicCore::RestoreContactInfo(int "
                 "agent_id) const needs to be implemented"
              << std::endl;
    return;
}

void cDeepMimicCore::SetAction(int agent_id, const std::vector<double> &action)
{
    // std::cout << "cDeepMimicCore::SetAction called" << std::endl;
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd in_action;
        ConvertVector(action, in_action);
        // std::cout << "set new action = " << in_action.transpose() << std::endl;
        rl_scene->SetAction(agent_id, in_action);
        // std::cout <<"set action !" << std::endl;
    }
}

void cDeepMimicCore::LogVal(int agent_id, double val)
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        rl_scene->LogVal(agent_id, val);
    }
}

int cDeepMimicCore::GetActionSpace(int agent_id) const
{
    /*
            DeepmimicCore的get action space是从GetRLScene中来的
    */
    eActionSpace action_space = eActionSpaceNull;
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        action_space = rl_scene->GetActionSpace(agent_id);
    }
    return static_cast<int>(action_space);
}

int cDeepMimicCore::GetStateSize(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetStateSize(agent_id);
    }
    return 0;
}

int cDeepMimicCore::GetGoalSize(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetGoalSize(agent_id);
    }
    return 0;
}

int cDeepMimicCore::GetActionSize(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetActionSize(agent_id);
    }
    return 0;
}

int cDeepMimicCore::GetNumActions(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetNumActions(agent_id);
    }
    return 0;
}

std::vector<double> cDeepMimicCore::BuildStateOffset(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        rl_scene->BuildStateOffsetScale(agent_id, offset, scale);

        std::vector<double> out_offset;
        ConvertVector(offset, out_offset);
        return out_offset;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildStateScale(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        rl_scene->BuildStateOffsetScale(agent_id, offset, scale);

        std::vector<double> out_scale;
        ConvertVector(scale, out_scale);
        return out_scale;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildGoalOffset(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        rl_scene->BuildGoalOffsetScale(agent_id, offset, scale);

        std::vector<double> out_offset;
        ConvertVector(offset, out_offset);
        return out_offset;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildGoalScale(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        rl_scene->BuildGoalOffsetScale(agent_id, offset, scale);

        std::vector<double> out_scale;
        ConvertVector(scale, out_scale);
        return out_scale;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildActionOffset(int agent_id) const
{
    // 获取action的mean(共计80个), mean在这里称offset,值得大概是从0点的漂移。
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        // 去rlsence中要offset
        rl_scene->BuildActionOffsetScale(agent_id, offset, scale);

        if (offset.hasNaN() == true || offset.allFinite() == false)
        {
            std::cout << "[error] cDeepMimicCore::BuildActionOffset illegal "
                      << offset.transpose() << std::endl;
            std::cout << "aborting..\n";
            exit(1);
        }

        std::vector<double> out_offset;
        ConvertVector(offset, out_offset);
        // std::cout <<"[scale] get offset(mean) from rl_scene:";
        // for(auto i : out_offset)
        // 	std::cout << i <<" ";
        // std::cout << std::endl;
        // exit(1);
        return out_offset;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildActionScale(int agent_id) const
{
    // 获取所有action(共80个)的scale
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd offset;
        Eigen::VectorXd scale;
        // 去rlscene中要scale
        rl_scene->BuildActionOffsetScale(agent_id, offset, scale);

        std::vector<double> out_scale;
        ConvertVector(scale, out_scale);
        // std::cout <<"scale = " << scale.transpose() << std::endl;
        if (scale.hasNaN() == true || scale.allFinite() == false)
        {
            std::cout << "[error] cDeepMimicCore::BuildActionScale illegal "
                      << scale.transpose() << std::endl;
            std::cout << "aborting..\n";
            exit(1);
        }
        // std::cout <<"[scale] get scale from rl_scene:";
        // for(auto i : out_scale)
        // 	std::cout << i <<" ";
        // exit(1);
        return out_scale;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildActionBoundMin(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd bound_min;
        Eigen::VectorXd bound_max;
        rl_scene->BuildActionBounds(agent_id, bound_min, bound_max);

        std::vector<double> out_min;
        ConvertVector(bound_min, out_min);
        return out_min;
    }
    return std::vector<double>(0);
}

std::vector<double> cDeepMimicCore::BuildActionBoundMax(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXd bound_min;
        Eigen::VectorXd bound_max;
        rl_scene->BuildActionBounds(agent_id, bound_min, bound_max);

        std::vector<double> out_max;
        ConvertVector(bound_max, out_max);
        return out_max;
    }
    return std::vector<double>(0);
}

std::vector<int> cDeepMimicCore::BuildStateNormGroups(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXi groups;
        rl_scene->BuildStateNormGroups(agent_id, groups);

        std::vector<int> out_groups;
        ConvertVector(groups, out_groups);
        return out_groups;
    }
    return std::vector<int>(0);
}

std::vector<int> cDeepMimicCore::BuildGoalNormGroups(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        Eigen::VectorXi groups;
        rl_scene->BuildGoalNormGroups(agent_id, groups);

        std::vector<int> out_groups;
        ConvertVector(groups, out_groups);
        return out_groups;
    }
    return std::vector<int>(0);
}

double cDeepMimicCore::CalcReward(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        double r = rl_scene->CalcReward(agent_id);
        //        std::cout <<"[get reward] reward = " << r << std::endl;
        return r;
    }
    return 0;
}

/**
 * \brief       Calculate the dervative of d(reward)/d(action)
 * note that the action must be normalized
*/
#include "scenes/DrawSceneDiffImitate.h"
#include "scenes/SceneDiffImitate.h"
std::vector<std::vector<double>> cDeepMimicCore::CalcDRewardDAction() const
{
    std::vector<std::vector<double>> mat(0);
    if (EnableDraw() == true)
    {
        auto draw_res =
            std::dynamic_pointer_cast<cDrawSceneDiffImitate>(mRLScene);
        auto res =
            std::dynamic_pointer_cast<cSceneDiffImitate>(draw_res->GetScene());
        res->Test();
        // MIMIC_ASSERT(res != nullptr);
    }
    else
    {
        MIMIC_ASSERT(false);
    }

    MIMIC_WARN("CalcDRewardDAction hasn't been implemented, only do test");
    return mat;
}

double cDeepMimicCore::GetRewardMin(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetRewardMin(agent_id);
    }
    return 0;
}

double cDeepMimicCore::GetRewardMax(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetRewardMax(agent_id);
    }
    return 0;
}

double cDeepMimicCore::GetRewardFail(int agent_id)
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetRewardFail(agent_id);
    }
    return 0;
}

double cDeepMimicCore::GetRewardSucc(int agent_id)
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        return rl_scene->GetRewardSucc(agent_id);
    }
    return 0;
}

bool cDeepMimicCore::IsEpisodeEnd() const
{
    // 返回是episode结束
    return mScene->IsEpisodeEnd();
}

bool cDeepMimicCore::CheckValidEpisode() const
{
    return mScene->CheckValidEpisode();
}

int cDeepMimicCore::CheckTerminate(int agent_id) const
{
    const auto &rl_scene = GetRLScene();
    cRLScene::eTerminate terminated = cRLScene::eTerminateNull;
    if (rl_scene != nullptr)
    {
        terminated = rl_scene->CheckTerminate(agent_id);
    }
    return static_cast<int>(terminated);
}

void cDeepMimicCore::SetMode(int mode)
{
    assert(mode >= 0 && mode < cRLScene::eModeMax);
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        rl_scene->SetMode(static_cast<cRLScene::eMode>(mode));
    }
}

void cDeepMimicCore::SetSampleCount(int count)
{
    const auto &rl_scene = GetRLScene();
    if (rl_scene != nullptr)
    {
        rl_scene->SetSampleCount(count);
    }
}

void cDeepMimicCore::SetupScene()
{
    // what does a scene include?
    ClearScene(); // 将mscene指针置空

    std::string scene_name = "";
    mArgParser->ParseString(
        "scene", scene_name); // 文件中指定的imitate / train / run之类的

    mScene = nullptr;
    mRLScene = nullptr;
    // 根据是否绘制，创建不同的Scene子类对象
    if (EnableDraw())
    {
        cSceneBuilder::BuildDrawScene(scene_name, mScene);
    }
    else
    {
        cSceneBuilder::BuildScene(scene_name, mScene);
    }

    if (mScene != nullptr)
    {
        // there is a dynamic_cast: it means that if the scene type is
        // kin_char(display motion),  this ptr "mRLScene" would be NULL.
        mRLScene = std::dynamic_pointer_cast<cRLScene>(mScene);
        mScene->ParseArgs(mArgParser);
        mScene->Init();
        printf("Loaded scene: %s\n", mScene->GetName().c_str());
    }
}

void cDeepMimicCore::ClearScene() { mScene = nullptr; }

int cDeepMimicCore::GetCurrTime() const { return glutGet(GLUT_ELAPSED_TIME); }

void cDeepMimicCore::InitFrameBuffer()
{
    mDefaultFrameBuffer = std::unique_ptr<cTextureDesc>(
        new cTextureDesc(0, 0, 0, 1, 1, 1, GL_RGBA, GL_RGBA));
}

void cDeepMimicCore::CalcDeviceCoord(int pixel_x, int pixel_y,
                                     double &out_device_x,
                                     double &out_device_y) const
{
    double w = GetWinWidth();
    double h = GetWinHeight();

    out_device_x = static_cast<double>(pixel_x) / w;
    out_device_y = static_cast<double>(pixel_y) / h;
    out_device_x = (out_device_x - 0.5f) * 2.f;
    out_device_y = (out_device_y - 0.5f) * -2.f;
}

double cDeepMimicCore::GetAspectRatio()
{
    double aspect_ratio = static_cast<double>(GetWinWidth()) / GetWinHeight();
    return aspect_ratio;
}

void cDeepMimicCore::CopyFrame(cTextureDesc &src) const
{
    cDrawUtil::CopyTexture(src);
}

const std::shared_ptr<cRLScene> &cDeepMimicCore::GetRLScene() const
{
    return mRLScene;
}

void cDeepMimicCore::ConvertVector(const Eigen::VectorXd &in_vec,
                                   std::vector<double> &out_vec) const
{
    int size = static_cast<int>(in_vec.size());
    out_vec.resize(size);
    std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(double));
}

void cDeepMimicCore::ConvertVector(const Eigen::VectorXi &in_vec,
                                   std::vector<int> &out_vec) const
{
    int size = static_cast<int>(in_vec.size());
    out_vec.resize(size);
    std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(int));
}

void cDeepMimicCore::ConvertVector(const std::vector<double> &in_vec,
                                   Eigen::VectorXd &out_vec) const
{
    int size = static_cast<int>(in_vec.size());
    out_vec.resize(size);
    std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(double));
}

void cDeepMimicCore::ConvertVector(const std::vector<int> &in_vec,
                                   Eigen::VectorXi &out_vec) const
{
    int size = static_cast<int>(in_vec.size());
    out_vec.resize(size);
    std::memcpy(out_vec.data(), in_vec.data(), size * sizeof(int));
}