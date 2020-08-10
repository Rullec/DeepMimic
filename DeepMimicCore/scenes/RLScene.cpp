#include "RLScene.h"
#include "util/LogUtil.hpp"
#include <iostream>
using namespace std;

cRLScene::cRLScene()
{
    mMode = eModeTrain;
    mSampleCount = 0;
}

cRLScene::~cRLScene() {}

void cRLScene::Init()
{
    cScene::Init();
    mSampleCount = 0;
}

void cRLScene::Clear()
{
    cScene::Clear();
    mSampleCount = 0;
}

double cRLScene::GetRewardFail(int agent_id) { return GetRewardMin(agent_id); }

double cRLScene::GetRewardSucc(int agent_id) { return GetRewardMax(agent_id); }

bool cRLScene::IsEpisodeEnd() const
{
    // episode是否结束?
    // std::cout <<"[end] bool cRLScene::IsEpisodeEnd() const" << std::endl;
    bool is_end = cScene::IsEpisodeEnd();
    if (is_end == true)
    {
        cTimer::tParams a = mTimer.GetParams();
        MIMIC_INFO("Timer said terminated, episode done, timer = {}, exp = {}",
                   mTimer.GetMaxTime(), a.mTimeExp);
    }
    eTerminate termin = eTerminateNull;
    for (int i = 0; i < GetNumAgents(); ++i)
    {
        termin = CheckTerminate(i); // 调用check Terminate函数
        if (termin !=
            eTerminateNull) // 只要不是无法判断，那么episode就是结束了。
        {
            is_end = true;

            MIMIC_INFO("Agent said terminated, episode done");
            break;
        }
    }
    return is_end;
}

cRLScene::eTerminate cRLScene::CheckTerminate(int agent_id) const
{
    return eTerminateNull;
}

void cRLScene::SetSampleCount(int count) { mSampleCount = count; }

void cRLScene::SetMode(eMode mode) { mMode = mode; }