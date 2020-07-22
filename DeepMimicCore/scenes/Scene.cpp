#include "Scene.h"
#include <iostream>

cScene::cScene()
{
	mRand.Seed(cMathUtil::RandUint());
	mRandSeed = 0;
	mHasRandSeed = false;
	mRetOptImpl = nullptr;
}

cScene::~cScene()
{
}

void cScene::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	mArgParser = parser;

	std::string timer_type_str = "";
	mArgParser->ParseString("timer_type", timer_type_str);
	mTimerParams.mType = cTimer::ParseTypeStr(timer_type_str);
	mArgParser->ParseDouble("time_lim_min", mTimerParams.mTimeMin);
	mArgParser->ParseDouble("time_lim_max", mTimerParams.mTimeMax);
	mArgParser->ParseDouble("time_lim_exp", mTimerParams.mTimeExp);
}

void cScene::Init()
{
	if (HasRandSeed())
	{
		SetRandSeed(mRandSeed);
	}

	InitTimers();
	ResetParams();
}

void cScene::Clear()
{
	ResetParams();
}

void cScene::Reset()
{
	ResetScene();
}

void cScene::Update(double timestep)
{
	UpdateTimers(timestep);
}

void cScene::Draw()
{
}

void cScene::Keyboard(unsigned char key, double device_x, double device_y)
{
}

void cScene::MouseClick(int button, int state, double device_x, double device_y)
{
}

void cScene::MouseMove(double device_x, double device_y)
{
}


void cScene::Reshape(int w, int h)
{
}

void cScene::Shutdown()
{
}

bool cScene::IsDone() const
{
	return false;
}

double cScene::GetTime() const
{
	return mTimer.GetTime();
}

bool cScene::HasRandSeed() const
{
	return mHasRandSeed;
}

void cScene::SetRandSeed(unsigned long seed)
{
	mHasRandSeed = true;
	mRandSeed = seed;
	mRand.Seed(seed);
}

unsigned long cScene::GetRandSeed() const
{
	return mRandSeed;
}

bool cScene::IsEpisodeEnd() const
{
	// std::cout << "bool cScene::IsEpisodeEnd() const " << mTimer.GetTime() <<" " << mTimer.GetMaxTime();
	bool is_end = mTimer.IsEnd();
	return is_end;
}

bool cScene::CheckValidEpisode() const
{
	return true;
}

void cScene::ResetParams()
{
	ResetTimers();
}

void cScene::ResetScene()
{
	ResetParams();
}

void cScene::InitTimers()
{
	mTimer.Init(mTimerParams);
}

void cScene::ResetTimers()
{
	mTimer.Reset();
}

void cScene::UpdateTimers(double timestep)
{
	// std::cout <<"void cScene::UpdateTimers(double timestep)" << std::endl;
	mTimer.Update(timestep);
}

void cScene::RunRetargeting(cRetOptImpl::tParam &param) {
    if (this->mRetOptImpl == nullptr) {
        mRetOptImpl = new cRetOptImpl();
    }
    mRetOptImpl->Run(param);
}

double cScene::GetMaxTime() const {
    return mTimer.GetMaxTime();
}
