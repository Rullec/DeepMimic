#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "util/MathUtil.h"
#include "util/ArgParser.h"
#include "util/Timer.h"

#include "ret_opt/RetOptImpl.h"

class cScene
{
public:

	virtual ~cScene();
	
	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();
	virtual void Clear();
	virtual void Reset();
	virtual void Update(double timestep);

	virtual void Draw();
	virtual void Keyboard(unsigned char key, double device_x, double device_y);
	virtual void MouseClick(int button, int state, double device_x, double device_y);
	virtual void MouseMove(double device_x, double device_y);
	virtual void Reshape(int w, int h);

	virtual void Shutdown();
	virtual bool IsDone() const;
	virtual double GetTime() const;
	virtual double GetMaxTime() const;

	virtual bool HasRandSeed() const;
	virtual void SetRandSeed(unsigned long seed);
	virtual unsigned long GetRandSeed() const;

	virtual bool IsEpisodeEnd() const;
	virtual bool CheckValidEpisode() const;

	virtual std::string GetName() const = 0;

	// ========================================================
	// For Body Shape Variation
	virtual void ChangeBodyShape(Eigen::VectorXd &body_param) {}
	virtual void RunRetargeting(cRetOptImpl::tParam& param);
	virtual void DumpMotionPool(const char* file) {}
    // ========================================================

protected:

	cRand mRand;
	bool mHasRandSeed;
	unsigned long mRandSeed;

	std::shared_ptr<cArgParser> mArgParser;
	cTimer::tParams mTimerParams;
	cTimer mTimer;

	cRetOptImpl* mRetOptImpl;

	cScene();

	virtual void ResetParams();
	virtual void ResetScene();

	virtual void InitTimers();
	virtual void ResetTimers();
	virtual void UpdateTimers(double timestep);
};