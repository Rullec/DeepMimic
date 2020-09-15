#include "Timer.h"
#include "LogUtil.h"
#include "MPIUtil.h"
#include "MathUtil.h"
#include <iostream>
using namespace std;

cTimer::tParams::tParams()
{
    mType = eTypeUniform;
    mTimeMin = std::numeric_limits<double>::infinity();
    mTimeMax = std::numeric_limits<double>::infinity();
    mTimeExp = 1;
}

cTimer::tParams cTimer::tParams::Blend(const tParams &other, double lerp)
{
    tParams blend_params;
    blend_params.mType = mType;
    blend_params.mTimeMin = cMathUtil::Lerp(lerp, mTimeMin, other.mTimeMin);
    blend_params.mTimeMax = cMathUtil::Lerp(lerp, mTimeMax, other.mTimeMax);
    blend_params.mTimeExp = cMathUtil::Lerp(lerp, mTimeExp, other.mTimeExp);
    return blend_params;
}

cTimer::cTimer() { Reset(); }

cTimer::eType cTimer::ParseTypeStr(const std::string &str)
{
    eType timer_type = eTypeUniform;
    if (str == "" || str == "uniform")
    {
    }
    else if (str == "exp")
    {
        timer_type = eTypeExp;
    }
    else
    {
        printf("Unsupported timer type %s\n", str.c_str());
        assert(false);
    }
    return timer_type;
}

cTimer::~cTimer() {}

void cTimer::Init(const tParams &params)
{
    SetParams(params);
    Reset();
}

void cTimer::Reset()
{

    mTime = 0;
    double max_time = 0;
    switch (mParams.mType)
    {
    case eTypeUniform:
    { // uniform: 设置最大时间为两者之间: min -
        // max之间。如果max和min相同，则这就是最大时间
        // *************Now We take this method, and the time min & time max is
        // the same***********
        max_time = cMathUtil::RandDouble(mParams.mTimeMin, mParams.mTimeMax);
        // MIMIC_DEBUG("Timer Reset uniform, min {}. max {}, max_time res {}",
        //             mParams.mTimeMin, mParams.mTimeMax, max_time);
        break;
    }
    case eTypeExp:
    { // 否则，最大时间被设置为 min +
        // 一个指数性质的随机数和正经max的最小值(也是在min
        // max之间，只是更加偏重min...)
        max_time =
            mParams.mTimeMin + cMathUtil::RandDoubleExp(1 / mParams.mTimeExp);
        double inter_res = max_time;
        max_time = std::min(max_time, mParams.mTimeMax);
        // MIMIC_DEBUG("Timer Reset exp, min {}. max {}, exp {}, intermediate
        // max "
        //             "{}, max_time res {}",
        //             mParams.mTimeMin, mParams.mTimeMax, mParams.mTimeExp,
        //             inter_res, max_time);
        break;
    }
    default:
        assert(false); // Unsupported timer type
        break;
    }
    // MIMIC_DEBUG("Timer Reset max time {}", max_time);
    /*
     Timer Reset max time 0.5000000000000017
        Timer Reset uniform, min 0.5000000000000017.
            max 0.5000000000000017,
            max_time res 0.5000000000000017
        Timer Reset uniform, min 0.5000000000000017.
            max 0.5000000000000017,
            max_time res 0.5000000000000017
        Timer Reset max time 0.5000000000000017
    */
    SetMaxTime(max_time);
    // std::cout <<"Reset cTime, max = " << max_time << std::endl;
    // abort();
}

void cTimer::Update(double timestep) { mTime += timestep; }

bool cTimer::IsEnd() const { return mTime >= mMaxTime; }

double cTimer::GetTime() const { return mTime; }

double cTimer::GetMaxTime() const { return mMaxTime; }

void cTimer::SetMaxTime(double time) { mMaxTime = time; }

const cTimer::tParams &cTimer::GetParams() const { return mParams; }

void cTimer::SetParams(const tParams &params)
{
    // MIMIC_DEBUG("Timer Set params, min {}, max {}, exp {}", params.mTimeMin,
    //             params.mTimeMax, params.mTimeExp);
    // now the time min and time max is the same, so we only take the exact
    // value

    // for example
    /*
    Timer Set params,
        min 0.500000000002324,
        max 0.500000000002324, exp 0.200000000001168
    Timer Set params,
        min 0.500000000002324,
        max 0.500000000002324, exp 0.200000000001168
    Timer Set params,
        min 0.500000000002324,
        max 0.500000000002324, exp 0.200000000001168
    */
    mParams = params;
}