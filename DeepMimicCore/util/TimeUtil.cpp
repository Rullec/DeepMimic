#include "TimeUtil.hpp"
#include <ctime>
#include <iostream>
#include <cassert>
using namespace std;
using namespace std::chrono;

void cTimeUtil::Begin(const std::string & name)
{
    mTimeTable[name] = high_resolution_clock::now();
}

void cTimeUtil::End(const std::string & name)
{
    time_it = mTimeTable.find(name);
    if(time_it == mTimeTable.end())
    {
        std::cout <<"[error] cTimeUtil::End No static info about " << name << std::endl;
        exit(1);
    }

    std::cout <<"[log] " << name << " cost time = " << \
    (high_resolution_clock::now() - time_it->second).count() * 1e-6 <<" ms\n";
    mTimeTable.erase(time_it);
}

std::string cTimeUtil::GetSystemTime()
{
    // not thread safe
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(30, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
}

void cTimeUtil::BeginLazy(const std::string & name)
{
    cTimeUtil::Begin(name);
}

void cTimeUtil::EndLazy(const std::string & name)
{
    time_it = mTimeTable.find(name);
    if(time_it == mTimeTable.end())
    {
        std::cout <<"[error] cTimeUtil::End No static info about " << name << std::endl;
        exit(1);
    }

    double t = (high_resolution_clock::now() - time_it->second).count() * 1e-6;
    mLazyTimeTable[name] += t;

}

void cTimeUtil::ClearLazy(const std::string & name)
{
    std::cout <<"[log] segment lazy " << name << " cost time = " << mLazyTimeTable[name] << " ms\n";
    mLazyTimeTable[name] = 0;
}

double cTimeUtil::GetAndClearTimeLazy(const std::string &name) {
    if (mLazyTimeTable.find(name) == mLazyTimeTable.end()) {
        return 0;
    }
    double t = mLazyTimeTable[name];
    mLazyTimeTable[name] = 0;
    return t;
}

void cTimeUtil::BeginAvgLazy(const string &name) {
    cTimeUtil::BeginLazy(name);
}

void cTimeUtil::EndAvgLazy(const string &name) {
    EndLazy(name);
    mLazyTimeCountTable[name] += 1;
}

double cTimeUtil::GetAndClearTimeAvgLazy(const string &name) {
    double t = GetAndClearTimeLazy(name);
    if (t == 0) return t;
    int counts = mLazyTimeCountTable[name];
    assert(counts != 0);
    mLazyTimeCountTable[name] = 0;
    return t / static_cast<double >(counts);
}

std::vector<std::string> cTimeUtil::GetNames() {
    std::vector<std::string> names;
    for(auto itr = mLazyTimeTable.begin(); itr != mLazyTimeTable.end();++itr) {
        names.push_back(itr->first);
    }
    return names;
}
