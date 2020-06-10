#include "TimeUtil.hpp"
#include <ctime>
#include <iostream>

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
    s.resize(s.find('\0'));
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

    mLazyTimeTable[name] += (high_resolution_clock::now() - time_it->second).count() * 1e-6;

}

void cTimeUtil::ClearLazy(const std::string & name)
{
    std::cout <<"[log] segment lazy " << name << " cost time = " << mLazyTimeTable[name] << " ms\n";
    mLazyTimeTable[name] = 0;
}