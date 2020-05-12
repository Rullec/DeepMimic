#include "cTimeUtil.hpp"
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>
#include <map>

using namespace std;
using namespace std::chrono;

std::map<const std::string, high_resolution_clock::time_point> mTimeTable;
std::map<const std::string, high_resolution_clock::time_point>::iterator time_it;
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