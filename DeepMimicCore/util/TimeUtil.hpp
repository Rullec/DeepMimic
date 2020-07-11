#ifndef _TIME_UTIL_H_
#define _TIME_UTIL_H_
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <ctime>
class cTimeUtil
{
public:
    // calculate a continuous segment of time 
    static void Begin(const std::string & name);
    static void End(const std::string & name);

    // calculate a discrete segment of time, lazy calculation until the final
    static void BeginLazy(const std::string & name);
    static void EndLazy(const std::string & name);
    static void ClearLazy(const std::string & name);
    static double GetAndClearTimeLazy(const std::string &name);

    static void BeginAvgLazy(const std::string& name);
    static void EndAvgLazy(const std::string& name);
    static double GetAndClearTimeAvgLazy(const std::string& name);
    static std::string GetSystemTime();
    static std::vector<std::string> GetNames();
private:
    inline static std::map<const std::string, std::chrono::high_resolution_clock::time_point> mTimeTable;          // record current time
    inline static std::map<const std::string, std::chrono::high_resolution_clock::time_point>::iterator time_it;

    inline static std::map<const std::string, double> mLazyTimeTable;      // record lazy accumulated time
    inline static std::map<const std::string, int> mLazyTimeCountTable;
};

#endif // # _TIME_UTIL_H_