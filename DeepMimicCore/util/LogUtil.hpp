#pragma once
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>
typedef std::shared_ptr<spdlog::logger> tLogger;
typedef spdlog::level::level_enum eLogLevel;

class cLogUtil{
public:
    static tLogger CreateLogger(const std::string & loggername);
    static void DropLogger(const std::string & loggername);
    static tLogger GetLogger(const std::string & loggername);
    static void Printf(const tLogger & logger, eLogLevel level, const char * fmt, va_list args);

private:
    inline const static size_t buf_size = 1000;
    inline static char buf[buf_size];
};

void InfoPrintf(const tLogger & logger, const char * fmt, ...);
void ErrorPrintf(const tLogger & logger, const char * fmt, ...);
void WarnPrintf(const tLogger & logger, const char * fmt, ...);
void DebugPrintf(const tLogger & logger, const char * fmt, ...);