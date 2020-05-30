#include "LogUtil.hpp"
#include <cstdarg>

tLogger cLogUtil::CreateLogger(const std::string & loggername)
{
    return spdlog::stdout_color_mt(loggername);
}

void cLogUtil::DropLogger(const std::string & loggername)
{
    spdlog::drop(loggername);
}

tLogger cLogUtil::GetLogger(const std::string & loggername)
{
    return spdlog::get(loggername);
}

void InfoPrintf(const tLogger & logger, const char * fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    // size_t real_size = vsnprintf(buf, buf_size, fmt, ap);
    
    cLogUtil::Printf(logger, eLogLevel::info, fmt, ap);
    va_end (ap);
}

void WarnPrintf(const tLogger & logger, const char * fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    // size_t real_size = vsnprintf(buf, buf_size, fmt, ap);
    
    cLogUtil::Printf(logger, eLogLevel::warn, fmt, ap);
    va_end (ap);
}

void ErrorPrintf(const tLogger & logger, const char * fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    // size_t real_size = vsnprintf(buf, buf_size, fmt, ap);
    
    cLogUtil::Printf(logger, eLogLevel::err, fmt, ap);
    va_end (ap);
}

void DebugPrintf(const tLogger & logger, const char * fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    // size_t real_size = vsnprintf(buf, buf_size, fmt, ap);
    
    cLogUtil::Printf(logger, eLogLevel::debug, fmt, ap);
    va_end (ap);
}

void cLogUtil::Printf(const tLogger & logger, eLogLevel level, const char * fmt, va_list args)
{
    size_t real_size = vsnprintf(buf, buf_size, fmt, args);
    
    const std::string & log = std::string(buf, std::min(real_size, buf_size));
    switch (level)
    {
        case eLogLevel::info: logger->info(log); break;
        case eLogLevel::debug: logger->debug(log); break;
        default:logger->error(log); break;
    }
}