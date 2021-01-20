#include "ArgParser.h"
#include "util/FileUtil.h"
#include "util/LogUtil.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

const char gKeyStart = '-';
const char gCommentStart = '#';

cArgParser::cArgParser() {}

cArgParser::~cArgParser() {}

cArgParser::cArgParser(const std::vector<std::string> &args) { LoadArgs(args); }

cArgParser::cArgParser(const std::string &file) { LoadFile(file); }

void cArgParser::LoadArgs(const std::vector<std::string> &arg_strs)
{
    /*
            之前在arg_strs 中存储了所有被分开的字符串
            但是，必须要区分key 和 value，Key前面是--
     */
    std::vector<std::string> vals; // vals是一个列表
    std::string curr_key = "";

    for (size_t i = 0; i < arg_strs.size(); ++i)
    {
        const std::string &str = arg_strs[i];
        if (!IsComment(str)) // 不是注释
            // (这也意味着注释只有一个效果，注释值的话后面还是1会算着)
            // 这样的逻辑其实不符合#后面啥都没有的原则
        {
            // 这个判断很有意思:
            // 如果str的开头是"keystart"的-，这个概念也应该抽象
            bool is_key = IsKey(str);
            if (is_key)
            {
                if (curr_key != "")
                {
                    bool in_table = mTable.find(curr_key) != mTable.end();
                    if (!in_table)
                    {
                        mTable[curr_key] = vals;
                        curr_key = "";
                    }
                }
                vals.clear();
                curr_key = str.substr(2, str.size());
            }
            else
            {
                vals.push_back(str);
            }
        }
    }

    if (curr_key != "")
    {
        bool in_table = mTable.find(curr_key) != mTable.end();
        if (!in_table)
        {
            mTable[curr_key] = vals;
        }
        curr_key = "";
    }
    vals.clear();
    // 就此建立完了一个好的参数map
}

void cArgParser::Clear() { mTable.clear(); }

bool cArgParser::LoadFile(const std::string &file)
{
    // file 传入: args/run_human_run_args.txt
    // LoadFile会解析该配置文件, 读取character, model等各种信息吧。
    mArgFilePath = file;
    FILE *file_ptr = cFileUtil::OpenFile(file.c_str(), "r");
    bool succ = (file_ptr != nullptr);

    // 文件输入流
    std::ifstream file_stream(file.c_str());
    std::string line_str;

    std::string str_buffer = "";
    std::vector<std::string> arg_strs;
    const std::string delims = " \t\n\r,";

    while (std::getline(file_stream, line_str))
    {
        if (line_str.size() > 0 && !IsComment(line_str))
        {
            // 该行有效而且不是注释
            for (size_t i = 0; i < line_str.size(); ++i)
            {
                char curr_char = line_str[i];
                // find first of
                // 返回curr_char任意字符在s中第一次出现的下标位置，没找到返回Npos
                // string::npos: -1，通常用来和返回值比较判字符串尾
                if (delims.find_first_of(curr_char) != std::string::npos)
                {
                    // 如果如果当前字符出现在delims黑名单中，代表间隔了，就要存储
                    if (str_buffer != "")
                    {
                        arg_strs.push_back(str_buffer);
                        str_buffer = "";
                    }
                }
                else
                {
                    // 否则继续存buffer
                    str_buffer += curr_char;
                }
            }

            // dump everything else out
            if (str_buffer != "")
            {
                // 拿到的都要放到str_buffer中
                arg_strs.push_back(str_buffer);
                str_buffer = "";
            }
        }
    }

    cFileUtil::CloseFile(file_ptr);

    // 把每行都存起来以后，要load Args
    LoadArgs(arg_strs);

    return succ;
}

bool cArgParser::ParseString(const std::string &key, std::string &out) const
{
    auto it = mTable.find(key); // mTable key -value pair for storing parameters
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        out = vals[0]; // 这只能取出一个参数来，就是vals[0]第一个
        return true;
    }
    return false;
}

bool cArgParser::ParseStringCritic(const std::string &key,
                                   std::string &out) const
{
    if (false == ParseString(key, out))
    {
        MIMIC_ERROR("parse string key {} failed", key);
    }
    return false;
}

bool cArgParser::ParseStrings(const std::string &key,
                              std::vector<std::string> &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        out = vals;
        return true;
    }
    return false;
}

bool cArgParser::ParseInt(const std::string &key, int &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        out = std::atoi(vals[0].c_str());
        return true;
    }
    return false;
}

bool cArgParser::ParseInts(const std::string &key, std::vector<int> &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        size_t num_vals = vals.size();
        out.clear();
        out.reserve(num_vals);
        for (int i = 0; i < num_vals; ++i)
        {
            out.push_back(std::atoi(vals[i].c_str()));
        }
        return true;
    }
    return false;
}

bool cArgParser::ParseDouble(const std::string &key, double &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        out = std::atof(vals[0].c_str());
        return true;
    }
    return false;
}

bool cArgParser::ParseDoubles(const std::string &key,
                              std::vector<double> &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        size_t num_vals = vals.size();
        out.clear();
        out.reserve(num_vals);
        for (int i = 0; i < num_vals; ++i)
        {
            out.push_back(std::atof(vals[i].c_str()));
        }
        return true;
    }
    return false;
}

bool cArgParser::ParseBool(const std::string &key, bool &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        out = ParseBool(vals[0]);
        return true;
    }
    return false;
}

bool cArgParser::ParseIntCritic(const std::string &key, int &out) const
{
    if (ParseInt(key, out) == false)
    {
        MIMIC_ERROR("parse key {} failed", key);
    }
    return true;
}

bool cArgParser::ParseDoubleCritic(const std::string &key, double &out) const
{
    if (ParseDouble(key, out) == false)
    {
        MIMIC_ERROR("parse key {} failed", key);
    }
    return true;
}

bool cArgParser::ParseBoolCritic(const std::string &key, bool &out) const
{
    if (ParseBool(key, out) == false)
    {
        MIMIC_ERROR("parse key {} failed", key);
    }

    return true;
}

bool cArgParser::ParseBools(const std::string &key,
                            std::vector<bool> &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        size_t num_vals = vals.size();
        out.clear();
        out.reserve(num_vals);
        for (int i = 0; i < num_vals; ++i)
        {
            out.push_back(ParseBool(vals[i]));
        }
        return true;
    }
    return false;
}

bool cArgParser::ParseVector(const std::string &key, tVector &out) const
{
    auto it = mTable.find(key);
    if (it != mTable.end())
    {
        const auto &vals = it->second;
        size_t num_vals =
            std::min(vals.size(), static_cast<size_t>(out.size()));
        for (int i = 0; i < num_vals; ++i)
        {
            out[i] = std::atof(vals[i].c_str());
        }
        return true;
    }
    return false;
}

bool cArgParser::IsComment(const std::string &str) const
{
    bool is_comment = false;
    if (str.size() > 0)
    {
        is_comment = str[0] == gCommentStart;
    }
    return is_comment;
}

bool cArgParser::IsKey(const std::string &str) const
{
    size_t len = str.size();
    if (len < 3)
    {
        return false;
    }
    else
    {
        if (str[0] == gKeyStart && str[1] == gKeyStart)
        {
            return true;
        }
    }
    return false;
}

int cArgParser::GetNumArgs() const { return static_cast<int>(mTable.size()); }

std::string cArgParser::GetArgFilePath() const { return mArgFilePath; }
bool cArgParser::ParseBool(const std::string &str) const
{
    bool val = false;
    if (str == "true" || str == "1" || str == "True" || str == "T" ||
        str == "t")
    {
        val = true;
    }
    return val;
}