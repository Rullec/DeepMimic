#include "JsonUtil.h"
#include "FileUtil.h"
#include <iostream>
#include <fstream>

std::string cJsonUtil::BuildVectorJson(const tVector& vec)
{
	std::string json = "";
	for (int i = 0; i < vec.size(); ++i)
	{
		if (i != 0)
		{
			json += ", ";
		}
		json += std::to_string(vec[i]);
	}
	json = "[" + json + "]";
	return json;
}

bool cJsonUtil::ReadVectorJson(const Json::Value& root, tVector& out_vec)
{
	bool succ = false;
	int num_vals = root.size();
	assert(num_vals <= 4);
	num_vals = std::min(num_vals, static_cast<int>(out_vec.size()));

	if (root.isArray())
	{
		out_vec.setZero();
		for (int i = 0; i < num_vals; ++i)
		{
			Json::Value json_elem = root.get(i, 0);
			out_vec[i] = json_elem.asDouble();
		}
		succ = true;
	}

	return succ;
}

std::string cJsonUtil::BuildVectorJson(const Eigen::VectorXd& vec)
{
	std::string json = BuildVectorString(vec);
	json = "[" + json + "]";
	return json;
}

std::string cJsonUtil::BuildVectorString(const Eigen::VectorXd& vec)
{
	std::string str = "";
	char str_buffer[32];
	for (int i = 0; i < vec.size(); ++i)
	{
		if (i != 0)
		{
			str += ",";
		}
		sprintf(str_buffer, "%20.10f", vec[i]);
		str += std::string(str_buffer);
	}
	return str;
}

bool cJsonUtil::ReadVectorJson(const Json::Value& root, Eigen::VectorXd& out_vec)
{
	bool succ = false;
	int num_vals = root.size();
	
	if (root.isArray())
	{
		out_vec.resize(num_vals);
		for (int i = 0; i < num_vals; ++i)
		{
			Json::Value json_elem = root.get(i, 0);
			out_vec[i] = json_elem.asDouble();
		}
		succ = true;
	}

	return succ;
}

bool cJsonUtil::ParseJson(const std::string & path, Json::Value & value)
{
    std::ifstream fin(path);
    if(fin.fail() == true)
    {
        std::cout << "[error] cJsonUtil::ParseJson file " << path <<" doesn't exist\n";
        return false;
    }
    Json::CharReaderBuilder rbuilder;
    std::string errs;
    bool parsingSuccessful = Json::parseFromStream(rbuilder, fin, &value, &errs);
    if (!parsingSuccessful)
    {
    // report to the user the failure and their locations in the document.
        std::cout  << "[error] cJsonUtil::ParseJson: Failed to parse configuration\n"
               << errs << std::endl;
               return false;
    }
    return true;
}

bool cJsonUtil::WriteJson(const std::string & path, Json::Value & value, bool indent/* = true*/)
{
    Json::StreamWriterBuilder builder;
	if(indent == false) builder.settings_["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	std::ofstream fout(path);
    writer->write(value, &fout);
	return fout.fail() == false;
}