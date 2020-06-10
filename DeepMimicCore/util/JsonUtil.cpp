#include "JsonUtil.h"
#include "FileUtil.h"
#include <iostream>
#include <fstream>
#include <memory>

tLogger cJsonUtil::mLogger = cLogUtil::CreateLogger("cJsonUtil");
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

bool cJsonUtil::LoadJson(const std::string & path, Json::Value & value)
{
	// cFileUtil::AddLock(path);
	// std::cout <<"parsing " << path << " begin \n";
    std::ifstream fin(path);
    if(fin.fail() == true)
    {
        std::cout << "[error] cJsonUtil::LoadJson file " << path <<" doesn't exist\n";
        return false;
    }
    Json::CharReaderBuilder rbuilder;
    std::string errs;
    bool parsingSuccessful = Json::parseFromStream(rbuilder, fin, &value, &errs);
    if (!parsingSuccessful)
    {
    // report to the user the failure and their locations in the document.
        std::cout  << "[error] cJsonUtil::LoadJson: Failed to parse json\n"
               << errs << std::endl;
               return false;
    }
	// std::cout <<"parsing " << path << " end \n";
	// cFileUtil::DeleteLock(path);
    return true;
}

bool cJsonUtil::WriteJson(const std::string & path, Json::Value & value, bool indent/* = true*/)
{
	// cFileUtil::AddLock(path);
    Json::StreamWriterBuilder builder;
	if(indent == false) builder.settings_["indentation"] = "";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	std::ofstream fout(path);
	if(fout.fail() == true)
	{
		mLogger->error("WriteJson open {} failed", path);
		exit(1);
	}
    writer->write(value, &fout);
	fout.close();
	// cFileUtil::DeleteLock(path);
	return fout.fail() == false;
}

#define JSONUTIL_ASSERT_NULL(root, data) (root.isMember(data))

int cJsonUtil::ParseAsInt(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsInt {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name].asInt();
}

std::string cJsonUtil::ParseAsString(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsString {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name].asString();
}

double cJsonUtil::ParseAsDouble(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsDouble {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name].asDouble();
}

float cJsonUtil::ParseAsFloat(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsFloat {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name].asFloat();
}

bool cJsonUtil::ParseAsBool(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsBool {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name].asBool();
}

Json::Value cJsonUtil::ParseAsValue(const std::string & data_field_name, const Json::Value & root)
{
	if(false == JSONUTIL_ASSERT_NULL(root, data_field_name))
	{
		mLogger->error("ParseAsValue {} failed", data_field_name.c_str());
		exit(0);
	}
	return root[data_field_name];
}