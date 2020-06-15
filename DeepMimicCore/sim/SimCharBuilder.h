#pragma once

#include "sim/SimCharacter.h"

class cSimCharBuilder
{
public:
	
	enum eCharType
	{
		eCharNone,
		cCharGeneral,
		cCharVarLinks,
		eCharMax
	};
	// 这个builder类里面就只有两个静态函数，所以说这个builder只是一个封装
	// 防止函数散落在外面。
	static void CreateCharacter(eCharType char_type, std::shared_ptr<cSimCharacter>& out_char);
	static void ParseCharType(const std::string& char_type_str, eCharType& out_char_type);
	
protected:

};
