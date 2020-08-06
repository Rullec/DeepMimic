#pragma once

#include "SimCharacter.h"

class cSimCharBuilder
{
public:
    enum eCharType
    {
        eCharInvalid,
        eCharBulletGeneral,
        eCharGeneralized,
        NUM_CHAR_TYPE
    };
    static void CreateCharacter(eCharType char_type,
                                std::shared_ptr<cSimCharacterBase> &out_char);
    static void ParseCharType(const std::string &char_type_str,
                              eCharType &out_char_type);

protected:
};
