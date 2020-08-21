#include "SimCharBuilder.h"
#include "sim/SimItems/SimCharGeneral.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "util/LogUtil.h"
const std::string gCharName[cSimCharBuilder::eCharType::NUM_CHAR_TYPE] = {
    "invalid", "general", "lagragian"};

void cSimCharBuilder::CreateCharacter(
    eCharType char_type, std::shared_ptr<cSimCharacterBase> &out_char)
{
    if (char_type == eCharType::eCharBulletGeneral)
    {
        out_char = std::shared_ptr<cSimCharacter>(new cSimCharacter());
    }
    else if (char_type == eCharGeneralized)
    {
        out_char = std::shared_ptr<cSimCharacterGen>(new cSimCharacterGen());
    }
    else
    {
        MIMIC_ERROR("invalid char type {}", char_type);
    }
}

void cSimCharBuilder::ParseCharType(const std::string &char_type_str,
                                    eCharType &out_char_type)
{
    out_char_type = eCharType::eCharInvalid;
    for (int i = 0; i < eCharType::NUM_CHAR_TYPE; ++i)
    {
        const std::string &name = gCharName[i];
        if (char_type_str == name)
        {
            out_char_type = static_cast<eCharType>(i);
            break;
        }
    }

    if (out_char_type == eCharType::eCharInvalid)
    {
        MIMIC_ERROR("char type \"{}\" is invalid", char_type_str);
    }
}