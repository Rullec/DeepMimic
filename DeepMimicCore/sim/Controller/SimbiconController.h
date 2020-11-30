#pragma once
#include "sim/Controller/CharController.h"
#include "sim/Controller/DeepMimicCharController.h"
#include "sim/Controller/ExpPDController.h"

/**
 * \brief           SimBiCon controller 
*/
namespace Json
{
class Value;
};

class cSimbiconController : public cDeepMimicCharController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cSimbiconController();
    virtual void Init(cSimCharacterBase *character,
                      const std::string &param_file) override;
    virtual ~cSimbiconController() override;
    virtual std::string GetName() const;
    
protected:
    cExpPDController mPDCtrl;
    virtual void InitFSM(const Json::Value &conf);
};