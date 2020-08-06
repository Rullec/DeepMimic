#pragma once

// #include "sim/Controller/CharController.h"
#include "sim/Controller/GeneralizedCharController.h"
#include "util/IndexManager.h"

class cSimCharacterGeneralizedeneralized;
class cAgentRegistryG
{
public:
    cAgentRegistryG();
    virtual ~cAgentRegistryG();

    virtual void Clear();
    virtual int GetNumAgents() const;
    virtual int AddAgent(const std::shared_ptr<cGCharController> &agent,
                         cSimCharacterGeneralized *character);

    virtual const std::shared_ptr<cGCharController> &GetAgent(int id) const;
    virtual cSimCharacterGeneralized *GetChar(int id) const;
    virtual void PrintInfo() const;

protected:
    cIndexManager mIDManager;
    std::vector<std::shared_ptr<cGCharController>> mAgents;
    std::vector<cSimCharacterGeneralized *> mChars;
};