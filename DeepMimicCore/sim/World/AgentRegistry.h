#pragma once

#include "sim/Controller/CharController.h"
#include "util/IndexManager.h"

class cAgentRegistry
{
public:
    cAgentRegistry();
    virtual ~cAgentRegistry();

    virtual void Clear();
    virtual int GetNumAgents() const;
    virtual int AddAgent(const std::shared_ptr<cCharController> &agent,
                         cSimCharacterBase *character);

    virtual const std::shared_ptr<cCharController> &GetAgent(int id) const;
    virtual cSimCharacterBase *GetChar(int id) const;
    virtual void PrintInfo() const;

protected:
    cIndexManager mIDManager;
    std::vector<std::shared_ptr<cCharController>> mAgents;
    std::vector<cSimCharacterBase *> mChars;
};