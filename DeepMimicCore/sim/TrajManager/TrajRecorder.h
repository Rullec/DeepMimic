#pragma once
#include "Trajectory.h"
#include "util/MathUtil.h"
class cSceneImitate;

// load mode: Set up different flag when we load different data.
// It takes an effect on the behavior of our ID Solver

// load info struct. Namely it is used for storaging the loaded info from
// fril.
class cCtPDFeaController;

/**
 * \brief                   Trajectory recorder
 */
class cTrajRecorder
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cTrajRecorder(cSceneImitate *scene, const std::string &conf);
    virtual ~cTrajRecorder();
    virtual void PreSim();
    virtual void PostSim();
    virtual void Reset();
    virtual void SetTimestep(double);

protected:
    cSimCharacterBase *mSimChar;
    cSceneImitate *mScene;
    tSaveInfo mSaveInfo;
    std::string mTrajSavePath;
    int mRecordMaxFrame;
    virtual void ParseConfig(const std::string &conf);
    virtual void RecordActiveForceGen();
    virtual void ReadContactInfo();
    virtual void ReadContactInfoGen();
    virtual void ReadContactInfoRaw();
    virtual void VerifyDynamicsEquation();
};
