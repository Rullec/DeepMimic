#pragma once
#include "SceneImitate.h"

/**
 * \brief       Differential imitate scene
*/
class cSimCharacterGen;
class cSceneDiffImitate : public cSceneImitate
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cSceneDiffImitate();
    virtual ~cSceneDiffImitate();
    virtual void Init() override;
    tVectorXd CalcDRewardDAction();
    void Test();
    virtual double CalcRewardImitate(cSimCharacterBase &sim_char,
                                     cKinCharacter &ref_char) const;

protected:
    // 1. calc pose reward deriv methods
    void TestDRootRotErrDpose0();
    void TestDJointPoseErrDpose0();
    void TestDPoseRewardDpose0();
    void TestDPoseRewardDq();
    tVectorXd CalcDPoseRewardDpose0();
    tMatrixXd CalcDPoseRewardDq();

    // 2. calc vel reward deriv methods
    void TestDVelRewardDvel0();
    void TestDVelRewardDqdot();
    tVectorXd CalcDVelRewardDvel0();
    tVectorXd CalcDVelRewardDqdot();

    // 3. calculate dpose0dq and dvel0/dqdot
    tVectorXd CalcDrDxcur();
    tVectorXd CalcDrDa(); // calc d(reward)/d(action)
    void TestDrDxcur();
    void TestDrDa();      // calc d(reward)/d(action)
    std::shared_ptr<cSimCharacterGen> GetDefaultGenChar();
};