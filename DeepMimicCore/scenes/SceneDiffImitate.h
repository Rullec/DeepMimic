#pragma once
#include "SceneImitate.h"
/**
 * \brief       Differential imitate scene
*/
class cSimCharacterGen;
class cCtPDGenController;
class cSceneDiffImitate : public cSceneImitate
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    enum eDerivMode // the mode for calculate drda
    {
        DERIV_SINGLE_STEP, // only use the nearest frame data to calcualte drda
        DERIV_MULTI_STEPS, // use the past, history data to calc drda (approximately)
        NUM_DERIV_MODE,
    };
    static const std::string gDerivModeStr[NUM_DERIV_MODE];

    cSceneDiffImitate();
    virtual ~cSceneDiffImitate();
    virtual void ParseArgs(const std::shared_ptr<cArgParser> &parser);
    virtual void Init() override;
    tVectorXd CalcDRewardDAction();
    void Test();
    virtual double CalcRewardImitate(cSimCharacterBase &sim_char,
                                     cKinCharacter &ref_char) const;
    static eDerivMode ParseDerivMode(std::string str);
    virtual void Reset() override;
    virtual void Update(double dt) override;
    virtual void SetAction(int agent_id,
                           const Eigen::VectorXd &action) override;

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
    tMatrixXd CalcDxurDa(); // calc d(xcur)/d(action)
    tMatrixXd CalcDxurDa_SingleStep();
    tVectorXd CalcDxurDa_MultiStep();
    void TestDrDxcur();
    void TestDRewardDAction(); // calc d(reward)/d(action)

    // 4. multistep buffer methods
    void TestP();
    tMatrixXd CalcP();
    tMatrixXd CalcQ(); // we do not need to test Q
    void ClearPQBuffer();
    tEigenArr<tMatrixXd> mPBuffer;
    tEigenArr<tMatrixXd> mQBuffer;

    std::shared_ptr<cSimCharacterGen> GetDefaultGenChar();
    std::shared_ptr<cCtPDGenController> GetDefaultGenCtrl();

    eDerivMode mDerivMode; // the mode for CalcDRewardDAction
    bool
        mEnableTestDRewardDAction; // enable testing the derivative d(reawrd)/d(action) when CalcReward
    bool mDebugOutput;
};