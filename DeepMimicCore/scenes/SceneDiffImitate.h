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
        DERIV_SINGLE_STEP_SUM, // sum all drda from previous action given
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
    tVectorXd CalcDPoseRewardDq();

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
    tMatrixXd CalcQ();    // we do not need to test Q
    void ClearPQBuffer(); // clear P buffer and Q buffer

    // 5. calc & test end effector reward deriv
    tVectorXd CalcDEndEffectorRewardDq() const;
    void TestDEndEffectorRewardDq();
    tVector CalcJointPosRel0(int id) const;
    tVector CalcJointPosRel1(int id) const;
    tMatrixXd CalcDJointPosRel0Dq(int id) const;
    void TestDJointPosRel0Dq(int id);
    double CalcEndEffectorErr(int id) const;
    tVectorXd CalcDEndEffectorErrDq(int id) const;
    void TestDEndEffectorErrDq(int id);

    // 6. calc & test root reward deriv
    double CalcRootErr() const;
    tVectorXd CalcDRootRewardDx();
    void TestDRootRewardDx();
    // d(root_rew)/d(root_err)
    // 6.1 root pos
    double CalcRootPosErr() const;
    tVectorXd CalcDRootPosErrDx() const;
    void TestDRootPosErrDx();
    // 6.2 root rot
    double CalcRootRotErr() const;
    tVectorXd CalcDRootRotErrDx();
    void TestDRootRotErrDx();
    // 6.3 root lin vel
    double CalcRootLinVelErr() const;
    tVectorXd CalcDRootLinVelErrDx() const;
    void TestDRootLinVelErrDx();
    // 6.4 root ang vel
    double CalcRootAngVelErr() const;
    tVectorXd CalcDRootAngVelErrDx() const;
    void TestDRootAngVelErrDx();

    // test the exponential relationship
    double CalcDEndEffectorRewardDErr(double err);
    void TestEndEffectorRewardByGivenErr();

    tEigenArr<tMatrixXd>
        mPBuffer; // buffer for storing P matrix, used in multistep mode
    tEigenArr<tMatrixXd>
        mQBuffer; // buffer for storing Q vector, used in multistep mode
    tEigenArr<tVectorXd> mDrdaSingleBuffer; // buffer for single step drda
    std::shared_ptr<cSimCharacterGen> GetDefaultGenChar() const;
    std::shared_ptr<cCtPDGenController> GetDefaultGenCtrl();

    eDerivMode mDerivMode; // the mode for CalcDRewardDAction
    bool
        mEnableTestDRewardDAction; // enable testing the derivative d(reawrd)/d(action) when CalcReward
    bool mDebugOutput;
};