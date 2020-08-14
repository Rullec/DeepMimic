#pragma once
#include "util/MathUtil.h"

class cSimCharacterGen;
/**
 * Stable PDController for SimCharaterGen (Lagragian based character)
 */
class cImpPDGenController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    cImpPDGenController();
    virtual ~cImpPDGenController();
    virtual void Init(cSimCharacterGen *gen_char, const tVectorXd &kp,
                      const tVectorXd &kd);

    virtual void Clear();
    virtual void UpdateControlForce(double dt, tVectorXd &out_tau);
    virtual void SetPDTarget(const tVectorXd &q, const tVectorXd &qdot);

protected:
    tVectorXd mKp, mKd; // Kp & Kd controll parameter
    cSimCharacterGen *mChar;
    tVectorXd mTarget_q, mTarget_qdot;

    void InitGains(const tVectorXd &kp, const tVectorXd &kd);
    void VerifyController();
    int GetPDTargetSize();
    void CheckVelExplode();
    void PostProcessControlForce(tVectorXd & out_tau);
};
