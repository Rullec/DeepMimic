#pragma once
#include "sim/Controller/CharController.h"
#include "sim/Controller/CtPDController.h"
#include "sim/Controller/ExpPDController.h"

namespace Json
{
class Value;
};

/**
 * \brief           SimBiCon controller 
*/
class cSimbiconController : public cCtPDController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cSimbiconController();
    virtual void Init(cSimCharacterBase *character,
                      const std::string &param_file) override;
    virtual ~cSimbiconController() override;
    virtual void UpdateCalcTau(double timestep,
                               Eigen::VectorXd &out_tau) override final;
    virtual void UpdateBuildTau(double time_step,
                                Eigen::VectorXd &out_tau) override final;
    virtual void UpdateApplyTau(const tVectorXd &out_tau) override final;
    virtual void Reset() override final;
    virtual void Clear() override final;

    virtual std::string GetName() const;
    virtual const tVectorXd &GetCurAction() const override final;
    virtual const tVectorXd &GetCurPDTargetPose() const override final;
    virtual void CalcActionByTargetPose(tVectorXd &pd_target) override final;
    virtual void CalcPDTargetByTorque(double dt, const tVectorXd &pose,
                                      const tVectorXd &vel,
                                      const tVectorXd &torque,
                                      tVectorXd &pd_target) override final;

protected:
    cExpPDController mPDCtrl;
    virtual void InitFSM(const Json::Value &conf);
    virtual void SetTargetTheta(const tVectorXd &tar_pose);
    virtual void SetTargetVel(const tVectorXd &tar_vel);
    virtual void VerifyControlForce(const tVectorXd &force);
};