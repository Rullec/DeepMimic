#pragma once
#include "sim/Controller/CtController.h"
class cCtPDController : public virtual cCtController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cCtPDController();
    virtual ~cCtPDController() = default;

    virtual void SetEnableSolvePDTargetTest(bool);
    virtual const tVectorXd &GetCurAction() const = 0;
    virtual const tVectorXd &GetCurPDTargetPose() const = 0;

    // virtual void CalcPDTarget(const Eigen::VectorXd &force,
    //                           Eigen::VectorXd out_pd_target) = 0;
    virtual void CalcActionByTargetPose(tVectorXd &pd_target) = 0;
    virtual void CalcPDTargetByTorque(double dt, const tVectorXd &pose,
                                      const tVectorXd &vel,
                                      const tVectorXd &torque,
                                      tVectorXd &pd_target) = 0;
    virtual void UpdateTimeOnly(double timestep);

protected:
    bool mEnableSolvePDTargetTest;
};