#pragma once

#include "CtPDController.h"
#include "sim/Controller/ImpPDController.h"

class cCtPDFeaController : public virtual cCtPDController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cCtPDFeaController();
    virtual ~cCtPDFeaController();

    virtual void Reset();
    virtual void Clear();

    virtual void SetEnableSolvePDTargetTest(bool);
    virtual void SetGravity(const tVector &gravity);
    virtual const tVectorXd &GetCurAction() const override;
    virtual const tVectorXd &GetCurPDTargetPose() const override;
    virtual std::string GetName() const;

    // virtual void CalcPDTarget(const Eigen::VectorXd &torque_,
    //                           Eigen::VectorXd out_pd_target) override;
    virtual void CalcPDTargetByTorque(double dt, const tVectorXd &pose,
                                      const tVectorXd &vel,
                                      const tVectorXd &torque,
                                      tVectorXd &pd_target) override;
    virtual void CalcActionByTargetPose(tVectorXd &pd_target) override;

protected:
    cImpPDController
        mPDCtrl; // 他才是关键的执行器，所有的ApplyAction都是要向这个里面放东西而已。

    tVector mGravity;
    tVectorXd mCurAction, mCurPDTargetPose; // record current action and pd
                                            // target pose of this char

    virtual bool ParseParams(const Json::Value &json);

    virtual void UpdateBuildTau(double time_step, Eigen::VectorXd &out_tau);
    virtual void SetupPDControllers(const Json::Value &json,
                                    const tVector &gravity);
    virtual void UpdatePDCtrls(double time_step, Eigen::VectorXd &out_tau);
    virtual void ApplyAction(const Eigen::VectorXd &action);
    virtual void BuildJointActionBounds(int joint_id, Eigen::VectorXd &out_min,
                                        Eigen::VectorXd &out_max) const;
    virtual void BuildJointActionOffsetScale(int joint_id,
                                             Eigen::VectorXd &out_offset,
                                             Eigen::VectorXd &out_scale) const;
    virtual void ConvertActionToTargetPose(int joint_id,
                                           Eigen::VectorXd &out_theta) const;
    virtual void ConvertTargetPoseToAction(int joint_id,
                                           Eigen::VectorXd &out_theta) const;

    virtual cKinTree::eJointType GetJointType(int joint_id) const;

    virtual void SetPDTargets(const Eigen::VectorXd &targets);
};