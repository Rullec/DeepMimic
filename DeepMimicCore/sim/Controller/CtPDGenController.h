#pragma once
#include "sim/Controller/CtController.h"

class cImpPDGenController;

/**
 * PD controller for SimCharacterGen. It controls the state and action define
 * for the python training agent
 */
struct tLoadInfo;
class cCtPDGenController : public virtual cCtController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    cCtPDGenController();
    virtual ~cCtPDGenController();

    virtual void Init(cSimCharacterBase *character,
                      const std::string &param_file);
    virtual void Reset() override final;
    virtual void Clear() override final;

    virtual void SetGravity(const tVector &g);

    virtual std::string GetName() const;

    virtual const tVectorXd &GetCurAction();

    virtual void
    BuildStateOffsetScale(Eigen::VectorXd &out_offset,
                          Eigen::VectorXd &out_scale) const override;
    virtual void BuildActionBounds(Eigen::VectorXd &out_min,
                                   Eigen::VectorXd &out_max) const override;
    virtual void
    BuildActionOffsetScale(Eigen::VectorXd &out_offset,
                           Eigen::VectorXd &out_scale) const override;
    virtual int GetActionSize() const override;
    virtual void SetGuidedControlInfo(bool enable,
                                      const std::string &guide_file);
    virtual void ConvertTargetPoseToActionFullsize(tVectorXd &pd_target);

protected:
    cImpPDGenController *mPDGenController;
    tVector mGravity;
    tVectorXd mCurAction, mCurPDTargetPose;
    tLoadInfo *mLoadInfo;
    bool mEnableGuidedAction;
    std::string mGuidedTrajFile;
    int mInternalFrameId; // internal frame id counting
    virtual bool ParseParams(const Json::Value &json);

    virtual void UpdateBuildTau(double time_step, Eigen::VectorXd &out_tau);
    virtual void UpdateBuildTauPD(double time_step, Eigen::VectorXd &out_tau);
    virtual void UpdateBuildTauGuided(double time_step,
                                      Eigen::VectorXd &out_tau);
    virtual void PostUpdateBuildTau();
    virtual void SetupPDControllers(const Json::Value &json,
                                    const tVector &gravity);
    virtual void UpdatePDCtrls(double time_step, Eigen::VectorXd &out_tau);
    virtual void ApplyAction(const Eigen::VectorXd &action);
    virtual void BuildJointActionBounds(int joint_id, Eigen::VectorXd &out_min,
                                        Eigen::VectorXd &out_max) const;
    virtual void BuildJointActionBoundsRevolute(int joint_id,
                                                Eigen::VectorXd &out_min,
                                                Eigen::VectorXd &out_max) const;
    virtual void
    BuildJointActionBoundsSpherical(int joint_id, Eigen::VectorXd &out_min,
                                    Eigen::VectorXd &out_max) const;
    virtual void BuildJointActionBoundsNone(int joint_id,
                                            Eigen::VectorXd &out_min,
                                            Eigen::VectorXd &out_max) const;
    virtual void BuildJointActionBoundsFixed(int joint_id,
                                             Eigen::VectorXd &out_min,
                                             Eigen::VectorXd &out_max) const;
    virtual void BuildJointActionOffsetScale(int joint_id,
                                             Eigen::VectorXd &out_offset,
                                             Eigen::VectorXd &out_scale) const;
    virtual void
    BuildJointActionOffsetScaleSphereical(int joint_id,
                                          Eigen::VectorXd &out_offset,
                                          Eigen::VectorXd &out_scale) const;
    virtual void
    BuildJointActionOffsetScaleRevolute(int joint_id,
                                        Eigen::VectorXd &out_offset,
                                        Eigen::VectorXd &out_scale) const;
    virtual void
    BuildJointActionOffsetScaleFixed(int joint_id, Eigen::VectorXd &out_offset,
                                     Eigen::VectorXd &out_scale) const;
    virtual void
    BuildJointActionOffsetScaleNone(int joint_id, Eigen::VectorXd &out_offset,
                                    Eigen::VectorXd &out_scale) const;

    virtual void ConvertActionToTargetPose(Eigen::VectorXd &out_theta) const;

    // virtual void ConvertTargetPoseToAction(int joint_id,
    //                                        Eigen::VectorXd &out_theta) const;

    virtual cKinTree::eJointType GetJointType(int joint_id) const;

    virtual void SetPDTargets(const Eigen::VectorXd &targets);
    virtual int GetJointActionSize(int id) const;

    void ConvertTargetPoseToAction(int joint_id,
                                   Eigen::VectorXd &out_theta) const;
};