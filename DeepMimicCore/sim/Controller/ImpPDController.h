#pragma once

#include "sim/Controller/ExpPDController.h"
#include "sim/Controller/RBDModel.h"

//#define IMP_PD_CTRL_PROFILER
class cReverseController;
class cImpPDController : public cExpPDController
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cImpPDController();
    virtual ~cImpPDController();

    virtual void Init(cSimCharacterBase *character,
                      const Eigen::MatrixXd &pd_params, const tVector &gravity);
    virtual void Init(cSimCharacterBase *character,
                      const std::shared_ptr<cRBDModel> &model,
                      const Eigen::MatrixXd &pd_params, const tVector &gravity);
    virtual void Clear();
    virtual void UpdateControlForce(double time_step, Eigen::VectorXd &out_tau);

    virtual void SetKp(int joint_id, double kp);
    virtual void SetKd(int joint_id, double kd);

    virtual void SetEnableSolvePDTargetTest(bool);
    virtual void SolvePDTargetByTorque(double timestep,
                                       const tVectorXd &char_pose,
                                       const tVectorXd &char_vel,
                                       const tVectorXd &torque,
                                       tVectorXd &PDTarget);

protected:
    Eigen::VectorXd mKp;
    Eigen::VectorXd mKd;
    tEigenArr<cPDController> mPDCtrls;

    tVector mGravity;
    bool mExternRBDModel;
    bool mEnableSolvePDTargetTest;

    std::shared_ptr<cRBDModel> mRBDModel;
    std::shared_ptr<cReverseController> mPDTargetSolver;

#if defined(IMP_PD_CTRL_PROFILER)
    double mPerfSolveTime;
    double mPerfTotalTime;
    int mPerfSolveCount;
    int mPerfTotalCount;
#endif // IMP_PD_CTRL_PROFILER

    virtual void InitGains();
    virtual std::shared_ptr<cRBDModel>
    BuildRBDModel(const cSimCharacter &character, const tVector &gravity) const;
    virtual void UpdateRBDModel();

    virtual void CalcControlForces(double time_step, Eigen::VectorXd &out_tau);
    virtual void BuildTargetPose(tVectorXd &out_pose) const;
    virtual void BuildTargetVel(tVectorXd &out_vel) const;
};