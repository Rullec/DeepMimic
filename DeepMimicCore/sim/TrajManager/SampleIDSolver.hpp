#include "InteractiveIDSolver.hpp"

class cSceneImitate;
class cSampleIDSolver : public cInteractiveIDSolver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit cSampleIDSolver(cSceneImitate *imitate_scene,
                             const std::string &config);
    virtual ~cSampleIDSolver();
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;
    virtual void SetTimestep(double) override final;

protected:
    void Parseconfig(const std::string &conf);
    void InitSampleSummaryTable();
    void RecordActionThetaDist(const tVectorXd &cur_action, double phase,
                               tMatrixXd &action_theta_dist_mat) const;

    bool mEnableIDTest; // enable solving Inverse Dynamics when sampling,
                        // usually for test
    bool mClearOldData; // clear the whole storaged trajs data dir before
                        // sampling
    bool
        mRecordThetaDist; // the action for spherical joints in our character is
                          // represented in axis-angle, an ambiguous rotation
                          // representation. If true, the SampleIDSolver will
                          // record the symbol of theta for each spherical joint
                          // when sampling this can help the following ID
                          // procedure and following supervised learning.
    bool mEnableSyncThetaDist; // sync the action theta distribution between
                               // processes
};