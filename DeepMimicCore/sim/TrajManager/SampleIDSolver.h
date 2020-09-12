#include "IDSolver.h"

class cSceneImitate;
class cSampleIDSolver : public cIDSolver
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
    // This struct are used in the derived cSampleIDSolver.
    // when we are workng in "sample" mode, usually tons of trajectories will be
    // sampled and storaged we need to specify the storage dir... for it.
    struct
    {
        int mSampleEpoches; // the epoche number of trajectoris we want.
        std::string mSampleTrajsDir;      // the saving dir of trajectoris, for
                                          // example "data/walk/"
        std::string mSampleTrajsRootName; // the root of trajectories' filename,
                                          // for example "traj_walk.json". Then
                                          // "traj_walk_xxx.json" will be
                                          // genearated and saved
        std::string mSummaryTableFilename; // These recorded trajs' info will be
                                           // recorded in this file. It can be
                                           // "summary_table.json"
        std::string
            mActionThetaDistFilename; // This file will record action theta
                                      // distribution for spherical joints
        eTrajFileVersion mSampleTrajVer;
    } mSampleInfo; // work for sample mode

    tSaveInfo mSaveInfo;
    tSummaryTable mSummaryTable;
    void Parseconfig(const std::string &conf);
    void InitSampleSummaryTable();
    void RecordActionThetaDist(const tVectorXd &cur_action, double phase,
                               tMatrixXd &action_theta_dist_mat) const;

    bool mEnableIDTest; // enable solving Inverse Dynamics when sampling,
                        // usually for test
    bool mClearOldData; // clear the whole storaged trajs data dir before
                        // sampling
    // bool
    //     mRecordThetaDist; // the action for spherical joints in our character
    //     is
    //                       // represented in axis-angle, an ambiguous rotation
    //                       // representation. If true, the SampleIDSolver will
    //                       // record the symbol of theta for each spherical
    //                       joint
    //                       // when sampling this can help the following ID
    //                       // procedure and following supervised learning.
    // bool mEnableSyncThetaDist; // sync the action theta distribution between
    //                            // processes
};