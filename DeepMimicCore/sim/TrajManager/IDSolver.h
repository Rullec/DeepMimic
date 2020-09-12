#pragma once
#include "Trajectory.h"
#include "sim/SimItems/SimCharacterBase.h"
#include <map>
#include <util/LogUtil.h>
#include <util/MathUtil.h>
enum eIDSolverType
{
    INVALID_IDSOLVER,
    Online,
    Display,
    OfflineSolve,
    Sample,
    SOLVER_TYPE_NUM
};

const std::string gIDSolverTypeStr[] = {
    "Invalid", "Online", "Display", "OfflineSolve", "Sample",
};

namespace btInverseDynamicsBullet3
{
class MultiBodyTree;
};
class btMultiBody;
class btMultiBodyDynamicsWorld;
class cCtPDController;
class cSceneImitate;
class cKinCharacter;
class btGenWorld;
struct tLoadInfo;
class cIDSolver
{
public:
    cIDSolver(cSceneImitate *imitate_scene, eIDSolverType type);
    virtual ~cIDSolver();
    eIDSolverType GetType();
    virtual void Reset() = 0;
    virtual void PreSim() = 0;
    virtual void PostSim() = 0;
    virtual void SetTimestep(double) = 0;
    static void RecordMultibodyInfo(const cSimCharacterBase *sim_char,
                                    tEigenArr<tMatrix> &local_to_world_rot,
                                    tEigenArr<tVector> &link_pos_world,
                                    tEigenArr<tVector> &link_omega_world,
                                    tEigenArr<tVector> &link_vel_world);
    static void RecordMultibodyInfo(cSimCharacterBase *sim_char,
                                    tEigenArr<tMatrix> &local_to_world_rot,
                                    tEigenArr<tVector> &link_pos_world);
    static void RecordGeneralizedInfo(cSimCharacterBase *sim_char, tVectorXd &q,
                                      tVectorXd &q_dot);
    static void SetGeneralizedPos(cSimCharacterBase *sim_char,
                                  const tVectorXd &q);
    virtual void ClearID();

protected:
    // tLogger mLogger;
    eIDSolverType mType;
    eSimCharacterType mSimCharType;
    cSceneImitate *mScene;
    cSimCharacterBase *mSimChar;
    cCtPDController *mCharController;
    cKinCharacter *mKinChar;
    cWorldBase *mWorld;

    double mWorldScale;
    bool mFloatingBase;
    int mDof;
    int mNumLinks; // including root

    // record functions
    void RecordReward(double &reward) const;
    void RecordRefTime(double &time) const;
    void RecordJointForces(tEigenArr<tVector> &mJointForces) const;
    void RecordAction(tVectorXd &action) const; // ball joints are in aas
    void RecordPDTarget(
        tVectorXd &pd_target) const; // ball joints are in quaternions
    void RecordContactForces(tEigenArr<tContactForceInfo> &mContactForces,
                             double mCurTimestep) const;
    void
    ApplyContactForcesToID(const tEigenArr<tContactForceInfo> &mContactForces,
                           const tEigenArr<tVector> &mLinkPos,
                           const tEigenArr<tMatrix> &mLinkRot) const;
    void ApplyExternalForcesToID(const tEigenArr<tVector> &link_poses,
                                 const tEigenArr<tMatrix> &link_rot,
                                 const tEigenArr<tVector> &ext_forces,
                                 const tEigenArr<tVector> &ext_torques) const;

    // set functions

    void SetGeneralizedVel(const tVectorXd &qdot);
    void SetGeneralizedVelFea(const tVectorXd &qdot);
    void SetGeneralizedVelGen(const tVectorXd &qdot);

    // calculation funcs
    tVectorXd CalcGeneralizedVel(const tVectorXd &q_before,
                                 const tVectorXd &q_after,
                                 double timestep) const;
    void CalcMomentum(const tEigenArr<tVector> &mLinkPos,
                      const tEigenArr<tMatrix> &mLinkRot,
                      const tEigenArr<tVector> &mLinkVel,
                      const tEigenArr<tVector> &mLinkOmega,
                      tVector &mLinearMomentum, tVector &mAngMomentum) const;
    void CalcDiscreteVelAndOmega(const tEigenArr<tVector> &mLinkPosCur,
                                 const tEigenArr<tMatrix> &mLinkRotCur,
                                 const tEigenArr<tVector> &mLinkPosNext,
                                 const tEigenArr<tMatrix> &mLinkRotNext,
                                 double timestep,
                                 tEigenArr<tVector> &mLinkDiscreteVel,
                                 tEigenArr<tVector> &mLinkDiscreteOmega) const;

    // solving single step
    virtual void
    SolveIDSingleStep(tEigenArr<tVector> &solved_joint_forces,
                      const tEigenArr<tContactForceInfo> &contact_forces,
                      const tEigenArr<tVector> &link_pos,
                      const tEigenArr<tMatrix> &link_rot,
                      const tVectorXd &mBuffer_q, const tVectorXd &mBuffer_u,
                      const tVectorXd &mBuffer_u_dot, int frame_id,
                      const tEigenArr<tVector> &mExternalForces,
                      const tEigenArr<tVector> &mExternalTorques) const;

    virtual double
    CalcAssembleJointForces(tEigenArr<tVector> &solved_joint_forces,
                            tVectorXd &total_joint_forces,
                            const tEigenArr<tVector> &ground_truth) const;

    virtual void PostProcessAction(tVectorXd &action,
                                   const tVectorXd &action_ground_truth) const;
    virtual double CalcActionError(const tVectorXd &solved_action,
                                   const tVectorXd &truth_action) const;

private:
    btMultiBody *mMultibody;
    btMultiBodyDynamicsWorld *mMultibodyWorld;
    btInverseDynamicsBullet3::MultiBodyTree *mInverseModel;
    // btGenWorld *mGenWorld;
    std::map<int, int> mWorldId2InverseId; // map the world array index to id in
                                           // inverse dynamics
    std::map<int, int> mInverseId2WorldId; // reverse map above

    // bullet vel calculation buffer
    btVector3 *omega_buffer, *vel_buffer;

    void InitFeaVariables();
    void InitGenVariables();
    static void RecordMultibodyInfoGen(const cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world,
                                       tEigenArr<tVector> &link_omega_world,
                                       tEigenArr<tVector> &link_vel_world);
    static void RecordMultibodyInfoFea(const cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world,
                                       tEigenArr<tVector> &link_omega_world,
                                       tEigenArr<tVector> &link_vel_world);
    static void RecordMultibodyInfoGen(cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world);
    static void RecordMultibodyInfoFea(cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world);
    static void RecordGeneralizedInfoGen(cSimCharacterBase *sim_char,
                                         tVectorXd &q, tVectorXd &q_dot);
    static void RecordGeneralizedInfoFea(cSimCharacterBase *sim_char,
                                         tVectorXd &q, tVectorXd &q_dot);
    static void SetGeneralizedPosGen(cSimCharacterBase *sim_char,
                                     const tVectorXd &q);
    void RecordJointForcesFea(tEigenArr<tVector> &mJointForces) const;
    void RecordJointForcesGen(tEigenArr<tVector> &mJointForces) const;
    void RecordContactForcesFea(tEigenArr<tContactForceInfo> &mContactForces,
                                double mCurTimestep) const;
    void RecordContactForcesGen(tEigenArr<tContactForceInfo> &mContactForces,
                                double mCurTimestep) const;
    tVectorXd CalcGeneralizedVelFea(const tVectorXd &q_before,
                                    const tVectorXd &q_after,
                                    double timestep) const;
    tVectorXd CalcGeneralizedVelGen(const tVectorXd &q_before,
                                    const tVectorXd &q_after,
                                    double timestep) const;
    virtual void
    SolveIDSingleStepFea(tEigenArr<tVector> &solved_joint_forces,
                         const tEigenArr<tContactForceInfo> &contact_forces,
                         const tEigenArr<tVector> &link_pos,
                         const tEigenArr<tMatrix> &link_rot,
                         const tVectorXd &mBuffer_q, const tVectorXd &mBuffer_u,
                         const tVectorXd &mBuffer_u_dot, int frame_id,
                         const tEigenArr<tVector> &mExternalForces,
                         const tEigenArr<tVector> &mExternalTorques) const;
    virtual void
    SolveIDSingleStepGen(tEigenArr<tVector> &solved_joint_forces,
                         const tEigenArr<tContactForceInfo> &contact_forces,
                         const tEigenArr<tVector> &link_pos,
                         const tEigenArr<tMatrix> &link_rot,
                         const tVectorXd &mBuffer_q, const tVectorXd &mBuffer_u,
                         const tVectorXd &mBuffer_u_dot, int frame_id,
                         const tEigenArr<tVector> &mExternalForces,
                         const tEigenArr<tVector> &mExternalTorques) const;
};
