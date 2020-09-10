#pragma once
#include "BulletInverseDynamics/IDConfig.hpp"
#include "TrajRecorder.h"
#include <BulletDynamics/Featherstone/btMultiBodyLink.h>
#include <BulletInverseDynamics/MultiBodyTree.hpp>
#include <BulletInverseDynamics/details/IDLinearMathInterface.hpp>
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

class btMultiBody;
class btMultiBodyDynamicsWorld;
class cSimCharacter;
class cCtPDController;
class cSceneImitate;
class cKinCharacter;

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
    // static void RecordMultibodyInfo(cSimCharacterBase *simchar,
    //                                 tEigenArr<tMatrix> &local_to_world_rot,
    //                                 tEigenArr<tVector> &link_pos_world,
    //                                 tEigenArr<tVector> &link_omega_world,
    //                                 tEigenArr<tVector> &link_vel_world);
    static void RecordMultibodyInfo(cSimCharacterBase *sim_char,
                                    tEigenArr<tMatrix> &local_to_world_rot,
                                    tEigenArr<tVector> &link_pos_world);
    static void RecordGeneralizedInfo(cSimCharacterBase *sim_char, tVectorXd &q,
                                      tVectorXd &q_dot);
    static void SetGeneralizedPos(cSimCharacterBase *sim_char,
                                  const tVectorXd &q);

protected:
    tLogger mLogger;
    eIDSolverType mType;

    cSceneImitate *mScene;
    cSimCharacter *mSimChar;
    cCtPDController *mCharController;
    cKinCharacter *mKinChar;
    btMultiBody *mMultibody;
    btMultiBodyDynamicsWorld *mWorld;
    btInverseDynamicsBullet3::MultiBodyTree *mInverseModel;
    double mWorldScale;
    bool mFloatingBase;
    int mDof;
    int mNumLinks;                         // including root
    std::map<int, int> mWorldId2InverseId; // map the world array index to id in
                                           // inverse dynamics
    std::map<int, int> mInverseId2WorldId; // reverse map above

    // bullet vel calculation buffer
    btVector3 *omega_buffer, *vel_buffer;

    // record functions
    void RecordReward(double &reward) const;
    void RecordRefTime(double &time) const;
    void RecordJointForces(tEigenArr<tVector> &mJointForces) const;
    void RecordAction(tVectorXd &action) const; // ball joints are in aas
    void RecordPDTarget(
        tVectorXd &pd_target) const; // ball joints are in quaternions
    void RecordContactForces(tEigenArr<tContactForceInfo> &mContactForces,
                             double mCurTimestep,
                             std::map<int, int> &mWorldId2InverseId) const;
    void
    ApplyContactForcesToID(const tEigenArr<tContactForceInfo> &mContactForces,
                           const tEigenArr<tVector> &mLinkPos,
                           const tEigenArr<tMatrix> &mLinkRot) const;
    void ApplyExternalForcesToID(const tEigenArr<tVector> &link_poses,
                                 const tEigenArr<tMatrix> &link_rot,
                                 const tEigenArr<tVector> &ext_forces,
                                 const tEigenArr<tVector> &ext_torques) const;

    // set functions

    void SetGeneralizedVel(const tVectorXd &q);

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
};
