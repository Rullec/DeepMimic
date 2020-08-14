#include "IDSolver.hpp"
#include <map>

class cSceneImitate;
class cOnlineIDSolver : public cIDSolver
{
public:
    enum eSolvingMode
    {
        VEL = 0,
        POS,
    };
    cOnlineIDSolver(cSceneImitate *scene);
    void ClearID();
    virtual void SetTimestep(double deltaTime) override final;
    virtual void PreSim() override final;
    virtual void PostSim() override final;
    virtual void Reset() override final;

protected:
    // ID vars
    bool mEnableExternalForce;
    bool mEnableExternalTorque;
    bool mEnableSolveID;

    double mCurTimestep;
    int mFrameId;
    eSolvingMode mSolvingMode;

    // ID buffer vars
    tEigenArr<tContactForceInfo> mContactForces;
    tEigenArr<tVector>
        mJointForces; // reference joint torque(except root) in link frame
    tEigenArr<tVector> mSolvedJointForces; // solved joint torques from
    tEigenArr<tVector>
        mExternalForces; // for each link, external forces in COM
    tEigenArr<tVector> mExternalTorques; // for each link, external torques
    tEigenArr<tMatrix> mLinkRot;         // local to world rotation mats
    tEigenArr<tVector> mLinkPos;         // link COM pos in world frame

    //// permanent memory
    tVectorXd mBuffer_q[MAX_FRAME_NUM]; // generalized coordinate "q" buffer,
                                        // storaged for each frame
    tVectorXd
        mBuffer_u[MAX_FRAME_NUM]; // q_dot = u buffer, storaged for each frame
    tVectorXd mBuffer_u_dot[MAX_FRAME_NUM]; //

    // tools
    void AddJointForces();
    void AddExternalForces();

    virtual void SolveIDSingleStep(
        tEigenArr<tVector> &solved_joint_forces,
        const tEigenArr<tContactForceInfo> &contact_forces,
        const tEigenArr<tVector> &link_pos,
        const tEigenArr<tMatrix> &link_rot, const tVectorXd &mBuffer_q,
        const tVectorXd &mBuffer_u, const tVectorXd &mBuffer_u_dot,
        int frame_id, const tEigenArr<tVector> &mExternalForces,
        const tEigenArr<tVector> &mExternalTorques) const override final;
    void ApplyExternalForcesToID() const;
    // tVectorXd CalculateGeneralizedVel(const tVectorXd & q_before, const
    // tVectorXd & q_after, double timestep) const;
};