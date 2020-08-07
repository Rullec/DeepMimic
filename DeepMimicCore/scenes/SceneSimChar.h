#pragma once

#include "scenes/DrawScene.h"

#include "sim/Controller/CtrlBuilder.h"
#include "sim/SimItems/SimCharBuilder.h"
#include "sim/SimItems/SimJoint.h"
#include "sim/World/Ground.h"
#include "sim/World/Perturb.h"
#include "sim/World/FeaWorld.h"
#include "util/IndexBuffer.h"

class cIDSolver;
class cTrajRecorder;
class cSceneSimChar : virtual public cScene
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct tObjEntry
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::shared_ptr<cSimRigidBody> mObj;
        double mEndTime;
        tVector mColor;
        bool mPersist;

        tObjEntry();
        bool IsValid() const;
    };

    struct tJointEntry
    {
        std::shared_ptr<cSimJoint> mJoint;

        tJointEntry();
        bool IsValid() const;
    };

    cSceneSimChar();
    virtual ~cSceneSimChar();

    virtual void ParseArgs(const std::shared_ptr<cArgParser> &parser);
    virtual void Init();
    virtual void Clear();
    virtual void Update(double time_elapsed);

    virtual int GetNumChars() const;
    virtual const std::shared_ptr<cSimCharacterBase> &GetCharacter() const;
    virtual const std::shared_ptr<cSimCharacterBase> &
    GetCharacter(int char_id) const;
    virtual const std::shared_ptr<cWorldBase> &GetWorld() const;
    virtual tVector GetCharPos() const;
    virtual const std::shared_ptr<cGround> &GetGround() const;
    virtual const tVector &GetGravity() const;
    virtual bool
    LoadControlParams(const std::string &param_file,
                      const std::shared_ptr<cSimCharacterBase> &out_char);

    virtual void AddPerturb(const tPerturb &perturb);
    virtual void ApplyRandForce(double min_force, double max_force,
                                double min_dur, double max_dur, cSimObj *obj);
    virtual void ApplyRandForce();
    virtual void ApplyRandForce(int char_id);
    virtual void RayTest(const tVector &beg, const tVector &end,
                         cWorldBase::tRayTestResult &out_result) const;

    virtual void SetGroundParamBlend(double lerp);
    virtual int GetNumParamSets() const;
    virtual void OutputCharState(const std::string &out_file) const;
    virtual void OutputGround(const std::string &out_file) const;
    virtual void ResolveCharGroundIntersect();

    virtual void SpawnProjectile();
    virtual void SpawnBigProjectile();
    virtual int GetNumObjs() const;
    virtual const std::shared_ptr<cSimRigidBody> &GetObj(int id) const;
    virtual const tObjEntry &GetObjEntry(int id) const;

    virtual void SetRandSeed(unsigned long seed);

    virtual std::string GetName() const;

protected:
    struct tPerturbParams
    {
        bool mEnableRandPerturbs;
        double mTimer;
        double mTimeMin;
        double mTimeMax;
        double mNextTime;
        double mMinPerturb;
        double mMaxPerturb;
        double mMinDuration;
        double mMaxDuration;
        std::vector<int> mPerturbPartIDs;

        tPerturbParams();
    };

    static const double gGroundSpawnOffset;

    cWorldBase::tParams mWorldParams;

    std::vector<cSimCharacterBase::tParams> mCharParams;
    std::vector<cCtrlBuilder::tCtrlParams> mCtrlParams;
    bool mEnableContactFall;
    bool mEnableRandCharPlacement;
    bool mEnableTorqueRecord;
    bool mEnablePDTargetSolveTest;
    std::string mTorqueRecordFile;
    bool mEnableJointTorqueControl;
    std::vector<int> mFallContactBodies; // 这是一个int列表，功能暂时不明。

    std::shared_ptr<cWorldBase> mWorld;
    std::shared_ptr<cGround> mGround;
    std::vector<std::shared_ptr<cSimCharacterBase>> mChars;
    std::vector<cSimCharBuilder::eCharType> mCharTypes;

    cGround::tParams mGroundParams;
    tPerturbParams mPerturbParams;

    cIndexBuffer<tObjEntry, Eigen::aligned_allocator<tObjEntry>> mObjs;
    // cIndexBuffer<tJointEntry> mJoints;

    // inverse dynamics info
    bool mEnableID;
    std::string mIDInfoPath;
    std::shared_ptr<cIDSolver> mIDSolver;

    // traj recoder info
    bool mEnableTrajRecord;
    std::string mTrajRecorderConfig;
    cTrajRecorder * mTrajRecorder;

    virtual bool
    ParseCharTypes(const std::shared_ptr<cArgParser> &parser,
                   std::vector<cSimCharBuilder::eCharType> &out_types) const;
    virtual bool
    ParseCharParams(const std::shared_ptr<cArgParser> &parser,
                    std::vector<cSimCharacterBase::tParams> &out_params) const;
    virtual bool ParseCharCtrlParams(
        const std::shared_ptr<cArgParser> &parser,
        std::vector<cCtrlBuilder::tCtrlParams> &out_params) const;

    virtual void BuildWorld();
    virtual bool BuildCharacters();
    virtual void BuildGround();
    virtual bool BuildController(const cCtrlBuilder::tCtrlParams &ctrl_params,
                                 std::shared_ptr<cCharController> &out_ctrl);

    virtual void SetFallContacts(const std::vector<int> &fall_bodies,
                                 std::shared_ptr<cSimCharacterBase> &out_char) const;
    virtual void InitCharacterPos();
    virtual void
    InitCharacterPos(const std::shared_ptr<cSimCharacterBase> &out_char);
    virtual void
    InitCharacterPosFixed(const std::shared_ptr<cSimCharacterBase> &out_char);
    virtual void BuildTrajManager();
    virtual void
    SetCharRandPlacement(const std::shared_ptr<cSimCharacterBase> &out_char);
    virtual void
    CalcCharRandPlacement(const std::shared_ptr<cSimCharacterBase> &out_char,
                          tVector &out_pos, tQuaternion &out_rot);
    virtual void ResolveCharGroundIntersect(
        const std::shared_ptr<cSimCharacterBase> &out_char) const;

    virtual void UpdateWorld(double time_step);
    virtual void UpdateCharacters(double time_step);
    virtual void PostUpdateCharacters(double time_step);
    virtual void UpdateGround(double time_elapsed);
    virtual void UpdateRandPerturb(double time_step);

    virtual void ResetScene();
    virtual void ResetCharacters();
    virtual void ResetWorld();
    virtual void ResetGround();

    virtual void PreUpdate(double timestep);
    virtual void PostUpdate(double timestep);

    virtual int
    GetRandPerturbPartID(const std::shared_ptr<cSimCharacterBase> &character);
    virtual void GetViewBound(tVector &out_min, tVector &out_max) const;
    virtual void ParseGroundParams(const std::shared_ptr<cArgParser> &parser,
                                   cGround::tParams &out_params) const;

    virtual void UpdateObjs(double timestep);
    // virtual void ClearJointForces();
    virtual void ClearObjs();
    virtual void CleanObjs();
    virtual int AddObj(const tObjEntry &obj_entry);
    virtual void RemoveObj(int handle);

    virtual bool HasFallen(const cSimCharacterBase &sim_char) const;

    virtual void SpawnProjectile(double density, double min_size,
                                 double max_size, double min_speed,
                                 double max_speed, double y_offset,
                                 double life_time);

    virtual void ResetRandPertrub();
};
