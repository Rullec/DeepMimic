//
// Created by ljf on 2020/6/19.
//

#ifndef DEEPMIMICCORE_CONTROLLER_H
#define DEEPMIMICCORE_CONTROLLER_H

#include <map>
#include <iostream>
#include <vector>

#include "util/MathUtil.h"

enum CONTACT_TYPE {
    INVALID,
    SELF_COLLISION,
    GROUND_CONTACT,
    TOTAL_CONTACT_TYPE
};

class BaseObject;
class BaseRender;
class RobotModel;
class TargetProject;
class BaseOptimizer;
struct ReTargetingNode;
class TargetPoint;

namespace Json {
    class Value;
}

typedef Eigen::Vector3d vec3;
typedef Eigen::Vector3f vec3f;
typedef Eigen::Vector3i vec3i;
typedef Eigen::Vector4f vec4f;
typedef Eigen::Vector4d vec4;
typedef Eigen::VectorXd vec;
typedef Eigen::Matrix3d mat3;
typedef Eigen::Matrix4d mat4;
typedef Eigen::Matrix4f mat4f;
typedef Eigen::MatrixXd mat;
typedef Eigen::Affine3d aff3;
typedef Eigen::Affine3f aff3f;
typedef Eigen::Quaterniond quaterniond;

typedef std::vector<vec3, Eigen::aligned_allocator<vec3>> EIGEN_V_VEC3;
typedef std::vector<EIGEN_V_VEC3, Eigen::aligned_allocator<EIGEN_V_VEC3>> EIGEN_VV_VEC3;
typedef std::vector<vec3f, Eigen::aligned_allocator<vec3f>> EIGEN_V_VEC3f;
typedef std::vector<vec4, Eigen::aligned_allocator<vec4>> EIGEN_V_VEC4;
typedef std::vector<vec, Eigen::aligned_allocator<vec>> EIGEN_V_VECXD;
typedef std::vector<mat4, Eigen::aligned_allocator<mat4>> EIGEN_V_MAT4D;
typedef std::vector<mat, Eigen::aligned_allocator<mat>> EIGEN_V_MATXD;
typedef std::vector<mat3, Eigen::aligned_allocator<mat3>> EIGEN_V_MAT3D;
typedef std::vector<EIGEN_V_MATXD, Eigen::aligned_allocator<EIGEN_V_MATXD>> EIGEN_VV_MATXD;
typedef std::vector< EIGEN_V_MAT4D, Eigen::aligned_allocator<EIGEN_V_MAT4D>> EIGEN_VV_MAT4D;
typedef std::vector<EIGEN_VV_MAT4D, Eigen::aligned_allocator<EIGEN_VV_MAT4D>> EIGEN_VVV_MAT4D;

struct StEndPair;

class SupportingArea;
using pro_map = std::map<std::string, TargetProject*>;
using mod_map = std::map<std::string, RobotModel*>;
using StLastPair = StEndPair;
using FIdMap = std::map<std::string, StLastPair>;


struct FreedomInfo {
    int id;
    const char * joint_name;
    char axis;
    double v;
};

using FV = std::vector<FreedomInfo>;

struct BatchParam {
    std::string motion_path;
    std::vector<std::string> tp_path;
    std::vector<std::string> asf_path;
    std::vector<std::string> coe_path;
    std::vector<std::string> amc_path; // for output
    void Clear() {
        tp_path.clear();
        asf_path.clear();
        coe_path.clear();
        amc_path.clear();
    }
};
class AngleConstraint {
public:
    int freedom_id;
    int frame;
    double v;

    AngleConstraint(int freedom_id, int frame, double v)
            : freedom_id(freedom_id),
              frame(frame),
              v(v) {
    }
};

enum ConstraintOrder {
    CYCLIC_CONSTRAINT = 0,
    FIX_FOOT_CONSTRAINT,
    VERTICAL_CONSTRAINT,
    JOINT_MIN_MAX_CONSTRAINT,
    KEY_FRAME_CONSTRAINT,
    MANUAL_CONSTRAINT,
    COP_CONSTRAINT,
    COM_FORWARD_CONSTRAINT,
    TARGET_SLIDING_CONSTRAINT,
    TARGET_ABOVE_CONSTRAINT,
    FIX_CONTACT_CONSTRAINT,
    FIX_ROOT_CONSTRAINT,
    LINEAR_MOMENTUM_CONSTRAINT,
    TOTAL_CONSTRAINTS
};

enum EnergyTermOrder {
    TARGET_FOLLOW = 0,
    Q_SMOOTH,
    ROOT_SMOOTH,
    COM_FOLLOW,
    COM_SMOOTH,
    CLOSE_TO_ORIGIN,
    TARGET_POINT_VERTICAL,
    TARGET_POINT_SIMILAR,
    ANGULAR_MOMENTUM,
    LINEAR_MOMENTUM,
    TORQUE_MIN,
    TOTAL_ENERGY_TERMS
};


struct OptimizerCoeff {
    int max_itr;
    int local_itr;

    std::vector<double> energy_term_coeff;
    std::vector<bool> constraint_flags;

    std::vector<int> key_frame_id;
    std::vector<AngleConstraint> angle_constraints_data;

    OptimizerCoeff() {
        max_itr = 0;
        local_itr = 0;
        constraint_flags.resize(TOTAL_CONSTRAINTS, false);
        energy_term_coeff.resize(TOTAL_ENERGY_TERMS, 0);
    }
};

struct OptimizationNode;

/*
	contact force info is storaged here.
	Members:
		link_id Type int: which link does this force belong to?
		force Type vec4: the 4d vector force, world space
		force_pos Type vec4: the applied pos of this force, world space
	Methods:
		GetForceDir: return force direction.
		GetForceValue(): return magnitute.
	Added by Xudong 2020/03/02
*/
struct tContactParam {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    const static int DATA_SIZE = 7;
    int				 link_id;
    vec4			 force, force_pos;
    CONTACT_TYPE	 contact_type;

    tContactParam() {
        link_id = -1;
        force_pos = force = vec4::Zero();
        contact_type = CONTACT_TYPE::INVALID;
    }

    vec4 GetForceDir() { return force; }
    double GetForceValue() { return force.norm(); }
};

typedef std::vector<tContactParam, Eigen::aligned_allocator<tContactParam>> EIGEN_V_CONTACT_PARAM;
typedef std::vector<EIGEN_V_CONTACT_PARAM, Eigen::aligned_allocator<EIGEN_V_CONTACT_PARAM>> EIGEN_VV_CONTACT_PARAM;

class Controller {

public:
    enum MotionType {
        EulerMotion=0,
        DeepMimicMotion=1,
        SimpleMotion=2,
        DeepMimicTrajectory = 3,
        MOTION_TYPE_NUM,
    };

    Controller();
    Controller(BaseRender* render);
    explicit Controller(const char* file);
    virtual			~Controller();

    void			Init(const char *file);

    void			MainLoop();

    void			UpdateProjectRenderData() const;
    void			UpdateModelRenderData() const;
    void			UpdateSupportingArea();
    void			UpdateSupportingAreaPos();

    void			SetKeyAcc(double acc) const;
    void			SetMouseAcc(double acc) const;

    void			BuildFreedomIdMap();
    void			BuildFreedomIdMap(RobotModel* model, FIdMap& map);

    void			Update(int frame);
    void			InitOptimizer();
    void			RunOptimizer();
    void			SetOptimizerCoeff(const OptimizerCoeff& coeff);

    void			LoadModel(const char * file, int type=0);
    void			LoadProject(const char *file);
    void			LoadAns(const char* file);
    void			LoadAnsOldStyle(const char* file);
    void			LoadAMC(const char* file);
    void			LoadAMC2(const char* file);
    void			LoadAmcFile(const char* file, RobotModel*, std::vector<double>& ans, int& frame);
    void			LoadCoeff(const char* file);
    void			LoadSimpleMotion(const char* file);							// simple motion ��ֱ�Ӱ�ans_q�浽һ���ļ���
    // �������Ż������л��õ�
    void			LoadJsonModel(const char* file);
    void			LoadBuffer(const char* file, bool show=false);
    void			LoadBuffer(const char* file, std::vector<double>& phase, std::vector<double>& motion, std::vector<std::vector<double>>& contact);
    void			LoadMotionFile(const char* file, std::vector<double>& ans, double & timestep);
    void			LoadMotion(const char* file);
    void			LoadMotion(std::vector<double>& motion);
    void            LoadMotion(const char* file, mat& motion);
    void            ConvertToDeepMimicMotionMat(const std::vector<double>& ans, RobotModel* model, mat& motion) const;
    void			ConvertToDeepMimicMotion(const std::vector<double>& ans, RobotModel* model, std::vector<double>& motion) const;
    void			ConvertFromDeepMimicMotion(std::vector<double>& motion, RobotModel* model, std::vector<double>& ans);
    void 			LoadTrajectory(const char * file, std::vector<double> & motion, int & num_of_frames, EIGEN_VV_CONTACT_PARAM & contact_info, std::vector<double> &phases);
    void 			SaveTrajectory(const char * orig_file, const char * target_file, TargetProject * project, const std::vector<double> & current_ans_q) const;
    void			LoadBatchOptimizationScript(const char* file);
    // void			LoadBufferReTargetingConfig(const char* file);
    void			LoadJointForce(const char* file, EIGEN_V_VECXD& tau);
    void			LoadJointForce(Json::Value root, EIGEN_V_VECXD& tau);


    void			ExportProject(const char* file) ;
    void			ExportTargetTrajectory(const char* file);
    void			ExportAns(const char* file);
    void			ExportAnsOldFormat(const char* file);
    void			ExportAmc(const char* file);
    void			ExportAmc(const char* file, RobotModel* model, std::vector<double>& ans);
    void			ExportCoeff(const char* file);

    void			ExportSimpleMotion(const char* file);
    void			ExportQuaternionAns(const char* file);
    void			SaveCycleImages();

    int				GetNumOfFrame() const { return num_of_frame; }
    FV&				GetFreedomInfo() { return freedom_info; }
    TargetProject*	GetTargetProject() const {return current_project;}
    int				GetCurrentFrame() const {return current_frame;}
    int				GetNumOfFreedom()const { return num_of_freedom; }
    OptimizerCoeff& GetOptimizerCoeff() { return optimizer_coeff; }

    void			UpdateFreedomValue(int order, double v);					// ����ֱ���޸�ĳ��freedom�ڵ�ǰ֡�µ�ֵ
    void			DetectContact();
    void			CutFrame(int st, int end);
    void			TargetPositionScale(double scale);

    void			AddKeyFrameConstraint(int frame = -1);
    void			RebuildTranslation();
    void			RebuildTranslation(ReTargetingNode* node);
    void			RebuildTranslation(RobotModel* mode, TargetProject* project, std::vector<double>& motion);
    void			Symmetric();
    void			ReCycle(int st_frame);
    void			ReCycle(int st_frame, int n_frame, int n_freedom, std::vector<double>& ans, int type=0);

    void			BatchModify(int freedom_id, double v, int start, int end);
    void			BatchSet(int freedom, double v);
    void			BatchReplace(int st_frame, int end_frame, int st_freedom, int end_freedom);
    void			BatchInterpolation(std::vector<int>& n_inters);

    void			Interpolation(int frame_id, int n_inter);
    void			DuplicateCycle(int times);
    void			SampleFrame(int inters);

    void			MoveTargetPoint(TargetPoint* tp, vec3 direction);
    void			MoveTargetPoint(int id, vec3 direction);
    void			AdjustTargetPointKeyFrames();

    void			StandOnGroundSingleFrame(int frame=0);
    void			StandOnGround();
    void			StandOnGround(RobotModel* model, TargetProject* project, std::vector<double> motion);
    void			DuplicateHalfCycle();
    void			CenterReCycle();
    void			PreProcess();
    void			PostProcess();
    void			AutoOptimization();
    void			BatchOptimization(const char* file);
    void			BatchOptimizationMultiProc(const char* file);
    void			BatchOptimizationMultiProcessing(const char* file);

    void			BlurInputMotion(ReTargetingNode* node, std::vector<double>& input);
    void			TrainingBufferReTargeting(const char* file);
    void			ComputeContactInfo(RobotModel* origin_model, RobotModel* target_model, std::vector<double>& motion, TargetProject* project, const EIGEN_VV_CONTACT_PARAM& contact);
    void			ComputeContactInfo(RobotModel* model, TargetProject* project, const EIGEN_VV_CONTACT_PARAM& contact);
    void			DetectTargetPointSliding(RobotModel* model, TargetProject* project, std::vector<double>& ans);
    void			DetectTargetPointContacting(RobotModel* model, TargetProject* project, std::vector<double>& ans);

    // void			ComputeContactInfo_Traj(ReTargetingNode * node);
    void			RunReTargeting(ReTargetingNode* re_targeting_node);
    void			ReTargetingPostProcessing(ReTargetingNode* node);

    void			SetSelectedTargetPointId(int id) { this->current_selected_tp = id; }
    void			SetShowSupportArea(bool show) {this->show_support_area = show;}
    void			SetShowCoP(bool show) { this->show_cop = show; }
    virtual void	Test();

    void			UpdateCopPoint();
    void			ComputeCop();
    void			ComputeCopNumerically();
    void			InitSupportingFace();

    void			ConcatMotion(const char* file);
    void			MoveRoot(vec3 direction);
    void			DeleteSupportingFace();
    void			SetTimeStep(double t) { std::cout << t << std::endl; this->time_step = t; }
    void			SetTargetPointHeight(double h) { this->tp_on_ground_height = h; }
    void			UpdateTargetProject();
    void			BufferReTargetingPreProcessing(ReTargetingNode* node);
    ReTargetingNode*LoadBufferReTargetingNode(const char* file);

    ReTargetingNode*LoadSimplificationNode(const char* file);

    void			TestJw();
    void			TestJwByNumGrad();
    void			TestRotMatToAxisAngle();
    void			TestMassMatrix();
    void			TestCoriolisMatrix();
    void			TestHessian();
    void			TestJDot();
    void			ComputeLinearVel(EIGEN_V_VEC3& t);
    void			ComputeLinearAcc(EIGEN_V_VEC3& vel, EIGEN_V_VEC3& acc);
    void			ComputeOmega(EIGEN_V_VEC4& t);
    void			ComputeQddot(const std::vector<double>& q_dot, std::vector<double>& q_ddot);
    void			ComputeQdot(std::vector<double>& t);
    void			ComputeQdot(RobotModel* model, TargetProject* p, std::vector<double>& ans, std::vector<double>& t);
    void			ExportAsfModel(const char* file);
    void			TestLinearMomentum();
    void			TestLinkJv();
    void			TestLinkJvTimesMassSum();
    void			TestLinkJvTotalDOF();
    void			TestTorqueMatrix();
    void 			TestDQDqNum();
    void			SimplifyModel(const char* file_path);
    void			TestAngularMomentum();
    void			TestTorqueMin();
    void			TestLoadTorque();
    void			TestTorqueComputation();
    void			TestJvOfContactPoint();

    ReTargetingNode* AutoReTargeting(const char* file);
    void			AutoReTargeting(ReTargetingNode* node);
    void			BatchBufferReTargeting(const char* file);
    void			BatchBufferReTargetingMultiProc(const char* file);
    void			BatchBufferReTargetingSubProc(const char* file, int task_id, int interval);

    void			LoadCommonReTargetingInfo(ReTargetingNode* node, Json::Value& json_root);
    void			LoadBuffer(ReTargetingNode* node);

    void			ComputeTargetPointHeightPortion(RobotModel* model, TargetProject* project, std::vector<double>& ans);
    void			ComputeTargetPoint3DPortion(RobotModel* model, TargetProject* project, std::vector<double>& ans);
    void			AdjustTargetPointToCornerPos();
    void			AdjustTargetPointToCornerPos(RobotModel* model, TargetProject* project);

    void            LoadJointMat(const char* file, mat& joint_mat);
    void 			LoadModelFromJointMat(const char* file);
    void            LoadModelFromJointMat(mat& joint_mat);
    virtual void	ClearController();

protected:
    void			Init();
    void			InitColor();
    void			InitNameMap();
    virtual void	InitMeshMap();
    void			InitTargetProject();

    void			UpdateNumOfFreedom(RobotModel* model);
    void			UpdateNumOfFrame(std::vector<double>& ans);
    void			UpdateTargetPoint();
    void			UpdateCoMPoint() const;
    void			UpdateContactPoint();
    void			DeleteContactPointInRender(int frame);

    void			UpdateFreedomInfo();
    void			UpdateCycleStep(bool forward);

    void			ReadConfig(const char * file);
    void			ReadConfigPostProcess();
    void			LoadAngleSmooth();				// smooth the angle data when loading from file.
    void			LoadAngleSmooth(RobotModel* mode, std::vector<double>& ans, int n_frames);
    void			LoadAns(const char* file, std::vector<double>& ans, int st_frame=0);

    void			GetLegsFreedomId(std::vector<int>& ids, BaseObject* start_object);
    void			CheckSymmetric();
    void			SymmetricRoot();
    vec3			GetTargetPointBase();
    double			GetLowestTargetPointValue(int frame);
    double			GetLowestTargetPointValue(RobotModel* model, TargetProject* project);
    double			GetLowestTargetPointValue(std::vector<double>& ans, RobotModel* model, TargetProject* project);

    double			GetTargetPointOnGroundValue(int frame);
    double			GetTargetPointOnGroundValue(int frame, std::vector<double>&ans, TargetProject* project, RobotModel* model) const ;

    void			InitCoMPointRenderData() const;
    void			InitTargetPointRenderData() const;
    void			InitContactPointRenderData() const;
    void			InitCoPPointRenderData() const;
    void			InitCoPData() const;

    void			LoadAnsPostProcess();

    void			LoadAmcPostProcess();

    static RobotModel*		CreateModel(const char* file, int type);
    TargetProject*	CreateTargetProject(const char* file, int num_of_frame);

    //================== for multi process batch opt ==================
//    void			LoadOptimizationNode(OptimizationNode::Param &param);
//    OptimizerCoeff*	LoadCoeffFromFile(const char* file);
//    void			StandOnGroundMultiProc(OptimizationNode* ko);
//    void			RebuildTranslationMultiProc(OptimizationNode* ko) const;
//    void			PreProcessMultiProc(OptimizationNode* ko);
//    void			PostProcessMultiProc(OptimizationNode* ko);
//    void			AutoOptimizationMultiProc(OptimizationNode* ko);
//    void			InitOptimizerMultiProc(OptimizationNode* ko);
//    void			RunOptimizerMultiProc(OptimizationNode* ko);
    void			BatchModifyMultiProc(std::vector<double>& ans,int freedom_id, double v, int start, int end);
    //=================================================================
    void			BatchModifyMultiProc(std::vector<double>& ans, int n_freedom, int n_frame, int freedom_id, double v, int start, int end);



    BaseRender			*render;
    mod_map				model_map;
    pro_map				project_map;
    RobotModel			*current_model;
    TargetProject		*current_project;
    BaseOptimizer		*optimizer;
    std::vector<double> ans_q;
    int					current_frame;
    int					num_of_freedom;
    int					num_of_frame;
    int					num_of_target;
    FV					freedom_info;

    vec3f				obj_color;
    vec3f				joint_color;
    vec3f				com_color;
    vec3f				cop_color;
    vec3f				tp_color;
    vec3f				tp_selected_color;
    vec3f				cp_color;

    vec3				cycle_step;
    FIdMap				freedom_id_map;
    OptimizerCoeff		optimizer_coeff;
    int					current_selected_tp;

    double				tp_on_ground_height;
    double				time_step;

    //std::map<EnergyTermOrder, std::string> energy_term_name_map;
    //std::map<ConstraintOrder, std::string> constraint_name_map;

    BatchParam			batch_param;
    bool				batch_opt_lock;

    std::vector<OptimizationNode*> optimization_nodes;
    std::vector<ReTargetingNode*> re_targeting_nodes;

    SupportingArea		*sp_area;
    bool				show_support_area;
    bool				show_cop;

};


#endif //DEEPMIMICCORE_CONTROLLER_H
