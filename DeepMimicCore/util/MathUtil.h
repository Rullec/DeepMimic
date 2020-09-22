#pragma once
#include "BulletGenDynamics/btGenUtil/MathUtil.h"
#include "Rand.h"
#include <random>

const int gInvalidIdx = -1;

enum eRotationOrder
{
    XYZ = 0, // first X, then Y, then Z. X->Y->Z. R_{total} = Rz * Ry * Rx;
    XZY,
    XYX,
    XZX, // x end
    YXZ,
    YZX,
    YXY,
    YZY, // y end
    ZXY,
    ZYX,
    ZYZ,
    ZXZ, // z end
};

// extern const enum eRotationOrder gRotationOrder;// rotation order. declared
// here and defined in LoboJointV2.cpp
const std::string ROTATION_ORDER_NAME[] = {
    "XYZ", "XZY", "XYX", "XZX", "YXZ", "YZX",
    "YXY", "YZY", "ZXY", "ZYX", "ZYZ", "ZXZ",
};

// for convenience define standard vector for rendering
typedef Eigen::Vector4d tVector;
typedef Eigen::VectorXd tVectorXd;
typedef Eigen::Vector3d tVector3d;
typedef Eigen::Vector3f tVector3f;
typedef Eigen::Matrix4d tMatrix;
typedef Eigen::Matrix3d tMatrix3d;
typedef Eigen::MatrixXd tMatrixXd;
typedef Eigen::Matrix4f tMatrix4f;
typedef Eigen::Quaterniond tQuaternion;
typedef Eigen::Affine3d aff3;
typedef Eigen::Affine3f aff3f;

template <typename T>
using tEigenArr = std::vector<T, Eigen::aligned_allocator<T>>;
typedef tEigenArr<tVector> tVectorArr;

const double gRadiansToDegrees = 57.2957795;
const double gDegreesToRadians = 1.0 / gRadiansToDegrees;
const tVector gGravity = tVector(0, -9.8, 0, 0);
// const tVector gGravity = tVector(0, 0, 0, 0);
const double gInchesToMeters = 0.0254;
const double gFeetToMeters = 0.3048;

class cMathUtil
{
public:
    enum eAxis
    {
        eAxisX,
        eAxisY,
        eAxisZ,
        eAxisMax
    };

    static int Clamp(int val, int min, int max);
    static void Clamp(const Eigen::VectorXd &min, const Eigen::VectorXd &max,
                      Eigen::VectorXd &out_vec);
    static double Clamp(double val, double min, double max);
    static double Saturate(double val);
    static double Lerp(double t, double val0, double val1);

    static double NormalizeAngle(double theta);

    // rand number
    static double RandDouble();
    static double RandDouble(double min, double max);
    static double RandDoubleNorm(double mean, double stdev);
    static double RandDoubleExp(double lambda);
    static double RandDoubleSeed(double seed);
    static int RandInt();
    static int RandInt(int min, int max);
    static int RandUint();
    static int RandUint(unsigned int min, unsigned int max);
    static int RandIntExclude(int min, int max, int exc);
    static void SeedRand(unsigned long int seed);
    static int RandSign();
    static bool FlipCoin(double p = 0.5);
    static double SmoothStep(double t);

    // matrices
    static tMatrix TranslateMat(const tVector &trans);
    static tMatrix ScaleMat(double scale);
    static tMatrix ScaleMat(const tVector &scale);
    static tMatrix
    RotateMat(const tVector &euler,
              const eRotationOrder gRotationOrder); // euler angles order rot(Z)
                                                    // * rot(Y) * rot(X)
    static tMatrix RotateMat(const tVector &axis, double theta);
    static tMatrix RotateMat(const tQuaternion &q);
    static tMatrix CrossMat(const tVector &a);
    // inverts a transformation consisting only of rotations and translations
    static tMatrix InvRigidMat(const tMatrix &mat);
    static tVector GetRigidTrans(const tMatrix &mat);
    static tVector InvEuler(const tVector &euler,
                            const eRotationOrder gRotationOrder);
    static void RotMatToAxisAngle(const tMatrix &mat, tVector &out_axis,
                                  double &out_theta);
    static tVector RotMatToEuler(const tMatrix &mat,
                                 const eRotationOrder gRotationOrder);
    static tQuaternion RotMatToQuaternion(const tMatrix &mat);
    static tVector EulerangleToAxisAngle(const tVector &euler,
                                         const eRotationOrder gRotationOrder);
    static void EulerToAxisAngle(const tVector &euler, tVector &out_axis,
                                 double &out_theta,
                                 const eRotationOrder gRotationOrder);
    static tVector AxisAngleToEuler(const tVector &axis, double theta);
    static tMatrix DirToRotMat(const tVector &dir, const tVector &up);

    static void DeltaRot(const tVector &axis0, double theta0,
                         const tVector &axis1, double theta1, tVector &out_axis,
                         double &out_theta);
    static tMatrix DeltaRot(const tMatrix &R0, const tMatrix &R1);

    static tQuaternion EulerToQuaternion(const tVector &euler,
                                         const eRotationOrder order);
    static tQuaternion CoefVectorToQuaternion(const tVector &coef);
    static tVector QuaternionToEuler(const tQuaternion &q,
                                     const eRotationOrder gRotationOrder);
    static tQuaternion AxisAngleToQuaternion(const tVector &axis, double theta);
    static tVector QuaternionToAxisAngle(const tQuaternion &q);
    static void QuaternionToAxisAngle(const tQuaternion &q, tVector &out_axis,
                                      double &out_theta);
    static tMatrix BuildQuaternionDiffMat(const tQuaternion &q);
    static tVector CalcQuaternionVel(const tQuaternion &q0,
                                     const tQuaternion &q1, double dt);
    static tVector CalcQuaternionVelRel(const tQuaternion &q0,
                                        const tQuaternion &q1, double dt);
    static tQuaternion VecToQuat(const tVector &v);
    static tVector QuatToVec(const tQuaternion &q);
    static tQuaternion QuatDiff(const tQuaternion &q0, const tQuaternion &q1);
    static double QuatDiffTheta(const tQuaternion &q0, const tQuaternion &q1);
    static double QuatTheta(const tQuaternion &dq);
    static tQuaternion VecDiffQuat(const tVector &v0, const tVector &v1);
    static tVector QuatRotVec(const tQuaternion &q, const tVector &dir);
    static tQuaternion MirrorQuaternion(const tQuaternion &q, eAxis axis);

    static double Sign(double val);
    static int Sign(int val);

    static double AddAverage(double avg0, int count0, double avg1, int count1);
    static tVector AddAverage(const tVector &avg0, int count0,
                              const tVector &avg1, int count1);
    static void AddAverage(const Eigen::VectorXd &avg0, int count0,
                           const Eigen::VectorXd &avg1, int count1,
                           Eigen::VectorXd &out_result);
    static void CalcSoftmax(const Eigen::VectorXd &vals, double temp,
                            Eigen::VectorXd &out_prob);
    static double EvalGaussian(const Eigen::VectorXd &mean,
                               const Eigen::VectorXd &covar,
                               const Eigen::VectorXd &sample);
    static double EvalGaussian(double mean, double covar, double sample);
    static double CalcGaussianPartition(const Eigen::VectorXd &covar);
    static double EvalGaussianLogp(double mean, double covar, double sample);
    static double EvalGaussianLogp(const Eigen::VectorXd &mean,
                                   const Eigen::VectorXd &covar,
                                   const Eigen::VectorXd &sample);
    static double Sigmoid(double x);
    static double Sigmoid(double x, double gamma, double bias);

    static int SampleDiscreteProb(const Eigen::VectorXd &probs);
    static tVector CalcBarycentric(const tVector &p, const tVector &a,
                                   const tVector &b, const tVector &c);

    static bool ContainsAABB(const tVector &pt, const tVector &aabb_min,
                             const tVector &aabb_max);
    static bool ContainsAABB(const tVector &aabb_min0, const tVector &aabb_max0,
                             const tVector &aabb_min1,
                             const tVector &aabb_max1);
    static bool ContainsAABBXZ(const tVector &pt, const tVector &aabb_min,
                               const tVector &aabb_max);
    static bool ContainsAABBXZ(const tVector &aabb_min0,
                               const tVector &aabb_max0,
                               const tVector &aabb_min1,
                               const tVector &aabb_max1);
    static void CalcAABBIntersection(const tVector &aabb_min0,
                                     const tVector &aabb_max0,
                                     const tVector &aabb_min1,
                                     const tVector &aabb_max1, tVector &out_min,
                                     tVector &out_max);
    static void CalcAABBUnion(const tVector &aabb_min0,
                              const tVector &aabb_max0,
                              const tVector &aabb_min1,
                              const tVector &aabb_max1, tVector &out_min,
                              tVector &out_max);
    static bool IntersectAABB(const tVector &aabb_min0,
                              const tVector &aabb_max0,
                              const tVector &aabb_min1,
                              const tVector &aabb_max1);
    static bool IntersectAABBXZ(const tVector &aabb_min0,
                                const tVector &aabb_max0,
                                const tVector &aabb_min1,
                                const tVector &aabb_max1);

    // check if curr_val and curr_val - delta belong to different intervals
    static bool CheckNextInterval(double delta, double curr_val,
                                  double int_size);

    static tVector SampleRandPt(const tVector &bound_min,
                                const tVector &bound_max);
    // samples a bound within the given bounds with a benter towards the focus
    // pt
    static tVector SampleRandPtBias(const tVector &bound_min,
                                    const tVector &bound_max);
    static tVector SampleRandPtBias(const tVector &bound_min,
                                    const tVector &bound_max,
                                    const tVector &focus);

    static void QuatSwingTwistDecomposition(const tQuaternion &q,
                                            const tVector &dir,
                                            tQuaternion &out_swing,
                                            tQuaternion &out_twist);
    static tQuaternion ProjectQuat(const tQuaternion &q, const tVector &dir);

    static void ButterworthFilter(double dt, double cutoff,
                                  Eigen::VectorXd &out_x);

    // added by myself
    static tMatrix RotMat(const tQuaternion &quater);
    // static tQuaternion RotMatToQuaternion(const tMatrix &mat);
    static tQuaternion CoefToQuaternion(const tVector &);
    static tQuaternion AxisAngleToQuaternion(const tVector &angvel);
    static tQuaternion EulerAnglesToQuaternion(const tVector &vec,
                                               const eRotationOrder &order);
    static tQuaternion MinusQuaternion(const tQuaternion &quad);
    static tVector QuaternionToCoef(const tQuaternion &quater);
    // static tVector QuaternionToAxisAngle(const tQuaternion &);
    static tVector CalcAngularVelocity(const tQuaternion &old_rot,
                                       const tQuaternion &new_rot,
                                       double timestep);
    static tVector CalcAngularVelocityFromAxisAngle(const tQuaternion &old_rot,
                                                    const tQuaternion &new_rot,
                                                    double timestep);
    static tVector QuaternionToEulerAngles(const tQuaternion &,
                                           const eRotationOrder &order);

    static tMatrix EulerAnglesToRotMat(const tVector &euler,
                                       const eRotationOrder &order);
    static tMatrix EulerAnglesToRotMatDot(const tVector &euler,
                                          const eRotationOrder &order);
    static tVector AngularVelToqdot(const tVector &omega, const tVector &cur_q,
                                    const eRotationOrder &order);
    static tMatrix VectorToSkewMat(const tVector &);
    static tVector SkewMatToVector(const tMatrix &);
    static bool IsSame(const tVector &v1, const tVector &v2, const double eps);
    static void ThresholdOp(tVectorXd &v, double threshold = 1e-6);
    template <typename T> static const std::string EigenToString(const T &mat)
    {
        std::stringstream ss;
        ss << mat;
        return ss.str();
    }
    static double Truncate(double num, int digits = 5);
    static tMatrixXd ExpandFrictionCone(int num_friction_dirs,
                                        const tVector &normal);
    static tMatrix InverseTransform(const tMatrix &);
    static double CalcConditionNumber(const tMatrixXd &mat);
    static tMatrixXd JacobPreconditioner(const tMatrixXd &mat);
    // static void RoundZero(tMatrixXd &mat, double threshold = 1e-10);

    template <typename T>
    static void RoundZero(T &mat, double threshold = 1e-10)
    {
        mat = (threshold < mat.array().abs()).select(mat, 0.0f);
    }
    template <typename T> static tVector Expand(const T &vec, double n)
    {
        return tVector(vec[0], vec[1], vec[2], n);
    }
    template <typename T> static tMatrix ExpandMat(const T &raw_mat)
    {
        tMatrix mat = tMatrix::Zero();
        mat.block(0, 0, 3, 3) = raw_mat.block(0, 0, 3, 3);
        return mat;
    }

private:
    static cRand gRand;

    template <typename T> static T SignAux(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    static tMatrix EulerAngleRotmatX(double x);
    static tMatrix EulerAngleRotmatY(double x);
    static tMatrix EulerAngleRotmatZ(double x);
    static tMatrix EulerAngleRotmatdX(double x);
    static tMatrix EulerAngleRotmatdY(double x);
    static tMatrix EulerAngleRotmatdZ(double x);
};
