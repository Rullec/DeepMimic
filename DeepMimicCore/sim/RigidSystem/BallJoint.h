#pragma once
#include "LoboJointV2.h"
#include "LoboLink.h"


/*
	Use R = Rx Ry Rz and the euler angle x y z as generalized coordinates
*/
class BallJoint :public LoboJointV2
{
public:
	BallJoint(LoboLink* link, Vector3d ori_position_, Vector3d ori_position_w_, Matrix3d ori_orientation_w_, Vector3d joint_position, int id_);
	~BallJoint();

	//assume the offset already get
	virtual void updateJointState(VectorXd &q, VectorXd &q_dot, bool updateDynamic = true);
	virtual void clampRotation(VectorXd& _q, VectorXd& _qdot);
	virtual void initGlobalTerm();

	virtual void updateJacobiByGivenPosition(Vector3d x_position, MatrixXd &JK_v_);
	virtual void updateJacobiByGivenRestPosition(Vector3d x_position, MatrixXd &JK_v_);

protected:
	///error function ,don't use it
	virtual void applyOriOrientation();

	virtual void applyOriOrientationtomTq();
	virtual void applyOriOrientationtomTqq();
	virtual void applyOriOrientationtomTqqq();

	virtual void updateRotationMatrix();

	//************************************
	// Method:    computeTransformFirstDerive
	// FullName:  BallJoint::computeTransformFirstDerive
	// Access:    virtual protected 
	// Returns:   void
	// Qualifier:
	// Will compute mTq then compute mWq
	// 
	// mWq  = p->mWq*T + p->W*mTq
	//************************************
	virtual void computeTransformFirstDerive();
	virtual void computeLocalTransformDerive();
	virtual void computeGlobalTransformDerive();

	//************************************
	// Method:    computeTransformSecondDerive
	// FullName:  BallJoint::computeTransformSecondDerive
	// Access:    virtual protected 
	// Returns:   void
	// Qualifier:
	// mW_qi_qj = p->mWqiqj*T + p->mWqi*Tqj + p->Wqj*mTqi + p->W*mTqiqj 
	//************************************
	virtual void computeTransformSecondDerive();
	virtual void computeLocalTSecondDerive();
	virtual void computeGlobalTSecondDerive();


	//virtual void computeTransformSecondDeriv();
	//virtual void computeLocalTSecondDeriv();
	//virtual void computeGlobalTSecondDeriv();

	//************************************
	// Method:    computeTransformThirdDerive
	// FullName:  BallJoint::computeTransformThirdDerive
	// Access:    virtual protected 
	// Returns:   void
	// Qualifier:
	// mW_qi_qj_qk = p->mWqiqjqk*T + p->mWqiqj*Tqk + p->mWqiqk*Tqj + p->mWqi*Tqjqk + p->Wqjqk*mTqi + p->Wqj*mTqiqk + p->Wqk*mTqiqj + p->W*mTqiqjqk
	//************************************
	virtual void computeTransformThirdDerive();
	virtual void computeLocalTThirdDerive();
	virtual void computeGlobalTThirdDerive();
	

	void computeJacobiW();
	void computeJacobiV();
	//
	void computeJacobiWdot(VectorXd &q_dot);
	void computeJacobiVdot(VectorXd &q_dot);

	void computeJacobiWdotq(VectorXd &q_dot);
	void computeJacobiVdotq(VectorXd &q_dot);

	void computeJacobiMatrix(VectorXd &q_dot);
	


	//************************************
	// Method:    getOrientationByq
	// FullName:  BallJoint::getOrientationByq
	// Access:    protected 
	// Returns:   void
	// Qualifier:
	// Parameter: Matrix3d & m = R(q[0])*R(q[1])*R(q[2]) //x-y-z euler angle
	// Parameter: VectorXd q 3X1 vector here
	//************************************
	virtual void getOrientationByq(Matrix3d &m,VectorXd q);


	virtual void initGeneralizedInfo();


public:

	std::vector<Matrix4d> R_m;	// R_m[0] = Rx, R_m[1] = Ry, R_m[2] = Rz, following the same
	std::vector<Matrix4d> R_m_firstDeriv;
	std::vector<Matrix4d> R_m_secondDerive;
	std::vector<Matrix4d> R_m_thirdDerive;

	
};

