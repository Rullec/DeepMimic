#pragma once
#include <util/MathUtil.h>
#include <sim/SimCharacter.h>
typedef std::pair<int, int> idx_pair;

class cReverseController {
public:
	cReverseController(cSimCharacter * sim_char);
	void CalcPDTarget(const tVectorXd & input_torque, const tVectorXd & input_cur_pose, const tVectorXd & input_cur_vel, tVectorXd & output_pd_target);
	void CalcAction(const tVectorXd & input_torque, const tVector & input_cur_pose, const tVectorXd & input_cur_vel, tVector & output_action);
	void SetParams(double timestep, const Eigen::MatrixXd &M, const Eigen::MatrixXd &C, const Eigen::VectorXd & kp, const Eigen::VectorXd & kd);

private:
	// pipeline control data
	bool mEnableSolving, mEnableFastSolving;

	// outsider data
	//Eigen::VectorXd mKp;
	//Eigen::VectorXd mKd;
	double mTimestep;
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> Kp_mat, Kd_mat;
	std::vector<idx_pair> mRawIndexLst, mNewIndexLst;	// used for the topo info of system matrix.
	int mRawMatSize, mNewMatSize;						// The size of new sys mat and raw sys mat
	Eigen::MatrixXd Kp_dense, Kd_dense;

	cSimCharacter* mChar;
	Eigen::MatrixXd M;
	Eigen::MatrixXd C;
	Eigen::MatrixXd I;

	// buffer data
	Eigen::MatrixXd M_s_inv, A, E, E_sub;
	Eigen::VectorXd b, f, f_sub;
	
	void CalcPoseDiffFromTorque(const tVectorXd & input_torque, const tVectorXd & input_pose, const tVectorXd & input_vel, tVectorXd & output_vel_diff);
	void BuildTopo();
	void MatrixTransfer(Eigen::MatrixXd & raw_mat, Eigen::MatrixXd & new_mat, int mode)const;
	void VectorTransfer(tVectorXd & old_vec, tVectorXd & new_vec, int mode) const;

	tVectorXd FastSolve(Eigen::MatrixXd & A, tVectorXd & b) const;
};