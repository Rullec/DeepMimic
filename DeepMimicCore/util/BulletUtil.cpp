#include "BulletUtil.h"
using namespace Eigen;

Eigen::Vector3d BT2EIGEN(const btVector3 & vec)
{
	return Eigen::Vector3d(vec[0], vec[1], vec[2]);
}

Eigen::Matrix3d BT2EIGEN(const btMatrix3x3 & mat_bt)
{
	Eigen::Matrix3d mat;
	mat.data();
	int vec[3] = { 0, 1, 2 };
	for (auto i : vec) 
		mat.block(i, 0, 1, 3) = BT2EIGEN(mat_bt[i]);
	return mat;
}