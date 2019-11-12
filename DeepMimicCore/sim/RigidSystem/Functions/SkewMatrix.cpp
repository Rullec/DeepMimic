#include "SkewMatrix.h"

void skewMatrix(Vector3d &w, Matrix3d &result)
{
	result.setZero();
	result.data()[1] = w.data()[2];
	result.data()[2] = -w.data()[1];
	result.data()[5] = w.data()[0];
	Matrix3d temp = result.transpose();
	result -= temp;
}

void skewVector(Vector3d &result, Matrix3d &w)
{
	Matrix3d A = (w - w.transpose()) / 2;
	result.setZero();
	result.data()[0] = A.data()[5];
	result.data()[1] = -A.data()[2];
	result.data()[2] = A.data()[1];
}

Vector3d fromSkewSysmmtric(const Matrix3d &R)
{
	Vector3d temp;
	temp.data()[0] = R.data()[5];
	temp.data()[1] = R.data()[6];
	temp.data()[2] = R.data()[1];
	return temp;
}
