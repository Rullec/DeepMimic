#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

void skewMatrix(Vector3d &w, Matrix3d &result);
void skewVector(Vector3d &result, Matrix3d &w);

Vector3d fromSkewSysmmtric(const Matrix3d &R);