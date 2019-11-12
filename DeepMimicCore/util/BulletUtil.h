#pragma once
#include <LinearMath/btTransformUtil.h>
#include <Eigen/Dense>

Eigen::Vector3d BT2EIGEN(const btVector3 & vec);
Eigen::Matrix3d BT2EIGEN(const btMatrix3x3 & mat);