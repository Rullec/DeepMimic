#include "EulerAngelRotationMatrix.h"
#include <math.h>
#include<iostream>
Matrix3d EulerAngelRoataion(double x, double y, double z, std::string rotation_order)
{
	Matrix3d m;
	if ("ZYX" == rotation_order)
	{
		m = xconventionRotation(x)
			* yconventionRotation(y)
			* zconventionRotation(z);
	}
	else if ("XYZ" == rotation_order)
	{
		m = zconventionRotation(z) *yconventionRotation(y) * xconventionRotation(x);
	}
	else
	{
		std::cout << "[error] Matrix3d EulerAngleRotation: Unsupported rotation order" << rotation_order << std::endl;
		exit(1);
	}

	return m;
}

Matrix3d xconventionRotation(double x)
{
	//return AngleAxisd(x, Vector3d::UnitX()).toRotationMatrix();

	Matrix3d m;

	double cosx = cos(x);
	double sinx = sin(x);

	m.setZero();
	m.data()[0] = 1;
	m.data()[4] = cosx;
	m.data()[5] = sinx;
	m.data()[7] = -sinx;
	m.data()[8] = cosx;
	return m;
}

Matrix3d yconventionRotation(double y)
{
	//return AngleAxisd(y, Vector3d::UnitY()).toRotationMatrix();
	Matrix3d m;

	double cosy = cos(y);
	double siny = sin(y);

	m.setZero();
	m.data()[0] = cosy;
	m.data()[2] = -siny;
	m.data()[4] = 1;
	m.data()[6] = siny;
	m.data()[8] = cosy;
	return m;

}

Matrix3d zconventionRotation(double z)
{
	//return AngleAxisd(z, Vector3d::UnitZ()).toRotationMatrix();
	Matrix3d m;
	m.setZero();

	double cosq = cos(z);
	double sinq = sin(z);

	m.data()[0] = cosq;
	m.data()[1] = sinq;
	m.data()[3] = -sinq;
	m.data()[4] = cosq;
	m.data()[8] = 1;

	return m;
}

void xconventionTransform(Matrix4d &output, double x)
{
	output.setZero();

	double cosx = cos(x);
	double sinx = sin(x);

	output.data()[0] = 1;
	output.data()[5] = cosx;
	output.data()[6] = sinx;
	output.data()[9] = -sinx;
	output.data()[10] = cosx;
	output.data()[15] = 1;
}

void xconventionRotation_dx(Matrix4d &output, double x)
{
	output.setZero();

	double cosx = cos(x);
	double sinx = sin(x);

	output.data()[0] = 0;
	output.data()[5] = -sinx;
	output.data()[6] = cosx;
	output.data()[9] = -cosx;
	output.data()[10] = -sinx;
}

void xconventionRotation_dxdx(Matrix4d &output, double x)
{
	output.setZero();

	double cosx = cos(x);
	double sinx = sin(x);

	output.data()[0] = 0;
	output.data()[5] = -cosx;
	output.data()[6] = -sinx;
	output.data()[9] = sinx;
	output.data()[10] = -cosx;
}

void xconventionRotation_dxdxdx(Matrix4d &output, double x)
{
	output.setZero();

	double cosx = cos(x);
	double sinx = sin(x);

	output.data()[0] = 0;
	output.data()[5] = sinx;
	output.data()[6] = -cosx;
	output.data()[9] = cosx;
	output.data()[10] = sinx;
}

void yconventionTransform(Matrix4d &output, double y)
{
	output.setZero();
	double cosy = cos(y);
	double siny = sin(y);

	output.data()[0] = cosy;
	output.data()[2] = -siny;
	output.data()[5] = 1;
	output.data()[8] = siny;
	output.data()[10] = cosy;
	output.data()[15] = 1;
}

void yconventionRotation_dy(Matrix4d & output, double y)
{
	output.setZero();


	double cosy = cos(y);
	double siny = sin(y);

	output.data()[0] = -siny;
	output.data()[2] = -cosy;
	output.data()[5] = 0;
	output.data()[8] = cosy;
	output.data()[10] = -siny;
}

void yconventionRotation_dydy(Matrix4d &output, double y)
{
	output.setZero();

	double cosy = cos(y);
	double siny = sin(y);

	output.data()[0] = -cosy;
	output.data()[2] = siny;
	output.data()[5] = 0;
	output.data()[8] = -siny;
	output.data()[10] = -cosy;
}


void yconventionRotation_dydydy(Matrix4d &output, double y)
{
	output.setZero();

	double cosy = cos(y);
	double siny = sin(y);

	output.data()[0] = siny;
	output.data()[2] = cosy;
	output.data()[5] = 0;
	output.data()[8] = -cosy;
	output.data()[10] = siny;
}

void zconventionTransform(Matrix4d &output, double z)
{
	output.setZero();

	double cosz = cos(z);
	double sinz = sin(z);

	output.data()[0] = cosz;
	output.data()[1] = sinz;
	output.data()[4] = -sinz;
	output.data()[5] = cosz;
	output.data()[10] = 1;
	output.data()[15] = 1;
}

void zconventionRotation_dz(Matrix4d & output, double z)
{
	output.setZero();
	double cosz = cos(z);
	double sinz = sin(z);

	output.data()[0] = -sinz;
	output.data()[1] = cosz;
	output.data()[4] = -cosz;
	output.data()[5] = -sinz;
	output.data()[10] = 0;
}

void zconventionRotation_dzdz(Matrix4d &output, double z)
{
	output.setZero();
	double cosz = cos(z);
	double sinz = sin(z);

	output.data()[0] = -cosz;
	output.data()[1] = -sinz;
	output.data()[4] = sinz;
	output.data()[5] = -cosz;
	output.data()[10] = 0;
}

void zconventionRotation_dzdzdz(Matrix4d &output, double z)
{
	output.setZero();
	double cosz = cos(z);
	double sinz = sin(z);

	output.data()[0] = sinz;
	output.data()[1] = -cosz;
	output.data()[4] = cosz;
	output.data()[5] = sinz;
	output.data()[10] = 0;
}

std::vector<Matrix3d> RotationDeriv(double x, double y, double z, const std::string rotation_order)
{
	std::vector<Matrix3d> m(3);
	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);
	
	// R = Rz * Ry * Rx, when the order is "XYZ", first X -> then Y -> then Z
	if ("XYZ" == rotation_order)
	{
		// R = Rz * Ry * Rx, when the order is "XYZ", first X -> then Y -> then Z
		
		// m[0] = dRdx
		m[0].setZero();
		m[0].data()[3] = sinx * sinz + cosx * cosz * siny;
		m[0].data()[4] = cosx * siny * sinz - cosz * sinx;
		m[0].data()[5] = cosx * cosy;
		m[0].data()[6] = cosx * sinz - cosz * sinx * siny;
		m[0].data()[7] = -cosx * cosz - sinx * siny * sinz;
		m[0].data()[8] = - cosy * sinx;

		// m[1] = dRdy
		m[1].setZero();
		m[1].data()[0] = -cosz * siny;
		m[1].data()[1] = -siny * sinz;
		m[1].data()[2] = -cosy;
		m[1].data()[3] = cosy * cosz * sinx;
		m[1].data()[4] = cosy * sinx * sinz;
		m[1].data()[5] = -sinx * siny;
		m[1].data()[6] = cosx * cosy * cosz;
		m[1].data()[7] = cosx * cosy * sinz;
		m[1].data()[8] = -cosx * siny;

		//respect z
		m[2].setZero();
		m[2].data()[0] = -cosy * sinz;
		m[2].data()[1] = cosy * cosz;
		m[2].data()[2] = 0;
		m[2].data()[3] = -cosx * cosz - sinx * siny * sinz;;
		m[2].data()[4] = cosz * sinx * siny - cosx * sinz;
		m[2].data()[5] = 0;
		m[2].data()[6] = cosz * sinx - cosx * siny * sinz;
		m[2].data()[7] = sinx * sinz + cosx * cosz *siny;
		m[2].data()[8] = 0;
		

	}
	else if ("ZYX" == rotation_order)
	{//repect to x
		m[0].setZero();
		m[0].data()[1] = cosx * cosz * siny - sinx * sinz;
		m[0].data()[2] = cosx * sinz + cosz * sinx * siny;
		m[0].data()[4] = -cosz * sinx - cosx * siny * sinz;
		m[0].data()[5] = cosx * cosz - sinx * siny * sinz;
		m[0].data()[7] = -cosx * cosy;
		m[0].data()[8] = -cosy * sinx ;


		//respect y
		m[1].setZero();
		m[1].data()[0] = -cosz * siny;
		m[1].data()[1] = cosy * cosz * sinx;
		m[1].data()[2] = -cosx * cosy * cosz;
		m[1].data()[3] = siny * sinz;
		m[1].data()[4] = -cosy * sinx * sinz;
		m[1].data()[5] = cosx * cosy * sinz;
		m[1].data()[6] = cosy;
		m[1].data()[7] = sinx * siny;
		m[1].data()[8] = -cosx * siny;

		//respect z
		m[2].setZero();
		m[2].data()[0] = -cosy * sinz;
		m[2].data()[1] = cosx * cosz - sinx * siny * sinz;
		m[2].data()[2] = cosz * sinx + cosx * siny * sinz;
		m[2].data()[3] = -cosy * cosz;
		m[2].data()[4] = -cosx * sinz - cosz * sinx * siny;
		m[2].data()[5] = cosx * cosz * siny - sinx * sinz ;


	}
	else
	{
		std::cout << "[error] Unsupported rotaiton order RotationDeriv : " << rotation_order << std::endl;
		exit(1);
	}

	return m;
}

void rotationFirstDerive_dx(Matrix4d& output, double x, double y, double z, const std::string rotation_order )
{
	output.topLeftCorner<3, 3>().setZero();
	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);
	// R = Rz * Ry * Rx, when the order is "XYZ", first X -> then Y -> then Z
	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if (rotation_order == "ZYX")
	{
		//output.data()[0] = 0;
		output.data()[1] = cosx * cosz * siny - sinx * sinz;
		output.data()[2] = cosx * sinz + cosz * sinx * siny;
		//output.data()[4] = 0;
		output.data()[5] = -cosz * sinx - cosx * siny * sinz;
		output.data()[6] = cosx * cosz - sinx * siny * sinz;
		//output.data()[8] = 0;
		output.data()[9] = -cosx * cosy;
		output.data()[10] = -cosy * sinx;
	}
	else if (rotation_order == "XYZ")
	{
		//output.data()[0] = 0;
		//output.data()[1] = 0;
		//output.data()[2] = 0;
		output.data()[4] = sinx * sinz + cosx * cosz * siny;
		output.data()[5] = cosx * siny * sinz - cosz * sinx;
		output.data()[6] = cosx * cosy;
		output.data()[8] = cosx * sinz - cosz * sinx * siny;
		output.data()[9] = -cosx * cosz - sinx * siny * sinz;
		output.data()[10] = -cosy * sinx;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationFirstDerive_dx: " << rotation_order << std::endl;
	}
	

	
}

void rotationFirstDerive_dy(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);
	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[0] = -cosz * siny;
		output.data()[1] = cosy * cosz * sinx;
		output.data()[2] = -cosx * cosy * cosz;
		output.data()[4] = siny * sinz;
		output.data()[5] = -cosy * sinx * sinz;
		output.data()[6] = cosx * cosy * sinz;
		output.data()[8] = cosy;
		output.data()[9] = siny * sinx;
		output.data()[10] = -cosx * siny;
	}
	else if ("XYZ" == rotation_order)
	{
		output.data()[0] = -cosz * siny;
		output.data()[1] = -siny * sinz;
		output.data()[2] = -cosy;
		output.data()[4] = cosy * cosz * sinx;
		output.data()[5] = cosy * sinx * sinz;
		output.data()[6] = -sinx * siny;
		output.data()[8] = cosx * cosy * cosz;
		output.data()[9] = cosx * cosy * sinz;
		output.data()[10] = -cosx * siny;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationFirstDerive_dy: " << rotation_order << std::endl;
		exit(1);
	}
	

}

void rotationFirstDerive_dz(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[0] = -cosy * sinz;
		output.data()[1] = cosx * cosz -sinx * siny * sinz;
		output.data()[2] = cosz * sinx + cosx * siny * sinz;
		output.data()[4] = -cosy * cosz;
		output.data()[5] = -cosx * sinz - sinx * siny * cosz;
		output.data()[6] = -sinz * sinx + cosx * siny * cosz;
	}
	else if ("XYZ" == rotation_order)
	{
		output.data()[0] = -cosy * sinz;
		output.data()[1] = cosy * cosz;
		//output.data()[2] = 0;
		output.data()[4] = -cosx * cosz - sinx * siny * sinz;
		output.data()[5] = cosz * sinx * siny - cosx * sinz;
		//output.data()[6] = 0;
		output.data()[8] = cosz * sinx - cosx * siny * sinz;
		output.data()[9] = sinx * sinz + cosx * cosz * siny;
		//output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationFirstDerive_dz: " << rotation_order << std::endl;
		exit(1);
	}
	
}

void rotationSecondDerive_dxdx(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[1] = -cosx * sinz - cosz * sinx * siny;
		output.data()[2] = cosx * cosz * siny - sinx * sinz;
		output.data()[5] = -cosx * cosz + sinx * siny * sinz;
		output.data()[6] = -cosz * sinx - cosx * siny * sinz;
		output.data()[9] = cosy * sinx;
		output.data()[10] = -cosx * cosy;
	}
	else if ("XYZ" == rotation_order)
	{
		//output.data()[0] = 0;
		//output.data()[1] = 0;
		//output.data()[2] = 0;
		output.data()[4] = cosx * sinz - cosz * sinx * siny;
		output.data()[5] = -cosx * cosz - sinx * siny * sinz;
		output.data()[6] = -cosy * sinx;
		output.data()[8] = -sinx * sinz - cosx * cosz * siny ;
		output.data()[9] = cosz * sinx - cosx * siny * sinz;
		output.data()[10] = -cosx * cosy;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dxdx: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dxdy(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if (rotation_order == "ZYX")
	{
		output.data()[1] = cosz * cosx * cosy;
		output.data()[2] = sinx * cosz * cosy;
		output.data()[5] = -cosx * cosy * sinz;
		output.data()[6] = -sinx * cosy * sinz;
		output.data()[9] = siny * cosx;
		output.data()[10] = sinx * siny;
	}
	else if (rotation_order == "XYZ")
	{
		output.data()[4] = cosx * cosy * cosz;
		output.data()[5] = cosx * cosy * sinz;
		output.data()[6] = -cosx * siny;
		output.data()[8] = -cosy * cosz * sinx;
		output.data()[9] = -cosy * sinx * sinz;
		output.data()[10] = sinx * siny;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dxdy: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dxdz(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if (rotation_order == "ZYX")
	{
		output.data()[1] = -sinz * cosx * siny - sinx * cosz;
		output.data()[2] = -sinx * sinz * siny + cosx * cosz;
		output.data()[5] = sinx * sinz - cosx * siny * cosz;
		output.data()[6] = -sinz * cosx - sinx * siny * cosz;

		// output.data()[9] = 0;
		// output.data()[10] = 0;
	}
	else if (rotation_order == "XYZ")
	{
		output.data()[4] = cosz * sinx - cosx * siny * sinz;
		output.data()[5] = sinx * sinz + cosx * cosz * siny;
		output.data()[6] = 0;
		output.data()[8] = cosx * cosz + sinx * siny * sinz;
		output.data()[9] = cosx * sinz - cosz * sinx * siny;
		output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dxdz: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dydx(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		//output.data()[0] = 0;
		output.data()[1] = cosz * cosx * cosy;
		output.data()[2] = sinx * cosz * cosy;
		//output.data()[4] = 0;
		output.data()[5] = -cosx * cosy * sinz;
		output.data()[6] = -sinx * cosy * sinz;
		//output.data()[8] = 0;
		output.data()[9] = siny * cosx;
		output.data()[10] = sinx * siny;
	}
	else if ("XYZ" == rotation_order)
	{
		//output.data()[0] = 0;
		//output.data()[1] = 0;
		//output.data()[2] = 0;
		output.data()[4] = cosx * cosy * cosz;
		output.data()[5] = cosx * cosy * sinz;
		output.data()[6] = -cosx * siny;
		output.data()[8] = -cosy * cosz *sinx ;
		output.data()[9] = -cosy * sinx *sinz;
		output.data()[10] = sinx * siny;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dydx: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dydy(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[0] = -cosy * cosz;
		output.data()[1] = -cosz * sinx * siny;
		output.data()[2] = cosx * cosz * siny;
		output.data()[4] = cosy * sinz;
		output.data()[5] = sinx * siny * sinz;
		output.data()[6] = -cosx * siny * sinz;
		output.data()[8] = -siny;
		output.data()[9] = cosy * sinx;
		output.data()[10] = -cosx * cosy;
	}
	else if ("XYZ" == rotation_order)
	{
		output.data()[0] = -cosy * cosz;
		output.data()[1] = -cosy * sinz;
		output.data()[2] = siny;
		output.data()[4] = -cosz * sinx * siny;
		output.data()[5] = -sinx * siny * sinz;
		output.data()[6] = -cosy * sinx;
		output.data()[8] = -cosx * cosz * siny;
		output.data()[9] = -cosx * siny * sinz;
		output.data()[10] = -cosx * cosy;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dydy: " << rotation_order << std::endl;
		exit(1);
	}
	
}

void rotationSecondDerive_dydz(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if (rotation_order == "ZYX")
	{
		output.data()[0] = siny * sinz;
		output.data()[1] = -sinz * sinx * cosy;
		output.data()[2] = cosx * sinz * cosy;
		output.data()[4] = siny * cosz;
		output.data()[5] = -sinx * cosy * cosz;
		output.data()[6] = cosx * cosy * cosz;
		//output.data()[8] = 0;
		//output.data()[9] = 0;
		//output.data()[10] = 0;
	}
	else if (rotation_order == "XYZ")
	{
		output.data()[0] = siny * sinz;
		output.data()[1] = -cosz * siny;
		output.data()[2] = 0;
		output.data()[4] = -cosy * sinx * sinz;
		output.data()[5] = cosy * cosz * sinx;
		output.data()[6] = 0;
		output.data()[8] = -cosx * cosy * sinz;
		output.data()[9] = cosx * cosy * cosz;
		output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dydz: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dzdx(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		//output.data()[0] = 0;
		output.data()[1] = -sinz * cosx * siny - sinx * cosz;
		output.data()[2] = -sinx * sinz * siny + cosx * cosz;
		//output.data()[4] = 0;
		output.data()[5] = sinx * sinz - cosx * siny * cosz;
		output.data()[6] = -sinz * cosx - sinx * siny * cosz;
	}
	else if ("XYZ" == rotation_order)
	{
		//output.data()[0] = 0;
		//output.data()[1] = 0;
		//output.data()[2] = 0;
		output.data()[4] = cosz * sinx - cosx * siny * sinz;
		output.data()[5] = sinx * sinz + cosx * cosz * siny;
		output.data()[6] = 0;
		output.data()[8] = cosx * cosz + sinx * siny * sinz;
		output.data()[9] = cosx * sinz - cosz * sinx * siny;
		output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dzdx: " << rotation_order << std::endl;
		exit(1);
	}
}

void rotationSecondDerive_dzdy(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3,3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[0] = siny * sinz;
		output.data()[1] = -sinz * sinx * cosy;
		output.data()[2] = cosx * sinz * cosy;
		output.data()[4] = siny * cosz;
		output.data()[5] = -sinx * cosy * cosz;
		output.data()[6] = cosx * cosy * cosz;
	}
	else if ("XYZ" == rotation_order)
	{
		output.data()[0] = siny * sinz;
		output.data()[1] = -cosz *siny;
		output.data()[2] = 0;
		output.data()[4] = -cosy * sinx * sinz;
		output.data()[5] = cosy * cosz * sinx;
		output.data()[6] = 0;
		output.data()[8] = -cosx * cosy * sinz;
		output.data()[9] = cosx * cosy *cosz;
		output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dzdy: " << rotation_order << std::endl;
		exit(1);
	}
}


void rotationSecondDerive_dzdz(Matrix4d& output, double x, double y, double z, const std::string rotation_order)
{
	output.topLeftCorner<3, 3>().setZero();

	double sinx = sin(x);
	double cosx = cos(x);

	double siny = sin(y);
	double cosy = cos(y);

	double sinz = sin(z);
	double cosz = cos(z);

	// for the meaning of "rotation_order", pleace check the explanation of it in LoboJointV2.h
	if ("ZYX" == rotation_order)
	{
		output.data()[0] = -cosy * cosz;
		output.data()[1] = -cosz * sinx * siny - cosx * sinz;
		output.data()[2] = cosx * cosz * siny - sinx * sinz;
		output.data()[4] = cosy * sinz;
		output.data()[5] = -cosx * cosz + sinx * siny * sinz;
		output.data()[6] = -cosz * sinx - cosx * siny * sinz;
	}
	else if ("XYZ" == rotation_order)
	{
		output.data()[0] = -cosy * cosz;
		output.data()[1] = -cosy * sinz;
		output.data()[2] = 0;
		output.data()[4] = cosx * sinz - cosz * sinx * siny;
		output.data()[5] = -cosx * cosz - sinx * siny *sinz;
		output.data()[6] = 0;
		output.data()[8] = -cosx * cosz * siny - sinx * sinz;
		output.data()[9] = cosz * sinx - cosx * siny * sinz;
		output.data()[10] = 0;
	}
	else
	{
		std::cout << "[error] Unsupported rotation order in rotationSecondDerive_dzdz: " << rotation_order << std::endl;
		exit(1);
	}
	
}