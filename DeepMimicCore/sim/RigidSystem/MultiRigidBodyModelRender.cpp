#include "stdafx.h"
#include "MultiRigidBodyModelRender.h"
#include<fstream>
//#include "Simulator\simulatorBase\simulatorRenderBase.h"
#include "Simulator/RigidBodyMotion/RigidSystem/MultiRigidBodyCore.h"
#include "ColorTable/Colortable.h"
MultiRigidBodyModelRender::MultiRigidBodyModelRender(MultiRigidBodyModel* multibody_)
{
	this->multibody = multibody_;
	sphere1 = new SphereRender("3dmodel/sphere.obj");
	cube = new SphereRender("3dmodel/rigidlink.obj", false);
	IsRenderMesh = true;
	IsRenderBox = false;
	selectedJointid = -1;
	selectedLinkid = -1;
}

MultiRigidBodyModelRender::~MultiRigidBodyModelRender()
{
	delete sphere1;
	delete cube;
}

void MultiRigidBodyModelRender::renderSystem(QOpenGLShaderProgram *program)
{
	if (nullptr == multibody)
		return;
	int numJoints = multibody->getnNumJoints();
	//double scale = 10.0;
	int DiffuseProduct_loc = program->uniformLocation("DiffuseProduct");
	double defaultcolor[3];
	getColor(0, defaultcolor);
	QVector4D color = QVector4D(defaultcolor[0], defaultcolor[1], defaultcolor[2], 0.5);
	QVector4D selectedcolor = QVector4D(0, 1, 0, 0.5);
	//Vector3d rigidoffset = Vector3d(0, 0.0, 0);
	for (int i = 0; i < numJoints; i++)
	{
		LoboJointV2* joint = multibody->getJoint(i);
		LoboLink* link = joint->getConnectedLink();

		Vector3d position = joint->position_w;
		if (i==selectedJointid)
		{
			sphere1->drawMesh(program, position, 0.005, 1);
		}
		else
		{
			sphere1->drawMesh(program, position, 0.005, 6);
		}

		Vector3d link_position = link->getPosition_world();
		Matrix3d rotation = link->getOrientation_world();
		//cube->drawMesh(program, link_position, rotation, 1.0, i+3);
		/*std::cout << "==========render function output==========" << std::endl;
		std::cout << "link " << link->getName() << " rotation \n" << rotation << std::endl;
		std::cout << "==========render function output end==========" << std::endl;*/
		if (i==selectedLinkid)
		{
			program->setUniformValue(DiffuseProduct_loc, selectedcolor);
		}
		else
		{
			program->setUniformValue(DiffuseProduct_loc, color);
		}
		

		Matrix3d visualOrientation = link->getVisual_orientation();
		Vector3d visualPosition = link->getVisual_position();
		//Vector3d oriposition = link->getOri_position();
		Vector3d t = rotation*visualPosition + link_position;
		Matrix3d R = rotation*visualOrientation;
		//std::cout << "-----------------" << std::endl;
		//std::cout<<
		//std::cout << visualOrientation << std::endl;
		if (link->getUseMeshRender() && IsRenderMesh)
		{
			//Vector3d box = link->getVisualbox();
			link->getLinkMesh()->drawMeshScaleGlobal(program, t, R, 1.0, 9);

			//draw rotate axis
			HingeJoint* hjoint = (HingeJoint*)joint;
			//Matrix3d hinge = hjoint->getJoint_parent()->orientation_w*hjoint->orientation;
			DrawFrame(program, hjoint);
		}
		if (IsRenderBox)
		{
			Vector3d box = link->getVisualbox();
			renderBoxs(program, box, t, R, i + 1);
			HingeJoint* hjoint = (HingeJoint*)joint;
			//Matrix3d hinge = hjoint->getJoint_parent()->orientation_w*hjoint->orientation;
			DrawFrame(program, hjoint);
		}
	}
}

void MultiRigidBodyModelRender::renderSystem_WireFrame(QOpenGLShaderProgram *program)
{
	if (nullptr == multibody)
		return;
	int numJoints = multibody->getnNumJoints();
	//double scale = 10.0;
	int DiffuseProduct_loc = program->uniformLocation("DiffuseProduct");
	double defaultcolor[3];
	getColor(0, defaultcolor);
	QVector4D color = QVector4D(defaultcolor[0], defaultcolor[1], defaultcolor[2], 0.5);
	QVector4D selectedcolor = QVector4D(0, 1, 0, 0.5);
	//Vector3d rigidoffset = Vector3d(0, 0.0, 0);
	for (int i = 0; i < numJoints; i++)
	{
		LoboJointV2* joint = multibody->getJoint(i);
		LoboLink* link = joint->getConnectedLink();

		Vector3d position = joint->position_w;
		Vector3d link_position = link->getPosition_world();
		Matrix3d rotation = link->getOrientation_world();
		Matrix3d visualOrientation = link->getVisual_orientation();
		Vector3d visualPosition = link->getVisual_position();
		//Vector3d oriposition = link->getOri_position();
		Vector3d t = rotation*visualPosition + link_position;
		Matrix3d R = rotation*visualOrientation;
		if (link->getUseMeshRender() && IsRenderMesh)
		{
			link->getLinkMesh()->drawMeshScaleGlobalWireFrame(program, t, R, 1.0, 9);
		}
	
	}
}

void MultiRigidBodyModelRender::renderBoxs(QOpenGLShaderProgram *program, Vector3d box, Vector3d translation, Matrix3d orientation, int colorid)
{
	int DiffuseProduct_loc = program->uniformLocation("DiffuseProduct");
	int BoolPoint_loc = program->uniformLocation("point");

	//use default color
	double defaultcolor[3];
	getColor(colorid, defaultcolor);
	QVector4D color = QVector4D(defaultcolor[0], defaultcolor[1], defaultcolor[2], 1.0);
	program->setUniformValue(DiffuseProduct_loc, color);

	double l = box.data()[0] / 2.0;
	double h = box.data()[1] / 2.0;
	double w = box.data()[2] / 2.0;

	Vector3d boxnodes[8];

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				double x, y, z;
				x = l*(i * 2 - 1);
				y = h*(j * 2 - 1);
				z = w*(k * 2 - 1);
				int index = i * 4 + j * 2 + k;

				boxnodes[index] = translation + orientation*Vector3d(x, y, z);
			}
		}
	}

	//top face
	Vector3d nodeTop[4];
	Vector3d topNorm = orientation*Vector3d(0, 1, 0);
	nodeTop[0] = boxnodes[3];
	nodeTop[1] = boxnodes[7];
	nodeTop[2] = boxnodes[6];
	nodeTop[3] = boxnodes[2];

	Vector3d nodeBottom[4];
	nodeBottom[0] = boxnodes[0];
	nodeBottom[1] = boxnodes[4];
	nodeBottom[2] = boxnodes[5];
	nodeBottom[3] = boxnodes[1];

	Vector3d nodeLeft[4];
	Vector3d leftNorm = orientation*Vector3d(-1, 0, 0);
	nodeLeft[0] = boxnodes[3];
	nodeLeft[1] = boxnodes[2];
	nodeLeft[2] = boxnodes[0];
	nodeLeft[3] = boxnodes[1];

	Vector3d nodeRight[4];
	nodeRight[0] = boxnodes[4];
	nodeRight[1] = boxnodes[6];
	nodeRight[2] = boxnodes[7];
	nodeRight[3] = boxnodes[5];

	Vector3d nodeFront[4];
	Vector3d frontNorm = orientation*Vector3d(0, 0, 1);
	nodeFront[0] = boxnodes[3];
	nodeFront[1] = boxnodes[1];
	nodeFront[2] = boxnodes[5];
	nodeFront[3] = boxnodes[7];

	Vector3d nodeBack[4];
	nodeBack[0] = boxnodes[0];
	nodeBack[1] = boxnodes[2];
	nodeBack[2] = boxnodes[6];
	nodeBack[3] = boxnodes[4];



	glBegin(GL_QUADS);
	glNormal3f(topNorm[0], topNorm[1], topNorm[2]);
	for (int i = 0; i < 4; i++)
	{
		glVertex3f(nodeTop[i].x(), nodeTop[i].y(), nodeTop[i].z());
	}

	glNormal3f(-topNorm[0], -topNorm[1], -topNorm[2]);
	for (int i = 0; i < 4; i++)
	{
		glVertex3f(nodeBottom[i].x(), nodeBottom[i].y(), nodeBottom[i].z());
	}

	glNormal3f(leftNorm[0], leftNorm[1], leftNorm[2]);
	for (int i = 0; i < 4; i++)
	{

		glVertex3f(nodeLeft[i].x(), nodeLeft[i].y(), nodeLeft[i].z());
	}

	glNormal3f(-leftNorm[0], -leftNorm[1], -leftNorm[2]);
	for (int i = 0; i < 4; i++)
	{
		glVertex3f(nodeRight[i].x(), nodeRight[i].y(), nodeRight[i].z());
	}

	glNormal3f(frontNorm[0], frontNorm[1], frontNorm[2]);

	for (int i = 0; i < 4; i++)
	{
		glVertex3f(nodeFront[i].x(), nodeFront[i].y(), nodeFront[i].z());
	}

	glNormal3f(-frontNorm[0], -frontNorm[1], -frontNorm[2]);

	for (int i = 0; i < 4; i++)
	{
		glVertex3f(nodeBack[i].x(), nodeBack[i].y(), nodeBack[i].z());
	}
	glEnd();
}

void MultiRigidBodyModelRender::DrawArrowWorld(QOpenGLShaderProgram* program, Vector3d start, Vector3d end, int colorindex /*= 0*/)
{
	int DiffuseProduct_loc = program->uniformLocation("DiffuseProduct");
	int BoolPoint_loc = program->uniformLocation("point");
	program->setUniformValue(BoolPoint_loc, 1);
	glLineWidth(1);
	double defaultcolor[3];
	getColor(colorindex, defaultcolor);
	QVector4D color = QVector4D(defaultcolor[0], defaultcolor[1], defaultcolor[2], 0.5);
	program->setUniformValue(DiffuseProduct_loc, color);
	glPushMatrix();
	glBegin(GL_LINES);
	glVertex3f(start.x(), start.y(), start.z());
	glVertex3f(end.x(), end.y(), end.z());
	glEnd();
	glPopMatrix();
	glLineWidth(1);
	program->setUniformValue(BoolPoint_loc, 0);
}

void MultiRigidBodyModelRender::DrawFrame(QOpenGLShaderProgram *program, HingeJoint * hjoint)
{
	Vector3d position = hjoint->position_w;
	Matrix3d hinge = hjoint->orientation_w;
	Vector3d Yaxis = hinge/*.transpose()*/*Vector3d::UnitY();
	Yaxis /= 15;
	this->DrawArrowWorld(program, position - Yaxis, position + Yaxis * 2, 1);
	Vector3d Xaxis = hinge/*.transpose()*/*Vector3d::UnitX();
	Xaxis /= 15;
	this->DrawArrowWorld(program, position - Xaxis, position + Xaxis * 2, 0);
	Vector3d Zaxis = hinge/*.transpose()*/*Vector3d::UnitZ();
	Zaxis /= 15;
	this->DrawArrowWorld(program, position - Zaxis, position + Zaxis * 2, 2);
}
