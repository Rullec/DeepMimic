#pragma once
 
#include <QOpenGLShaderProgram>
#include "Render/SphereRender.h"
#include "Simulator\RigidBodyMotion\RigidSystem\HingeJoint.h"

class MultiRigidBodyModel;
class MultiRigidBodyModelRender
{
public:
	MultiRigidBodyModelRender(MultiRigidBodyModel* multibody);
	~MultiRigidBodyModelRender();

	virtual void renderSystem(QOpenGLShaderProgram *program);
	virtual void renderSystem_WireFrame(QOpenGLShaderProgram *program);
	//void renderSystem_WireFrame(QOpenGLShaderProgram * program);

	void setRenderMesh(bool val) { IsRenderMesh = val; }
	void setRenderBox(bool val) { IsRenderBox = val; }

	void setSelectedLink(int id) { selectedLinkid = id; }
	void setSelectedJoint(int id) { selectedJointid = id; }

protected:

	virtual void renderBoxs(QOpenGLShaderProgram *program, Vector3d box, Vector3d translation, Matrix3d orientation,int colorid);
	void DrawArrowWorld(QOpenGLShaderProgram* program, Vector3d start, Vector3d end, int colorindex = 0);
	void DrawFrame(QOpenGLShaderProgram *program,HingeJoint*hjoint);
	//render sphere
	bool IsRenderMesh;
	bool IsRenderBox;
	int selectedLinkid;
	int selectedJointid;
	MultiRigidBodyModel* multibody;
	SphereRender* sphere1;
	SphereRender* cube;
};

