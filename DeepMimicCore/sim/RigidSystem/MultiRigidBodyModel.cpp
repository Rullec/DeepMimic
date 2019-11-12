#include "MultiRigidBodyModel.h"
#include "LoboLink.h"
#include "BallJoint.h"
#include "HingeJoint.h"
#include "FixedJoint.h"
#include "UniversalJoint.h"

#include "Functions/SkewMatrix.h"
#include "Functions/EulerAngelRotationMatrix.h"

#include "XML/tinyxml2.h"
#include <queue>
#include <fstream>
#include <iostream>
#include <iterator>
#define PI_2 1.57079632679
//#define GAZEBO_DATA_VERSION

//#define PI_2 1.57079632679
#ifndef GAZEBO_DATA_VERSION
#pragma message("如果使用炳坤学长的网格，就打开宏GAZEBO_DATA_VERSION")
#endif

using namespace std;

MultiRigidBodyModel::MultiRigidBodyModel()
{
	useGravity = true;
	root_joint = nullptr;
	gravity_force = Vector3d(0, -9.8, 0);
}


MultiRigidBodyModel::~MultiRigidBodyModel()
{
	deleteModel();
	std::cout << "test" << std::endl;
}

void MultiRigidBodyModel::initGeneralizedInfo()
{
	buildJointTreeIndex();
	//if(this->numofDOFs==0)
	numofDOFs = 0;
	/*if (joint_tree_index.size() == 0)
	return;*/
	generalized_offset.resize(joint_tree_index.size());
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		generalized_offset[jointid] = numofDOFs;
		joints_list[jointid]->setGeneralized_offset(numofDOFs);
		numofDOFs += joints_list[jointid]->getR();

		joints_list[jointid]->initOriPositionInWorld();
	}

	std::cout << "[log] total DOFs = >" << numofDOFs << std::endl;

	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		joints_list[jointid]->setGlobalR(numofDOFs);
		joints_list[jointid]->updateChainIndex();
		joints_list[jointid]->initGlobalTerm();
	}


}

void MultiRigidBodyModel::loadXML(const char* filename, double massscale, double meshscale)
{
	std::cout << "[log] LOAD RIGIDBODY MODEL BEGIN..." << std::endl;
	deleteModel();

	std::cout << "[log] rigid mass scale = " << massscale << std::endl;
	tinyxml2::XMLDocument doc;
	if (doc.LoadFile(filename))
	{
		std::cout << "[log] Read XML file:" << filename << "failed! " << std::endl;
		return;
	}

	// handle a exception between different XML version
	tinyxml2::XMLElement* model = doc.FirstChildElement("rm");
	if (nullptr == model)
	{
		std::cout << "[warn] the XML file have no <rm> notation, maybe you should read '<rom>' notation" << std::endl;
		model = doc.FirstChildElement("rom");
	}
	model = model->FirstChildElement("world")->FirstChildElement("model");
	name = model->Attribute("name");
	tinyxml2::XMLElement* link = model->FirstChild()->ToElement();

	std::vector<LinkXML> links;
	std::vector<JointXML> joints;

	//transform from Gazebo to Opengl
	Matrix3d xyzXZYtransform = xconventionRotation(-PI_2);
	Matrix3d inverseY;
	inverseY.setIdentity();
	inverseY.data()[4] = 1;
	xyzXZYtransform = inverseY * xyzXZYtransform;
#ifndef GAZEBO_DATA_VERSION
	xyzXZYtransform.setIdentity();
#endif // !1

	//xyzXZYtransform.setIdentity();
	int root_link = -1;
	std::vector<const char*> linkName;

	for (link; link; link = link->NextSiblingElement())
	{
		const char* elementName = link->Name();
		if (elementName&&strcmp(elementName, "link") == 0)
		{
			//std::cout << link->Name() << std::endl;
			const char* pose = link->FirstChildElement("pose")->GetText();	// pose是link的局部坐标系的全局坐标
			std::istringstream iss(pose);
			std::istream_iterator<double> it(iss), end;
			std::vector<double> poseDouble(it, end);
			double mass = link->FirstChildElement("inertial")->FirstChildElement("mass")->DoubleText();
			tinyxml2::XMLElement* inertia = link->FirstChildElement("inertial")->FirstChildElement("inertia");
			Matrix3d inertia_matrix;
			inertia_matrix.setZero();
#ifndef GAZEBO_DATA_VERSION
			inertia_matrix.data()[0] = inertia->FirstChildElement("ixx")->DoubleText();
			inertia_matrix.data()[3] = inertia->FirstChildElement("ixz")->DoubleText();
			inertia_matrix.data()[6] = inertia->FirstChildElement("ixy")->DoubleText();
			inertia_matrix.data()[4] = inertia->FirstChildElement("izz")->DoubleText();
			inertia_matrix.data()[7] = inertia->FirstChildElement("iyz")->DoubleText();
			inertia_matrix.data()[8] = inertia->FirstChildElement("iyy")->DoubleText();
#else
			inertia_matrix.data()[0] = inertia->FirstChildElement("ixx")->DoubleText();
			inertia_matrix.data()[3] = inertia->FirstChildElement("ixz")->DoubleText();
			inertia_matrix.data()[6] = inertia->FirstChildElement("ixy")->DoubleText();
			inertia_matrix.data()[4] = inertia->FirstChildElement("izz")->DoubleText();
			inertia_matrix.data()[7] = inertia->FirstChildElement("iyz")->DoubleText();
			inertia_matrix.data()[8] = inertia->FirstChildElement("iyy")->DoubleText();
#endif // !GAZEBO_DATA_VERSION
			const char* masspose = link->FirstChildElement("inertial")->FirstChildElement("pose")->GetText();
			std::istringstream issmass(masspose);
			std::istream_iterator<double> itmass(issmass), endmass;
			std::vector<double> massposeDouble(itmass, endmass);	// 6个数字, inertia的pose, 前3个是网格坐标系原点在




			LinkXML linkxml;
			linkxml.linkname = link->Attribute("name");
			linkxml.InertialTensor = inertia_matrix * massscale/**mass*/;

			linkxml.mass = mass * massscale;
			linkxml.link_parent = -1; // initial value

			// the vector must have 6 element
			assert(poseDouble.size() == 6);
#ifndef GAZEBO_DATA_VERSION
			linkxml.pose[0] = poseDouble[0] * meshscale;
			linkxml.pose[1] = poseDouble[1] * meshscale;
			linkxml.pose[2] = poseDouble[2] * meshscale;
			linkxml.pose[3] = poseDouble[3] * meshscale;
			linkxml.pose[4] = poseDouble[4] * meshscale;
			linkxml.pose[5] = poseDouble[5] * meshscale;
#else
			linkxml.pose[0] = poseDouble[0] * meshscale;
			linkxml.pose[1] = poseDouble[2] * meshscale;
			linkxml.pose[2] = -poseDouble[1] * meshscale;
			linkxml.pose[3] = poseDouble[3] * meshscale;
			linkxml.pose[4] = -poseDouble[5] * meshscale;
			linkxml.pose[5] = -poseDouble[4] * meshscale;
#endif // !GAZEBO_DATA_VERSION

#ifndef GAZEBO_DATA_VERSION
			linkxml.masspose[0] = massposeDouble[0] * meshscale;
			linkxml.masspose[1] = massposeDouble[1] * meshscale;
			linkxml.masspose[2] = massposeDouble[2] * meshscale;
			linkxml.masspose[3] = massposeDouble[3] * meshscale;
			linkxml.masspose[4] = massposeDouble[4] * meshscale;
			linkxml.masspose[5] = massposeDouble[5] * meshscale;
#else
			linkxml.masspose[0] = massposeDouble[0] * meshscale;
			linkxml.masspose[1] = massposeDouble[2] * meshscale;
			linkxml.masspose[2] = -massposeDouble[1] * meshscale;
			linkxml.masspose[3] = massposeDouble[3] * meshscale;
			linkxml.masspose[4] = -massposeDouble[5] * meshscale;
			linkxml.masspose[5] = -massposeDouble[4] * meshscale;
#endif // !GAZEBO_DATA_VERSION


			const char* box = NULL;

			if (link->FirstChildElement("visual")->FirstChildElement("geometry")->FirstChildElement("box"))
			{
				box = link->FirstChildElement("visual")->FirstChildElement("geometry")->FirstChildElement("box")->FirstChildElement("size")->GetText();

				std::istringstream issbox(box);
				std::istream_iterator<double> itbox(issbox), endbox;
				std::vector<double> boxDouble(itbox, endbox);

				// the vector must have 3 element
				assert(boxDouble.size() == 3);
#ifndef GAZEBO_DATA_VERSION
				linkxml.box[0] = boxDouble[0];
				linkxml.box[1] = boxDouble[1];
				linkxml.box[2] = boxDouble[2];
#else
				linkxml.box[0] = boxDouble[0];
				linkxml.box[1] = boxDouble[2];
				linkxml.box[2] = -boxDouble[1];
#endif // !GAZEBO_DATA_VERSION
				linkxml.useBoxRender = true;

			}
			else
			{
				linkxml.box[0] = 0.1;
				linkxml.box[1] = 0.1;
				linkxml.box[2] = 0.1;
				linkxml.useBoxRender = false;
			}
			if (link->FirstChildElement("visual")->FirstChildElement("geometry")->FirstChildElement("mesh"))
			{

				tinyxml2::XMLElement* mesh = link->FirstChildElement("visual")->FirstChildElement("geometry")->FirstChildElement("mesh");

				if (mesh)
				{
					linkxml.meshFilePath = mesh->FirstChildElement("uri")->GetText();
					linkxml.useTriMeshRender = true;
				}
				else
				{
					linkxml.useTriMeshRender = false;
				}

				const char* visualpose = link->FirstChildElement("visual")->FirstChildElement("pose")->GetText();

				std::istringstream iss(visualpose);
				std::istream_iterator<double> it(iss), end;
				std::vector<double> visualposeDouble(it, end);

				// the vector must have 6 element
				assert(visualposeDouble.size() == 6);
#ifndef GAZEBO_DATA_VERSION
				linkxml.visualpose[0] = visualposeDouble[0] * meshscale;
				linkxml.visualpose[1] = visualposeDouble[1] * meshscale;
				linkxml.visualpose[2] = visualposeDouble[2] * meshscale;
				linkxml.visualpose[3] = visualposeDouble[3];
				linkxml.visualpose[4] = visualposeDouble[4];
				linkxml.visualpose[5] = visualposeDouble[5];
#else
				linkxml.visualpose[0] = visualposeDouble[0] * meshscale;
				linkxml.visualpose[1] = visualposeDouble[1] * meshscale;
				linkxml.visualpose[2] = visualposeDouble[2] * meshscale;
				linkxml.visualpose[3] = visualposeDouble[3];
				linkxml.visualpose[4] = visualposeDouble[4];
				linkxml.visualpose[5] = visualposeDouble[5];
#endif // !GAZEBO_DATA_VERSION

			}
			links.push_back(linkxml);
		}
	}

	//search joint
	link = model->FirstChild()->ToElement();
	for (link; link; link = link->NextSiblingElement())
	{

		const char* elementName = link->Name();
		if (elementName&&strcmp(elementName, "joint") == 0)
		{
			JointXML joint;
			joint.type = 0;
			joint.axis.setZero();

			const char* parentname = link->FirstChildElement("parent")->GetText();
			const char* childname = link->FirstChildElement("child")->GetText();
			const char* pose = link->FirstChildElement("pose")->GetText();
			std::istringstream iss(pose);
			std::istream_iterator<double> it(iss), end;
			std::vector<double> poseDouble(it, end);
			const char* axis = NULL;
			if (link->FirstChildElement("axis"))
				axis = link->FirstChildElement("axis")->FirstChildElement("xyz")->GetText();
			if (link->FirstChildElement("axis")->FirstChildElement("limit"))
			{
				joint.lower = std::atoi(link->FirstChildElement("axis")->FirstChildElement("limit")->FirstChildElement("lower")->GetText());
				joint.upper = std::atoi((link->FirstChildElement("axis")->FirstChildElement("limit")->FirstChildElement("upper")->GetText()));
			}
			if (axis)
			{
				std::istringstream axisiss(axis);
				std::istream_iterator<double> axisit(axisiss), axisend;
				std::vector<double> axisDouble(axisit, axisend);
#ifndef GAZEBO_DATA_VERSION
				joint.axis.data()[0] = axisDouble[0];
				joint.axis.data()[1] = axisDouble[1];
				joint.axis.data()[2] = axisDouble[2];
#else
				joint.axis.data()[0] = axisDouble[0];
				joint.axis.data()[1] = axisDouble[2];
				joint.axis.data()[2] = -axisDouble[1];
#endif // !GAZEBO_DATA_VERSION

			}

			joint.name = link->FirstAttribute()->Value();
			const char* jointtype = link->Attribute("type");

			if (strcmp(jointtype, "revolute") == 0)
			{
				joint.type = 2;
			}
			else if (strcmp(jointtype, "fixed") == 0)
			{
				joint.type = 3;
			}
			else if (strcmp(jointtype, "ball") == 0)
			{
				joint.type = 1;
			}

			int parentid = -1;
			int childid = -1;

			for (int i = 0; i < links.size(); i++)
			{
				if (strcmp(parentname, links[i].linkname) == 0)
				{
					parentid = i;
					//link_child[i].push_back()
				}

				if (strcmp(childname, links[i].linkname) == 0)
				{
					childid = i;
				}
			}

#ifndef GAZEBO_DATA_VERSION
			joint.pose[0] = poseDouble[0] * meshscale;
			joint.pose[1] = poseDouble[1] * meshscale;
			joint.pose[2] = poseDouble[2] * meshscale;
			joint.pose[3] = poseDouble[3] * meshscale;
			joint.pose[4] = poseDouble[4] * meshscale;
			joint.pose[5] = poseDouble[5] * meshscale;
#else
			joint.pose[0] = poseDouble[0] * meshscale;
			joint.pose[1] = poseDouble[2] * meshscale;
			joint.pose[2] = -poseDouble[1] * meshscale;
			joint.pose[3] = poseDouble[3] * meshscale;
			joint.pose[4] = -poseDouble[5] * meshscale;
			joint.pose[5] = -poseDouble[4] * meshscale;
#endif // !GAZEBO_DATA_VERSION


			joint.childid = childid;
			joint.parentid = parentid;

			links[parentid].link_child.push_back(childid);
			links[childid].link_parent = parentid;
			links[parentid].jointid = joints.size(); // id + 1 because we will add root later.

			joints.push_back(joint);
		}
	}

	// 以上是读取joint_list
	for (int i = 0; i < links.size(); i++)
	{
		//create link 
		if (links[i].useTriMeshRender&&links[i].useBoxRender)
		{
			addLink(links[i].InertialTensor, links[i].mass, links[i].meshFilePath, Vector3d(links[i].box[0], links[i].box[1], links[i].box[2]));
		}
		//addLink(links[i].InertialTensor, links[i].mass, Vector3d(links[i].box[0], links[i].box[1], links[i].box[2]));
		else if (links[i].useTriMeshRender)
		{
			addLink(links[i].InertialTensor, links[i].mass, links[i].meshFilePath);
		}
		else if (links[i].useBoxRender)
		{
			addLink(links[i].InertialTensor, links[i].mass, Vector3d(links[i].box[0], links[i].box[1], links[i].box[2]));
		}
		else
		{
			std::cout << "link " << i << " mesh info miss\n";
		}
		links_list[i]->setName(links[i].linkname);

		//std::ofstream fout("1.txt", std::ios::app);

		//fout << "\n************link name: " << links[i].linkname << "************" << std::endl;
		if (links[i].useTriMeshRender)
		{
			Matrix3d visualOrientation;
			//fout << "visualPose:";
			std::cout << "[log] link " << i << " visual pose: ";
			for (int j = 0; j < 6; j++)	std::cout << links[i].visualpose[j] << " ";
			std::cout << std::endl;

			
			visualOrientation = xconventionRotation(links[i].visualpose[3])*yconventionRotation(links[i].visualpose[4])*zconventionRotation(links[i].visualpose[5]);
			// 绕 x->y->z 轴的顺序旋转
			//visualOrientation = zconventionRotation(links[i].visualpose[5]) * yconventionRotation(links[i].visualpose[4]) *xconventionRotation(links[i].visualpose[3]);
			//fout << "visualOrientation:\n" << visualOrientation << std::endl;

			//fout << "xyztransform:\n" << xyzXZYtransform << std::endl;
			//xyzXZYtransform << 1, 0, 0, 0, 0, -1, 0, 1, 0;
			visualOrientation = xyzXZYtransform * visualOrientation;
			//fout << "xyzXZYtransform * visualOrientation:\n" << visualOrientation << std::endl;

			Vector3d visualTranslation;
			visualTranslation.data()[0] = links[i].visualpose[0];
			visualTranslation.data()[1] = links[i].visualpose[1];
			visualTranslation.data()[2] = links[i].visualpose[2];

			visualTranslation = xyzXZYtransform * visualTranslation;
			//fout << "visualTranslation:\n" << visualOrientation << std::endl;
			// 这里设置link的visual pos, 是网格在局部坐标系下的拜访
			links_list[links_list.size() - 1]->setVisual_orientation(visualOrientation);
			links_list[links_list.size() - 1]->setVisual_position(visualTranslation);


		}

		//fout.close();
		if (links[i].link_parent == -1)
		{
			root_link = i;
		}
	}

	//add the root joint
	Vector3d position;
	Matrix3d orientation;
	orientation.setIdentity();
	position.data()[0] = links[root_link].pose[0] ;	// link 质心在全局的坐标，也是link局部坐标系原点在全局坐标
	position.data()[1] = links[root_link].pose[1] ;
	position.data()[2] = links[root_link].pose[2] ;
	Vector3d position_w = position;	// position world
	links[root_link].pose[0] = 0;	// link质心的世界坐标变成0
	links[root_link].pose[1] = 0;
	links[root_link].pose[2] = 0;

	// 把link的质心位置拿到
	Vector3d link_visual_position(-links[root_link].masspose[0], -links[root_link].masspose[1], -links[root_link].masspose[2]);

	addUniversalJoint(-1, position, position_w, orientation, -link_visual_position, root_link);
	joints_list[0]->name = links[root_link].linkname;
	links_list[root_link]->setVisual_position(link_visual_position);
	//addHingeJoint(-1, position, orientation, Vector3d(0, 0, 0), root_link, 1);
	links[root_link].jointid = 0;

	for (int i = 0; i < joints.size(); i++)
	{
		int linkid = joints[i].childid;
		int parentLink = joints[i].parentid;

		Vector3d position_w_;
		Vector3d position_;
		Vector3d joint_pose_;
		joint_pose_.data()[0] = joints[i].pose[0] /*+ links[linkid].masspose[0]*/;
		joint_pose_.data()[1] = joints[i].pose[1] /*+ links[linkid].masspose[1]*/;
		joint_pose_.data()[2] = joints[i].pose[2] /*+ links[linkid].masspose[2]*/;


		Vector3d mass_position_;
		Matrix3d orientation_w_;
		orientation_w_.setIdentity();

		// from zyx to xyz
		orientation_w_ = (xconventionRotation(links[linkid].pose[3]) * yconventionRotation(links[linkid].pose[4]) * 
zconventionRotation(links[linkid].pose[5])).transpose();

		mass_position_ = -joint_pose_;//mass center in local link frame

									  //这里给的position_在初始设想中应该是在上一个joint的局部坐标中的位置
									  //这里给的position其实还是相对link的，实际上是不对的
									  //但是这里没有joint的链接关系，所以也没法计算这个，在下面会重新计算reset
		position_ = orientation_w_ * joint_pose_;
		position_w_ = position_;
		//joint position at rest pose in world frame
		position_w_.data()[0] += links[linkid].pose[0];
		position_w_.data()[1] += links[linkid].pose[1];
		position_w_.data()[2] += links[linkid].pose[2];

		std::cout << "[log] joint " << links[linkid].linkname << " : " << position_w_.transpose() << std::endl;

		//在构建joint的时候，xml文件中的joint position后三个代表joint朝向的数据实际上没有使用到
		//joint的orientation和link的是一致的
		//实际上应该是给出joint相对parent joint的orientation，但是这一块先放着，暂时都是identity矩阵
		LoboJointV2 * joint = NULL;

		if (joints[i].type == 2)
		{
			Matrix3d hingeOrientation;
			hingeOrientation.setZero();
			Vector3d axis;
			axis.data()[0] = joints[i].axis.data()[0];
			axis.data()[1] = joints[i].axis.data()[1];
			axis.data()[2] = joints[i].axis.data()[2];
			if (abs(axis.norm()) < 1e-8)
			{
				std::cout << "warning: joint rotation axis = Vector3d(0,0,0)" << std::endl;
			}
			axis.normalize();
			Vector3d ytoAxis = Vector3d::UnitY().cross(axis);

			double ydotaxis = Vector3d::UnitY().dot(axis);
			double cosangle = ydotaxis / (axis.norm());
			double angle = std::acos(cosangle);


			if (fabs(ytoAxis.norm()) < 1e-8)
			{
				ytoAxis = Vector3d::UnitZ();
				//hingeOrientation.setIdentity();
			}
			ytoAxis.normalize();
			hingeOrientation = (AngleAxisd(angle, ytoAxis)).toRotationMatrix();
			//hingeOrientation.setIdentity();
			mass_position_ = hingeOrientation.transpose()*mass_position_;
			Matrix3d link_visual_orientation = hingeOrientation.transpose()*orientation_w_*links_list[linkid]->getVisual_orientation();
			/*Vector3d link_visual_position = links_list[linkid]->getVisual_position();
			link_visual_position[0] -= links[linkid].masspose[0];
			link_visual_position[1] -= links[linkid].masspose[1];
			link_visual_position[2] -= links[linkid].masspose[2];*/
			links_list[linkid]->setVisual_orientation(link_visual_orientation);
			//links_list[linkid]->setVisual_position(link_visual_position);
			//hingeOrientation = hingeOrientation.transpose().eval();


			//std::cout << (hingeOrientation* Vector3d::UnitY()).transpose() << std::endl;
			orientation_w_ = orientation_w_ * hingeOrientation;

			joint = new HingeJoint(links_list[linkid], position_, position_w_, orientation_w_, mass_position_, joints_list.size(), 1);

		}
		else if (joints[i].type == 3)
		{
			joint = new FixedJoint(links_list[linkid], position_, position_w_, orientation_w_, mass_position_, joints_list.size());
		}
		else if (joints[i].type == 1)
		{
			joint = new BallJoint(links_list[linkid], position_, position_w_, orientation_w_, mass_position_, joints_list.size());
		}
		joint->name = joints[i].name;
		joint->LimitLower = joints[i].lower;
		joint->LimitUpper = joints[i].upper;
		joints_list.push_back(joint);
		links[linkid].jointid = joint->getJoint_id();
	}

	//add the joints graph info 
	for (int i = 0; i < joints.size(); i++)
	{
		int parentJointid;
		int linkid = joints[i].childid;
		int linkparent = links[linkid].link_parent;

		parentJointid = links[linkparent].jointid;
		int jointid = links[linkid].jointid;

		joints_list[jointid]->setJoint_parent(joints_list[parentJointid]);
		joints_list[parentJointid]->addChild(joints_list[jointid]);
	}

	buildJointTreeIndex();

	//set joint ori_orientation_w
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		LoboJointV2* joint = joints_list[jointid];
		if (joint->getJoint_parent() == nullptr)
		{
			joint->ori_orientation = joint->ori_orientation_w;
			joint->ori_orientation_4d.topLeftCorner<3, 3>() = joint->ori_orientation;
			joint->ori_orientation_4d(0, 3) = joint->ori_position_w(0);
			joint->ori_orientation_4d(1, 3) = joint->ori_position_w(1);
			joint->ori_orientation_4d(2, 3) = joint->ori_position_w(2);
			joint->ori_orientation_4d(3, 3) = 1.0;
		}
		else
		{
			joint->ori_orientation = joint->getJoint_parent()->ori_orientation_w.transpose()*joint->ori_orientation_w;
			joint->ori_orientation_4d.topLeftCorner<3, 3>() = joint->ori_orientation;
		}
		//joint->position_w = joint->position;
	}

	//set joint ori_position
	for (int i = 1; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		LoboJointV2* joint = joints_list[jointid];
		LoboJointV2* parentjoint = joint->getJoint_parent();

		Vector3d position_in_parent = parentjoint->ori_orientation_w.transpose()*(joint->ori_position_w - parentjoint->ori_position_w);
		//Matrix3d orientation_in_parent = parentjoint->orientation_w*joint->orientation;

		joint->resetPose(position_in_parent);
		joint->ori_orientation_4d(0, 3) = position_in_parent(0);
		joint->ori_orientation_4d(1, 3) = position_in_parent(1);
		joint->ori_orientation_4d(2, 3) = position_in_parent(2);
		joint->ori_orientation_4d(3, 3) = 1.0;
	}
	initGeneralizedInfo();
	printMultiBodyInfo();
	for (size_t i = 0; i < joints_list.size(); i++)
	{
		joints_list[i]->getConnectedLink()->setConnectJoint(joints_list[i]);
	}
	std::cout << "[log] LOAD RIGIDBODY MODEL END" << std::endl;
}

void MultiRigidBodyModel::convertRestpos2Worldpos(Vector3d & pos_w, const  Vector3d & pos_r, int jointid)
{
	LoboJointV2* joint = this->getJoint(jointid);
	Vector3d t = joint->position_w;
	Matrix3d R = joint->orientation_w;

	pos_w = R * pos_r + t;
	/*std::cout << "t: " << t.transpose() << std::endl;
	std::cout << "R: " << R<< std::endl;
	std::cout << "pos_w : " << pos_w.transpose() << std::endl;*/

}

void MultiRigidBodyModel::addLink(Matrix3d Inertia_tensor, double mass)
{

	Vector3d pos = Vector3d(0, 0, 0);
	Matrix3d rotation = Matrix3d::Identity();
	LoboLink* link = new LoboLink(pos, rotation, links_list.size());
	link->setMass(mass);
	link->setInertiaTensor(Inertia_tensor);
	links_list.push_back(link);
}

void MultiRigidBodyModel:: addLink(Matrix3d Inertia_tensor, double mass,const std::string link_name)
{
	addLink(Inertia_tensor, mass);
	links_list[links_list.size() - 1]->setName(link_name);
}

void MultiRigidBodyModel::addLink(Matrix3d Inertia_tensor, double mass, Vector3d box)
{

	Vector3d pos = Vector3d(0, 0, 0);
	Matrix3d rotation = Matrix3d::Identity();
	LoboLink* link = new LoboLink(pos, rotation, links_list.size());
	link->setVisualbox(box);
	link->setMass(mass);
	link->setInertiaTensor(Inertia_tensor);
	links_list.push_back(link);
}

void MultiRigidBodyModel::addLink(Matrix3d Inertia_tensor, double mass, const char* meshfilepath)
{

	Vector3d pos = Vector3d(0, 0, 0);
	Matrix3d rotation = Matrix3d::Identity();
	LoboLink* link = new LoboLink(pos, rotation, links_list.size());
	link->setMass(mass);
	link->setInertiaTensor(Inertia_tensor);
	link->initLinkRender(meshfilepath);
	link->setUseMeshRender(true);
	links_list.push_back(link);
}

void MultiRigidBodyModel::addLink(Matrix3d Inertia_tensor, double mass, const char * meshfilepath, Vector3d box)
{

	Vector3d pos = Vector3d(0, 0, 0);
	Matrix3d rotation = Matrix3d::Identity();
	LoboLink* link = new LoboLink(pos, rotation, links_list.size());
	link->setVisualbox(box);
	link->setMass(mass);
	link->setInertiaTensor(Inertia_tensor);
	link->initLinkRender(meshfilepath);
	link->setUseMeshRender(true);
	links_list.push_back(link);
}

void MultiRigidBodyModel::addJoint(LoboJointV2* parent, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, LoboLink* link)
{
	LoboJointV2* joint = new BallJoint(link, position, position_w, orientation, mass_position_, joints_list.size());
	if (parent == NULL)
	{
		root_joint = joint;
	}
	else
	{
		joint->setJoint_parent(parent);
		parent->addChild(joint);
	}

	joints_list.push_back(joint);
}

void MultiRigidBodyModel::addJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid)
{
	if (jointid == -1)
	{
		addJoint(NULL, position, position_w, orientation, mass_position_, links_list[linkid]);
	}
	else
		addJoint(joints_list[jointid], position, position_w, orientation, mass_position_, links_list[linkid]);
}

void MultiRigidBodyModel::addHingeJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid, int hingeType)
{
	LoboJointV2 * joint = new HingeJoint(links_list[linkid], position, position_w, orientation, mass_position_, joints_list.size(), hingeType);

	if (jointid == -1)
	{
		root_joint = joint;
	}
	else
	{
		joint->setJoint_parent(joints_list[jointid]);
		joints_list[jointid]->addChild(joint);
	}

	joints_list.push_back(joint);
}

void MultiRigidBodyModel::addUniversalJoint(int jointid, Vector3d position, Vector3d position_w, Matrix3d orientation, Vector3d mass_position_, int linkid)
{
	LoboJointV2 * joint = new UniversalJoint(links_list[linkid], position, position_w, orientation, mass_position_, joints_list.size());

	if (jointid == -1)
	{
		root_joint = joint;
	}
	else
	{
		joint->setJoint_parent(joints_list[jointid]);
		joints_list[jointid]->addChild(joint);
	}

	joints_list.push_back(joint);
}

void MultiRigidBodyModel::setModelState(VectorXd &q, VectorXd &q_dot, bool updateDynamic)
{
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		joints_list[jointid]->updateJointState(q, q_dot, updateDynamic);
	}
}

Vector3d MultiRigidBodyModel::getNodeVelocity(int jointid, const Vector3d & nodepos_w, const VectorXd & qvel)
{
	MatrixXd Jci;
	Vector3d a = nodepos_w;
	getJacobiVByGivenPosition(Jci, a, jointid);
	return Jci * qvel;

}

void MultiRigidBodyModel::getMatrix(VectorXd&q, VectorXd &q_dot, MatrixXd &massMatrix, MatrixXd &C, VectorXd &Cq)
{
	this->setIfComputeSecondDerive(true);
	setModelState(q, q_dot, true);
	massMatrix.resize(numofDOFs, numofDOFs);
	massMatrix.setZero();
	C.resize(numofDOFs, numofDOFs);
	C.setZero();

	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int k = joint_tree_index[i];
		LoboLink* link_k = joints_list[k]->getConnectedLink();

		//compute mass Matrix
		MatrixXd massMatrix_cartesian;
		link_k->computeMassMatrix(massMatrix_cartesian);

		massMatrix += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK;
		//std::cout << "JK:\n" << joints_list[k]->JK << "\n";
		//compute 
		MatrixXd w_skew(6, 6);
		w_skew.setZero();
		Vector3d wi = joints_list[k]->angular_velocity;

		Matrix3d wi_skew;
		skewMatrix(wi, wi_skew);
		w_skew.block(3, 3, 3, 3) = wi_skew;
		C += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK_dot
			+ joints_list[k]->JK.transpose()*w_skew*massMatrix_cartesian*joints_list[k]->JK;
	}

	Cq = C * q_dot;
}

void MultiRigidBodyModel::getMatrix(VectorXd&q, VectorXd &q_dot, VectorXd &massQ_,
	VectorXd &cQ_, MatrixXd &dMassdq_q, MatrixXd& dCdq_q, MatrixXd &massMatrix, MatrixXd &C, double dqdotdq)
{
	/*QElapsedTimer timer;
	timer.start();*/

	setModelState(q, q_dot, true);

	//std::cout << "setModelState " << timer.nsecsElapsed() / 1e6 << std::endl;

	//massMatrix.resize(numofDOFs, numofDOFs);
	massMatrix.setZero();
	//C.resize(numofDOFs, numofDOFs);
	C.setZero();
	dMassdq_q.setZero();
	dCdq_q.setZero();
	//dMassdq_q.resize(numofDOFs, numofDOFs);

	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int k = joint_tree_index[i];
		LoboLink* link_k = joints_list[k]->getConnectedLink();

		//compute mass Matrix
		MatrixXd massMatrix_cartesian(6, 6);
		//std::cout << "before computeMassMatrix " << timer.nsecsElapsed() / 1e6 << std::endl;

		link_k->computeMassMatrix(massMatrix_cartesian);
		//std::cout << "computeMassMatrix " << timer.nsecsElapsed() / 1e6 << std::endl;


		massMatrix += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK;
		/*double jointmass = link_k->getMass();
		Matrix3d inertia = link_k->getInertiaTensor();
		massMatrix += jointmass*joints_list[k]->JK_v.transpose()*joints_list[k]->JK_v;*/


		//compute 
		MatrixXd w_skew(6, 6);
		w_skew.setZero();
		Vector3d wi = joints_list[k]->angular_velocity;

		Matrix3d wi_skew;
		skewMatrix(wi, wi_skew);
		w_skew.block(3, 3, 3, 3) = wi_skew;
		C += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK_dot
			+ joints_list[k]->JK.transpose()*w_skew*massMatrix_cartesian*joints_list[k]->JK;
		//std::cout << "C " << timer.nsecsElapsed() / 1e6 << std::endl;


		joints_list[k]->computedMassdQ(link_k->getMass(), link_k->getInertiaTensor(), dMassdq_q, massQ_);
		//std::cout << "computedMassdQ " << timer.nsecsElapsed() / 1e6 << std::endl;

		joints_list[k]->computedCoriolisdQ(link_k->getMass(), link_k->getInertiaTensor(), dCdq_q, cQ_, dqdotdq);
		//std::cout << "computedCoriolisdQ " << timer.nsecsElapsed() / 1e6 << std::endl;

	}

}

void MultiRigidBodyModel::getMatrix(VectorXd & q, VectorXd & qvel, VectorXd&qaccel, MatrixXd & dMdq_qaccel, MatrixXd & dCdq_qvel, MatrixXd & M, MatrixXd & C)
{
	setModelState(q, qvel, true);
	M.setZero();
	//C.resize(numofDOFs, numofDOFs);
	C.setZero();
	dMdq_qaccel.setZero();
	dCdq_qvel.setZero();
	//dMassdq_q.resize(numofDOFs, numofDOFs);

	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int k = joint_tree_index[i];
		LoboLink* link_k = joints_list[k]->getConnectedLink();

		//compute mass Matrix
		MatrixXd massMatrix_cartesian(6, 6);
		//std::cout << "before computeMassMatrix " << timer.nsecsElapsed() / 1e6 << std::endl;

		link_k->computeMassMatrix(massMatrix_cartesian);

		M += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK;


		//compute 
		MatrixXd w_skew(6, 6);
		w_skew.setZero();
		Vector3d wi = joints_list[k]->angular_velocity;

		Matrix3d wi_skew;
		skewMatrix(wi, wi_skew);
		w_skew.block(3, 3, 3, 3) = wi_skew;
		C += joints_list[k]->JK.transpose()*massMatrix_cartesian*joints_list[k]->JK_dot
			+ joints_list[k]->JK.transpose()*w_skew*massMatrix_cartesian*joints_list[k]->JK;
		//std::cout << "C " << timer.nsecsElapsed() / 1e6 << std::endl;


		joints_list[k]->computedMassdQ(link_k->getMass(), link_k->getInertiaTensor(), dMdq_qaccel, qaccel);
		//std::cout << "computedMassdQ " << timer.nsecsElapsed() / 1e6 << std::endl;

		joints_list[k]->computedCoriolisdQ(link_k->getMass(), link_k->getInertiaTensor(), dCdq_qvel, qvel);
		//std::cout << "computedCoriolisdQ " << timer.nsecsElapsed() / 1e6 << std::endl;

	}
}

void MultiRigidBodyModel::getJacobiVByGivenPosition(MatrixXd &jacobi_q, Vector3d x_position, int jointid)
{
	joints_list[jointid]->updateJacobiByGivenPosition(x_position, jacobi_q);
}

void MultiRigidBodyModel::getJacobiVByGivenRestPosition(MatrixXd & jacobi_q, Vector3d x_position_rest, int jointid)
{
	joints_list[jointid]->updateJacobiByGivenRestPosition(x_position_rest, jacobi_q);
}

void MultiRigidBodyModel::convertExternalForce(VectorXd f_cartesian, VectorXd &Q, bool ifgravity)
{
	assert(numofDOFs == Q.rows());//首先对Q进行重置
	Q.setZero();
	// 把一个6*铰链个数维的笛卡尔力，转化成广义力Q(numDOFs*1)。
	// 这涉及到力的转换，我还不理解。
	for (int i = 0; i < joints_list.size(); i++)
	{
		LoboLink* link_k = joints_list[i]->getConnectedLink();// 拿到after this joint的link指针
		Vector3d force, torque;

		force.data()[0] = f_cartesian.data()[i * 6 + 0];
		force.data()[1] = f_cartesian.data()[i * 6 + 1];
		force.data()[2] = f_cartesian.data()[i * 6 + 2];

		torque.data()[0] = f_cartesian.data()[i * 6 + 3];
		torque.data()[1] = f_cartesian.data()[i * 6 + 4];
		torque.data()[2] = f_cartesian.data()[i * 6 + 5];


		//std::cout << f_cartesian << std::endl;

		if (ifgravity)
		{
			force += gravity_force * link_k->getMass();
		}
		/*std::cout <<"------Jv-----" << std::endl;
		std::cout << joints_list[i]->JK_v << std::endl;
		std::cout << "------Jw-----" << std::endl;
		std::cout << joints_list[i]->JK_w << std::endl;*/
		Q += joints_list[i]->JK_v.transpose()*force		// Jk_v是3*numofDOFs的，Jk_w也是3*numofDOFs的。
			+ joints_list[i]->JK_w.transpose()*torque;	// force和torque都是3*1的
														//std::cout << Q << std::endl;					// 做了乘法之后是numofDOFs * 1的
	}

	//damping 
}

void MultiRigidBodyModel::clampRotation(VectorXd& _q, VectorXd& _qdot)
{
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		joints_list[jointid]->clampRotation(_q, _qdot);
	}
}

//void MultiRigidBodyModel::UpdateRigidMesh()
//{
//	//cout <<"[error] Xudong 20190315: this function should be considered carefuuly"<<endl;
//	//cout <<"because at first this function is commented, but in the lastest version, they works"<<endl;
//	//cout <<"I don't know why, and I also don't know what will happen now"<<endl;
//	int numjoint = getnNumJoints();
//
//
//	for (size_t i = 0; i < numjoint; i++)
//	{
//
//		LoboLink* link = this->getJoint(i)->getConnectedLink();
//		LoboTriMesh* mesh = this->getJoint(i)->getConnectedLink()->getLinkMesh()->getLoboTriMesh();
//		Vector3d link_position = link->getPosition_world();
//		Matrix3d rotation = link->getOrientation_world();
//		Matrix3d visualOrientation = link->getVisual_orientation();
//		Vector3d visualPosition = link->getVisual_position();
//
//		Vector3d t = rotation * visualPosition + link_position;
//		Matrix3d R = rotation * visualOrientation;
//
//		mesh->transformMesh(t, R);
//	}
//}

void MultiRigidBodyModel::setIfComputeThirdDerive(bool b)
{
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		joints_list[jointid]->setIfComputeThirdDerive(b);
	}
}

void MultiRigidBodyModel::setIfComputeSecondDerive(bool b)
{
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		int jointid = joint_tree_index[i];
		joints_list[jointid]->setIfComputeSecondDerive(b);
	}
}

int MultiRigidBodyModel::getnNumJoints()
{
	return joints_list.size();
}

LoboJointV2* MultiRigidBodyModel::getJoint(int id)
{
	return joints_list[id];
}

LoboJointV2 * MultiRigidBodyModel::getJoint(std::string jointname)
{
	LoboJointV2* joint = NULL;
	for (size_t i = 0; i < joints_list.size(); i++)
	{
		if (jointname == joints_list[i]->name)
		{
			joint = joints_list[i];
			break;
		}
	}
	return joint;
}

LoboLink* MultiRigidBodyModel::getLink(int id)
{
	return links_list[id];
}

LoboLink * MultiRigidBodyModel::getLink(std::string linkname)
{
	LoboLink*link = nullptr;
	for (size_t i = 0; i < links_list.size(); i++)
	{
		if (linkname == links_list[i]->getName())
		{
			link = links_list[i];
			break;
		}
	}
	return link;
}

void MultiRigidBodyModel::printMultiBodyInfo()
{
	std::cout << "******************print multi rigid body model******************" << std::endl;
	std::cout << "num joints" << joints_list.size() << std::endl;
	std::cout << "print tree: " << std::endl;
	for (int i = 0; i < joint_tree_index.size(); i++)
	{
		std::cout << joint_tree_index[i] << " ";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < joints_list.size(); i++)
	{
		std::cout << "joint name : " << joints_list[i]->name << " r = " << joints_list[i]->getR() << "offset = " << joints_list[i]->getGeneralized_offset() << " upper limit: " << joints_list[i]->LimitUpper << " lower limit: " << joints_list[i]->LimitLower << std::endl;
	}
	std::cout << std::endl;
	std::cout << "print end." << std::endl;

	std::cout << "Total DOFs: " << numofDOFs << std::endl;
	
	double mass = 0;
	for (size_t i = 0; i < joints_list.size(); i++)
	{
		mass += joints_list[i]->getConnectedLink()->getMass();
	}
	std::cout << "Total Mass: " << mass<< std::endl;
	std::cout << std::endl;
	std::cout << "************************************" << std::endl;
}

void MultiRigidBodyModel::deleteModel()
{
	for (int i = 0; i < links_list.size(); i++) delete links_list[i];
	for (int i = 0; i < joints_list.size(); i++) delete joints_list[i];

}

void MultiRigidBodyModel::buildJointTreeIndex()
{
	joint_tree_index.clear();
	std::queue<LoboJointV2*> queue_;
	if (root_joint == nullptr)
		return;
	queue_.push(root_joint);

	LoboJointV2* joint_pointer;

	while (queue_.size() != 0)
	{
		joint_pointer = queue_.front();
		queue_.pop();

		int jointid = joint_pointer->getJoint_id();
		joint_tree_index.push_back(jointid);

		int numChild = joint_pointer->getNumChild();

		for (int i = 0; i < numChild; i++)
		{
			queue_.push(joint_pointer->getChild(i));
		}
	}
}

