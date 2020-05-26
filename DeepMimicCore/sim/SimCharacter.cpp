#include "SimCharacter.h"
#include <iostream>

#include "SimBox.h"
#include "SimCapsule.h"
#include "SimSphere.h"
#include "RBDUtil.h"
// #include "RigidSystem/MultiRigidBodyModel.h"

#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "util/BulletUtil.h"

cSimCharacter::tParams::tParams()
{
	mID = gInvalidIdx;
	mCharFile = "";
	mStateFile = "";
	mInitPos = tVector(0, 0, 0, 0);
	mLoadDrawShapes = true;

	mEnableContactFall = true;
}


cSimCharacter::cSimCharacter()
{
	mFriction = 0.9;
	mEnableContactFall = true;
	mEnableJointTorqueControl = true;
}

cSimCharacter::~cSimCharacter()
{
	RemoveFromWorld();
}

/*
	@Function: cSimCharacter::Init

	Init the whole cSimCharacter.
	It can be divided into 3 parts widely.
	1. cCharcter::Init(): Load Joints Skleton by calling cCharcter::Init(). Read Json["Skeletons"]["Joints"] value and parse it.
		set up the final joint info matrix, which is so called "mJointMat"

	2. cCharcter::Init(): Load Draw Shapes, in the second half part of cCharacter::Init(). Prepared well to do rendering

	3. BuildSimBody(): first load simulation info Json["BodyDefs"], then set up bullet simulation body etc
		including: 
			succ &= BuildMultiBody(mMultiBody);		// bullet multibody class
			succ &= BuildConstraints(mMultiBody);	// constraints connect between links
			succ &= BuildBodyLinks();				// build class SimBodyLink
			succ &= BuildJoints();					// build class SimBodyJoint
*/
bool cSimCharacter::Init(const std::shared_ptr<cWorld>& world, const tParams& params)
{
	bool succ = true;
	// 在sim char后面还有一个Character
	bool succ_skeleton = cCharacter::Init(params.mCharFile, params.mLoadDrawShapes);
	succ &= succ_skeleton;
	// std::cout <<"2 pose = " << mPose.transpose() << std::endl;

	SetID(params.mID);
	RemoveFromWorld();
	mWorld = world;
	mEnableContactFall = params.mEnableContactFall;
	
	if (succ_skeleton)
	{
		// build simbody, load bodydefs
		succ &= BuildSimBody(params);
	}

	if (params.mStateFile != "")
	{
		bool succ_state = ReadState(params.mStateFile);

		if (!succ_state)
		{
			printf("Failed to load character state from %s\n", params.mStateFile.c_str());
		}
	}

	if (succ)
	{
		mPose0 = mPose;
		mVel0 = mVel;

		SetPose(mPose);
		SetVel(mVel);
	}

	return succ;
}

void cSimCharacter::Clear()
{
	cCharacter::Clear();

	RemoveFromWorld();
	mBodyDefs.resize(0, 0);

	if (HasController())
	{
		mController->Clear();
	}
}


void cSimCharacter::Reset()
{
	cCharacter::Reset();
	if (HasController())
	{
		mController->Reset();
	}
	
	ClearJointTorques();
}

static int gCurFrame = 0;
void cSimCharacter::Update(double timestep)
{
	ClearJointTorques();

	if (HasController())
	{
		// 在这里对controller进行更新
		// std::cout <<"void cSimCharacter::Update(double timestep)" << std::endl;
		mController->Update(timestep);
	}

	// dont clear torques until next frame since they can be useful for visualization
	UpdateJoints();	

}

void cSimCharacter::PostUpdate(double time_step)
{
	//std::cout << "[warning] void cSimCharacter::PostUpdate(double time_step) update link pos " << std::endl;
	UpdateLinkPos();
	UpdateLinkVel();
	BuildPose(mPose);
	BuildVel(mVel);

	if (HasController())
	{
		mController->PostUpdate(time_step);
	}
}

tVector cSimCharacter::GetRootPos() const
{
	// 获取root position
	int root_id = GetRootID();
	const cSimBodyJoint& root = GetJoint(root_id);
	tVector pos = root.CalcWorldPos();	// root joint在世界坐标系下的位置
	return pos;
}

void cSimCharacter::GetRootRotation(tVector& out_axis, double& out_theta) const
{
	tQuaternion rot = GetRootRotation();
	cMathUtil::QuaternionToAxisAngle(rot, out_axis, out_theta);
}

tQuaternion cSimCharacter::GetRootRotation() const
{
	int root_id = GetRootID();
	const cSimBodyJoint& root = GetJoint(root_id);
	tQuaternion rot = root.CalcWorldRotation();
	// std::cout <<"error root rot.calcworldrot before = " << rot.coeffs().transpose() << std::endl;
	// std::cout <<"error mInvRootAttachRot = " << mInvRootAttachRot.coeffs().transpose() << std::endl;
	rot = mInvRootAttachRot * rot;
	if(mInvRootAttachRot.coeffs().segment(0,3).norm() > 1e-5)
	{
		std::cout <<"[error] cSimCharacter::GetRootRotation doesn't support non-zero rest pose " << mInvRootAttachRot.coeffs().transpose() << std::endl;
		exit(1);
	}
	// std::cout <<"error root rot.calcworldrot after = " << rot.coeffs().transpose() << std::endl;
	return rot;
}

tVector cSimCharacter::GetRootVel() const
{
	int root_id = GetRootID();
	const cSimBodyJoint& root = GetJoint(root_id);
	tVector vel = root.CalcWorldVel();
	return vel;
}

tVector cSimCharacter::GetRootAngVel() const
{
	int root_id = GetRootID();
	const cSimBodyJoint& root = GetJoint(root_id);
	tVector ang_vel = root.CalcWorldAngVel();
	return ang_vel;
}

const Eigen::MatrixXd& cSimCharacter::GetBodyDefs() const
{
	return mBodyDefs;
}

void cSimCharacter::SetRootPos(const tVector& pos)
{
	cCharacter::SetRootPos(pos);
	SetPose(mPose);
}

void cSimCharacter::SetRootRotation(const tVector& axis, double theta)
{
	tQuaternion q = cMathUtil::AxisAngleToQuaternion(axis, theta);
	SetRootTransform(GetRootPos(), q);
}

void cSimCharacter::SetRootRotation(const tQuaternion& q)
{
	SetRootTransform(GetRootPos(), q);
}

void cSimCharacter::SetRootTransform(const tVector& pos, const tQuaternion& rot)
{
	// std::cout <<"set root transform begin\n";
	// std::cout <<"pos = " << pos.transpose()<<" rot = " << rot.coeffs().transpose() << std::endl;
	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, mPose);// pose中的rotation
	// std::cout << "root rot = " << root_rot.coeffs().transpose() << std::endl;
	tVector root_vel = cKinTree::GetRootVel(mJointMat, mVel);
	tVector root_ang_vel = cKinTree::GetRootAngVel(mJointMat, mVel);
	tQuaternion delta_rot = rot * root_rot.inverse();
	// std::cout << "delta rot = " << delta_rot.coeffs().transpose() << std::endl;

	root_vel = cMathUtil::QuatRotVec(delta_rot, root_vel);
	root_ang_vel = cMathUtil::QuatRotVec(delta_rot, root_ang_vel);

	cKinTree::SetRootPos(mJointMat, pos, mPose);
	cKinTree::SetRootRot(mJointMat, rot, mPose);
	cKinTree::SetRootVel(mJointMat, root_vel, mVel);
	cKinTree::SetRootAngVel(mJointMat, root_ang_vel, mVel);

	SetPose(mPose);
	SetVel(mVel);
}

void cSimCharacter::SetRootVel(const tVector& vel)
{
	cCharacter::SetRootVel(vel);
	SetVel(mVel);
}

void cSimCharacter::SetRootAngVel(const tVector& ang_vel)
{
	cCharacter::SetRootAngVel(ang_vel);
	SetVel(mVel);
}

tQuaternion cSimCharacter::CalcHeadingRot() const
{
	tVector ref_dir = tVector(1, 0, 0, 0);
	tQuaternion root_rot = GetRootRotation();
	tVector rot_dir = cMathUtil::QuatRotVec(root_rot, ref_dir);
	double heading = std::atan2(-rot_dir[2], rot_dir[0]);
	return cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), heading);
}

int cSimCharacter::GetNumBodyParts() const
{
	return static_cast<int>(mBodyParts.size());
}

void cSimCharacter::SetVel(const Eigen::VectorXd& vel)
{
	cCharacter::SetVel(vel);

	double world_scale = mWorld->GetScale();
	int root_id = cKinTree::GetRoot(mJointMat);
	tVector root_vel = cKinTree::GetRootVel(mJointMat, vel);
	tVector root_ang_vel = cKinTree::GetRootAngVel(mJointMat, vel);
	cKinTree::eJointType root_type = cKinTree::GetJointType(mJointMat, root_id);

	if (!mMultiBody->hasFixedBase())
	{
		mMultiBody->setBaseVel(world_scale * btVector3(root_vel[0], root_vel[1], root_vel[2]));
		mMultiBody->setBaseOmega(btVector3(root_ang_vel[0], root_ang_vel[1], root_ang_vel[2]));
	}
	else
	{
		cSimBodyJoint& root_joint = GetJoint(root_id);
		tQuaternion world_to_root = root_joint.CalcWorldRotation().inverse();
		tVector local_root_vel = cMathUtil::QuatRotVec(world_to_root, root_vel);
		tVector local_root_ang_vel = cMathUtil::QuatRotVec(world_to_root, root_ang_vel);
		Eigen::VectorXd root_params = Eigen::VectorXd::Zero(GetParamSize(root_id));

		switch (root_type)
		{
		case cKinTree::eJointTypeRevolute:
		{
			root_params[0] = local_root_ang_vel[2];
			break;
		}
		case cKinTree::eJointTypePlanar:
		{
			root_params[0] = local_root_vel[0];
			root_params[1] = local_root_vel[1];
			root_params[2] = local_root_ang_vel[2];
			break;
		}
		case cKinTree::eJointTypePrismatic:
		{
			root_params[0] = local_root_vel[0];
			break;
		}
		case cKinTree::eJointTypeFixed:
		{
			break;
		}
		case cKinTree::eJointTypeSpherical:
		{
			root_params[0] = local_root_ang_vel[0];
			root_params[1] = local_root_ang_vel[1];
			root_params[2] = local_root_ang_vel[2];
			break;
		}
		default:
			assert(false); // unsupported joint type
			break;
		}

		root_joint.SetVel(root_params);
	}

	for (int j = 1; j < GetNumJoints(); ++j)
	{
		cSimBodyJoint& curr_jont = GetJoint(j);
		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		Eigen::VectorXd curr_params = vel.segment(param_offset, param_size);
		curr_jont.SetVel(curr_params);
	}

	UpdateLinkVel();

	if (HasController())
	{
		mController->HandleVelReset();
	}
}

void cSimCharacter::SetPose_xudong(const Eigen::VectorXd & pose)
{
	
	std::cout << "[error] cSimCharacter::SetPose_xudong: bug and deprecated " << std::endl;
	exit(1);
	cCharacter::SetPose(pose);

	// 获取root信息，获取root type
	double world_scale = mWorld->GetScale();
	int root_id = cKinTree::GetRoot(mJointMat);
	tVector root_pos = cKinTree::GetRootPos(mJointMat, pose);
	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, pose);
	cKinTree::eJointType root_type = cKinTree::GetJointType(mJointMat, root_id);

	// 设置mMultiBody中的root pos & orientation
	tVector euler = cMathUtil::QuaternionToEuler(root_rot, eRotationOrder::XYZ);	// 从quaternion转化为旋转矩阵
	mMultiBody->setBasePos(world_scale * btVector3(root_pos[0], root_pos[1], root_pos[2]));	// 设置root joint位置
	mMultiBody->setWorldToBaseRot(btQuaternion(root_rot.x(), root_rot.y(), root_rot.z(), root_rot.w()));	// 这就是设置root joint rot，因为没有直接设置root orientation名字的API

	// simulation的root_joint 旋转设置为单位阵
	cSimBodyJoint & root_joint = GetJoint(root_id);
	Eigen::VectorXd root_pose = Eigen::VectorXd::Zero(GetParamSize(root_id));
	if (cKinTree::eJointTypeSpherical == root_type)	// 如果root是ball joint
	{
		root_pose = cMathUtil::QuatToVec(tQuaternion::Identity());	//[w, x, y, z]
	}
	root_joint.SetPose(root_pose);	// root作为None类型，是没有数据的。他只有一个全局位置，和一个世界坐标系到它的变换。

	for (int j = 1; j < GetNumJoints(); j++)
	{
		cSimBodyJoint & curr_joint = GetJoint(j);
		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		Eigen::VectorXd curr_params = pose.segment(param_offset, param_size);
		curr_joint.SetPose(curr_params);
	}

	UpdateLinkPos();
	UpdateLinkVel();

}

void cSimCharacter::SetVel_xudong(const Eigen::VectorXd & pose)
{

}

tVector cSimCharacter::CalcJointPos(int joint_id) const
{
	const cSimBodyJoint& joint = GetJoint(joint_id);
	tVector pos;
	
	if (joint.IsValid())
	{
		pos = joint.CalcWorldPos();
	}
	else
	{
		int parent_id = cKinTree::GetParent(mJointMat, joint_id);
		assert(parent_id != cKinTree::gInvalidJointID);
		assert(IsValidBodyPart(parent_id));

		tVector attach_pt = cKinTree::GetAttachPt(mJointMat, joint_id);
		tVector part_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, parent_id);
		attach_pt -= part_attach_pt;

		const auto& parent_part = GetBodyPart(parent_id);
		pos = parent_part->LocalToWorldPos(attach_pt);
	}
	return pos;
}

tVector cSimCharacter::CalcJointVel(int joint_id) const
{
	const cSimBodyJoint& joint = GetJoint(joint_id);
	tVector vel;

	if (joint.IsValid())
	{
		vel = joint.CalcWorldVel();
	}
	else
	{
		int parent_id = cKinTree::GetParent(mJointMat, joint_id);
		assert(parent_id != cKinTree::gInvalidJointID);
		assert(IsValidBodyPart(parent_id));

		tVector attach_pt = cKinTree::GetAttachPt(mJointMat, joint_id);
		tVector part_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, parent_id);
		attach_pt -= part_attach_pt;

		const auto& parent_part = GetBodyPart(parent_id);
		vel = parent_part->GetLinearVelocity(attach_pt);
	}

	return vel;
}

void cSimCharacter::CalcJointWorldRotation(int joint_id, tVector& out_axis, double& out_theta) const
{
	const auto& joint = GetJoint(joint_id);
	if (joint.IsValid())
	{
		joint.CalcWorldRotation(out_axis, out_theta);
	}
	else
	{
		int parent_id = cKinTree::GetParent(mJointMat, joint_id);
		assert(parent_id != cKinTree::gInvalidJointID);
		assert(IsValidBodyPart(parent_id));

		const auto& parent_part = GetBodyPart(parent_id);
		parent_part->GetRotation(out_axis, out_theta);
	}
}

tQuaternion cSimCharacter::CalcJointWorldRotation(int joint_id) const
{
	tQuaternion rot = tQuaternion::Identity();
	const auto& joint = GetJoint(joint_id);
	if (joint.IsValid())
	{
		rot = joint.CalcWorldRotation();
	}
	else
	{
		int parent_id = cKinTree::GetParent(mJointMat, joint_id);
		assert(parent_id != cKinTree::gInvalidJointID);
		assert(IsValidBodyPart(parent_id));

		const auto& parent_part = GetBodyPart(parent_id);
		rot = parent_part->GetRotation();
	}

	return rot;
}

tVector cSimCharacter::CalcCOM() const
{
	tVector com = tVector::Zero();
	double total_mass = 0;
	for (int i = 0; i < static_cast<int>(mBodyParts.size()); ++i)
	{
		if (IsValidBodyPart(i))
		{
			const auto& part = mBodyParts[i];
			double mass = part->GetMass();
			tVector curr_com = part->GetPos();

			com += mass * curr_com;
			total_mass += mass;
		}
	}
	com /= total_mass;
	return com;
}

tVector cSimCharacter::CalcCOMVel() const
{
	tVector com_vel = tVector::Zero();
	double total_mass = 0;
	for (int i = 0; i < static_cast<int>(mBodyParts.size()); ++i)
	{
		if (IsValidBodyPart(i))
		{
			const auto& part = mBodyParts[i];
			double mass = part->GetMass();
			tVector curr_vel = part->GetLinearVelocity();	// 充分证明这是质心处的线速度

			com_vel += mass * curr_vel;
			total_mass += mass;
		}
	}
	com_vel /= total_mass;
	return com_vel;
}

void cSimCharacter::CalcAABB(tVector& out_min, tVector& out_max) const
{
	out_min[0] = std::numeric_limits<double>::infinity();
	out_min[1] = std::numeric_limits<double>::infinity();
	out_min[2] = std::numeric_limits<double>::infinity();

	out_max[0] = -std::numeric_limits<double>::infinity();
	out_max[1] = -std::numeric_limits<double>::infinity();
	out_max[2] = -std::numeric_limits<double>::infinity();

	for (int i = 0; i < GetNumBodyParts(); ++i)
	{
		if (IsValidBodyPart(i))
		{
			const auto& part = GetBodyPart(i);

			tVector curr_min = tVector::Zero();
			tVector curr_max = tVector::Zero();
			part->CalcAABB(curr_min, curr_max);

			out_min = out_min.cwiseMin(curr_min);
			out_max = out_max.cwiseMax(curr_max);
		}
	}
}

tVector cSimCharacter::GetSize() const
{
	tVector aabb_min;
	tVector aabb_max;
	CalcAABB(aabb_min, aabb_max);
	return aabb_max - aabb_min;
}

const cSimBodyJoint& cSimCharacter::GetJoint(int joint_id) const
{
	return mJoints[joint_id];
}

cSimBodyJoint& cSimCharacter::GetJoint(int joint_id)
{
	return mJoints[joint_id];
}

void cSimCharacter::GetChildJoint(int joint_id, tVectorXd & out_child_id)
{
	cKinTree::GetChild(mJointMat, joint_id, out_child_id);
}

int cSimCharacter::GetParentJoint(int joint_id)
{
	return cKinTree::GetParent(mJointMat, joint_id);
}

const std::shared_ptr<cSimBodyLink>& cSimCharacter::GetBodyPart(int idx) const
{
	return mBodyParts[idx];
}

std::shared_ptr<cSimBodyLink>& cSimCharacter::GetBodyPart(int idx)
{
	return mBodyParts[idx];
}

tVector cSimCharacter::GetBodyPartPos(int idx) const
{
	auto& part = GetBodyPart(idx);
	tVector pos = part->GetPos();
	return pos;
}

tVector cSimCharacter::GetBodyPartVel(int idx) const
{
	auto& part = GetBodyPart(idx);
	tVector vel = part->GetLinearVelocity();
	return vel;
}

const std::shared_ptr<cSimBodyLink>& cSimCharacter::GetRootPart() const
{
	int root_idx = GetRootID();
	return mBodyParts[root_idx];
}

std::shared_ptr<cSimBodyLink> cSimCharacter::GetRootPart()
{
	int root_idx = GetRootID();
	return mBodyParts[root_idx];
}

void cSimCharacter::RegisterContacts(int contact_flags, int filter_flags)
{
	for (int i = 0; i < static_cast<int>(mBodyParts.size()); ++i)
	{
		if (IsValidBodyPart(i))
		{
			std::shared_ptr<cSimBodyLink>& part = mBodyParts[i];
			part->RegisterContact(contact_flags, filter_flags);
		}
	}
}

void cSimCharacter::UpdateContact(int contact_flags, int filter_flags)
{
	for (int i = 0; i < static_cast<int>(mBodyParts.size()); ++i)
	{
		if (IsValidBodyPart(i))
		{
			std::shared_ptr<cSimBodyLink>& part = mBodyParts[i];
			part->UpdateContact(contact_flags, filter_flags);
		}
	}
}

bool cSimCharacter::HasFallen() const
{
	bool fallen = false;
	if (mEnableContactFall)
	{
		fallen |= CheckFallContact();
	}
	return fallen;
}

bool cSimCharacter::HasStumbled() const
{
	bool stumbled = false;
	for (int j = 0; j < GetNumJoints(); ++j)
	{
		if (!IsEndEffector(j))
		{
			const auto& curr_part = GetBodyPart(j);
			bool contact = curr_part->IsInContact();
			if (contact)
			{
				stumbled = true;
				break;
			}
		}
	}
	return stumbled;
}

bool cSimCharacter::HasVelExploded(double vel_threshold /*= 100.0*/) const
{
	// 有速度探索?什么意思?输入还是一个门槛
	int num_bodies = GetNumBodyParts();
	for (int b = 0; b < num_bodies; ++b)
	{
		// 对于每一个body, 如果速度的最大值超过了100.0，就是true
		const auto& body = GetBodyPart(b);
		tVector vel = body->GetLinearVelocity();
		tVector ang_vel = body->GetAngularVelocity();
		double max_val = std::max(vel.cwiseAbs().maxCoeff(), ang_vel.cwiseAbs().maxCoeff());
		if (max_val > vel_threshold)
		{
			std::cout <<"invalid episode, body"<<b<<", velocity exploded = " << max_val << " > " << vel_threshold << std::endl;
			return true;
		}
	}
	return false;
}

bool cSimCharacter::IsInContact() const
{
	for (int i = 0; i < GetNumBodyParts(); ++i)
	{
		if (IsValidBodyPart(i))
		{
			if (IsInContact(i))
			{
				return true;
			}
		}
	}
	return false;
}

bool cSimCharacter::IsInContact(int idx) const
{
	return GetBodyPart(idx)->IsInContact();
}

const tEigenArr<cContactManager::tContactPt>& cSimCharacter::GetContactPts(int idx) const
{
	return GetBodyPart(idx)->GetContactPts();
}

const tEigenArr<cContactManager::tContactPt>& cSimCharacter::GetContactPts() const
{
	return GetContactPts(GetRootID());
}

const cContactManager::tContactHandle& cSimCharacter::GetContactHandle() const
{
	return GetRootPart()->GetContactHandle();
}

const void cSimCharacter::GetTotalContactPts(Eigen::VectorXd & out_contact) const
{
	// for aligned, max 10 contact point
	const int contact_max_size = 70;
	out_contact.resize(contact_max_size);
	out_contact.setConstant(-1);

	std::shared_ptr<cWorld> cur_world = GetWorld();
	// 1. contact_manager is supposed to manage all the contact infos.
	const cContactManager & contact_manager = cur_world->GetContactManager();
	if (contact_manager.GetNumTotalContactPts() * 7 > contact_max_size)
	{
		std::cout << "[error] cSimCharacter::GetTotalContactPts: exceed max contact size " << contact_manager.GetNumTotalContactPts() * 7 << std::endl;
		exit(1);
	}
	

	// 2. for body i, its contact info is located in contact_managed[i]
	for (int i = 0, cur_index = 0; i < contact_manager.GetNumEntries(); i++)
	{
		// i是第N个contact force点
		// 但是他属于那个link呢?
		const tEigenArr<cContactManager::tContactPt> &p = contact_manager.GetContactPts(i);
		for (auto &cur_pt : p)
		{
			int link_id = i;
			tVector3 contact_point_world = cur_pt.mPos.segment(0, 3),
				contact_force_world = cur_pt.mForce.segment(0, 3);

			// format: body_id(1) + pt_pos(3) + pt_force(3) = 7
			out_contact[cur_index + 0] = i;
			out_contact.segment(cur_index + 1, 3) = contact_point_world;
			out_contact.segment(cur_index + 1 + 3, 3) = contact_force_world;
			cur_index += 7;

			if (true == contact_point_world.hasNaN() || true == contact_force_world.hasNaN())
			{
				std::cout << "[error] cSimCharacter::GetTotalContactPts: Nan for link " << i << ", pof = "\
					<< contact_point_world.transpose() << ", force = " << cur_pt.mForce.transpose() << std::endl;
				//exit(1);
			}
		}
	}
}

void cSimCharacter::SetController(std::shared_ptr<cCharController> ctrl)
{
	RemoveController();
	mController = ctrl;
}

void cSimCharacter::RemoveController()
{
	if (HasController())
	{
		mController.reset();
	}
}

bool cSimCharacter::HasController() const
{
	return mController != nullptr;
}

const std::shared_ptr<cCharController>& cSimCharacter::GetController()
{
	return mController;
}

const std::shared_ptr<cCharController>& cSimCharacter::GetController() const
{
	return mController;
}

void cSimCharacter::EnableController(bool enable)
{
	if (HasController())
	{
		mController->SetActive(enable);
	}
}

void cSimCharacter::ApplyForce(const tVector& force)
{
	ApplyForce(force, tVector::Zero());
}

void cSimCharacter::ApplyForce(const tVector& force, const tVector& local_pos)
{
	const auto& root_body = GetRootPart();
	const auto& joint = GetJoint(GetRootID());

	tVector world_pos = local_pos;
	world_pos[3] = 1;
	tMatrix joint_to_world = joint.BuildWorldTrans();
	world_pos = joint_to_world * world_pos;

	tVector body_local = root_body->WorldToLocalPos(world_pos);
	root_body->ApplyForce(force, body_local);
}

void cSimCharacter::ApplyTorque(const tVector& torque)
{
	const auto& root_body = GetRootPart();
	root_body->ApplyTorque(torque);
}

void cSimCharacter::ClearForces()
{
	int num_parts = GetNumBodyParts();
	for (int b = 0; b < num_parts; ++b)
	{
		if (IsValidBodyPart(b))
		{
			auto& part = GetBodyPart(b);
			part->ClearForces();
		}
	}
}

void cSimCharacter::ApplyControlForces(const Eigen::VectorXd& tau)
{
	// std::cout <<"void cSimCharacter::ApplyControlForces(const Eigen::VectorXd& tau)" << std::endl;
	// std::cout <<"\n****************************************"<<std::endl;
	assert(tau.size() == GetNumDof());
	for (int j = 1; j < GetNumJoints(); ++j)
	{
		cSimBodyJoint& joint = GetJoint(j);
		if (joint.IsValid())
		{
			int param_offset = GetParamOffset(j);
			int param_size = GetParamSize(j);
			if (param_size > 0)
			{
				Eigen::VectorXd curr_tau = tau.segment(param_offset, param_size);
				joint.AddTau(curr_tau);
				// std::cout <<"joint " << j <<", ";
			}
		}
	}
}

void cSimCharacter::PlayPossum()
{
	if (HasController())
	{
		mController->SetMode(cController::eModePassive);
	}
}

void cSimCharacter::SetPose(const Eigen::VectorXd& pose)
{
	cCharacter::SetPose(pose);

	double world_scale = mWorld->GetScale();
	int root_id = cKinTree::GetRoot(mJointMat);
	tVector root_pos = cKinTree::GetRootPos(mJointMat, pose);
	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, pose);
	cKinTree::eJointType root_type = cKinTree::GetJointType(mJointMat, root_id);

	tVector euler = cMathUtil::QuaternionToEuler(root_rot, eRotationOrder::XYZ);
	mMultiBody->setBasePos(world_scale * btVector3(root_pos[0], root_pos[1], root_pos[2]));
	mMultiBody->setWorldToBaseRot(btQuaternion(root_rot.x(), root_rot.y(), root_rot.z(), root_rot.w()).inverse());

	cSimBodyJoint& root_joint = GetJoint(root_id);
	Eigen::VectorXd root_pose = Eigen::VectorXd::Zero(GetParamSize(root_id));
	if (root_type == cKinTree::eJointTypeSpherical)
	{
		root_pose = cMathUtil::QuatToVec(tQuaternion::Identity());
	}
	root_joint.SetPose(root_pose);

	for (int j = 1; j < GetNumJoints(); ++j)
	{
		cSimBodyJoint& curr_joint = GetJoint(j);
		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		Eigen::VectorXd curr_params = pose.segment(param_offset, param_size);
		curr_joint.SetPose(curr_params);
	}

	UpdateLinkPos();
	UpdateLinkVel();

	if (HasController())
	{
		mController->HandlePoseReset();
	}
}

bool cSimCharacter::BuildSimBody(const tParams& params)
{
	bool succ = true;
	// load important key "BodyDefs" , it is essential for simulation, including mass, inertia, etc.
	succ = LoadBodyDefs(params.mCharFile, mBodyDefs);

	if (succ)
	{
		mInvRootAttachRot = cMathUtil::EulerToQuaternion(cKinTree::GetAttachTheta(mJointMat, GetRootID()), eRotationOrder::XYZ).inverse();
		
		succ &= BuildMultiBody(mMultiBody);
		succ &= BuildJointLimits(mMultiBody);
		succ &= BuildBodyLinks();
		succ &= BuildJoints();

		mWorld->AddCharacter(*this);

		mVecBuffer0.resize(mMultiBody->getNumLinks() + 1);
		mVecBuffer1.resize(mMultiBody->getNumLinks() + 1);
		mRotBuffer.resize(mMultiBody->getNumLinks() + 1);
	}

	return succ;
}

bool cSimCharacter::BuildMultiBody(std::shared_ptr<cMultiBody>& out_body)
{
	// build class MultiBody inherited from btMultiBody
	bool succ = true;

	// base is always identity and the mass/ inertia is zero.
	// the normal "root" is the child of base
	double world_scale = mWorld->GetScale();
	int num_joints = GetNumJoints();
	bool fixed_base = FixedBase();
	btVector3 base_intertia = btVector3(0, 0, 0);
	btScalar base_mass = 0;
	mMultiBody = std::shared_ptr<cMultiBody>(new cMultiBody(num_joints, base_mass, base_intertia, fixed_base, false));
	mMultiBody->setBaseName("root_bullet");

	btTransform base_trans;
	base_trans.setIdentity();
	mMultiBody->setBaseWorldTransform(base_trans);

	for (int j = 0; j < num_joints; ++j)
	{
		// 对于每一个joint j, 从cKinTree中拿到BodyShape
		cShape::eShape body_shape = cKinTree::GetBodyShape(mBodyDefs, j);	// bodyshape就说明他是个球还是个胶囊还是个圆柱
		tVector body_size = cKinTree::GetBodySize(mBodyDefs, j);			// bodysize就是json中的param1 - param2
		double mass = cKinTree::GetBodyMass(mBodyDefs, j);					// body也有质量
		cKinTree::eJointType joint_type = cKinTree::GetJointType(mJointMat, j);
		int parent_joint = cKinTree::GetParent(mJointMat, j);				// 父铰链

		bool is_root = cKinTree::IsRoot(mJointMat, j);
		
		// 描述文件大小，没有scale
		btCollisionShape * col_shape = BuildCollisionShape(body_shape, body_size);
		btVector3 inertia = btVector3(0, 0, 0);
		col_shape->calculateLocalInertia(static_cast<btScalar>(mass), inertia);	// local inertia是本地轴向的
		// inertia: 没有经过scale

		// arg so many transforms...
		tQuaternion this_to_parent = cMathUtil::EulerToQuaternion(cKinTree::GetAttachTheta(mJointMat, j), eRotationOrder::XYZ);	// 这个joint到parent的变换的欧拉角
		tQuaternion body_to_this = cMathUtil::EulerToQuaternion(cKinTree::GetBodyAttachTheta(mBodyDefs, j), eRotationOrder::XYZ);//　本地网格到joint 变换的欧拉角到四元数
		tQuaternion this_to_body = body_to_this.inverse();	// 从local joint到网格

		tQuaternion parent_body_to_parent = tQuaternion::Identity();
		if (parent_joint != gInvalidIdx)
		{
			parent_body_to_parent = cMathUtil::EulerToQuaternion(cKinTree::GetBodyAttachTheta(mBodyDefs, parent_joint), eRotationOrder::XYZ);
		}
		tQuaternion parent_to_parent_body = parent_body_to_parent.inverse();

		tQuaternion body_to_parent_body = parent_to_parent_body * this_to_parent * body_to_this;
		tQuaternion parent_body_to_body = body_to_parent_body.inverse();

		// parent body attachment point in body's coordinate frame
		tVector parent_body_attach_pt = tVector::Zero();
		if (parent_joint != gInvalidIdx)
		{
			// 拿到parent body和本body的连接点, 在body坐标系下，连接点的坐标
			// body's coordinate什么意思?
			parent_body_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, parent_joint);
		}
		// 做了一个到parent到parent body的坐标系
		parent_body_attach_pt = cMathUtil::QuatRotVec(parent_to_parent_body, parent_body_attach_pt);

		tVector joint_attach_pt = cKinTree::GetAttachPt(mJointMat, j);		// 铰链的连接点，在parent joint坐标系下
		tVector body_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, j);	// body也有连接点!
		joint_attach_pt = cMathUtil::QuatRotVec(parent_to_parent_body, joint_attach_pt);	// 铰链的连接点也做了变换
		joint_attach_pt -= parent_body_attach_pt;//减去了parent body attach point是为什么? 运算说明在同一坐标系下..
		body_attach_pt = cMathUtil::QuatRotVec(body_to_this.inverse(), body_attach_pt);

		btTransform parent_body_to_body_trans = btTransform::getIdentity();
		parent_body_to_body_trans.setRotation(btQuaternion(parent_body_to_body.x(), parent_body_to_body.y(), parent_body_to_body.z(), parent_body_to_body.w()));

		bool disable_parent_collision = true;

		//Eigen::Vector3d tmp_inertia = Eigen::Vector3d(inertia[0], inertia[1], inertia[2]);
		//std::cout<<"[init body] link " << j <<", mass = " << mass <<", inertia = " << tmp_inertia.transpose() << std::endl;
		if (is_root && !fixed_base)
		{
			mMultiBody->setupFixed(j, static_cast<btScalar>(mass), inertia, parent_joint,
				parent_body_to_body_trans.getRotation(),
				world_scale * btVector3(joint_attach_pt[0], joint_attach_pt[1], joint_attach_pt[2]),
				world_scale * btVector3(body_attach_pt[0], body_attach_pt[1], body_attach_pt[2]),
				disable_parent_collision);
		}
		else
		{
			switch (joint_type)
			{
			case cKinTree::eJointTypeRevolute:
			{
				tVector axis = tVector(1, 0, 0, 0);
				axis = cMathUtil::QuatRotVec(this_to_body, axis);

				mMultiBody->setupRevolute(j, static_cast<btScalar>(mass), inertia, parent_joint,
					parent_body_to_body_trans.getRotation(),
					btVector3(axis[0], axis[1], axis[2]),
					world_scale * btVector3(joint_attach_pt[0], joint_attach_pt[1], joint_attach_pt[2]),
					world_scale * btVector3(body_attach_pt[0], body_attach_pt[1], body_attach_pt[2]),
					disable_parent_collision);
			}
			break;
			case cKinTree::eJointTypePlanar:
			{
				tVector offset = parent_body_attach_pt + joint_attach_pt
								+ cMathUtil::QuatRotVec(body_to_parent_body, body_attach_pt);
				
				tVector axis = tVector(0, 0, 1, 0);
				axis = cMathUtil::QuatRotVec(this_to_body, axis);

				mMultiBody->setupPlanar(j, static_cast<btScalar>(mass), inertia, parent_joint,
					parent_body_to_body_trans.getRotation(),
					btVector3(axis[0], axis[1], axis[2]),
					world_scale * btVector3(offset[0], offset[1], offset[2]),
					disable_parent_collision);
			}
			break;
			case cKinTree::eJointTypePrismatic:
			{
				tVector axis = tVector(1, 0, 0, 0);
				axis = cMathUtil::QuatRotVec(this_to_body, axis);
				
				mMultiBody->setupPrismatic(j, static_cast<btScalar>(mass), inertia, parent_joint,
					parent_body_to_body_trans.getRotation(),
					btVector3(axis[0], axis[1], axis[2]),
					world_scale * btVector3(joint_attach_pt[0], joint_attach_pt[1], joint_attach_pt[2]),
					world_scale * btVector3(body_attach_pt[0], body_attach_pt[1], body_attach_pt[2]),
					disable_parent_collision);
			}
			break;
			case cKinTree::eJointTypeFixed:
			{
				mMultiBody->setupFixed(j, static_cast<btScalar>(mass), inertia, parent_joint,
					parent_body_to_body_trans.getRotation(),
					world_scale * btVector3(joint_attach_pt[0], joint_attach_pt[1], joint_attach_pt[2]),
					world_scale * btVector3(body_attach_pt[0], body_attach_pt[1], body_attach_pt[2]),
					disable_parent_collision);
			}
			break;
			case cKinTree::eJointTypeSpherical:
			{
				// inertia didn't scale
				mMultiBody->setupSpherical(j, static_cast<btScalar>(mass), inertia, parent_joint,
					parent_body_to_body_trans.getRotation(),
					world_scale * btVector3(joint_attach_pt[0], joint_attach_pt[1], joint_attach_pt[2]),
					world_scale * btVector3(body_attach_pt[0], body_attach_pt[1], body_attach_pt[2]),
					disable_parent_collision);
			}
			break;
			default:
				printf("Unsupported joint type");
				assert(false);
				break;
			}
		}

		// set up link name
		mMultiBody->getLink(j).m_linkName = mBodyDefsName[j].c_str();
		mMultiBody->getLink(j).m_jointName = mSkeletonJointsName[j].c_str();

		// 从这里开始，添加碰撞。
		// 如果有问题，只有可能是添加的时候初始化有问题，就从这里排查，把log打出来，然后看他们究竟有什么不一样的地方。
		btMultiBodyLinkCollider* col_obj = new btMultiBodyLinkCollider(mMultiBody.get(), j);

		col_obj->setCollisionShape(col_shape);
		btTransform col_obj_trans;
		col_obj_trans.setIdentity();
		col_obj->setWorldTransform(col_obj_trans);
		col_obj->setFriction(mFriction);

		int collisionFilterGroup = GetPartColGroup(j);
		int collisionFilterMask = GetPartColMask(j);
		//std::cout << "[add collider] filter group " << j << " = " << collisionFilterGroup << std::endl;
		//std::cout << "[add collider] filter mask " << j << " = " << collisionFilterMask << std::endl;
		mWorld->AddCollisionObject(col_obj, collisionFilterGroup, collisionFilterMask);
		mMultiBody->getLink(j).m_collider = col_obj;
	}
	
	mMultiBody->finalizeMultiDof();

	return succ;
}

bool cSimCharacter::BuildJointLimits(std::shared_ptr<cMultiBody>& out_body)
{
	// 这里增加的约束，并不是bullet中rigid body之间的point2point等约束
	// 而是角度限制约束, 对于btMultiBody而言，这个约束是必要的。
	double world_scale = mWorld->GetScale();
	for (int j = 0; j < GetNumJoints(); ++j)
	{
		cKinTree::eJointType joint_type = cKinTree::GetJointType(mJointMat, j);
		if (joint_type == cKinTree::eJointTypeRevolute || joint_type == cKinTree::eJointTypePrismatic)
		{
			tVector lim_low = cKinTree::GetJointLimLow(mJointMat, j);
			tVector lim_high = cKinTree::GetJointLimHigh(mJointMat, j);
			if (lim_low[0] <= lim_high[1])
			{
				if (joint_type == cKinTree::eJointTypePrismatic)
				{
					lim_low *= world_scale;
					lim_high *= world_scale;
				}
				std::cout <<"[warning] cSimCharacter::BuildJointLimits: Adding joint constraints is not allowed because it destroys the inverse dynamics of the solution for joint " << GetBodyName(j) << std::endl;
				std::cout <<"continuing\n";
				continue;
				auto joint_cons = std::shared_ptr<btMultiBodyJointLimitConstraint>(new btMultiBodyJointLimitConstraint(mMultiBody.get(), j, lim_low[0], lim_high[0]));
				joint_cons->finalizeMultiDof();
				mCons.push_back(joint_cons);
			}
		}
	}
	return true;
}

bool cSimCharacter::LoadBodyDefs(const std::string& char_file, Eigen::MatrixXd& out_body_defs)
{
	bool succ = cKinTree::LoadBodyDefs(char_file, out_body_defs, mBodyDefsName);
	int num_joints = GetNumJoints();
	int num_body_defs = static_cast<int>(out_body_defs.rows());
	assert(num_joints == num_body_defs);
	return succ;
}

bool cSimCharacter::BuildBodyLinks()
{
	int num_joints = GetNumJoints();
	mBodyParts.clear();
	mBodyParts.resize(num_joints);

	for (int j = 0; j < num_joints; ++j)
	{
		std::shared_ptr<cSimBodyLink>& curr_part = mBodyParts[j];

		cSimBodyLink::tParams params;
		params.mMass = cKinTree::GetBodyMass(mBodyDefs, j);
		params.mJointID = j;

		curr_part = std::shared_ptr<cSimBodyLink>(new cSimBodyLink());

		short col_group = GetPartColGroup(j);
		short col_mask = GetPartColMask(j);
		curr_part->SetColGroup(col_group);
		curr_part->SetColMask(col_mask);

		curr_part->Init(mWorld, mMultiBody, params);
	}

	return true;
}

btCollisionShape* cSimCharacter::BuildCollisionShape(const cShape::eShape shape, const tVector& shape_size)
{
	btCollisionShape* col_shape = nullptr;
	switch (shape)
	{
	case cShape::eShapeBox:
		col_shape = mWorld->BuildBoxShape(shape_size);
		break;
	case cShape::eShapeCapsule:
		col_shape = mWorld->BuildCapsuleShape(0.5 * shape_size[0], shape_size[1]);
		break;
	case cShape::eShapeSphere:
		col_shape = mWorld->BuildSphereShape(0.5 * shape_size[0]);
		break;
	case cShape::eShapeCylinder:
		col_shape = mWorld->BuildCylinderShape(0.5 * shape_size[0], shape_size[1]);
		break;
	default:
		printf("Unsupported body link shape\n");
		assert(false);
		break;
	}
	return col_shape;
}

bool cSimCharacter::BuildJoints()
{
	int num_joints = GetNumJoints();
	mJoints.clear();
	mJoints.resize(num_joints);
	
	for (int j = 0; j < num_joints; ++j)
	{
		bool is_root = cKinTree::IsRoot(mJointMat, j);
		cSimBodyJoint& curr_joint = GetJoint(j);
		
		cSimBodyJoint::tParams joint_params;
		joint_params.mID = j;
		joint_params.mLimLow = cKinTree::GetJointLimLow(mJointMat, j);
		joint_params.mLimHigh = cKinTree::GetJointLimHigh(mJointMat, j);
		joint_params.mTorqueLimit = (is_root) ? 0 : cKinTree::GetTorqueLimit(mJointMat, j);
		joint_params.mForceLimit = (is_root) ? 0 : cKinTree::GetForceLimit(mJointMat, j);

		tVector child_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, j);		// 拿到BodyDefs里面的pt, 说的是网格
		tVector child_attach_theta = cKinTree::GetBodyAttachTheta(mBodyDefs, j);	// BodyDefs里面的theta, 说的是网格, 这个角度说的是child joint
		tMatrix child_to_joint = cMathUtil::TranslateMat(child_attach_pt) * cMathUtil::RotateMat(child_attach_theta, eRotationOrder::XYZ);
		tMatrix joint_to_child = cMathUtil::InvRigidMat(child_to_joint);// 从joint 坐标系到link坐标系

		joint_params.mChildRot = cMathUtil::RotMatToQuaternion(joint_to_child);	// 从joint 到link坐标系
		joint_params.mChildPos = cMathUtil::GetRigidTrans(joint_to_child);	// joint在link坐标系位置
		
		std::shared_ptr<cSimBodyLink> child_link = GetBodyPart(j);
		std::shared_ptr<cSimBodyLink> parent_link = nullptr;

		int parent_id = cKinTree::GetParent(mJointMat, j);
		if (parent_id != gInvalidIdx)
		{
			/*
				JointMat中的角度t，是在parent系下 joint坐标系原点的角度和位置
				此时，如果要让joint坐标系下任何一个点P，变换到parent坐标系下面，就只需要Trans(t) * P了
			*/
			tVector joint_attach_pt = cKinTree::GetAttachPt(mJointMat, j);
			tVector joint_attach_theta = cKinTree::GetAttachTheta(mJointMat, j);
			tVector parent_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs, parent_id);
			tVector parent_attach_theta = cKinTree::GetBodyAttachTheta(mBodyDefs, parent_id);
			
			tMatrix parent_to_parent_joint = cMathUtil::TranslateMat(parent_attach_pt) * cMathUtil::RotateMat(parent_attach_theta, eRotationOrder::XYZ);
			tMatrix joint_to_parent_joint = cMathUtil::TranslateMat(joint_attach_pt) * cMathUtil::RotateMat(joint_attach_theta, eRotationOrder::XYZ);
			tMatrix joint_to_parent = cMathUtil::InvRigidMat(parent_to_parent_joint) * joint_to_parent_joint;

			parent_link = GetBodyPart(parent_id);
			joint_params.mParentRot = cMathUtil::RotMatToQuaternion(joint_to_parent);	// from joint coord to parent joint coord, for a vec
			joint_params.mParentPos = cMathUtil::GetRigidTrans(joint_to_parent);		// joint pos in parent joint coord
		}

		curr_joint.Init(mWorld, mMultiBody, parent_link, child_link, joint_params);
	}

	return true;
}

void cSimCharacter::BuildConsFactor(int joint_id, tVector& out_linear_factor, tVector& out_angular_factor) const
{
	cKinTree::eJointType joint_type = cKinTree::GetJointType(mJointMat, joint_id);
	bool is_root = cKinTree::IsRoot(mJointMat, joint_id);
	out_linear_factor.setOnes();
	out_angular_factor.setOnes();

	if (is_root)
	{
		BuildRootConsFactor(joint_type, out_linear_factor, out_angular_factor);
	}
}

void cSimCharacter::BuildRootConsFactor(cKinTree::eJointType joint_type, tVector& out_linear_factor, tVector& out_angular_factor) const
{
	out_linear_factor = tVector::Ones();
	out_angular_factor = tVector::Ones();

	switch (joint_type)
	{
	case cKinTree::eJointTypeRevolute:
		out_linear_factor = tVector::Zero();
		out_angular_factor = tVector(1, 0, 0, 0);
		break;
	case cKinTree::eJointTypePlanar:
		out_linear_factor = tVector(1, 1, 0, 0);
		out_angular_factor = tVector(0, 0, 1, 0);
		break;
	case cKinTree::eJointTypePrismatic:
		out_linear_factor = tVector(0, 0, 1, 0);
		out_angular_factor = tVector::Zero();
		break;
	case cKinTree::eJointTypeFixed:
		out_linear_factor = tVector::Zero();
		out_angular_factor = tVector::Zero();
		break;
	case cKinTree::eJointTypeNone:
		out_linear_factor = tVector::Ones();
		out_angular_factor = tVector::Ones();
		break;
	case cKinTree::eJointTypeSpherical:
		out_linear_factor = tVector::Zero();
		out_angular_factor = tVector(1, 1, 1, 0);
		break;
	default:
		assert(false); // unsupported joint type
		break;
	}
}

bool cSimCharacter::FixedBase() const
{
	int root_id = GetRootID();
	cKinTree::eJointType joint_type = cKinTree::GetJointType(mJointMat, root_id);
	return joint_type != cKinTree::eJointTypeNone;
}

void cSimCharacter::RemoveFromWorld()
{
	if (mWorld != nullptr)
	{
		mWorld->RemoveCharacter(*this);
		mJoints.clear();
		mBodyParts.clear();
		mCons.clear();
		mMultiBody.reset();
		mWorld.reset();
	}
}

bool cSimCharacter::IsValidBodyPart(int idx) const
{
	return mBodyParts[idx] != nullptr;
}

bool cSimCharacter::EnableBodyPartFallContact(int idx) const
{
	assert(idx >= 0 && idx < GetNumBodyParts());
	return cKinTree::GetBodyEnableFallContact(mBodyDefs, idx);
}

void cSimCharacter::SetBodyPartFallContact(int idx, bool enable)
{
	assert(idx >= 0 && idx < GetNumBodyParts());
	cKinTree::SetBodyEnableFallContact(idx, enable, mBodyDefs);
}

tMatrix cSimCharacter::BuildJointWorldTrans(int joint_id) const
{
	const cSimBodyJoint& joint = GetJoint(joint_id);
	if (joint.IsValid())
	{
		return joint.BuildWorldTrans();
	}
	else
	{
		return cCharacter::BuildJointWorldTrans(joint_id);
	}
}

void cSimCharacter::ClearJointTorques()
{
	int num_joints = GetNumJoints();
	for (int j = 1; j < num_joints; ++j)
	{
		cSimBodyJoint& joint = GetJoint(j);
		if (joint.IsValid())
		{
			joint.ClearTau();
		}
	}
}

void cSimCharacter::UpdateJoints()
{
	// std::cout << "void cSimCharacter::UpdateJoints()" << std::endl;
	if(mEnableJointTorqueControl == false) std::cout <<"[debug] cSimCharacter::UpdateJoints disable joint forces\n";
	int num_joints = GetNumJoints();
	for (int j = 1; j < num_joints; ++j)
	{
		cSimBodyJoint& joint = GetJoint(j);
		if (joint.IsValid() && mEnableJointTorqueControl)
		{
			joint.ApplyTau();
		}
	}
}

void cSimCharacter::UpdateLinkPos()
{
	mMultiBody->updateCollisionObjectWorldTransforms(mRotBuffer, mVecBuffer0);
}

void cSimCharacter::UpdateLinkVel()
{
	static double max_vel_err = 0;
	static double max_omega_err = 0;

	btAlignedObjectArray<btVector3>& ang_vel_buffer = mVecBuffer0;
	btAlignedObjectArray<btVector3>& vel_buffer = mVecBuffer1;

	// 利用bullet计算速度
	mMultiBody->compTreeLinkVelocities(&ang_vel_buffer[0], &vel_buffer[0]);

	double world_scale = mWorld->GetScale();
	for (int b = 0; b < GetNumBodyParts(); ++b)
	{
		if (IsValidBodyPart(b))
		{
			const btVector3& bt_vel = vel_buffer[b + 1];
			const btVector3& bt_ang_vel = ang_vel_buffer[b + 1];

			// 从bullet中获取各个body对固定点质心的线速度和角速度(局部坐标系下)
			tVector com_vel = tVector(bt_vel[0], bt_vel[1], bt_vel[2], 0);
			tVector com_omega = tVector(bt_ang_vel[0], bt_ang_vel[1], bt_ang_vel[2], 0);
			com_vel /= world_scale;

			// velocities are in the body's local coordinates
			// so need to transform them into world coords
			// 局部坐标系转化为全局坐标系下
			//std::cout << "link " << b << std::endl;
			//std::cout << "[before] " << com_vel.transpose() << std::endl;
			//std::cout << "[before] " << com_omega.transpose() << std::endl;
			tQuaternion world_rot = GetBodyPart(b)->GetRotation();
			com_vel = cMathUtil::QuatRotVec(world_rot, com_vel);
			com_omega = cMathUtil::QuatRotVec(world_rot, com_omega);
			//std::cout << "[after] " << com_vel.transpose() << std::endl;
			//std::cout << "[after] " << com_omega.transpose() << std::endl;
			const auto& link = GetBodyPart(b);
			link->UpdateVel(com_vel, com_omega);
		}
	}
}

short cSimCharacter::GetPartColGroup(int part_id) const
{
	return GetPartColMask(part_id);
}

short cSimCharacter::GetPartColMask(int part_id) const
{
	int col_group = cKinTree::GetBodyColGroup(mBodyDefs, part_id);
	assert(col_group < static_cast<int>(sizeof(short) * 8));

	short flags;
	if (col_group == 0)
	{
		flags = cContactManager::gFlagNone;
	}
	else if (col_group == -1)
	{
		flags = cContactManager::gFlagAll;
	}
	else
	{
		flags = 1 << col_group;
	}
	return flags;
}

tVector cSimCharacter::GetPartColor(int part_id) const
{
	return cKinTree::GetBodyColor(mBodyDefs, part_id);
}

double cSimCharacter::CalcTotalMass() const
{
	double m = 0;
	for (int i = 0; i < GetNumBodyParts(); ++i)
	{
		if (IsValidBodyPart(i))
		{
			m += GetBodyPart(i)->GetMass();
		}
	}
	return m;
}

void cSimCharacter::SetLinearDamping(double damping)
{
	mMultiBody->setLinearDamping(damping);
}

void cSimCharacter::SetAngularDamping(double damping)
{
	mMultiBody->setAngularDamping(damping);
}

tVector cSimCharacter::GetPos() const
{
	return GetRootPos();
}

void cSimCharacter::SetPos(const tVector& pos)
{
	SetRootPos(pos);
}

void cSimCharacter::GetRotation(tVector& out_axis, double& out_theta) const
{
	return GetRootRotation(out_axis, out_theta);
}

tQuaternion cSimCharacter::GetRotation() const
{
	return GetRootRotation();
}

void cSimCharacter::SetRotation(const tVector& axis, double theta)
{
	SetRootRotation(axis, theta);
}

void cSimCharacter::SetRotation(const tQuaternion& q)
{
	SetRootRotation(q);
}

tMatrix cSimCharacter::GetWorldTransform() const
{
	return GetJoint(GetRootID()).BuildWorldTrans();
}

tVector cSimCharacter::GetLinearVelocity() const
{
	return GetRootVel();
}

tVector cSimCharacter::GetLinearVelocity(const tVector& local_pos) const
{
	const auto& root_body = GetRootPart();
	const auto& joint = GetJoint(GetRootID());

	tVector world_pos = local_pos;
	world_pos[3] = 1;
	tMatrix joint_to_world = joint.BuildWorldTrans();
	world_pos = joint_to_world * world_pos;

	tVector body_local = root_body->WorldToLocalPos(world_pos);
	return root_body->GetLinearVelocity(body_local);
}

void cSimCharacter::SetLinearVelocity(const tVector& vel)
{
	SetRootVel(vel);
}

tVector cSimCharacter::GetAngularVelocity() const
{
	return GetRootAngVel();
}

void cSimCharacter::SetAngularVelocity(const tVector& vel)
{
	SetRootAngVel(vel);
}

short cSimCharacter::GetColGroup() const
{
	return GetRootPart()->GetColGroup();
}

void cSimCharacter::SetColGroup(short col_group)
{
	for (int i = 0; i < GetNumBodyParts(); ++i)
	{
		if (IsValidBodyPart(i))
		{
			GetBodyPart(i)->SetColGroup(col_group);
		}
	}
}

short cSimCharacter::GetColMask() const
{
	return GetRootPart()->GetColMask();
}

void cSimCharacter::SetColMask(short col_mask)
{
	for (int i = 0; i < GetNumBodyParts(); ++i)
	{
		if (IsValidBodyPart(i))
		{
			GetBodyPart(i)->SetColMask(col_mask);
		}
	}
}

void cSimCharacter::SetEnablejointTorqueControl(bool v_)
{
	mEnableJointTorqueControl = v_;
	// std::cout <<"[debug] cSimCharacter::SetEnablejointTorqueControl " << v_ << std::endl;
	// exit(1);
}

std::string cSimCharacter::GetCharFilename()
{
	return mCharFilename;
}

const std::shared_ptr<cWorld>& cSimCharacter::GetWorld() const
{
	return mWorld;
}

const std::shared_ptr<cMultiBody>& cSimCharacter::GetMultiBody() const
{
	return mMultiBody;
}

const std::vector<std::shared_ptr<btMultiBodyJointLimitConstraint>>& cSimCharacter::GetConstraints() const
{
	return mCons;
}

void cSimCharacter::BuildJointPose(int joint_id, Eigen::VectorXd& out_pose) const
{
	bool is_root = cKinTree::IsRoot(mJointMat, joint_id);
	if (is_root)
	{
		int param_size = GetParamSize(joint_id);
		out_pose = Eigen::VectorXd::Zero(param_size);
		assert(out_pose.size() == cKinTree::gRootDim);

		tVector root_pos = GetRootPos();
		tQuaternion root_rot = GetRootRotation();

		out_pose.segment(0, cKinTree::gPosDim) = root_pos.segment(0, cKinTree::gPosDim);
		out_pose(cKinTree::gPosDim) = root_rot.w();
		out_pose(cKinTree::gPosDim + 1) = root_rot.x();
		out_pose(cKinTree::gPosDim + 2) = root_rot.y();
		out_pose(cKinTree::gPosDim + 3) = root_rot.z();
	}
	else
	{
		const cSimBodyJoint& joint = GetJoint(joint_id);
		joint.BuildPose(out_pose);
	}
}

void cSimCharacter::BuildJointVel(int joint_id, Eigen::VectorXd& out_vel) const
{
	bool is_root = cKinTree::IsRoot(mJointMat, joint_id);
	if (is_root)
	{
		int param_size = GetParamSize(joint_id);
		out_vel = Eigen::VectorXd::Zero(param_size);
		assert(out_vel.size() == cKinTree::gRootDim);

		tVector root_vel = GetRootVel();
		tVector ang_vel = GetRootAngVel();
		out_vel.segment(0, cKinTree::gPosDim) = root_vel.segment(0, cKinTree::gPosDim);
		out_vel.segment(cKinTree::gPosDim, cKinTree::gRotDim) = ang_vel.segment(0, cKinTree::gRotDim);
	}
	else
	{
		const cSimBodyJoint& joint = GetJoint(joint_id);
		joint.BuildVel(out_vel);
	}
}

void cSimCharacter::BuildPose(Eigen::VectorXd& out_pose) const
{
	int num_joints = GetNumJoints();
	int num_dof = cKinTree::GetNumDof(mJointMat);
	out_pose.resize(num_dof);
	for (int j = 0; j < num_joints; ++j)
	{
		Eigen::VectorXd joint_pose;
		BuildJointPose(j, joint_pose);

		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		assert(joint_pose.size() == param_size);
		out_pose.segment(param_offset, param_size) = joint_pose;
	}
}

void cSimCharacter::BuildVel(Eigen::VectorXd& out_vel) const
{
	int num_joints = GetNumJoints();
	int num_dof = cKinTree::GetNumDof(mJointMat);
	out_vel.resize(num_dof);

	for (int j = 0; j < num_joints; ++j)
	{
		Eigen::VectorXd joint_vel;
		BuildJointVel(j, joint_vel);

		int param_offset = GetParamOffset(j);
		int param_size = GetParamSize(j);
		assert(joint_vel.size() == param_size);
		out_vel.segment(param_offset, param_size) = joint_vel;
	}
}

bool cSimCharacter::CheckFallContact() const
{
	// 角色: 检查摔倒接触
	int num_parts = GetNumBodyParts();	// 对于每一个接触
	for (int b = 0; b < num_parts; ++b)
	{
		if (IsValidBodyPart(b) && EnableBodyPartFallContact(b))	// 如果他有效，而且fall contact是真. 所以说那个config里面填写的fall contact是: 如果他落在地上就算失败了。
		{
			// 脚应该返回假，所以脚应该是0
			// 该落地的地方都应该把EnableFallContact设置为0
			// 脚不应该开，该落地的地方不应该开：脚的enablebody part fall contact是0才对
			const auto& curr_part = GetBodyPart(b);
			bool has_contact = curr_part->IsInContact();	// 看这个link是否摔倒了, 所以脚是不是应该...关了.
			if (has_contact)
			{
				// std::cout <<"[end] detect part " << b <<" contact with ground, so episode end" << std::endl;
				return true;
			}
		}
	}
	return false;
}

const btCollisionObject* cSimCharacter::GetCollisionObject() const
{
	return nullptr;
}

btCollisionObject* cSimCharacter::GetCollisionObject()
{
	return nullptr;
}

void btvector2eigen(btAlignedObjectArray<btScalar> & r, Eigen::VectorXd &e)
{
	e.resize(r.size());
	for (int i = 0; i < r.size(); i++)	e[i] = r[i];
}
