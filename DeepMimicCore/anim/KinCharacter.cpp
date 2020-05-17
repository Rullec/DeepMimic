#include "KinCharacter.h"
#include <assert.h>
#include <functional>
#include "util/FileUtil.h"
#include <iostream>

const double gDiffTimeStep = 1 / 600.0;

cKinCharacter::tParams::tParams()
{
	mID = gInvalidIdx;
	mCharFile = "";
	mMotionFile = "";
	mOrigin.setZero();
	mLoadDrawShapes = true;
}

cKinCharacter::cKinCharacter()
{
	mOrigin.setZero();
	mOriginRot.setIdentity();
	mCycleRootDelta.setZero();
}

cKinCharacter::~cKinCharacter()
{
}

bool cKinCharacter::Init(const tParams& params)
{
	mID = params.mID;
	bool succ = cCharacter::Init(params.mCharFile, params.mLoadDrawShapes);
	if (succ)
	{
		// load motion data for kin_char
		if (params.mMotionFile != "")
		{
			LoadMotion(params.mMotionFile);
		}

		if (params.mStateFile != "")
		{
			bool succ_state = ReadState(params.mStateFile);

			if (!succ_state)
			{
				printf("Failed to load character state from %s\n", params.mStateFile.c_str());
			}
			else
			{
				mPose0 = mPose;
				mVel0 = mVel;
				SetPose(mPose);
				SetVel(mVel);
			}
		}

		SetOriginPos(params.mOrigin);
	}
	else
	{
		printf("Failed to build character from char_file: %s\n", params.mCharFile.c_str());
	}
	return succ;
}

void cKinCharacter::Clear()
{
	cCharacter::Clear();
	mMotion.Clear();
}

void cKinCharacter::Update(double time_step)
{
	cCharacter::Update(time_step);
	mTime += time_step;
	std::cout << "[debug] cKinChar update time = " << mTime << std::endl;
	Pose(mTime);
}

void cKinCharacter::Reset()
{
	cCharacter::Reset();
}

bool cKinCharacter::LoadMotion(const std::string& motion_file)
{
	cMotion::tParams motion_params;
	motion_params.mMotionFile = motion_file;
	motion_params.mBlendFunc = std::bind(&cKinCharacter::BlendFrames, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mMirrorFunc = std::bind(&cKinCharacter::MirrorFrame, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
	motion_params.mVelFunc = std::bind(&cKinCharacter::CalcFrameVel, this,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
											std::placeholders::_4);
	motion_params.mPostProcessFunc = std::bind(&cKinCharacter::PostProcessFrame, this, std::placeholders::_1);
	bool succ = mMotion.Load(motion_params);

	if (succ)
	{
		int char_dof = GetNumDof();
		int motion_dof = mMotion.GetNumDof();

		if (char_dof != motion_dof)
		{
			printf("DOF mismatch, char dof: %i, motion dof: %i\n", char_dof, motion_dof);
			mMotion.Clear();
			succ = false;
			exit(1);
		}
	}

	if (succ)
	{
		mCycleRootDelta = CalcCycleRootDelta();
		Pose(mTime);
		mPose0 = GetPose();
		mVel0 = GetVel();
	}

	if (!succ)
	{
		printf("Failed to load motion from: %s\n", motion_file.c_str());
	}
	return succ;
}

const cMotion& cKinCharacter::GetMotion() const
{
	return mMotion;
}

double cKinCharacter::GetMotionDuration() const
{
	if (mMotion.IsValid())
	{
		return mMotion.GetDuration();
	}
	return 0;
}

void cKinCharacter::SetMotionDuration(double dur)
{
	return mMotion.SetDuration(dur);
}

void cKinCharacter::SetTime(double time)
{
	// std::cout << "[debug] cKinChar set time = " << time << std::endl;
	mTime = time;
}

double cKinCharacter::GetTime() const
{
	return mTime;
}

int cKinCharacter::GetCycle() const
{
	int cycle = 0;
	if (mMotion.EnableLoop())
	{
		double phase = mTime / mMotion.GetDuration();
		cycle = static_cast<int>(std::floor(phase));
	}
	return cycle;
}

double cKinCharacter::GetPhase() const
{
	double phase = mTime / mMotion.GetDuration();
	if (mMotion.EnableLoop())
	{
		phase -= static_cast<int>(phase);
		phase = (phase < 0) ? (1 + phase) : phase;
	}
	else
	{
		phase = cMathUtil::Clamp(phase, 0.0, 1.0);
	}
	return phase;
}

void cKinCharacter::Pose(double time)
{
	// given time, change the Pose acoording to the given motion file
	// std::cout <<"void cKinCharacter::Pose(double time)" << time << std::endl;
	
	CalcPose(time, mPose);
	SetPose(mPose);

	// compute velocity
	CalcVel(time, mVel);
	SetVel(mVel);
}

void cKinCharacter::BuildAcc(Eigen::VectorXd& out_acc) const
{
	CalcAcc(mTime, out_acc);
}

bool cKinCharacter::HasMotion() const
{
	return mMotion.IsValid();
}

/**
 * \brief					Set Root Pos to a specified place pos
 * \param					pos, new root position
*/
void cKinCharacter::SetRootPos(const tVector& pos)
{
	// 1. first get current root position
	tVector root_pos = GetRootPos();

	// 2. calculate the vector from current root pos to ideal root position "pos".
	tVector delta = pos - root_pos;

	// 3. Move origin point along with this vector
	MoveOrigin(delta);
}

void cKinCharacter::SetRootRotation(const tQuaternion& q)
{
	tQuaternion root_rot = GetRootRotation();
	tQuaternion dq = q * root_rot.conjugate();
	RotateOrigin(dq);
}

const tVector& cKinCharacter::GetOriginPos() const
{
	return mOrigin;
}

void cKinCharacter::SetOriginPos(const tVector& origin)
{
	// input origin: 0, 0, 0, 相当于mOrigin变成了0, 0,0，但是mPose0和mPose都相应的移动了一个位移
	// after the first epoch, the mOrigin has became to 
	// std::cout <<"kinchar set origin pose = " << origin.transpose() << std::endl;
	// std::cout <<"cur ideal origin = " << origin.transpose() << std::endl;
	// std::cout <<"cur true mOrigin = " << mOrigin.transpose() << std::endl;
	tVector delta = origin - mOrigin;
	// std::cout <<"delta = " << delta.transpose() << std::endl;
	MoveOrigin(delta);
	mOrigin = origin; // this is needed in canse of NaNs
	// std::cout <<"final mOrigin = " << mOrigin.transpose() << std::endl;
}

/**
 * \brief			Move origin along with a given vector "delta"
 * 	Usually this vector is from current pos to an ideal pos
*/
void cKinCharacter::MoveOrigin(const tVector& delta)
{
	// 1. change the origin to ideal pos
	mOrigin += delta;

	// 2. move the mPose0 to the ideal pos
	tVector root0 = cKinTree::GetRootPos(mJointMat, mPose0);
	root0 += delta;
	cKinTree::SetRootPos(mJointMat, root0, mPose0);

	// 3. move the current mPose to the ideal pos
	tVector root = cKinTree::GetRootPos(mJointMat, mPose);
	root += delta;
	cKinTree::SetRootPos(mJointMat, root, mPose);
}

const tQuaternion& cKinCharacter::GetOriginRot() const
{
	return mOriginRot;
}

void cKinCharacter::SetOriginRot(const tQuaternion& rot)
{
	tQuaternion delta_rot = cMathUtil::QuatDiff(mOriginRot, rot);
	RotateOrigin(delta_rot);
	mOriginRot = rot; // this is needed in case of NaNs
}

void cKinCharacter::RotateOrigin(const tQuaternion& rot)
{
	mOriginRot = rot * mOriginRot;
	mOriginRot.normalize();

	tVector root_pos = GetRootPos();
	tVector root_pos_delta = mOrigin - root_pos;
	root_pos_delta = cMathUtil::QuatRotVec(rot, root_pos_delta);
	mOrigin = root_pos + root_pos_delta;

	tQuaternion root_rot0 = cKinTree::GetRootRot(mJointMat, mPose0);
	root_rot0 = rot * root_rot0;
	root_rot0.normalize();
	cKinTree::SetRootRot(mJointMat, root_rot0, mPose0);

	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, mPose);
	root_rot = rot * root_rot;
	root_rot.normalize();
	cKinTree::SetRootRot(mJointMat, root_rot, mPose);

	tVector vel0 = cKinTree::GetRootVel(mJointMat, mVel0);
	vel0 = cMathUtil::QuatRotVec(rot, vel0);
	cKinTree::SetRootVel(mJointMat, vel0, mVel0);

	tVector vel = cKinTree::GetRootVel(mJointMat, mVel);
	vel = cMathUtil::QuatRotVec(rot, vel);
	cKinTree::SetRootVel(mJointMat, vel, mVel);

	tVector ang_vel0 = cKinTree::GetRootAngVel(mJointMat, mVel0);
	ang_vel0 = cMathUtil::QuatRotVec(rot, ang_vel0);
	cKinTree::SetRootAngVel(mJointMat, ang_vel0, mVel0);

	tVector ang_vel = cKinTree::GetRootAngVel(mJointMat, mVel);
	ang_vel = cMathUtil::QuatRotVec(rot, ang_vel);
	cKinTree::SetRootAngVel(mJointMat, ang_vel, mVel);
}

void cKinCharacter::ResetParams()
{ 
	cCharacter::ResetParams();
	mTime = 0;
}

tVector cKinCharacter::GetCycleRootDelta() const
{
	tVector delta = cMathUtil::QuatRotVec(mOriginRot, mCycleRootDelta);
	return delta;
}

tVector cKinCharacter::CalcCycleRootDelta() const
{
	int num_frames = mMotion.GetNumFrames();
	Eigen::VectorXd frame_beg = mMotion.GetFrame(0);
	Eigen::VectorXd  frame_end = mMotion.GetFrame(num_frames - 1);

	tVector root_pos_beg = cKinTree::GetRootPos(mJointMat, frame_beg);
	tVector root_pos_end = cKinTree::GetRootPos(mJointMat, frame_end);

	tVector delta = root_pos_end - root_pos_beg;
	return delta;
}

void cKinCharacter::CalcPose(double time, Eigen::VectorXd& out_pose) const
{
	std::cout <<"------------begin to calculate pose for time " << time << std::endl;
	// given a time, how to compute the pose accordly
	tVector root_delta = tVector::Zero();	// root delta translation is zero
	tQuaternion root_delta_rot = tQuaternion::Identity();	// root delta rotation is identity

	// 1. get a purely ref motion from mMotion 
	if (HasMotion())
	{
		// if motion exists
		mMotion.CalcFrame(time, out_pose);
		// std::cout <<"void cKinCharacter::CalcPose: pose size = " << out_pose.size() << std::endl;
		// std::cout <<"void cKinCharacter::CalcPose: pose = " << out_pose.transpose() << std::endl;
		if (mMotion.EnableLoop())
		{
			int cycle_count = mMotion.CalcCycleCount(time);
			root_delta = cycle_count * mCycleRootDelta;
		}
	}
	else
	{
		out_pose = mPose0;
	}
	
	// 2. fetch the root_pos and root_rot in out_pose, which is purely ref motion
	tVector root_pos = cKinTree::GetRootPos(mJointMat, out_pose);
	tQuaternion root_rot = cKinTree::GetRootRot(mJointMat, out_pose);	// then find out current root_rot in the ref motion
	std::cout <<"root pos from motion data = " << root_pos.transpose() << std::endl;

	// 3. root_rot = mOriginRot * root_ref_rot
	root_delta_rot = mOriginRot * root_delta_rot;
	root_rot = root_delta_rot * root_rot;
	root_pos += root_delta;
	std::cout <<"root delta = " << root_delta.transpose() << std::endl;
	std::cout <<"root pos after root delta = " << root_pos.transpose() << std::endl;
	std::cout <<"root delta rot = " << root_delta_rot.coeffs().transpose() << std::endl;

	// 4. rotate root_pos with mOriginRot, strange?
	root_pos = cMathUtil::QuatRotVec(root_delta_rot, root_pos);
	std::cout <<"root pos after root_delta rot = " << root_pos.transpose() << std::endl;
	std::cout <<"mOrigin = " << mOrigin.transpose() << std::endl;

	// 5. move root_pos with mOrigin? stange...
	root_pos += mOrigin;
	std::cout <<"root pos after mOrigin = " << root_pos.transpose() << std::endl;

	// 6. write new root_pos and root_rot to out_pose
	// final_root_pos = mOriginRot * root_ref_pos  + mOrigin
	cKinTree::SetRootPos(mJointMat, root_pos, out_pose);
	cKinTree::SetRootRot(mJointMat, root_rot, out_pose);
	std::cout <<"------------end to calculate pose for time " << time << std::endl;
}

void cKinCharacter::CalcVel(double time, Eigen::VectorXd& out_vel) const
{
	if (HasMotion())
	{
		mMotion.CalcFrameVel(time, out_vel);
	}
	else
	{
		out_vel = Eigen::VectorXd::Zero(GetNumDof());
	}
}

void cKinCharacter::CalcAcc(double time, Eigen::VectorXd& out_acc) const
{
	Eigen::VectorXd vel0;
	Eigen::VectorXd vel1;
	CalcVel(time - gDiffTimeStep, vel0);
	CalcVel(time, vel1);
	out_acc = (vel1 - vel0) / gDiffTimeStep;
}

bool cKinCharacter::IsMotionOver() const
{
	bool over = true;
	if (HasMotion())
	{
		over = mMotion.IsOver(mTime);
	}

	return over;
}

void cKinCharacter::BlendFrames(const cMotion::tFrame* a, const cMotion::tFrame* b, double lerp, cMotion::tFrame* out_frame) const
{
	cKinTree::LerpPoses(mJointMat, *a, *b, lerp, *out_frame);
}

void cKinCharacter::MirrorFrame(const std::vector<int>* right_joints, const std::vector<int>* left_joints, cMotion::tFrame* out_frame) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::MirrorPoseStance(joint_mat, *left_joints, *right_joints, *out_frame);
}

void cKinCharacter::CalcFrameVel(const cMotion::tFrame* a, const cMotion::tFrame* b, double timestep, cMotion::tFrame* out_vel) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::CalcVel(joint_mat, *a, *b, timestep, *out_vel);
}

void cKinCharacter::PostProcessFrame(cMotion::tFrame* out_frame) const
{
	const auto& joint_mat = GetJointMat();
	cKinTree::PostProcessPose(joint_mat, *out_frame);
}