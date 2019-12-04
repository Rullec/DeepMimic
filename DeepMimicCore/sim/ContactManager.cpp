#include "ContactManager.h"
#include "sim/World.h"
#include "SimObj.h"
#include <util/BulletUtil.h>
#include <iostream>

const int cContactManager::gInvalidID = -1;

cContactManager::tContactPt::tContactPt()
{
	mPos.setZero();
	mForce.setZero();
}

cContactManager::tContactHandle::tContactHandle()
{
	mID = gInvalidID;
	mFlags = gFlagAll;
	mFilterFlags = gFlagAll;
}

cContactManager::tContactEntry::tContactEntry()
{
	mFlags = gFlagAll;
	mFilterFlags = gFlagAll;
	mContactPts.clear();
}

bool cContactManager::tContactHandle::IsValid() const
{
	return mID != gInvalidID;
}

cContactManager::cContactManager(cWorld& world)
	: mWorld(world)
{

}

cContactManager::~cContactManager()
{

}

void cContactManager::Init()
{
	Clear();
}

void cContactManager::Reset()
{
	for (int i = 0; i < GetNumEntries(); ++i)
	{
		tContactEntry& entry = mContactEntries[i];
		entry.mContactPts.clear();
	}
}

void cContactManager::Clear()
{
	mContactEntries.clear();
}

void b3printf(btVector3 data)
{
	for (int i = 0; i < 3; i++) std::cout << data[i] << " ";
	std::cout << std::endl;
	return;
}

void cContactManager::Update()
{
	ClearContacts();
	double world_scale = mWorld.GetScale();
	double timestep = mWorld.GetTimeStep();
	std::unique_ptr<btMultiBodyDynamicsWorld>& bt_world = mWorld.GetInternalWorld();	// 获取Internal world

	int num_manifolds = bt_world->getDispatcher()->getNumManifolds();	// btDispatcher是什么?Mainfolds又是什么?
	 //std::cout <<"num_mainfolds = " << num_manifolds << std::endl;
	// abort();
	for (int i = 0; i < num_manifolds; ++i)
	{
		btPersistentManifold* mani = bt_world->getDispatcher()->getManifoldByIndexInternal(i);	// 对于这个Mainfold
		const btCollisionObject* obj0 = static_cast<const btCollisionObject*>(mani->getBody0());	// 获取碰撞对象1
		const btCollisionObject* obj1 = static_cast<const btCollisionObject*>(mani->getBody1());	// 获取碰撞对象2

		int num_contacts = mani->getNumContacts();	// 获取接触点个数
		//std::cout << " there are " << num_contacts << "contact pts" << std::endl;
		for (int j = 0; j < num_contacts; ++j)
		{
			// 对于每个接触点
			btManifoldPoint& pt = mani->getContactPoint(j);
			//btScalar dist_tol = static_cast<btScalar>(0.5 * world_scale);	// 规定接触距离

			btScalar linear_friction_force1 = pt.m_appliedImpulseLateral1 / timestep;
			btScalar linear_friction_force2 = pt.m_appliedImpulseLateral2 / timestep;
			tVector lateral_friction_dir1 = cBulletUtil::btVectorTotVector0(pt.m_lateralFrictionDir1),
				lateral_friction_dir2 = cBulletUtil::btVectorTotVector0(pt.m_lateralFrictionDir2);

			//std::cout << "later print" << std::endl; bt2eigen(pt.m_lateralFrictionDir1);
			//std::cout << "force 1 = " << linear_friction_force1 << ", direction = " << lateral_friction_dir1.transpose() << std::endl;
			//std::cout << "force 2 = " << linear_friction_force2 << ", direction = " << lateral_friction_dir2.transpose() << std::endl;

			//if (pt.getDistance() <= dist_tol)
			{
				const cSimObj* sim_obj0 = static_cast<const cSimObj*>(obj0->getUserPointer());
				const cSimObj* sim_obj1 = static_cast<const cSimObj*>(obj1->getUserPointer());


				const tContactHandle& h0 = sim_obj0->GetContactHandle();
				const tContactHandle& h1 = sim_obj1->GetContactHandle();

				bool valid_contact = IsValidContact(h0, h1);	// 判定接触有效性
				if (valid_contact)
				{
					double impulse = pt.getAppliedImpulse() / world_scale;
					tVector normal = tVector(pt.m_normalWorldOnB[0], pt.m_normalWorldOnB[1], pt.m_normalWorldOnB[2], 0);
					tVector force0 = (impulse / timestep) * normal;	// A受的力?
					//std::cout << "m_appliedImpulse = " << pt.m_appliedImpulse << std::endl;
					tVector friction = linear_friction_force1 * lateral_friction_dir1 + linear_friction_force2 * lateral_friction_dir2;
					//std::cout << "contact pt " << j << " = " << friction.transpose() << std::endl;
					if (force0.hasNaN() == true)
					{
						std::cout << "[error] impulse = " << impulse << ", normal = " << normal.transpose() << ", timestep = " << timestep << std::endl;
						exit(1);
					}
					if (h0.IsValid())
					{
						tContactPt pt0;
						const btVector3& contact_pt = pt.getPositionWorldOnA();
						pt0.mPos = tVector(contact_pt[0], contact_pt[1], contact_pt[2], 0) / world_scale;
						pt0.mForce = force0;
						pt0.mForce += friction;
						mContactEntries[h0.mID].mContactPts.push_back(pt0);	// 施加力
					}

					if (h1.IsValid())
					{
						std::cout << "[error] h1 is valid, take care of friction here." << std::endl;
						exit(1);
						tContactPt pt1;
						const btVector3& contact_pt = pt.getPositionWorldOnB();
						pt1.mPos = tVector(contact_pt[0], contact_pt[1], contact_pt[2], 0) / world_scale;
						pt1.mForce = -force0;
						pt1.mForce -= friction;
						mContactEntries[h1.mID].mContactPts.push_back(pt1);
					}

					// 看摩擦力: 没有用，只是一些系数
					//std::cout << "------------" << std::endl;
					//std::cout << pt.m_combinedFriction << std::endl;;
					//std::cout << pt.m_combinedRollingFriction << std::endl;;
					////std::cout << pt.m_lateralFrictionDir1[0] << std::endl;;
					////std::cout << pt.m_lateralFrictionDir2 << std::endl;;
					//b3printf(pt.m_lateralFrictionDir1);
					//b3printf(pt.m_lateralFrictionDir2);

					//std::cout << pt.m_combinedSpinningFriction << std::endl;;
					//std::cout << pt.m_frictionCFM << std::endl;;
				}
			}
		}
	}
}

cContactManager::tContactHandle cContactManager::RegisterContact(int contact_flags, int filter_flags)
{
	tContactHandle handle;
	handle.mFlags = contact_flags;
	handle.mFilterFlags = filter_flags;
	handle.mID = RegisterNewID();

	tContactEntry& entry = mContactEntries[handle.mID];
	entry.mFlags = contact_flags;
	entry.mFilterFlags = filter_flags;

	assert(handle.IsValid());
	return handle;
}

void cContactManager::UpdateContact(const cContactManager::tContactHandle& handle)
{
	assert(handle.IsValid());
	tContactEntry& entry = mContactEntries[handle.mID];
	entry.mFlags = handle.mFlags;
	entry.mFilterFlags = handle.mFilterFlags;
}

int cContactManager::GetNumEntries() const
{
	return static_cast<int>(mContactEntries.size());
}

int cContactManager::GetNumTotalContactPts() const
{
	int num_total_pts = 0;
	for (auto &i : mContactEntries)
	{

		num_total_pts += i.mContactPts.size();
	}
	return num_total_pts;
}

bool cContactManager::IsInContact(const tContactHandle& handle) const
{
	if (handle.IsValid())
	{
		return mContactEntries[handle.mID].mContactPts.size() > 0;
	}
	return false;
}

const tEigenArr<cContactManager::tContactPt>& cContactManager::GetContactPts(const tContactHandle& handle) const
{
	return GetContactPts(handle.mID);
}

const tEigenArr<cContactManager::tContactPt>& cContactManager::GetContactPts(int handle_id) const
{
	assert(handle_id != gInvalidID);
	return mContactEntries[handle_id].mContactPts;
}

int cContactManager::RegisterNewID()
{
	// 每次mContactEntries增加一个点
	int id = gInvalidID;
	id = static_cast<int>(mContactEntries.size());
	mContactEntries.resize(id + 1);
	return id;
}

void cContactManager::ClearContacts()
{
	int num_entries = GetNumEntries();	// 一共有多少个接触区域?
	for (int i = 0; i < num_entries; ++i)
	{
		tContactEntry& curr_entry = mContactEntries[i];
		curr_entry.mContactPts.clear();	// 每个接触区域中所有的点，都要clear
	}
}

bool cContactManager::IsValidContact(const tContactHandle& h0, const tContactHandle& h1) const
{
	bool valid_h0 = ((h0.mFilterFlags & h1.mFlags) != 0);	// 这两个flag是cWorld::eCharacterXX / cWorld::eEnvironmentXX之类的
	bool valid_h1 = ((h1.mFilterFlags & h0.mFlags) != 0);	// 代表的是哪2类相撞。如果相撞类别不同，那h0和h1就是无效的。
	bool valid_contact = valid_h0 && valid_h1;
	return valid_contact;	// 才有效
}
