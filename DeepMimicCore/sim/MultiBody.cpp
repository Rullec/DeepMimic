#include "MultiBody.h"


void cMultiBody::SpatialTransform(const btMatrix3x3 &rotation_matrix, const btVector3 &displacement, 
							const btVector3 &top_in, const btVector3 &bottom_in, btVector3 &top_out, 
							btVector3 &bottom_out)
{
	top_out = rotation_matrix * top_in;
	bottom_out = -displacement.cross(top_out) + rotation_matrix * bottom_in;
}

cMultiBody::cMultiBody(int n_links, btScalar mass, const btVector3 &inertia,
						bool fixedBase, bool canSleep, bool deprecatedMultiDof) 
	: btMultiBody(n_links, mass, inertia, fixedBase, canSleep, deprecatedMultiDof)
{}

cMultiBody::~cMultiBody()
{
}

#include <iostream>
void cMultiBody::compTreeLinkVelocities(btVector3 *omega, btVector3 *vel) const
{
	// 但是这个函数必须保证, 父亲的序号必然小于儿子的序号
	int num_links = getNumLinks();
	// Calculates the velocities of each link (and the base) in its local frame
	const btQuaternion& base_rot = getWorldToBaseRot();
	omega[0] = quatRotate(base_rot, getBaseOmega());
	vel[0] = quatRotate(base_rot, getBaseVel());

	for (int i = 0; i < num_links; ++i)
	{
		const btMultibodyLink& link = getLink(i);
		const int parent = link.m_parent;
		if (parent > i)
		{
			std::cout << "[error] cMultiBody::compTreeLinkVelocities: parent id should < child id" << std::endl;
			exit(1);
		}
		// transform parent vel into this frame, store in omega[i+1], vel[i+1]
		// bullet是link和link连接的模型，每个link坐标系都建在COM处
		// m_cachedRotParentToThis 是从parent link坐标系到 child link坐标系的变换阵
		// m_cachedRVector 是child link坐标系下，vector from COM of parent to COM of this link
		// 对于角速度omega来说，只是变换了一下
		// 但是对于线速度来说，
		// top: angular terms, bottom: linear terms
		SpatialTransform(btMatrix3x3(link.m_cachedRotParentToThis), link.m_cachedRVector,
			omega[parent + 1], vel[parent + 1],
			omega[i + 1], vel[i + 1]);

		// now add qidot * shat_i
		//only supported for revolute/prismatic joints, todo: spherical and planar joints
		const btScalar* jointVel = getJointVelMultiDof(i);
		for (int dof = 0; dof < link.m_dofCount; ++dof)
		{
			omega[i + 1] += jointVel[dof] * link.getAxisTop(dof);
			vel[i + 1] += jointVel[dof] * link.getAxisBottom(dof);
		}
	}
}

void cMultiBody::setupPlanar(int i,
	btScalar mass,
	const btVector3 &inertia,
	int parent,
	const btQuaternion &rotParentToThis,
	const btVector3 &rotationAxis,
	const btVector3 &parentComToThisComOffset,
	bool disableParentCollision)
{

	btMultiBody::setupPlanar(i, mass, inertia, parent, rotParentToThis, rotationAxis, parentComToThisComOffset,
							disableParentCollision);
	auto& link = getLink(i);
	link.setAxisBottom(1, link.getAxisBottom(1).normalized());
	link.setAxisBottom(2, link.getAxisBottom(2).normalized());
}