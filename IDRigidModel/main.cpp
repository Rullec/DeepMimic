#include "../DeepMimicCore/sim/RigidSystem/MultiRigidBodyModel.h"
#include <iostream>
#include <memory>
using namespace std;

int main()
{	
	shared_ptr<MultiRigidBodyModel> model = (shared_ptr<MultiRigidBodyModel>)(new MultiRigidBodyModel());
	std::cout << model->getGravity_force() << std::endl;
}
