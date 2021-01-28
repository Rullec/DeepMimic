#include "DrawSceneDiffImitate.h"
#include "SceneDiffImitate.h"
#include "util/LogUtil.h"

cDrawSceneDiffImitate::cDrawSceneDiffImitate() {}
cDrawSceneDiffImitate::~cDrawSceneDiffImitate() {}

void cDrawSceneDiffImitate::BuildScene(
    std::shared_ptr<cSceneSimChar> &out_scene) const
{
    out_scene = std::shared_ptr<cSceneDiffImitate>(new cSceneDiffImitate());
}

/**
 * \brief           Draw kin character for debugging, shift it a little along with the z axis for a better look
*/
void cDrawSceneDiffImitate::DrawKinCharacter(
    std::shared_ptr<cKinCharacter> &kin_char) const
{
    // 1. get pose
    const tMatrixXd &joint_mat = kin_char->GetJointMat();
    tVectorXd raw_pose = kin_char->GetPose();
    tVectorXd new_pose = raw_pose;

    // 2. shift
    int root_id = cKinTree::GetRoot(joint_mat);
    switch (cKinTree::GetJointType(joint_mat, root_id))
    {
    case cKinTree::eJointTypeBipedalNone:
        new_pose[1] += 0.2;
        break;

    default:
        MIMIC_ERROR("unsupported joint type");
        break;
    }
    kin_char->SetPose(new_pose);
    cDrawSceneImitate::DrawKinCharacter(kin_char);

    // 3. resotre raw pose
    kin_char->SetPose(raw_pose);
}