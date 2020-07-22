//
// Created by ljf on 2020/6/15.
//

#ifndef DEEPMIMICCORE_SIMCHARVARSHAPE_H
#define DEEPMIMICCORE_SIMCHARVARSHAPE_H

#include "SimCharacter.h"
class cSimCharVarShape : public cSimCharacter
{
public:
    cSimCharVarShape();

    bool Init(const std::shared_ptr<cWorld> &world, const tParams &params) override;
    bool LoadVarLinksFile(const char* file);
    void ChangeBodyShape(Eigen::VectorXd& body_param) override;
    void ChangeBodyShape(Eigen::VectorXd& body_param, Eigen::MatrixXd& joint_mat);


protected:
    std::vector<int> var_joint_ids, var_body_ids, var_draw_shape_ids;// the id of changeable links

};


#endif //DEEPMIMICCORE_SIMCHARVARSHAPE_H
