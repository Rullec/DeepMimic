//
// Created by ljf on 2020/6/15.
//

#include "SimCharVarShape.h"
#include <iostream>
cSimCharVarShape::cSimCharVarShape() {
    std::cout << "[log] SimCharVarShape class  constructed\n";
}

bool cSimCharVarShape::Init(const std::shared_ptr<cWorld> &world, const cSimCharacter::tParams &params) {
    bool succ = cSimCharacter::Init(world, params);

    succ &= LoadVarLinksFile(params.mVarLinksFile.data());

    return succ;
}

bool cSimCharVarShape::LoadVarLinksFile(const char *file) {
    bool succ = true;
    std::ifstream f_stream(file);
    Json::Reader reader;
    Json::Value root;
    succ &= reader.parse(f_stream, root);
    auto& var_links_name = root["var_links"];
//    std::cout << var_links_name.size() << std::endl;
//    exit(1);
    if (var_links_name.empty()) {
        var_body_ids.clear();
        var_draw_shape_ids.clear();
        return succ;
    }

    for(auto itr = var_links_name.begin(); itr != var_links_name.end(); ++itr) {
        auto links_name = itr->asString();
        bool found = false;
        for(int i = 0; i < GetNumBodyParts(); ++i) {
            if (GetBodyName(i) == links_name) {
                var_body_ids.push_back(i);
                var_joint_ids.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "[Error] Cannot find variable link: " << links_name << std::endl;
            succ = false;
        }
        if (GetDrawShapeDefs().rows() > 0) {
            found = false;
            for(size_t i = 0; i < GetDrawShapeDefs().rows(); ++i) {
                if (GetDrawShapeName(i) == links_name) {
                    var_draw_shape_ids.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "[Error] Cannot find variable link: " << links_name << std::endl ;
                succ = false;
            }
        }

    }
    std::cout << "variable links: \n";
    for(auto& i: var_body_ids) {
        std::cout << "i: " << i << ' ' << GetBodyName(i) << std::endl;
    }
    std::cout << "==========\n";
    if (GetDrawShapeDefs().rows() > 0) {
        std::cout << "variable draw defs: \n";
        for(auto& i: var_draw_shape_ids) {
            std::cout << "i: " << i << ' ' << GetDrawShapeName(i) << std::endl;
        }
        std::cout << "==========\n";
    }

    return succ;
}

void cSimCharVarShape::ChangeBodyShape(Eigen::VectorXd& param) {
//    std::cout << "[log] cSimCharVarShape::ChangeBodyShape() is called\n";
    assert (!var_body_ids.empty());
    assert(param.size() == 3 * var_joint_ids.size());
    cCharacter::Reset();

    for(size_t i = 0; i < var_body_ids.size(); ++i) {
        tVector scale(param[i * 3], param[i * 3 + 1], param[i * 3 + 2], 0);
        tVector body_shape = cKinTree::GetBodySize(mBodyDefs0, var_body_ids[i]);
        body_shape.noalias() = body_shape.cwiseProduct(scale);

        double mass = cKinTree::GetBodyMass(mBodyDefs0, var_body_ids[i]);
        mass *= scale[0] * scale[1] * scale[2];
        // 1. set shape param
        // do not modify body mass for now
        cKinTree::SetBodySize(mBodyDefs, body_shape, var_body_ids[i]);
//        cKinTree::SetBodyMass(mBodyDefs, mass, var_body_ids[i]);
        // 2. set attach param
        tVector body_attach_pt = cKinTree::GetBodyAttachPt(mBodyDefs0, var_body_ids[i]);
        body_attach_pt.noalias() = body_attach_pt.cwiseProduct(scale);
        cKinTree::SetBodyAttachPt(mBodyDefs, body_attach_pt, var_body_ids[i]);

        if (GetDrawShapeDefs().rows() > 0) {
            cKinTree::SetDrawShapeSize(mDrawShapeDefs, body_shape, var_draw_shape_ids[i]);
            cKinTree::SetDrawShapeAttachPt(mDrawShapeDefs, body_attach_pt, var_draw_shape_ids[i]);
        }

        // todo: fix here by using child joint id
        tVector joint_attach_pt = cKinTree::GetJointAttachPt(mJointMat0, var_joint_ids[i] + 1);
        joint_attach_pt.noalias() = joint_attach_pt.cwiseProduct(scale);
        cKinTree::SetJointAttachPt(mJointMat, joint_attach_pt, var_joint_ids[i] + 1);
    }
    UpdateBodyShape();
//    std::cout << "[log] cSimCharVarShape::ChangeBodyShape() finished\n";
}

