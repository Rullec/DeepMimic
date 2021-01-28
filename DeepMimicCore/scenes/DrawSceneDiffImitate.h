#pragma once

#include "DrawSceneImitate.h"

class cDrawSceneDiffImitate : public cDrawSceneImitate
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cDrawSceneDiffImitate();
    virtual ~cDrawSceneDiffImitate();

    // virtual void Init();
    // virtual void Clear();
    // virtual bool IsEpisodeEnd() const;
    // virtual bool CheckValidEpisode() const;

    // virtual void Keyboard(unsigned char key, double device_x, double device_y);
    // virtual void DrawKinChar(bool enable);

    // virtual std::string GetName() const;

protected:
    // bool mDrawKinChar;

    // virtual cRLScene *GetRLScene() const;

    virtual void BuildScene(std::shared_ptr<cSceneSimChar> &out_scene) const;
    virtual void DrawKinCharacter(
        std::shared_ptr<cKinCharacter> &kin_char) const override final;
};
