//
// Created by Omid Ghotbi (TAO) base on mega-vpt by ChenXin.
//

#include <base/integrator.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <util/medium_tracker.h>
#include <base/medium.h>
#include <util/rng.h>
#include <base/phase_function.h>

namespace luisa::render {

using namespace compute;

class MegakernelVolumetricPathTracing final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    MegakernelVolumePathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
};

};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelVolumePathTracing)