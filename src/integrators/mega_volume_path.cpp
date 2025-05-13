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
    MegakernelVolumetricPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 20u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelVolumetricPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        //        pipeline().printer().set_log_dispatch_id(make_uint2(330, 770));
        Instance::_render_one_camera(command_buffer, camera);
    }

    [[nodiscard]] UInt event(const SampledWavelengths &swl, luisa::shared_ptr<Interaction> it, Expr<float> time,
                             Expr<float3> wo, Expr<float3> wi) const noexcept {
        Float3 wo_local, wi_local;
        $if (it->shape().has_surface()) {
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                auto shading = closure->it().shading();
                wo_local = shading.world_to_local(wo);
                wi_local = shading.world_to_local(wi);
            });
        }
    }

};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelVolumetricPathTracing)