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
        $else {
            auto shading = it->shading();
            wo_local = shading.world_to_local(wo);
            wi_local = shading.world_to_local(wi);
        };
        device_log(
            "wo_local: ({}, {}, {}), wi_local: ({}, {}, {})",
            wo_local.x, wo_local.y, wo_local.z,
            wi_local.x, wi_local.y, wi_local.z);
        return ite(
            wo_local.z * wi_local.z > 0.f,
            Surface::event_reflect,
            ite(
                wi_local.z > 0.f,
                Surface::event_exit,
                Surface::event_enter));
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension(), 0.f};
        SampledSpectrum r_u{swl.dimension(), 1.f}, r_l{swl.dimension(), 1.f};
        auto rr_depth = node<MegakernelVolumetricPathTracing>()->rr_depth();
        MediumTracker medium_tracker;

        // functions
        auto le_zero = [](auto b) noexcept { return b <= 0.f; };

        // initialize medium tracker
        auto env_medium_tag = pipeline().environment_medium_tag();
        pipeline().media().dispatch(env_medium_tag, [&](auto medium) {
            medium_tracker.enter(medium->priority(), make_medium_info(medium->priority(), env_medium_tag));
        });

        auto ray = camera_ray;
        // TODO: bug in initialization of medium tracker where the angle between shared edge is small
        auto depth_track = def<uint>(0u);
		auto max_iterations = 644u;

        $while (true) {
            auto it = pipeline().geometry()->intersect(ray);
            $if (!it->valid()) { $break; };
			
			depth_track += 1u;
			$if (depth_track > max_iterations) {
				device_log("[WARNING] Max iteration limit reached in geometry intersect loop. Breaking forcefully.");
				$break;
			};

            device_log("depth={}", depth_track);

            $if (it->shape().has_medium()) {
                auto surface_tag = it->shape().surface_tag();
                auto medium_tag = it->shape().medium_tag();

                auto medium_priority = def<uint>(0u);
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    medium_priority = medium->priority();
                });
                auto medium_info = make_medium_info(medium_priority, medium_tag);

                // deal with medium tracker
                auto surface_event = event(swl, it, time, -ray->direction(), ray->direction());
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) {
                    device_log("surface event={}", surface_event);
                    // update medium tracker
                    $switch (surface_event) {
                        $case (Surface::event_enter) {
                            medium_tracker.enter(medium_priority, medium_info);
                            device_log("enter: priority={}, medium_tag={}", medium_priority, medium_tag);
                        };
                        $case (Surface::event_exit) {
                            $if (medium_tracker.exist(medium_priority, medium_info)) {
                                medium_tracker.exit(medium_priority, medium_info);
                                device_log("exit exist: priority={}, medium_tag={}", medium_priority, medium_tag);
                            }
                            $else {
                                medium_tracker.enter(medium_priority, medium_info);
                                device_log("exit nonexistent: priority={}, medium_tag={}", medium_priority, medium_tag);
                            };
                        };
                    };
                });
            };
            device_log("medium tracker size={}", medium_tracker.size());
            auto dir = ray->direction();
            auto origin = ray->origin();
            device_log("ray->origin()=({}, {}, {})", origin.x, origin.y, origin.z);
            device_log("ray->direction()=({}, {}, {})", dir.x, dir.y, dir.z);
            device_log("it->p()=({}, {}, {})", it->p().x, it->p().y, it->p().z);
            device_log("it->shape().has_medium()={}", it->shape().has_medium());
            ray = it->spawn_ray(ray->direction());
            depth_track += 1u;
        };
        device_log("Final medium tracker size={}", medium_tracker.size());

        ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto eta_scale = def(1.f);
        auto depth = def(0u);
		auto iteration_count = def(0u);
        auto max_depth = node<MegakernelVolumePathTracing>()->max_depth();

        $while (depth < max_depth) {
			// Increment and check iteration count to prevent infinite loops
            iteration_count += 1u;
			$if (iteration_count > max_depth * 10u) {  // Safety threshold
				device_log("Breaking loop due to iteration limit");
				$break;  // Force break to prevent hanging
			};
            auto eta = def(1.f);
            auto u_rr = def(0.f);
            Bool scattered = def(false);
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
        
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelVolumetricPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelVolumetricPathTracingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelVolumetricPathTracing)