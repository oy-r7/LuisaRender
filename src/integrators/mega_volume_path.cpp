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

            // trace
            auto it = pipeline().geometry()->intersect(ray);
            auto has_medium = it->shape().has_medium();

            device_log("depth={}", depth + 1u);
            device_log("before: medium tracker size={}, priority={}, tag={}",
                       medium_tracker.size(), medium_tracker.current().priority, medium_tracker.current().medium_tag);
            device_log("it->p(): ({}, {}, {})", it->p().x, it->p().y, it->p().z);

            // sample the participating medium
            $if (!medium_tracker.vacuum()) {
                // Sample the participating medium
                auto t_max = ite(it->valid(), length(it->p() - ray->origin()), Interaction::default_t_max);

                // Initialize RNG for sampling the majorant transmittance
                auto u1 = as<uint2>(sampler()->generate_2d());
                auto u2 = as<uint2>(sampler()->generate_2d());
                PCG32 rng((cast<ulong>(u1.x) << 32ull) | cast<ulong>(u1.y),
                          (cast<ulong>(u2.x) << 32ull) | cast<ulong>(u2.y));

                // Sample medium using delta tracking
                Float u = sampler()->generate_1d();
                Float u_behavior = sampler()->generate_1d();

                auto medium_tag = medium_tracker.current().medium_tag;
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    auto closure = medium->closure(ray, swl, time);
                    eta = closure->eta();

                    if (!closure->instance()->node()->is_vacuum()) {
                        Bool terminated = def(false);
                        SampledSpectrum T_maj = closure->sampleT_maj(
                            t_max, u, rng,
                            [&](luisa::unique_ptr<Medium::Closure> closure_p,
                                SampledSpectrum sigma_maj, SampledSpectrum T_maj) -> Bool {
                                    Bool ans = def(true);

                                // Handle medium scattering event for ray
                                $if (beta.all([](auto b) noexcept { return b <= 0.f; })) {
                                    terminated = true;
                                    ans = false;
                                }
                                $else {
                                    // Add emission from medium scattering event
                                    $if ((depth < max_depth) & !closure_p->le().is_zero()) {
                                        // Compute beta' at new path vertex
                                        Float pdf = sigma_maj[0u] * T_maj[0u];
                                        SampledSpectrum betap = beta * T_maj / pdf;

                                        // Compute rescaled path probability for absorption at path vertex
                                        SampledSpectrum r_e = r_u * sigma_maj * T_maj / pdf;

                                        // Update Li for medium emission
                                        auto Le_medium = betap * closure_p->sigma_a() * closure_p->le() / r_e.average();
                                        Li += ite(!r_e.is_zero(), Le_medium, 0.f);
                                    };

                                    // Compute medium event probabilities for interaction
                                    Float pAbsorb = closure_p->sigma_a()[0u] / sigma_maj[0u];
                                    Float pScatter = closure_p->sigma_s()[0u] / sigma_maj[0u];
                                    Float pNull = max(0.f, 1 - pAbsorb - pScatter);

                                    // Sample medium scattering event type and update path
                                    Float um = rng.uniform_float();
                                    UInt medium_event = Medium::sample_event(pAbsorb, pScatter, pNull, um);
                                    // don't use switch-case here, because of local variable definition
                                    $if (medium_event == Medium::event_absorb) {
                                        device_log("Absorb");
                                        // Handle absorption along ray path
                                        terminated = true;
                                        ans = false;
                                    }
                                    $elif (medium_event == Medium::event_scatter) {
                                        device_log("Scatter");
                                        // Handle scattering along ray path
                                        // Stop path sampling if maximum depth has been reached
                                        depth += 1u;
                                        $if (depth >= max_depth) {
                                            terminated = true;
                                            ans = false;
                                        }
                                        $else {
                                            // Update beta and r_u for real scattering event
                                            Float pdf = T_maj[0u] * closure_p->sigma_s()[0u];
                                            beta *= T_maj * closure_p->sigma_s() / pdf;
                                            r_u *= T_maj * closure_p->sigma_s() / pdf;

                                            Bool Ld_medium_zero = def(false);
                                            $if (!beta.is_zero() & !r_u.is_zero()) {
                                                // Sample direct lighting at volume scattering event
                                                // generate uniform samples
                                                auto u_light_selection = sampler()->generate_1d();
                                                auto u_light_surface = sampler()->generate_2d();

                                                // sample one light
                                                Interaction light_it{};
                                                auto light_sample = light_sampler()->sample(
                                                    light_it, u_light_selection, u_light_surface, swl, time);

                                                // direct lighting
                                                $if (light_sample.eval.pdf > 0.0f) {
                                                    auto wo = closure->ray()->direction();
                                                    auto wi = light_sample.shadow_ray->direction();

                                                    auto light_ray = make_ray(closure->ray()->origin(), wi, 0.f, one_minus_epsilon);
                                                    SampledSpectrum T_ray{swl.dimension(), 1.f}, r_l{swl.dimension(), 1.f}, r_u{swl.dimension(), 1.f};

                                                    auto shadow_iterations = def(0u);
													$while (any(light_ray->direction() != 0.f)) {
                                                        // Add iteration guard:
														shadow_iterations += 1u;
														$if (shadow_iterations > 64u) {
															device_log("[WARNING] Exceeded shadow iteration limit. Forcing break.");
															Ld_medium_zero = true;
															$break;
														};

                                                        auto si = pipeline().geometry()->intersect(light_ray);
                                                        $if (si->valid() & si->shape().has_surface()) {
                                                            Ld_medium_zero = true;
                                                            $break;
                                                        };

                                                        Float t_max = ite(si->valid(), length(si->p() - light_ray->origin()), one_minus_epsilon);
                                                        Float u = rng.uniform_float();
                                                        SampledSpectrum T_maj = closure_p->sampleT_maj(
                                                            t_max, u, rng,
                                                            [&](luisa::unique_ptr<Medium::Closure> closure_p,
                                                                SampledSpectrum sigma_maj, SampledSpectrum T_maj) -> Bool {
																device_log("[WARNING] SampledSpectrum T_maj.");
                                                                // Update ray transmittance estimate at sampled point
                                                                // Update T_ray and PDFs using ratio-tracking estimator
                                                                SampledSpectrum sigma_n = max(sigma_maj - closure_p->sigma_a() - closure_p->sigma_s(), 0.f);
                                                                Float pdf = T_maj[0u] * sigma_maj[0u];
                                                                T_ray *= T_maj * sigma_n / pdf;
                                                                r_l *= T_maj * sigma_maj / pdf;
                                                                r_u *= T_maj * sigma_n / pdf;

                                                                // Possibly terminate transmittance computation using
                                                                // Russian roulette
                                                                SampledSpectrum Tr = T_ray / (r_l + r_u).average();
                                                                Float q = 0.75f;
                                                                T_ray = ite(Tr.max() < 0.05f, ite(rng.uniform_float() < q, 0.f, T_ray / (1 - q)), T_ray);

                                                                return ite(T_ray.is_zero(), false, true);
                                                            });

                                                        auto normalization_factor = max(T_maj[0u], 1e-10f);
														T_ray *= T_maj / normalization_factor;
														r_l *= T_maj / normalization_factor;
														r_u *= T_maj / normalization_factor;

                                                        // More robust check for early termination
														$if (T_ray.max() < 1e-4f) {
															T_ray = SampledSpectrum{swl.dimension(), 0.f};  // Set to zero explicitly
															Ld_medium_zero = true;
															$break;
														};

                                                        $if (!si->valid()) {
                                                            $break;
                                                        };
                                                        light_ray = si->spawn_ray_to(light_sample.shadow_ray->origin());
                                                    };

                                                    $if (!Ld_medium_zero) {
                                                        auto phase_function = closure->phase_function();
                                                        auto f_hat = phase_function->p(wo, wi);
                                                        auto scatter_pdf = phase_function->pdf(wo, wi);

                                                        r_l *= r_u * light_sample.eval.pdf;
                                                        r_u *= r_u * scatter_pdf;

                                                        Li += beta * f_hat * T_ray * light_sample.eval.L / (r_l + r_u).average();
                                                    };
                                                };

                                                // Sample new direction at real scattering event
                                                Float2 u = sampler()->generate_2d();
                                                auto ps = closure->phase_function()->sample_p(-ray->direction(), u);
                                                $if (!ps.valid | (ps.pdf == 0.f)) {
                                                    terminated = true;
                                                }
                                                $else {
                                                    // Update ray path state for indirect volume scattering
                                                    beta *= ps.p / ps.pdf;
                                                    r_l = r_u / ps.pdf;
                                                    scattered = true;
                                                    auto p = closure_p->ray()->origin();
                                                    ray = make_ray(p, ps.wi);
                                                    device_log(
                                                        "Medium scattering event at depth={}, p=({}, {}, {})",
                                                        depth, p.x, p.y, p.z);
                                                };
                                            };
                                            ans = false;
                                        };
                                    }
                                    $elif (medium_event == Medium::event_null) {
                                        device_log("Null");
                                        // Handle null scattering along ray path
                                        SampledSpectrum sigma_n = max(sigma_maj - closure_p->sigma_a() - closure_p->sigma_s(), 0.f);
                                        Float pdf = T_maj[0u] * sigma_n[0u];
                                        beta *= T_maj * sigma_n / pdf;
                                        $if (pdf == 0.f) {
                                            beta = 0.f;
                                        };
                                        r_u *= T_maj * sigma_n / pdf;
                                        r_l *= T_maj * sigma_maj / pdf;
                                        ans = !beta.is_zero() & !r_u.is_zero();
                                    };
                                };
                                return ans;
                            });
                        // Handle terminated, scattered, and unscattered medium rays
                        $if (terminated | beta.all(le_zero) | r_u.all(le_zero)) {
                            // Terminate path sampling if ray has been terminated
                            $break;
                        };
                        $if (scattered) {
                            $continue;
                        };
                        device_log("T_maj=({}, {}, {})", T_maj[0u], T_maj[1u], T_maj[2u]);

                        beta *= T_maj / T_maj[0u];
                        r_u *= T_maj / T_maj[0u];
                        r_l *= T_maj / T_maj[0u];

                    }
                });
            };

            // sample the surface
            // miss, environment light
            $if (!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    $if (depth == 0u) {
                        Li += beta * eval.L / r_u.average();
                    }
                    $else {
                        r_l /= balance_heuristic(pdf_bsdf, eval.pdf);
                        Li += beta * eval.L / (r_u + r_l).average();
                    };
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if (it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    $if (depth == 0u) {
                        Li += beta * eval.L / r_u.average();
                    }
                    $else {
                        r_l /= balance_heuristic(pdf_bsdf, eval.pdf);
                        Li += beta * eval.L / (r_u + r_l).average();
                    };
                };
            }

            // hit ordinary surface
            $if (!it->shape().has_surface()) {
                // TODO: if shape has no surface, we cannot get the right normal direction
                //      so we cannot deal with medium tracker correctly (enter/exit)
                ray = it->spawn_ray(ray->direction());
                pdf_bsdf = 1e16f;
            }
            $else {
                // generate uniform samples
                auto u_light_selection = sampler()->generate_1d();
                auto u_light_surface = sampler()->generate_2d();
                auto u_lobe = sampler()->generate_1d();
                auto u_bsdf = sampler()->generate_2d();

                // sample one light
                auto light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);

                 // trace shadow ray
                auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

                auto medium_tag = it->shape().medium_tag();
                auto medium_priority = def(0u);
                auto eta_next = def(1.f);
                pipeline().media().dispatch(medium_tag, [&](auto medium) {
                    auto closure = medium->closure(ray, swl, time);
                    eta_next = closure->eta();
                });

                $if (has_medium) {
                    pipeline().media().dispatch(medium_tag, [&](auto medium) {
                        medium_priority = medium->priority();
                        auto closure = medium->closure(ray, swl, time);
                        device_log("eta={}", closure->eta());
                    });
                };

                auto medium_info = make_medium_info(medium_priority, medium_tag);
                medium_info.medium_tag = medium_tag;

                // evaluate material
                auto surface_tag = it->shape().surface_tag();
                auto surface_event_skip = event(swl, it, time, -ray->direction(), ray->direction());
                auto wo = -ray->direction();

                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, eta, time);
                });

                call.execute([&](auto closure) noexcept {
                    UInt surface_event;
					$if (medium_tag != medium_tracker.current().medium_tag) {
						surface_event = surface_event_skip;
						ray = it->spawn_ray(ray->direction());
						pdf_bsdf = 1e16f;
						// Reset medium tracker if we detect inconsistency
						$if (medium_tracker.size() > 10) {  // Arbitrary threshold for likely corruption
							medium_tracker = MediumTracker{};  // Reset tracker
							auto env_medium_tag = pipeline().environment_medium_tag();
							pipeline().media().dispatch(env_medium_tag, [&](auto medium) {
								medium_tracker.enter(medium->priority(), make_medium_info(medium->priority(), env_medium_tag));
							});
						};
					}
                    $else {
                        if (auto dispersive = closure->is_dispersive()) {
                            $if (*dispersive) { swl.terminate_secondary(); };
                        }
                        // direct lighting
                        // TODO: add medium to direct lighting
                        $if (light_sample.eval.pdf > 0.0f & !occluded) {
                            auto wi = light_sample.shadow_ray->direction();
                            auto eval = closure->evaluate(wo, wi);
                            auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                     light_sample.eval.pdf;
                            Li += w * beta * eval.f * light_sample.eval.L;
                        };

                        // sample material
                        auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                        surface_event = surface_sample.event;

                        ray = it->spawn_ray(surface_sample.wi);
                        pdf_bsdf = surface_sample.eval.pdf;
                        auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                        beta *= w * surface_sample.eval.f;
                        r_l = r_u * w;

                        // apply eta scale & update medium tracker
                        $if (has_medium) {
							// More robust eta handling with additional checks
							auto valid_eta = (eta > 0.001f) & (eta_next > 0.001f);
							$switch (surface_event) {
								$case (Surface::event_enter) {
									eta_scale = ite(valid_eta, sqr(eta_next / eta), 1.0f);
								};
								$case (Surface::event_exit) {
									eta_scale = ite(valid_eta, sqr(eta / eta_next), 1.0f);
								};
								$default {
									// For reflection events, no change to eta_scale
									device_log("==================================> eta_scale no change");
								};
							};
						};
                    };

                    $if (has_medium) {
                        $switch (surface_event) {
                            $case (Surface::event_enter) {
                                medium_tracker.enter(medium_priority, medium_info);
                            };
                            $case (Surface::event_exit) {
                                medium_tracker.exit(medium_priority, medium_info);
                            };
                        };
                    };
            };

            beta = zero_if_any_nan(beta);
            $if (beta.all(le_zero)) { $break; };

            auto rr_threshold = node<MegakernelVolumePathTracing>()->rr_threshold();
            auto luminance = beta.average() * eta_scale;  // Using average as approximation for luminance
			auto q = clamp(max(luminance, rr_threshold), 0.05f, 1.0f);
			$if (depth + 1u >= rr_depth) {
				$if (u_rr >= q) { $break; };  // Simpler termination condition
				beta *= 1.0f / q;  // Always scale by 1/q when continuing
			};
            depth += 1u;

            device_log(
                "scattered={}, beta=({}, {}, {}), pdf_bsdf={}, Li: ({}, {}, {})",
                scattered, beta[0u], beta[1u], beta[2u], pdf_bsdf, Li[0u], Li[1u], Li[2u]);
            device_log("after: medium tracker size={}, priority={}, tag={}",
                       medium_tracker.size(), medium_tracker.current().priority, medium_tracker.current().medium_tag);
        
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