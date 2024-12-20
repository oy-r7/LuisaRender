//
// Created by Mike Smith on 2022/1/10.
//

#include <gui/framerate.h>
#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

using namespace compute;

static const luisa::unordered_map<luisa::string_view, uint>
    aov_component_to_channels{{"color", 3u},
                              {"normal", 3u},
                              {"albedo", 3u},
                              {"depth", 1u},
                              {"position", 3u},
                              {"visibility", 1u},
                              {"diffuse", 3u},
                              {"specular", 3u}};

class AuxiliaryBufferPathTracing final : public Integrator {

public:
    enum struct DumpStrategy {
        POWER2,
        ALL,
        FINAL,
    };

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _noisy_count;
    DumpStrategy _dump_strategy{};
    luisa::unordered_set<luisa::string> _enabled_aov;

public:
    AuxiliaryBufferPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _noisy_count{std::max(desc->property_uint_or_default("noisy_count", 8u), 1u)} {
        auto components = desc->property_string_list_or_default("components", {"all"});
        for (auto &comp : components) {
            for (auto &c : comp) { c = static_cast<char>(std::tolower(c)); }
            if (comp == "all") {
                for (auto &[name, _] : aov_component_to_channels) {
                    _enabled_aov.emplace(name);
                }
            } else if (aov_component_to_channels.contains(comp)) {
                _enabled_aov.emplace(comp);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Ignoring unknown AOV component '{}'. [{}]",
                    comp, desc->source_location().string());
            }
        }
        for (auto &&comp : _enabled_aov) {
            LUISA_INFO("Enabled AOV component '{}'.", comp);
        }
        auto dump = desc->property_string_or_default("dump", "power2");
        for (auto &c : dump) { c = static_cast<char>(std::tolower(c)); }
        if (dump == "all") {
            _dump_strategy = DumpStrategy::ALL;
        } else if (dump == "final") {
            _dump_strategy = DumpStrategy::FINAL;
        } else {
            if (dump != "power2") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown dump strategy '{}'. "
                    "Fallback to power2 strategy. [{}]",
                    dump, desc->source_location().string());
            }
            _dump_strategy = DumpStrategy::POWER2;
        }
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto noisy_count() const noexcept { return _noisy_count; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto dump_strategy() const noexcept { return _dump_strategy; }
    [[nodiscard]] auto is_component_enabled(luisa::string_view component) const noexcept {
        return _enabled_aov.contains(component);
    }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class AuxiliaryBufferPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

public:
    explicit AuxiliaryBufferPathTracingInstance(
        const AuxiliaryBufferPathTracing *node,
        Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<AuxiliaryBufferPathTracing>();
        CommandBuffer command_buffer{&stream};
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, camera);
            command_buffer << compute::synchronize();
            camera->film()->release();
        }
    }
};

luisa::unique_ptr<Integrator::Instance> AuxiliaryBufferPathTracing::build(
    Pipeline &pipeline, CommandBuffer &cmd_buffer) const noexcept {
    return luisa::make_unique<AuxiliaryBufferPathTracingInstance>(
        this, pipeline, cmd_buffer);
}

class AuxiliaryBuffer {

private:
    Pipeline &_pipeline;
    Image<float> _image;

private:
    static constexpr auto clear_shader_name = luisa::string_view{"__aux_buffer_clear_shader"};

public:
    AuxiliaryBuffer(Pipeline &pipeline, uint2 resolution, uint channels, bool enabled = true) noexcept
        : _pipeline{pipeline} {
        _pipeline.register_shader<2u>(
            clear_shader_name, [](ImageFloat image) noexcept {
                image.write(dispatch_id().xy(), make_float4(0.f));
            });
        if (enabled) {
            _image = pipeline.device().create_image<float>(
                channels == 1u ?// TODO: support FLOAT2
                    PixelStorage::FLOAT1 :
                    PixelStorage::FLOAT4,
                resolution);
        }
    }
    void clear(CommandBuffer &command_buffer) const noexcept {
        if (_image) {
            command_buffer << _pipeline.shader<2u, Image<float>>(clear_shader_name, _image)
                                  .dispatch(_image.size());
        }
    }
    [[nodiscard]] auto save(CommandBuffer &command_buffer,
                            std::filesystem::path path, uint total_samples) const noexcept
        -> luisa::function<void()> {
        if (!_image) { return {}; }
        auto host_image = luisa::make_shared<luisa::vector<float>>();
        auto nc = pixel_storage_channel_count(_image.storage());
        host_image->resize(_image.size().x * _image.size().y * nc);
        command_buffer << _image.copy_to(host_image->data());
        return [host_image, total_samples, nc, size = _image.size(), path = std::move(path)] {
            auto scale = static_cast<float>(1. / total_samples);
            for (auto &p : *host_image) { p *= scale; }
            LUISA_INFO("Saving auxiliary buffer to '{}'.", path.string());
            save_image(path.string(), host_image->data(), size, nc);
        };
    }
    void accumulate(Expr<uint2> p, Expr<float4> value, bool squared = false) noexcept {
        if (_image) {
            $if (!any(isnan(value))) {
                auto old = _image->read(p);
                auto threshold = 256.f;
                auto abs_v = abs(value);
                auto strength = max(max(max(abs_v.x, abs_v.y), abs_v.z), 0.f);
                auto c = value.xyz() * (threshold / max(threshold, strength));
                if (squared) {
                    _image->write(p, old + make_float4(c * c, value.w));
                } else {
                    _image->write(p, old + make_float4(c, value.w));
                }
            };
        }
    }
};

void AuxiliaryBufferPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {

    auto spp = node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << synchronize();

    using namespace luisa::compute;

    // 3 diffuse, 3 specular, 3 normal, 1 depth, 3 albedo, 1 roughness, 1 emissive, 1 metallic, 1 transmissive, 1 specular-bounce
    luisa::unordered_map<luisa::string, luisa::unique_ptr<AuxiliaryBuffer>> aux_buffers;
    for (auto [comp, nc] : aov_component_to_channels) {
        auto enabled = node<AuxiliaryBufferPathTracing>()->is_component_enabled(comp);
        LUISA_INFO("Component {} is {}.", comp, enabled ? "enabled" : "disabled");
        auto v = luisa::make_unique<AuxiliaryBuffer>(pipeline(), resolution, nc, enabled);
        aux_buffers.emplace(comp, std::move(v));
        // Add ^2 buffers
        auto v_2 = luisa::make_unique<AuxiliaryBuffer>(pipeline(), resolution, nc, enabled);
        aux_buffers.emplace(fmt::format("{}_2", comp), std::move(v_2));
    }

    // clear auxiliary buffers
    auto clear_auxiliary_buffers = [&] {
        for (auto &[_, buffer] : aux_buffers) {
            buffer->clear(command_buffer);
        }
    };

    Kernel2D render_auxiliary_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto camera_sample = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li{swl.dimension()};

        SampledSpectrum beta_diffuse{swl.dimension(), camera_sample.weight};
        SampledSpectrum Li_diffuse{swl.dimension()};

        auto ray = camera_sample.ray;
        auto pdf_bsdf = def(1e16f);
        auto pdf_bsdf_diffuse = def(1e16f);
        auto specular_bounce = def(true);
        auto specular_depth = def(-1);

        auto albedo = def(make_float3());
        auto normal = def(make_float3());
        auto roughness = def(make_float2());
        auto position = def(make_float3());
        auto visibility = def(0.f);
        auto visibility_tmp = def(0.f);

        $for (depth, node<AuxiliaryBufferPathTracing>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if (depth == 0 & it->valid()) {
                normal = it->shading().n();
                auto distance = length(it->p() - ray->origin());
                aux_buffers.at("depth")->accumulate(dispatch_id().xy(), make_float4(distance));
                aux_buffers.at("depth_2")->accumulate(dispatch_id().xy(), make_float4(distance), true);
            };

            // miss
            $if (!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if (specular_bounce | depth == 0) {
                        albedo = spectrum->srgb(swl, eval.L);
                    }
                    $else {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf_diffuse, eval.pdf);
                    };
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if (it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(
                        *it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    $if (specular_bounce | depth == 0) {
                        albedo = spectrum->srgb(swl, eval.L);
                        position = it->p();
                    }
                    $else {
                        Li_diffuse += beta_diffuse * eval.L * balance_heuristic(pdf_bsdf_diffuse, eval.pdf);
                    };
                    // just after the first diffuse bounce
                    $if (depth == specular_depth + 1) {
                        // Hit by bsdf
                        visibility += balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                };
            }

            $if (!it->shape().has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto eta_scale = def(1.f);

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }

                // apply opacity map
                auto alpha_skip = def(false);
                // if (auto o = closure->opacity()) {
                //     auto opacity = saturate(*o);
                //     alpha_skip = u_lobe >= opacity;
                //     u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                // }

                $if (alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {

                    // direct lighting
                    $if (light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                        Li_diffuse += beta_diffuse * eval.f_diffuse * light_sample.eval.L *
                                      balance_heuristic(light_sample.eval.pdf, eval.pdf) / light_sample.eval.pdf;
                        visibility_tmp = balance_heuristic(light_sample.eval.pdf, eval.pdf);
                    };

                    // sample material
                    auto sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(sample.wi);
                    pdf_bsdf = sample.eval.pdf;
                    pdf_bsdf_diffuse = sample.eval.pdf;
                    auto w = ite(sample.eval.pdf > 0.f, 1.f / sample.eval.pdf, 0.f);
                    auto w_diffuse = ite(sample.eval.pdf_diffuse > 0.f, 1.f / sample.eval.pdf_diffuse, 0.f);
                    beta *= w * sample.eval.f;

                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch (sample.event) {
                        $case (Surface::event_enter) { eta_scale = sqr(eta); };
                        $case (Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };

                    // first non-specular bounce
                    auto rough = closure->roughness();
                    $if (specular_bounce & any(rough > .05f)) {
                        specular_depth = depth;
                        specular_bounce = false;
                        albedo = spectrum->srgb(swl, closure->albedo());
                        normal = closure->it().shading().n();
                        roughness = rough;
                        position = it->p();
                        // Hit by nee
                        visibility += visibility_tmp;
                        // split the diffuse component
                        beta_diffuse *= w * sample.eval.f_diffuse;
                    }
                    $else {
                        beta_diffuse *= w * sample.eval.f;
                    };
                };
            });
        };
        auto exposure = camera->film()->node()->exposure();
        auto radiance = spectrum->srgb(swl, Li) * shutter_weight * luisa::exp2(exposure);
        auto diffuse = min(radiance, spectrum->srgb(swl, Li_diffuse) * shutter_weight * luisa::exp2(exposure));
        auto specular = radiance - diffuse;
        aux_buffers.at("color")->accumulate(pixel_id, make_float4(radiance, 1.f));
        aux_buffers.at("normal")->accumulate(pixel_id, make_float4(normal, 1.f));
        aux_buffers.at("albedo")->accumulate(pixel_id, make_float4(albedo, 1.f));
        aux_buffers.at("color_2")->accumulate(pixel_id, make_float4(radiance, 1.f), true);
        aux_buffers.at("normal_2")->accumulate(pixel_id, make_float4(normal, 1.f), true);
        aux_buffers.at("albedo_2")->accumulate(pixel_id, make_float4(albedo, 1.f), true);
        aux_buffers.at("position")->accumulate(pixel_id, make_float4(position, 1.f));
        aux_buffers.at("position_2")->accumulate(pixel_id, make_float4(position, 1.f), true);
        aux_buffers.at("visibility")->accumulate(pixel_id, make_float4(visibility));
        aux_buffers.at("visibility_2")->accumulate(pixel_id, make_float4(visibility), true);
        aux_buffers.at("diffuse")->accumulate(pixel_id, make_float4(diffuse, 1.f));
        aux_buffers.at("diffuse_2")->accumulate(pixel_id, make_float4(diffuse, 1.f), true);
        aux_buffers.at("specular")->accumulate(pixel_id, make_float4(specular, 1.f));
        aux_buffers.at("specular_2")->accumulate(pixel_id, make_float4(specular, 1.f), true);
    };

    Clock clock_compile;
    auto render_auxiliary = pipeline().device().compile(render_auxiliary_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.0);

    auto dispatch_count = 0u;
    auto dispatches_per_commit = 32u;
    auto aux_spp = node<AuxiliaryBufferPathTracing>()->noisy_count();
    auto should_dump = [this, aux_spp](uint32_t n) -> bool {
        auto strategy = node<AuxiliaryBufferPathTracing>()->dump_strategy();
        if (strategy == AuxiliaryBufferPathTracing::DumpStrategy::POWER2) {
            return n > 0 && ((n & (n - 1)) == 0);
        }
        if (strategy == AuxiliaryBufferPathTracing::DumpStrategy::ALL) {
            return true;
        }
        return n == aux_spp;
    };
    LUISA_ASSERT(shutter_samples.size() == 1u || camera->node()->spp() == aux_spp,
                 "AOVIntegrator is not compatible with motion blur "
                 "if rendered with different spp from the camera.");
    if (aux_spp != camera->node()->spp()) {
        Camera::ShutterSample ss{
            .point = {.time = camera->node()->shutter_span().x,
                      .weight = 1.f},
            .spp = aux_spp};
        shutter_samples = {ss};
    }
    auto sample_count = 0u;
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);
        clear_auxiliary_buffers();
        auto parent_path = camera->node()->file().parent_path();
        auto filename = camera->node()->file().stem().string();
        auto ext = camera->node()->file().extension().string();
        LUISA_WARNING("spp: {}", s.spp);
        for (auto i = 0u; i < 2 * s.spp; i++) {
            command_buffer << render_auxiliary(sample_count++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (sample_count % s.spp == 0) {
                LUISA_INFO("Saving AOVs at sample #{}.", sample_count);
                luisa::vector<luisa::function<void()>> savers;
                for (auto &[component, buffer] : aux_buffers) {
                    auto flag = sample_count == s.spp ? "A" : "B";
                    auto total_samples = s.spp;
                    if (component.ends_with("_2")) {
                        // skip var saving at first half
                        flag = "";
                        total_samples = sample_count;
                        if (sample_count == s.spp)
                            continue;
                    }
                    auto path = node<AuxiliaryBufferPathTracing>()->dump_strategy() ==
                                        AuxiliaryBufferPathTracing::DumpStrategy::FINAL ?
                                    parent_path / fmt::format("{}_{}{}{}", filename, component, flag, ext) :
                                    parent_path / fmt::format("{}_{}_{:05}{}{}", filename, component, sample_count, flag, ext);
                    if (auto saver = buffer->save(command_buffer, path, total_samples)) {
                        savers.emplace_back(std::move(saver));
                    }
                }
                if (!savers.empty()) {
                    command_buffer << [&] { for (auto &s : savers) { s(); } }
                                   << synchronize();
                }
                if (sample_count == s.spp) {
                    // clean the first half
                    for (auto &[component, buffer] : aux_buffers) {
                        if (!component.ends_with("_2")) buffer->clear(command_buffer);
                    }
                }
            }
            if (sample_count % 16u == 0u) { command_buffer << commit(); }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::AuxiliaryBufferPathTracing)
