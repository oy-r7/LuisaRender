//
// Created by Mike Smith on 2022/4/9.
//

#include <numbers>

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class Grouped final : public Environment {

private:
    luisa::vector<const Environment*> environments;

public:
    Grouped(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc} {
        auto env_node_list = desc->property_node_list("environments");
        environments.clear();
        for(auto &item: env_node_list) {
            environments.emplace_back(scene->load_environment(item));
        }
    }
    [[nodiscard]] bool is_black() const noexcept override {
        for(auto &item: environments) {
            if (!item->is_black()) {
                return false;
            }
        }
        return true;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GroupedInstance final : public Environment::Instance {

private:
    luisa::vector<luisa::unique_ptr<Instance>> _envs;

public:
    GroupedInstance(Pipeline &pipeline, const Environment *env,
                    luisa::vector<luisa::unique_ptr<Instance>> envs) noexcept
        : Environment::Instance{pipeline, env}, _envs{std::move(envs)} {}

    [[nodiscard]] Environment::Evaluation evaluate(Expr<float3> wi,
                                                   const SampledWavelengths &swl,
                                                   Expr<float> time) const noexcept override {
        auto scale = 1.f / _envs.size();
        auto world_to_env = transpose(transform_to_world());
        auto wi_local = normalize(world_to_env * wi);
        auto L = SampledSpectrum{swl.dimension()};
        auto pdf = def(0.f);
        for(auto &item: _envs) {
            auto eval = item->evaluate(wi_local, swl, time);
            L += scale * eval.L;
            pdf += scale * eval.pdf;
        }
        return {.L = L, .pdf = pdf};
    }

    [[nodiscard]] Environment::Sample sample(const SampledWavelengths &swl,
                                             Expr<float> time,
                                             Expr<float2> u_in) const noexcept override {
        auto u = make_float2(u_in);
        auto scale = 1.f / _envs.size();
        auto sample = Environment::Sample::zero(swl.dimension());
        auto sample_id = cast<int>(floor(u.x * static_cast<float>(_envs.size())));
        u.x = fract(u.x * static_cast<float>(_envs.size())); // Remapped
        $switch (sample_id) {
            for (auto i = 0; i < _envs.size(); i++) {
                $case(i) {
                    sample = _envs[i]->sample(swl, time, u);
                    auto L = SampledSpectrum{swl.dimension()};
                    auto pdf = def(0.f);
                    for(auto &item: _envs){
                        auto eval_item = item->evaluate(sample.wi, swl, time);
                        L += scale * eval_item.L;
                        pdf += scale * eval_item.pdf;
                    }
                    sample.eval.L = L;
                    sample.eval.pdf = pdf;
                };
            }
            $default { unreachable(); };
        };
        sample.wi = normalize(transform_to_world() * sample.wi);
        return sample;
    }
};

luisa::unique_ptr<Environment::Instance> Grouped::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    luisa::vector<luisa::unique_ptr<Instance>> envs;
    for(auto &item: environments) {
        envs.emplace_back(item->build(pipeline, command_buffer));
    }
    return luisa::make_unique<GroupedInstance>(pipeline, this, std::move(envs));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Grouped)
