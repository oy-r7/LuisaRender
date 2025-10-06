//
// Created by Mike Smith on 2022/1/10.
//
#include <util/rng.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <util/progress_bar.h>
#include <util/counter_buffer.h>
#include <base/integrator.h>
#include <base/camera.h>
#include <queue>
#include <iostream>
#include <memory>
#include <optional>
//#include "stb_image_write.h"
constexpr auto X = -2;
constexpr auto Y = -2;

#define LIS_EXPERIMENT

struct DoFRecord {
    luisa::float2 accumulation;
    luisa::float4 position;
};

LUISA_STRUCT(DoFRecord, accumulation, position) {

};

struct Onb {
    luisa::float3 tangent;
    luisa::float3 binormal;
    luisa::float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    void Construct(luisa::compute::Float3 pos_dir)
    {
        normal = normalize(pos_dir);
        binormal = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        tangent = normalize(cross(binormal, normal));
    }

    luisa::compute::Float3 GetTransform(luisa::compute::Float3& v)
    {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
    luisa::compute::Float3 GetInverseTransform(luisa::compute::Float3 & w_v) {
        return make_float3(dot(w_v, tangent), dot(w_v, binormal), dot(w_v, normal));
    }

};

/*
class AdamOptimizer {
public:
    AdamOptimizer(float lr = 0.01, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
        : alpha(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void initialize() {
        size_t size = 20;
        m.assign(size, 0.0);
        v.assign(size, 0.0);
        grad_buffer.assign(size, 0.0);
    }

    void zero_grad() {
        std::fill(grad_buffer.begin(), grad_buffer.end(), 0.0);
    }

    void write(uint index, float gradiant) {
        grad_buffer[index] = gradiant;
    }

    void accumulate_grad(const std::vector<float> &grad) {
        for (size_t i = 0; i < grad.size(); ++i) {
            grad_buffer[i] += grad[i];
        }
    }

    void Step(float a) {
        //std::vector<float> params;
         if (m.empty()) {
            initialize();
        }

        t += 1;
        for (size_t i = 0; i < grad_buffer.size(); ++i){
            
            m[i] = beta1 * m[i] + (1 - beta1) * grad_buffer[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grad_buffer[i] * grad_buffer[i];

            float m_hat = m[i] / (1 - std::pow(beta1, t));
            float v_hat = v[i] / (1 - std::pow(beta2, t));

            grad_buffer[i] -= alpha * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        
        
    }

private:
    float alpha, beta1, beta2, epsilon;
    int t;
    std::vector<float> m;
    std::vector<float> v;
    std::vector<float> grad_buffer;
};*/

struct Gaussian3D
{
    luisa::float3 mu;
    float sigma;
    float sigma_1d;
    float mean_1d;
    luisa::float3 directional_mu;
    luisa::float3 pos_dir;
    luisa::float3 pos_center;
};

LUISA_STRUCT(Gaussian3D, mu, sigma, sigma_1d, mean_1d, directional_mu, pos_dir, pos_center) {
    void construct_directional(const luisa::compute::Float3 & origin)
    {
        directional_mu = mu - origin;
    }
    luisa::compute::Float erf_approx(luisa::compute::Float x)
    {
        /* return $lambda({
            // Coefficients for a minimax polynomial approximation
            constexpr float a1 = 0.254829592f;
            constexpr float a2 = -0.284496736f;
            constexpr float a3 = 1.421413741f;
            constexpr float a4 = -1.453152027f;
            constexpr float a5 = 1.061405429f;
            constexpr float p = 0.3275911f;

            // Save the sign of x
            auto sign = ite(x >= 0.f, 1.f, -1.f);
            x = abs(x);

            // A&S formula 7.1.26
            auto t = 1.0f / (1.0f + p * x);
            auto y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

            return sign * ite(x > 3.55f, 1.f, y);
        })();*/

        constexpr float a1 = 0.254829592f;
        constexpr float a2 = -0.284496736f;
        constexpr float a3 = 1.421413741f;
        constexpr float a4 = -1.453152027f;
        constexpr float a5 = 1.061405429f;
        constexpr float p = 0.3275911f;

        // Save the sign of x
        auto sign = ite(x >= 0.f, 1.f, -1.f);
        x = abs(x);

        // A&S formula 7.1.26
        auto t = 1.0f / (1.0f + p * x);
        auto y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

        return sign * ite(x > 3.55f, 1.f, y);

    }

    luisa::compute::Float sample_normal_dist(const luisa::compute::Float& sigma, const luisa::compute::Float2& u)
    {
        //Box
        auto mag = sqrt(-2.0f * log(u[0])) * sigma;
        auto z0 = mag * cos(2.0f * luisa::constants::pi * u[1]);
        return z0;
    }

    luisa::compute::Float3 sample_g3d(luisa::compute::Var<uint> & state)
    {
        auto x = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        auto y = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        auto z = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        return mu + make_float3(x, y, z);
    }

    luisa::compute::Float pdf_directional(const luisa::compute::Float3& dir)
    {
        /* return $lambda({
            constexpr float sqrt_pi_over_2 = 1.2533141373155002512078826f;
            auto d = length(directional_mu);
            auto cosTheta = dot(normalize(directional_mu), dir);
            auto cos2Theta = luisa::render::sqr(cosTheta);
            auto sin2Theta = 1.f - cos2Theta;
            auto sigma2 = sigma * sigma;
            auto d2 = d * d;

            auto t1 = sigma2 * exp(-d2 / (2.f * sigma2)) * d * cosTheta;
            auto t2 = sqrt_pi_over_2 * sigma * exp(-d2 * sin2Theta / (2.f * sigma2));
            auto t3 = sigma2 + d2 * cos2Theta;
            auto t4 = 1.f + erf_approx(d * cosTheta / (1.41421356f * sigma));

            auto atten = t1 + t2 * t3 * t4;

            constexpr float pi_2_pow_3_over_2 = 15.7496099457f;
            auto normalizingTerm = pi_2_pow_3_over_2 * sigma * sigma * sigma;
            luisa::compute::Float ret = max(0.f, atten / normalizingTerm);
            return ret;
        })();*/

        constexpr float sqrt_pi_over_2 = 1.2533141373155002512078826f;
        auto d = length(directional_mu);
        auto cosTheta = dot(normalize(directional_mu), dir);
        auto cos2Theta = luisa::render::sqr(cosTheta);
        auto sin2Theta = 1.f - cos2Theta;
        auto sigma2 = sigma * sigma;
        auto d2 = d * d;

        auto t1 = sigma2 * exp(-d2 / (2.f * sigma2)) * d * cosTheta;
        auto t2 = sqrt_pi_over_2 * sigma * exp(-d2 * sin2Theta / (2.f * sigma2));
        auto t3 = sigma2 + d2 * cos2Theta;
        auto t4 = 1.f + erf_approx(d * cosTheta / (1.41421356f * sigma));

        auto atten = t1 + t2 * t3 * t4;

        constexpr float pi_2_pow_3_over_2 = 15.7496099457f;
        auto normalizingTerm = pi_2_pow_3_over_2 * sigma * sigma * sigma;
        luisa::compute::Float ret = max(0.f, atten / normalizingTerm);
        return ret;

        

    }

    void sample_directional(luisa::compute::Var<uint> &state, luisa::compute::Float3& wi)
    {
       
        auto x = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        auto y = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        auto z = sample_normal_dist(sigma, luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state))));
        wi = normalize(directional_mu + make_float3(x, y, z));
    }

    void construct_pos(const luisa::compute::Float3& center, const luisa::compute::Float3& dir)
    {
        pos_dir = dir;
        luisa::compute::Var<Onb> onb{};
        onb->Construct(pos_dir);
        auto pc = mu - center;
        auto offset = onb->GetTransform(pc);
        offset.z = 0.f;
        pos_center = center + onb->GetInverseTransform(offset);
    }

    luisa::compute::Float3 sample_pos(const luisa::compute::Float2& u0, const luisa::compute::Float2& u1)
    {
        luisa::compute::Var<Onb> onb{};
        onb->Construct(pos_dir);
        auto x = sample_normal_dist(sigma, u0);
        auto y = sample_normal_dist(sigma, u1);
        auto v = make_float3(x, y, 0.f);
        return pos_center + onb->GetInverseTransform(v);
    }

    luisa::compute::Float pdf_pos(const luisa::compute::Float3& p)
    {
        auto pointTo = p - pos_center;
        auto d2 = dot(pointTo, pointTo);
        auto sigma2 = luisa::render::sqr(sigma);
        auto atten = exp(-d2 / (2.f * sigma2));
        auto normalizingTerm = (sigma2 * luisa::constants::pi);
        return atten / normalizingTerm;
    }

    luisa::compute::Float pdf(const luisa::compute::Float3& p)
    {
        auto pointTo = p - pos_center;
        auto d2 = dot(pointTo, pointTo);
        auto sigma2 = luisa::render::sqr(sigma);
        auto atten = exp(-d2 / (2.f * sigma2));
        auto normalizingTerm = pow((sigma2 * luisa::constants::pi), 1.5f);
        return atten / normalizingTerm;
    }

    luisa::compute::Float construct_1d(luisa::compute::Float3 origin, luisa::compute::Float3 w) {
        auto a = sqrt(1.f / sigma / sigma);
        auto mu_new = mu - origin;
        return dot(w, mu_new);
        sigma_1d = 1.f;
        $if (a > 0.f) {
            sigma_1d = sqrt(1.f / 6.f);
        };
        mean_1d = dot(w, mu_new);
        //$if(mean_1d > 0.f)
        //{
        //	ret = 2.f;
        //	//ret = lerp(1.f, 5.f, exp(-length(w * mean_1d - mu_new)));
        //};
        return 5.f;
    }

    luisa::compute::Float alpha_t(const luisa::compute::Float &t) {
        //Peak is actually the normalizing constant
        //Float peak = 1.f / sqrt(2.f * CL_PI * sigma_1d * sigma_1d);
        //Float pdf_t = peak * exp(-(sqr(t - mean_1d)) / (2 * sigma_1d * sigma_1d));
        //return pdf_t / peak;
        //We return pdf/peak, as pdf/peak = exp*peak/peak, it simplifies to the exp term only
        auto mean = mean_1d + sigma_1d;
        return exp(-(luisa::render::sqr(t - mean_1d)) / (2 * sigma_1d * sigma_1d));
    }

};

template<int N>
struct G3DMixture
{
    luisa::compute::Float threshold = 0.f;
    luisa::compute::ArrayVar<Gaussian3D, N> gaussian{};
    luisa::compute::ArrayVar<float, N> weights{};
    void construct_directional(const luisa::compute::Float3& origin)
    {
        for (int i = 0; i < N; i++)
        {
            gaussian[i]->construct_directional(origin);
        }
    }

    luisa::compute::Float pdf_directional(const luisa::compute::Float3& dir)
    {
        luisa::compute::Float pdf = 0.f;
        for (int i = 0; i < N; i++)
        {
            pdf += gaussian[i]->pdf_directional(dir) * weights[i];
        }
        return pdf;
    }

    void sample_directional(luisa::compute::Var<uint> state, luisa::compute::Float3 &wi, luisa::compute::Float &pdf) {
        auto targetWeight = luisa::render::lcg(state);
        luisa::compute::UInt selectedComponent = 0;
        $for (i, N) {
            $if (targetWeight < weights[i]) {
                selectedComponent = i;
                $break;
            };
            targetWeight -= weights[i];
        };
        gaussian[selectedComponent]->sample_directional(state, wi);
        pdf = pdf_directional(wi);
    }

    void construct_pos(const luisa::compute::Float3& center, const luisa::compute::Float3& dir)
    {
        for (int i = 0; i < N; i++)
        {
            gaussian[i]->construct_pos(center, dir);
        }
    }

    luisa::compute::Float pdf_pos(const luisa::compute::Float3& p)
    {
        luisa::compute::Float pdf = 0.f;
        for (int i = 0; i < N; i++)
        {
            pdf += gaussian[i]->pdf_pos(p) * weights[i];
        }
        return pdf;
    }

    luisa::compute::Float pdf(const luisa::compute::Float3& p)
    {
        luisa::compute::Float pdf = 0.f;
        for (int i = 0; i < N; i++)
        {
            pdf += gaussian[i]->pdf(p) * weights[i];
        }
        return pdf;
    }

    luisa::compute::Float3 sample_g3d(luisa::compute::Var<uint> state)
    {
        auto targetWeight = luisa::render::lcg(state);
        luisa::compute::UInt selectedComponent = 0;
        $for (i, N) {
            $if (targetWeight < weights[i]) {
                selectedComponent = i;
                $break;
            };
            targetWeight -= weights[i];
        };
        return gaussian[selectedComponent]->sample_g3d(state);
    }

    luisa::compute::Float3 sample_pos(luisa::compute::Var<uint> state)
    {
        auto targetWeight = luisa::render::lcg(state);
        luisa::compute::UInt selectedConponent = 0;
        $for (i, N) {
            $if (targetWeight < weights[i]) {
                selectedConponent = i;
                $break;
            };
            targetWeight -= weights[i];
        };
        auto u0 = luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state)));
        auto u1 = luisa::compute::make_float2(luisa::render::lcg(state), (luisa::render::lcg(state)));
        return gaussian[selectedConponent]->sample_pos(u0, u1);
    }

    luisa::compute::Float construct_1d(const luisa::compute::Float3& origin, const luisa::compute::Float3& v)
    {
        luisa::compute::Float scale = 0.f;
        for (int i = 0; i < N; i++)
        {
            scale += weights[i] * gaussian[i]->construct_1d(origin, v);
        }
        return scale;
    }

    luisa::compute::Float alpha_t(const luisa::compute::Float& t)
    {
        luisa::compute::Float alpha = 0.f;
        for (int i = 0; i < N; i++)
        {
            alpha += weights[i] * gaussian[i]->alpha_t(t);
        }
        return alpha;
    }
};





#ifdef LIS_EXPERIMENT
//For test purpose
luisa::compute::Buffer<float> testRadianceBuffer{};
luisa::compute::Buffer<unsigned> sampleCountBuffer{};
std::optional<luisa::compute::UInt> radianceCoord{};
unsigned dist_res = 128u;
#endif



constexpr unsigned BUFFER_ITER = 64u;
constexpr unsigned COMPONENT_COUNT = 16;
constexpr float WIDTH_SCALE = 0.5f;
constexpr unsigned POLYNOMIAL_DEGREE = 2u;
constexpr unsigned POLYNOMIAL_PARAMS_PER_OUTPUT(unsigned dim) {
    //unsigned params = 0; // Start without the constant term (degree 0)
    unsigned params = 1;// Start with the constant term (degree 0)
    for (int i = 1; i <= POLYNOMIAL_DEGREE; ++i) {
        // Calculate combinations (dim + i - 1) choose i
        unsigned num = 1;
        for (unsigned j = 0; j < i; ++j) {
            num *= (dim + j);
            num /= (j + 1);
        }
        params += num;
    }
    return params;
}
constexpr unsigned PARAMS_PER_G3D = POLYNOMIAL_PARAMS_PER_OUTPUT(2) * 4u;
constexpr unsigned SCREEN_SPACE_RECORD_RES = 256u;
constexpr unsigned MAX_G3D_COUNT = SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES;



namespace luisa::render {

using namespace compute;



#ifdef LIS_EXPERIMENT
constexpr uint2 TARGET_PIXEL = make_uint2(289u, 285u);
#endif

class OwnkernelPathTracing final : public ProgressiveIntegrator {

private:
    

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    float _aperture;
    float _focal_length;
    float _focus_distance;
    float3 _position;
    float3 _look_at;
    float3 _up;
    uint2 _resolution{};
    

   
   


public:


    OwnkernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _resolution{desc->property_uint2_or_default(
              "resolution", lazy_construct([desc] {
                  return make_uint2(desc->property_uint_or_default("resolution", 1024u));
              }))},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _aperture{desc->property_float_or_default("aperture", 2.f)},
          _focal_length{desc->property_float_or_default("focal_length", 35.f)},
          _position{desc->property_float3_or_default("position", make_float3(0.f, 0.f, 1.f))},
          _look_at{ desc->property_float3_or_default("look_at", make_float3(0.f,0.f,1.f))},
          _up{desc->property_float3_or_default("up", make_float3(0.f, 1.f, 0.f))},
          _focus_distance{desc->property_float_or_default(
              "focus_distance", lazy_construct([desc] {
                  auto target = desc->property_float3("look_at");
                  auto position = desc->property_float3("position");
                  return length(target - position);
              }))} {
        _focus_distance = std::max(std::abs(_focus_distance), 1e-4f);
        

    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto aperture() const noexcept { return _aperture; }
    [[nodiscard]] auto focal_length() const noexcept { return _focal_length; }
    [[nodiscard]] auto focus_distance() const noexcept { return _focus_distance; }
    [[nodiscard]] auto look_at() const noexcept { return _look_at; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] uint2 resolution() const noexcept { return _resolution; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

    


};

class OwnkernelPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    bool polynomialFitting = false;
private:
    Device &_device = pipeline().device();
    luisa::compute::Buffer<luisa::float4> g3dBuffer = _device.create_buffer<float4>(MAX_G3D_COUNT * 2);
    //Bool positionrecord;
    mutable Buffer<DoFRecord> records = _device.create_buffer<DoFRecord>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
    mutable Buffer<int> labelBuffer = _device.create_buffer<int>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);

    std::optional<luisa::compute::Bool> everBounced{};

    std::optional<luisa::compute::Float3> firstBounce{};
    mutable std::optional<luisa::compute::Float> lensPdf{};
    std::optional<float> tryTimes{};
    std::optional<luisa::compute::Int> selectedComponent{};

    std::optional<luisa::compute::Bool> positionRecorded{};
    //luisa::compute::Buffer<float> result_buf = _device.create_buffer<float>(2);

    std::optional<luisa::compute::Float3> pdfRecords{};
    luisa::uint2 canvasSize = make_uint2(SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
    // luisa::compute::Buffer<DoFRecord> records{};
    mutable luisa::compute::Buffer<luisa::float2> countBuffer = _device.create_buffer<float2>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
    mutable luisa::compute::Buffer<uint> countB = _device.create_buffer<uint>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
    mutable luisa::compute::Buffer<int> recordCB = _device.create_buffer<int>(1);

    luisa::compute::Buffer<unsigned> g3dCountBuffer = _device.create_buffer<unsigned>(2);
    luisa::compute::Buffer<float> selectionProbabilities = _device.create_buffer<float>(canvasSize.x * canvasSize.y);
    //std::unique_ptr<AdamOptimizer> selectionProbOptimizer = std::make_unique<AdamOptimizer>();
    luisa::compute::Buffer<float> polynomialParamsBuffer = _device.create_buffer<float>(MAX_G3D_COUNT * PARAMS_PER_G3D);
    //std::unique_ptr<AdamOptimizer> polynomialOptimizer = std::make_unique<AdamOptimizer>();

    //luisa::compute::Buffer<int> labelBuffer{};
    luisa::compute::Buffer<luisa::compute::AABB> componentBoundBuffer = _device.create_buffer<AABB>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);

    Kernel2D<> accountPotentialPointsKernel = [&]() {
        auto coord = make_int2(dispatch_x().cast<int>(), dispatch_y().cast<int>());
        //auto size = dispatch_size().xy();
        Int2 size = node<OwnkernelPathTracing>()->resolution();
        auto coord1D = coord.y * size.x + coord.x;
        auto count = countBuffer->read(coord1D);
       
        
        /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            auto rcon = recordCB->read(0);
            luisa::compute::device_log("rcon = {}", rcon);
            recordCB->write(0, 0);
        };*/
        
        labelBuffer->write(coord1D, -1);
        $if (count.x == 0.f) {
            $return();
        };
        auto record = records->read(coord1D);
        auto posz = record.position.z;
        auto accy = record.accumulation.y;
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("countx = {}, county = {}", count.x, count.y);
        };


        auto g3d = record.position / count.y;

        //acc.x: screen space [0,1] radius; acc.y: intensity
        auto acc = record.accumulation / count.x;
        auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(),
                                         coord.y.cast<float>() / size.y.cast<float>());

         $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("accx = {}, accy = {}", acc.x, acc.y);
        };


        $if (acc.x > RADIUS_THRESHOLD & acc.y > HIGHLIGHT_THRESHOLD) {
            labelBuffer->write(coord1D, 0);
        };

        auto label = labelBuffer->read(coord1D);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("label = {} id = {}", label, coord1D);
        };


    };
    luisa::compute::Shader2D<> accountPotentialPointsShader = _device.compile(accountPotentialPointsKernel);

    Kernel1D<> clearAABBKernel = [&]() {
        Var<AABB> ab{};
        ab.packed_max = {-9999999.f, -9999999.f, -9999999.f};
        ab.packed_min = {9999999.f, 9999999.f, 9999999.f};
        componentBoundBuffer->write(dispatch_x(), ab);
    };
    luisa::compute::Shader1D<> clearAABBShader = _device.compile(clearAABBKernel);

    Kernel2D<> generateAABBKernel = [&]() {
        auto coord = make_int2(dispatch_x().cast<int>(), dispatch_y().cast<int>());
       // auto size = dispatch_size().xy();
        Int2 size = node<OwnkernelPathTracing>()->resolution();
        auto coord1D = coord.y * size.x + coord.x;
        auto label = labelBuffer->read(coord1D);
        auto record = records->read(coord1D);
        auto count = countBuffer->read(coord1D);
        $if (label > 0) {
            auto g3d = record.position / count.y;
            auto radius = record.position.w / count.y;
            UInt g3dId = label - 1;
            auto boundAtomic = componentBoundBuffer->atomic(g3dId);
            boundAtomic.packed_max[0].fetch_max(g3d[0] + radius);
            boundAtomic.packed_min[0].fetch_min(g3d[0] - radius);

            boundAtomic.packed_max[1].fetch_max(g3d[1] + radius);
            boundAtomic.packed_min[1].fetch_min(g3d[1] - radius);

            boundAtomic.packed_max[2].fetch_max(g3d[2] + radius);
            boundAtomic.packed_min[2].fetch_min(g3d[2] - radius);
        };
    };
    luisa::compute::Shader2D<> generateAABBShader = _device.compile(generateAABBKernel);

    Kernel1D<> AABB2G3DKernel = [&]() {
        auto bound = componentBoundBuffer->read(dispatch_x());
        auto center = make_float3(
            (bound.packed_max[0] + bound.packed_min[0]) * 0.5f,
            (bound.packed_max[1] + bound.packed_min[1]) * 0.5f,
            (bound.packed_max[2] + bound.packed_min[2]) * 0.5f);
        auto size = length(make_float3(
                        (bound.packed_max[0] - bound.packed_min[0]),
                        (bound.packed_max[1] - bound.packed_min[1]),
                        (bound.packed_max[2] - bound.packed_min[2]))) *
                    0.5f;
        //size *= 0.5f;
        $if ((0u == luisa::compute::dispatch_x())) {
           // luisa::compute::device_log("center = {}, {}, {}, size = {}", center.x, center.y, center.z, size);
        };
        g3dBuffer->write(dispatch_x(), make_float4(center, size));
    };
    luisa::compute::Shader1D<> AABB2G3DShader = _device.compile(AABB2G3DKernel);
    std::vector<int> hostLabels;
    std::vector<DoFRecord> check;
    int f = 1;
    //luisa::compute::Device device;

    //const DeviceResource *resource;

    //int ConnectedComponents(std::vector<int> &buffer, unsigned width, unsigned height);

    float RADIUS_THRESHOLD = 1e-4f;
    float HIGHLIGHT_THRESHOLD = 1.f;
    /*#ifdef LIS_EXPERIMENT
    //For test purpose
    luisa::compute::Buffer<float> testRadianceBuffer{};
    luisa::compute::Buffer<unsigned> sampleCountBuffer{};
    std::optional<luisa::compute::UInt> radianceCoord{};
    unsigned dist_res = 128u;
#endif*/

    void prepare(uint2 size) {
        canvasSize = size;

        g3dBuffer = _device.create_buffer<float4>(MAX_G3D_COUNT * 2);
        g3dCountBuffer = _device.create_buffer<unsigned>(2);
        records = _device.create_buffer<DoFRecord>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
        countBuffer = _device.create_buffer<float2>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
        labelBuffer = _device.create_buffer<int>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);
        //hostLabels.resize(labelBuffer.size());
        componentBoundBuffer = _device.create_buffer<AABB>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);

        selectionProbabilities = _device.create_buffer<float>(size.x * size.y);
        //selectionProbOptimizer = std::make_unique<AdamOptimizer>();

        polynomialParamsBuffer = _device.create_buffer<float>(MAX_G3D_COUNT * PARAMS_PER_G3D);
        //polynomialOptimizer = std::make_unique<AdamOptimizer>();

        //Accel init
        /* auto stream = device->create_stream();
        stream << SharedKernels::Manager()->ZeroOut(g3dBuffer)
               << SharedKernels::Manager()->ZeroOut(g3dCountBuffer)
               << SharedKernels::Manager()->ZeroOut(countBuffer);
        //<<SharedKernels::Manager()->ZeroOut(defocusImageBuffer);
        stream << synchronize();

#ifdef LIS_EXPERIMENT
        //Experiment purpose
        testRadianceBuffer = device->create_buffer<float>(dist_res * dist_res);
        sampleCountBuffer = device->create_buffer<unsigned>(dist_res * dist_res);
        stream << SharedKernels::Manager()->ZeroOut(testRadianceBuffer) << SharedKernels::Manager()->ZeroOut(sampleCountBuffer) << synchronize();
        //------------------------------
#endif*/

        Kernel1D clearAABBKernel = [&]() {
            Var<AABB> ab{};
            ab.packed_max = {-9999999.f, -9999999.f, -9999999.f};
            ab.packed_min = {9999999.f, 9999999.f, 9999999.f};
            componentBoundBuffer->write(dispatch_x(), ab);
        };
        luisa::compute::Shader1D<> clearAABBShader = _device.compile(clearAABBKernel);
        Kernel2D generateAABBKernel = [&]() {
            auto coord = make_int2(dispatch_x().cast<int>(), dispatch_y().cast<int>());
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto coord1D = coord.y * size.x + coord.x;
            auto label = labelBuffer->read(coord1D);
            auto record = records->read(coord1D);
            auto count = countBuffer->read(coord1D);
            $if (label > 0) {
                auto g3d = record.position / count.y;
                auto radius = record.position.w / count.y;
                UInt g3dId = label - 1;
                auto boundAtomic = componentBoundBuffer->atomic(g3dId);
                boundAtomic.packed_max[0].fetch_max(g3d[0] + radius);
                boundAtomic.packed_min[0].fetch_min(g3d[0] - radius);

                boundAtomic.packed_max[1].fetch_max(g3d[1] + radius);
                boundAtomic.packed_min[1].fetch_min(g3d[1] - radius);

                boundAtomic.packed_max[2].fetch_max(g3d[2] + radius);
                boundAtomic.packed_min[2].fetch_min(g3d[2] - radius);
            };
        };
        generateAABBShader = _device.compile(generateAABBKernel);

        Kernel1D AABB2G3DKernel = [&]() {
            auto bound = componentBoundBuffer->read(dispatch_x());
            auto center = make_float3(
                (bound.packed_max[0] + bound.packed_min[0]) * 0.5f,
                (bound.packed_max[1] + bound.packed_min[1]) * 0.5f,
                (bound.packed_max[2] + bound.packed_min[2]) * 0.5f);
            auto size = length(make_float3(
                            (bound.packed_max[0] - bound.packed_min[0]),
                            (bound.packed_max[1] - bound.packed_min[1]),
                            (bound.packed_max[2] - bound.packed_min[2]))) *
                        0.5f;
            g3dBuffer->write(dispatch_x() , make_float4(center, size));
            
            };
        AABB2G3DShader = _device.compile(AABB2G3DKernel);

        Kernel2D accountPotentialPointsKernel = [&]() {
            auto coord = make_int2(dispatch_x().cast<int>(), dispatch_y().cast<int>());
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto coord1D = coord.y * size.x + coord.x;
            auto count = countBuffer->read(coord1D);
            labelBuffer->write(coord1D, -1);
            $if (count.x == 0.f) {
                $return();
            };
            auto record = records->read(coord1D);
            auto g3d = record.position / count.y;

            //acc.x: screen space [0,1] radius; acc.y: intensity
            auto acc = record.accumulation / count.x;
            auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(),
                                             coord.y.cast<float>() / size.y.cast<float>());

            $if (acc.x > RADIUS_THRESHOLD & acc.y > HIGHLIGHT_THRESHOLD) {
                labelBuffer->write(coord1D, 0);
            };
        };
        accountPotentialPointsShader = _device.compile(accountPotentialPointsKernel);
    }

    Float2 GetRasterPosition(/* const SceneNodeDesc *desc,*/ Float3 pos, UInt frame, UInt2 size, Float scale) const {

        Float2 rasterPos;
        Float focalLength = node<OwnkernelPathTracing>()->focal_length() * float(1e-3);
        Float aperture = node<OwnkernelPathTracing>()->aperture();
        auto focalLengthScale = 1.f / focalLength;
        focalLength = 1.f;
        aperture *= focalLengthScale;

        //auto sensorDistance = focalLength;
        //Float sensorDistance = 1.f / (1.f / node<OwnkernelPathTracing>()->focal_length() * float(1e-3) - 1 / node<OwnkernelPathTracing>()->focus_distance());
        //sensorDistance *= scale;
        auto fd = node<OwnkernelPathTracing>()->focus_distance();
        auto fl = node<OwnkernelPathTracing>()->focal_length() * 1e-3;
        Float sensorDistance = 1. / (1. / fl - 1. / fd);

        auto targetRatio = Float(size.x) / Float(size.y);
        auto currentRatio = aperture / aperture;
        $if (aperture > 0.f) {
            $if (currentRatio > targetRatio) {
                aperture = aperture * targetRatio;
            }
            $elif (currentRatio < targetRatio) {
                aperture = aperture / targetRatio;
            };
        }
        $else {
            $if (currentRatio > targetRatio) {
                aperture = aperture / targetRatio;
            }
            $elif (currentRatio < targetRatio) {
                aperture = aperture * targetRatio;
            };
        };
        auto origin = node<OwnkernelPathTracing>()->position();
        auto direction = node<OwnkernelPathTracing>()->look_at();
        auto horizontal = make_float3(1.f, 0.f, 0.f);
        auto vertical = make_float3(0.f, 1.f, 0.f);

        auto v = pos - origin;
        auto distanceToPlane = dot(v, direction);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("raster = {}, {}, {}", distanceToPlane, aperture, sensorDistance);
        };
        //Direction is inversed
        auto y = dot(v, vertical) / distanceToPlane * sensorDistance / (aperture * .5f);
        auto x = dot(v, horizontal) / distanceToPlane * sensorDistance / (aperture * .5f);
        rasterPos.y = (y * .5f + .5f);
        rasterPos.x = (x * .5f + .5f);
        return rasterPos;
    }

    UInt fnv1a_hash(const UInt &num) {
        const uint32_t fnv_prime = 16777619u;
        const uint32_t offset_basis = 2166136261u;

        UInt hash = offset_basis;
        for (int i = 0; i < 4; ++i) {
            hash ^= (num >> (i * 8)) & 0xFF;// Process byte by byte
            hash *= fnv_prime;
        }

        return hash;
    }

    Float EvaluatePolynomial(const Float2 &uv, const BufferView<float> &params, const UInt &offset) const {
        Float result = 0.f; // Constant term (degree 0)
        auto index = offset;// Start from the first parameter after the constant term

        // Evaluate terms of increasing degrees
        result += params->read(index);
        index += 1u;
        for (unsigned degree = 1; degree <= POLYNOMIAL_DEGREE; ++degree) {
            // Iterate over terms of the current degree
            for (unsigned i = 0; i <= degree; ++i) {
                unsigned j = degree - i;// j + i = degree (i.e., x^i * y^j)
                Float term = params->read(index) * pow(uv.x, float(i)) * pow(uv.y, float(j));
                result += term;
                index += 1u;
            }
        }

        return result;
    }

    Float EvaluatePolynomial(const Float2 &uv, const ArrayFloat<PARAMS_PER_G3D> &params, const UInt &offset) const {
        Float result = 0.f; // Constant term (degree 0)
        auto index = offset;// Start from the first parameter after the constant term

        // Evaluate terms of increasing degrees
        result += params[index];
        index += 1u;
        for (unsigned degree = 1; degree <= POLYNOMIAL_DEGREE; ++degree) {
            // Iterate over terms of the current degree
            for (unsigned i = 0; i <= degree; ++i) {
                unsigned j = degree - i;// j + i = degree (i.e., x^i * y^j)
                Float term = params[index] * pow(uv.x, float(i)) * pow(uv.y, float(j));
                result += term;
                index += 1u;
            }
        }

        return result;
    }

     Float G3DPdf(const Float3 &dir, G3DMixture<COMPONENT_COUNT> &g3ds, Int &usedComponent) const{
        Float pdf = 0.f;
        Float maxPdf = 0.f;
        for (int i = 0; i < COMPONENT_COUNT; i++) {
            auto p = g3ds.gaussian[i]->pdf_directional(dir);
            $if (p > 1e-1f & maxPdf < p) {
                usedComponent = i;
                maxPdf = p;
            };
            pdf += g3ds.weights[i] * p;
        }
        return pdf;
    }

    int ConnectedComponents(std::vector<int> &buffer, unsigned int width, unsigned int height) const{
        int label = 1;                                           // Start labeling from 1, assuming -1 is for non-masked pixels and 0 for unvisited masked pixels
        std::vector<int> directions = {-1, 0, 1, 0, 0, -1, 0, 1};// Directions for 4-neighborhood: left, right, up, down

        // Lambda function to perform a BFS for labeling the connected component
        auto bfs = [&](int x, int y) {
            std::queue<std::pair<int, int>> q;
            q.push({x, y});
            buffer[y * width + x] = label;// Set the initial point to the current label

            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();

                // Explore the 4-neighborhood (left, right, up, down)
                for (int d = 0; d < 4; ++d) {
                    int nx = cx + directions[d * 2];
                    int ny = cy + directions[d * 2 + 1];

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && buffer[ny * width + nx] == 0) {
                        buffer[ny * width + nx] = label;// Label the pixel
                        q.push({nx, ny});               // Add to the queue to explore its neighbors
                    }
                }
            }
        };

        auto bfs_isotropic = [&](int x, int y) {
            std::queue<std::pair<int, int>> q;
            q.push({x, y});
            buffer[y * width + x] = label;// Set the initial point to the current label

            // Track the bounding box of the component
            int min_x = x, max_x = x;
            int min_y = y, max_y = y;

            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();

                // Explore the 4-neighborhood (left, right, up, down)
                for (int d = 0; d < 4; ++d) {
                    int nx = cx + directions[d * 2];
                    int ny = cy + directions[d * 2 + 1];

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && buffer[ny * width + nx] == 0) {
                        // Calculate new bounding box if we add this pixel
                        int new_min_x = std::min(min_x, nx);
                        int new_max_x = std::max(max_x, nx);
                        int new_min_y = std::min(min_y, ny);
                        int new_max_y = std::max(max_y, ny);

                        // Calculate the width and height of the bounding box
                        int width_component = new_max_x - new_min_x + 1;
                        int height_component = new_max_y - new_min_y + 1;

                        // Check if the component is becoming too elongated (e.g., width/height ratio)
                        if (std::max(width_component, height_component) <= std::min(width_component, height_component) + 1) {
                            // If the component is still "square-like", continue expanding
                            buffer[ny * width + nx] = label;// Label the pixel
                            q.push({nx, ny});               // Add to the queue to explore its neighbors

                            // Update the bounding box
                            min_x = new_min_x;
                            max_x = new_max_x;
                            min_y = new_min_y;
                            max_y = new_max_y;
                        }
                    }
                }
            }
        };
        // Iterate over all pixels in the buffer
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                if (buffer[y * width + x] == 0) {
                    // Found an unvisited masked pixel
                    if (false/* polynomialFitting*/) {
                        bfs(x, y);// Perform BFS to label the entire component
                    } else {
                        bfs_isotropic(x, y);
                    }
                    ++label;// Increment the label for the next component
                }
            }
        }
        return label - 1;
    }
    
    void postsampling( CommandBuffer &command_buffer, unsigned int i) {
         constexpr float selectionProbabilityLr = 2e-1f;
        //Var<uint> BI= BUFFER_ITER;
         
         hostLabels.resize(labelBuffer.size(), 0);
             
         

         
         //auto center = dispatch_size().xy() / 2u;
         check.resize(records.size());
         command_buffer << records.copy_to(check.data()) << synchronize();
             //CommandList cmds{};
            //std::vector<float> sel_copy{selectionProbabilityLr};
            //selectionProbOptimizer->Step(selectionProbabilityLr);
            command_buffer << accountPotentialPointsShader().dispatch(SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
            command_buffer << commit();
            
            command_buffer << labelBuffer.copy_to(hostLabels.data()) << synchronize();
            //std::cout << "host: " << hostLabels[32896 + X] << std::endl;
            /* if (i == 400u) {
                saveMatrixToFile(hostLabels, "label.txt");
            }*/
            
            unsigned int componentCount = 0;
            componentCount = ConnectedComponents(hostLabels, SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
            //std::cout << "label: " << hostLabels[32896 + X] << " com:" << componentCount << std::endl;
            //std::string filename = "label'_" + std::to_string(i) + ".txt";
            //std::cout << filename << std::endl;
            //saveMatrixToFile(hostLabels, filename);
             command_buffer
                << labelBuffer.copy_from(hostLabels.data())
                << clearAABBShader().dispatch(componentBoundBuffer.size())
                << generateAABBShader().dispatch(SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES)
                << AABB2G3DShader().dispatch(componentCount)
                << g3dCountBuffer.view(0, 1).copy_from(&componentCount)
                << synchronize();

            /* command_buffer << countB.copy_to(hostLabels.data()) << synchronize();
             unsigned int checkcom = 0;
             checkcom = ConnectedComponents(hostLabels, SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
             command_buffer << labelBuffer.copy_from(hostLabels.data()) << synchronize();*/
             //logger::message("gaussians: {}", componentCount);
         
         /* $else {
            //selectionProbOptimizer->Step(selectionProbabilityLr);
            if (polynomialFitting) {
                //polynomialOptimizer->Step((sampleIdx - BUFFER_ITER < 2) ? 0.f : (1e-2f * expf(static_cast<float>(sampleIdx - BUFFER_ITER) * -1e-2f)));
            }
        };*/
        //experiment

   }

   

    void InKernelFinalize(const UInt sampleIdx, const Float3 &sample, Var<Ray> &camera_ray, shared_ptr<Interaction> first) const {

        auto energy = clamp(reduce_sum(sample) * 0.3333333f, 0.f, 200.f);
        Bool positionRecorded = false;
#ifdef LIS_EXPERIMENT
        //Experiment
        $if (all(dispatch_id().xy() == TARGET_PIXEL)) {
            sampleCountBuffer->atomic(*radianceCoord).fetch_add(1u);
            testRadianceBuffer->atomic(*radianceCoord).fetch_max(energy);
        };
#endif

        $if (positionRecorded) {
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto d = length(camera_ray->origin() - first->p());
            auto scale = (d - node<OwnkernelPathTracing>()->focus_distance()) / (d * (node<OwnkernelPathTracing>()->focus_distance() - node<OwnkernelPathTracing>()->focal_length() * float(1e-3)));
            auto diameter = abs(scale * (node<OwnkernelPathTracing>()->focal_length() * float(1e-3) * node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / node<OwnkernelPathTracing>()->aperture()));
            auto radius = diameter * 0.5f;
            auto radiusRatio = constants::pi * sqr(radius) / (node<OwnkernelPathTracing>()->aperture() * node<OwnkernelPathTracing>()->aperture());

            auto screenSpaceRadius = radius / node<OwnkernelPathTracing>()->aperture();

            auto coord = dispatch_id().xy();
            auto coord1D = coord.y * size.x + coord.x;
            auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(), coord.y.cast<float>() / size.y.cast<float>());

            $if (!polynomialFitting | sampleIdx < BUFFER_ITER) {
                auto posScreen = GetRasterPosition(first->p(), 0, size, scale);
                Float energyWeight = 1.f;
                {
                    energyWeight = exp(-16.f * length(normalizedPos - posScreen));
                }
                posScreen = posScreen * 0.8f + 0.1f;
                auto distance = length(first->p() - camera_ray->origin());
                auto pointToBounce = normalize(first->p() - camera_ray->origin());
                auto cosTheta0 = dot(node<OwnkernelPathTracing>()->look_at(), pointToBounce);
                auto width = ((node<OwnkernelPathTracing>()->aperture() / node<OwnkernelPathTracing>()->focal_length() * float(1e-3)) * d * cosTheta0) / float(SCREEN_SPACE_RECORD_RES);

                $if (screenSpaceRadius > RADIUS_THRESHOLD & all(posScreen > 0.f) & all(posScreen < 1.f)) {
                    auto recordId = (posScreen.x * SCREEN_SPACE_RECORD_RES).cast<uint>() + (posScreen.y * SCREEN_SPACE_RECORD_RES).cast<uint>() * SCREEN_SPACE_RECORD_RES;

                    countBuffer->atomic(recordId).x.fetch_add(energyWeight);
                    countBuffer->atomic(recordId).y.fetch_add(energy);
                    Float3 position = first->p() * energy;
                    auto atomicRef = records->atomic(recordId);
                    atomicRef.accumulation.x.fetch_add(screenSpaceRadius * energyWeight);
                    atomicRef.accumulation.y.fetch_add(energy * energyWeight);
                    atomicRef.position.x.fetch_add(position.x);
                    atomicRef.position.y.fetch_add(position.y);
                    atomicRef.position.z.fetch_add(position.z);
                    atomicRef.position.w.fetch_add(width * energy);
                };
            }
            $else {
                if (polynomialFitting) {
                    auto offset = (*selectedComponent) * PARAMS_PER_G3D;
                    $if (*selectedComponent > -1 & energy > 1.f) {
                        ArrayFloat<PARAMS_PER_G3D> params{};
                        ArrayFloat<PARAMS_PER_G3D> gradiants{};
                        for (int i = 0; i < PARAMS_PER_G3D; i++) {
                            params[i] = polynomialParamsBuffer->read(offset + i);
                        }
                        auto g3d = g3dBuffer->read(*selectedComponent);
                        Var<Gaussian3D> g;
                        normalizedPos -= make_float2(.5f);
                        $autodiff {
                            requires_grad(params);
                            g.mu[0] = g3d[0] + EvaluatePolynomial(normalizedPos, params, 0);
                            g.mu[1] = g3d[1] + EvaluatePolynomial(normalizedPos, params, POLYNOMIAL_PARAMS_PER_OUTPUT(2));
                            g.mu[2] = g3d[2] + EvaluatePolynomial(normalizedPos, params, POLYNOMIAL_PARAMS_PER_OUTPUT(2) * 2);
                            g.sigma = g3d.w * WIDTH_SCALE;

                            //neural network
                            //auto sigmaScale
                        };
                    };
                }
            };

            $if (energy > 0.f & pdfRecords->y > 0.f) {
                auto linearIndex = dispatch_x() + dispatch_y() * dispatch_size_x();
                auto parameter = pdfRecords->z;
                auto alpha = 1.0f / (1.0f + exp(-parameter));
                auto x = pdfRecords->x;
                auto y = pdfRecords->y;
                auto eval_pdf = x + alpha * (y - x);
                auto loss = -energy * log(eval_pdf);

                //Machine learning
            };
        };
    }
    /*
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_render_one_camera(command_buffer, camera);
    }*/

    //add
    void _sample_thin_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_sample_thin_camera(command_buffer, camera);
    }

    //base
    [[nodiscard]] Float3 Li_COC(const Camera::Instance *camera, Expr<uint> frame_index,
                                Expr<uint2> pixel_id, Expr<float> time) const noexcept {

        uint2 size = camera->film()->node()->resolution();
        Bool positionRecorded = false;

        //Float3 origin = node<OwnkernelPathTracing>()->position();
        auto direction = node<OwnkernelPathTracing>()->look_at();
        auto horizontal = make_float3(1.f, 0.f, 0.f);
        auto vertical = make_float3(0.f, 1.f, 0.f);
        auto aperture = node<OwnkernelPathTracing>()->aperture();
        
        std::optional<luisa::compute::Int> selectedComponent{};
        std::optional<luisa::compute::Float3> pdfRecords{};
        pdfRecords.emplace(make_float3(0.f));

        selectedComponent.emplace(-1);
        //gecerate_camera
        auto tragetRatio = Float(size.x) / Float(size.y);
        auto currentRatio = 1.f;

        //no crop

        //auto rasterPos = GetRasterPosition(first->p(), 0, size);

        //generate camera ray
        //tryTimes.emplace(1.f);
        

        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        Bool positionrec = false;
        //records = _device.create_buffer<DoFRecord>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);

        
        //Float3 origin = camera_ray->origin();
        Float3 origin = node<OwnkernelPathTracing>()->position();
        auto first = pipeline().geometry()->intersect(camera_ray);

        auto d = length(origin - first->p());
        auto scale = (d - node<OwnkernelPathTracing>()->focus_distance()) / (d * (node<OwnkernelPathTracing>()->focus_distance() - node<OwnkernelPathTracing>()->focal_length() * float(1e-3)));
        auto diameter = abs(scale * (node<OwnkernelPathTracing>()->focal_length() * float(1e-3) * node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / node<OwnkernelPathTracing>()->aperture()));
        //auto radius = diameter * 0.5f;
        auto coord = dispatch_id().xy();
        auto coord1D = coord.y * size.x + coord.x;
        auto con = countBuffer->read(coord1D);

        auto lab = labelBuffer->read(coord1D);

        auto accp = records->read(coord1D);
/*
        Var<float3> pos = make_float3(0.f);
        auto judge = countB->read(coord1D);
        $if (judge > 0) {
            auto g3df = g3dBuffer->read(judge - 1);
            auto g3dCenterf = g3df.xyz();
            auto g3dRadiusf = g3df.w;
            pos = g3dCenterf;
        };

        
        //pos = node<OwnkernelPathTracing>()->look_at();
        $if (con.y != 0) {
            pos = accp.position.xyz() / con.y;
        };
        
        Float3 po = make_float3(0.f, 0.f, -5.f);

        $if (lab > 0) {
            Var<int> cho = lab - 1;
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("cho = {}", cho);
            };
            auto g3df = g3dBuffer->read(cho);
            auto g3dCenterf = g3df.xyz();
            auto g3dRadiusf = g3df.w;
            pos = g3dCenterf;
            
            
        };*/

        
        auto resolution = make_float2(node<OwnkernelPathTracing>()->resolution());
        Float resox = resolution.x;
        Float resoy = resolution.y;
        Float2 pixel_offset = .5f * resolution;
        auto pix = make_float2(coord.x.cast<Float>(), coord.y.cast<Float>());
        auto coord_focal = (pix - pixel_offset);
        auto pos = node<OwnkernelPathTracing>()->position() + make_float3(coord_focal, 3.f);
        
        

        //weight
        auto rasterPos = GetRasterPosition(pos, 0, size, scale);
        rasterPos = rasterPos - .5f;
        rasterPos *= aperture;
        //auto sensorDistance = node<OwnkernelPathTracing>()->focal_length() * float(1e-3);
        //Float sensorDistance = 1.f / (1.f / node<OwnkernelPathTracing>()->focal_length() * float(1e-3) - 1 / node<OwnkernelPathTracing>()->focus_distance());
        //sensorDistance *= scale;
        auto fd = node<OwnkernelPathTracing>()->focus_distance();
        auto fl = node<OwnkernelPathTracing>()->focal_length() * 1e-3;
        Float sensorDistance = 1. / (1. / fl - 1. / fd);
        Float weight = 1.f;
        Var<Ray> g_ray; 
        Float3 pssSample = normalize(make_float3(0.5f));
        Float3 dir = make_float3(0.f);
        
        const auto focusDir = normalize(direction * sensorDistance + rasterPos.x * horizontal + rasterPos.y * vertical);
        const auto focusDistance = node<OwnkernelPathTracing>()->focus_distance() / dot(focusDir, direction);
        const auto focusPoint = origin + focusDir * focusDistance;
        const auto lensRadius = node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / (2.f * node<OwnkernelPathTracing>()->aperture());

         $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
             luisa::compute::device_log("pos = {}, {}, {}", rasterPos.x, rasterPos.y, focusDistance);
        };



        auto g3dCount = g3dCountBuffer->read(0);
        G3DMixture<COMPONENT_COUNT> g3ds{};
        ArrayVar<int, COMPONENT_COUNT> g3dIds{};
        Var<uint> usedG3ds = 0u;
        auto cosThetaRef = abs(dot(focusDir,normalize(focusPoint - (origin + vertical * lensRadius))));
        auto theta0 = acos(cosThetaRef);
        
        $for (i, g3dCount) {
            /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("g3Dc = {}", g3dCount);
            };*/
            auto g3d = g3dBuffer->read(i);
            auto g3dCenter = g3d.xyz();
            auto g3dRadius = g3d.w;
            


            auto pointTo = g3dCenter - focusPoint;
            auto d = length(pointTo);
            auto v = pointTo / d;

            auto cosTheta = abs(dot(v, focusDir));
            auto theta = acos(cosTheta);

            auto sinPsi = g3dRadius / d;
            auto cosPsi = ite(sinPsi >= 1.f, 0.f, sqrt(1.f - sqr(sinPsi)));
            auto psi = acos(cosPsi);
            auto thetaMin = theta - psi;

            $if (max(0.f, thetaMin) <= theta0) {
                Var<uint> g = usedG3ds;
                usedG3ds += 1;
                $if (g >= COMPONENT_COUNT) {
                    g = (sampler()->generate_1d() * usedG3ds);
                };
                $if (g < COMPONENT_COUNT) {
                    g3dIds[g] = i;
                    g3ds.gaussian[g]->mu = g3dCenter;
                    g3ds.gaussian[g]->sigma = g3dRadius * WIDTH_SCALE;
                };
            };
        };
        usedG3ds = min(COMPONENT_COUNT, usedG3ds);
        int z = 1;
        $if (usedG3ds > 0u) {
            //printf("Hello g3d\n");
            //auto coord = dispatch_id().xy();
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            //auto coord1D = coord.y * size.x + coord.x;
            auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(), coord.y.cast<float>() / size.y.cast<float>());
            Float compo_weight = 1.f / usedG3ds.cast<float>();
            $for (i, usedG3ds) {
                g3ds.weights[i] = compo_weight;
                //polynomial fitting
            };
            $for (i, COMPONENT_COUNT - usedG3ds) {
                g3ds.gaussian[usedG3ds + i].mu = make_float3(5.f);
                g3ds.gaussian[usedG3ds + i].sigma = 1.f;
                g3ds.weights[usedG3ds + i] = 0.f;
            };
            g3ds.construct_directional(focusPoint);
            Float pdf = 0.f;

            auto linearIndex = dispatch_x() + dispatch_y() * dispatch_size_x();
            Float selectionProbability = 0.8f;
            Float selectionProbabilityRaw = 0.f;

            //if fixed
            /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("camera_dir = {}, {}, {}", camera_ray->direction().x, camera_ray->direction().y, camera_ray->direction().z);
                luisa::compute::device_log("camera_origin = {}, {}, {}", camera_ray->origin().x, camera_ray->origin().y, camera_ray->origin().z);
                luisa::compute::device_log("camera_weight = {}", camera_weight);
            };*/

            const Float uniformPdf = 1.f / (lensRadius * lensRadius * constants::pi);
            $while (true) {
               Bool sampleG3d = sampler()->generate_1d() < selectionProbability;
                
                $if (sampleG3d) { 
                    Float3 sampledDir;
                    Float pdf_g3d = 1.f;
                    Float sam = sampler()->generate_1d();
                    Var<uint> ran = (sam * 1000);
                    g3ds.sample_directional(ran, sampledDir, pdf_g3d);

                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("gaussian = {}, {}, {}", g3ds.gaussian[0].mu[0], g3ds.gaussian[0].mu[1], g3ds.gaussian[0].mu[2]);
                        //luisa::compute::device_log("origin = {}, {}, {}", origin.x, origin.y, origin.z);
                        //luisa::compute::device_log("weight = {}", weight);
                    };

                    Int usedId = -1;
                    pdf_g3d = G3DPdf(sampledDir, g3ds, *selectedComponent);
                    auto pp = G3DPdf(-sampledDir, g3ds, usedId);
                    $if (pp > pdf_g3d) {
                        *selectedComponent = usedId;
                    };
                    pdf_g3d += pp;

                    auto cosTheta = dot(sampledDir, direction);
                    $if (cosTheta < 0.f) {
                        cosTheta *= -1.f;
                        sampledDir *= -1.f;
                    };
                    auto distance = node<OwnkernelPathTracing>()->focus_distance() / cosTheta;
                    origin = focusPoint - sampledDir * distance;
                    auto adjust = dispatch_size().xy() / 2u;
                    auto pixel_id = make_float3(coord.x.cast<float>() - adjust.x.cast<float>(), coord.y.cast<float>() - adjust.y.cast<float>(), 0.f) * sensorDistance;
                    auto pixel_center = node<OwnkernelPathTracing>()->position();
                    auto offsetVector = pixel_center - origin;
                    auto pointDistance = length(offsetVector);
                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("offset = {}, {}, {}", offsetVector.x, offsetVector.y, offsetVector.z);
                        luisa::compute::device_log("origin = {}, {}, {}", origin.x, origin.y, origin.z);
                        luisa::compute::device_log("distance = {}", pointDistance);
                    };
                    dir = sampledDir;
                    
                    $if (pointDistance <= lensRadius) {
                        //experiment
                        
                        pdf = (pdf_g3d * abs(cosTheta)) / (distance * distance);
                        pdfRecords->y = pdf;
                        pdfRecords->x = uniformPdf;
                        pdfRecords->z = selectionProbabilityRaw;
                        pdf = lerp(uniformPdf, pdf, selectionProbability);
                        $if (*selectedComponent > -1) {
                            *selectedComponent = g3dIds[*selectedComponent];
                        };
                        //weight = uniformPdf / pdf;
                        g_ray = make_ray(origin, dir);
                        
                         $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                            luisa::compute::device_log("CQC");
                            //luisa::compute::device_log("dir = {}, {}, {}", dir.x, dir.y, dir.z);
                            //luisa::compute::device_log("first = {}, {}, {}", first->p().x, first->p().y, first->p().z);
                            //luisa::compute::device_log("weight = {}, pdf = {}, unifiom = {}", weight, pdf, uniformPdf);
                        };



                        $break;
                    };
                    //trytimes
                }
                $else {
                    LUISA_INFO("coming.");
                    /* Float2 sample = sampler()->generate_2d();
                    Float2 c = sample_uniform_disk_concentric(sample);*/
                    origin = camera_ray->origin();
                    dir = camera_ray->direction();

                    Int usedId = -1;
                    auto pdf_g3d = G3DPdf(dir, g3ds, *selectedComponent);
                    auto pp = G3DPdf(-dir, g3ds, usedId);
                    $if (pp > pdf_g3d) {
                        *selectedComponent = usedId;
                    };
                    pdf_g3d += pp;

                    auto cosTheta = dot(focusDir, dir);
                    auto distance = focusDistance / cosTheta;
                    pdf = (pdf_g3d * abs(cosTheta)) / (distance * distance);
                    pdfRecords->y = pdf;
                    pdfRecords->x = uniformPdf;
                    pdfRecords->z = selectionProbabilityRaw;
                    pdf = lerp(uniformPdf, pdf, selectionProbability);

                    $if (*selectedComponent > -1) {
                        *selectedComponent = g3dIds[*selectedComponent];
                    };
                    //experiment
                    //weight = camera_weight;
                    g_ray = camera_ray;

                     $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("CU");
                        //luisa::compute::device_log("dir = {}, {}, {}", dir.x, dir.y, dir.z);
                        //luisa::compute::device_log("first = {}, {}, {}", first->p().x, first->p().y, first->p().z);
                        //luisa::compute::device_log("weight = {}, pdf = {}, unifiom = {}", weight, pdf, uniformPdf);
                    };

                    $break;
                };
            };
            weight = uniformPdf / pdf;
            
            
            
        }
        $else {
            
             pdfRecords->y = 0.f;
            /* Float2 sample = sampler()->generate_2d();
            Float pdf2d = 1.f;
            Float2 c = sample_uniform_disk_concentric(sample);
            weight = 1.f / pdf2d;
            origin += lensRadius * horizontal * c.x + lensRadius * vertical * c.y;
            dir = normalize(focusPoint - origin);
            g_ray = make_ray(origin, dir);*/
            
            g_ray = camera_ray;
            weight = camera_weight;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("U");
                //luisa::compute::device_log("dir = {}, {}, {}", dir.x, dir.y, dir.z);
                //luisa::compute::device_log("first = {}, {}, {}", first->p().x, first->p().y, first->p().z);
                //luisa::compute::device_log("weight = {}, pdf = {}, unifiom = {}", weight, pdf, uniformPdf);
            };
            //experiment
        };
        
        //if内

        lensPdf.emplace(1.f / weight);
        
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        //auto g_ray;
        //SampledSpectrum beta{swl.dimension(), camera_weight};
        //SampledSpectrum Li{swl.dimension()};
        /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("weight = {}", weight);
        };*/
        SampledSpectrum beta{swl.dimension(), weight};
        SampledSpectrum Li{swl.dimension()};
        //result_buf->write(0, weight);

         $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("camera_dir = {}, {}, {}", camera_ray->direction().x, camera_ray->direction().y, camera_ray->direction().z);
                //luisa::compute::device_log("id = {}, camera_origin = {}, {}, {}",coord1D, camera_ray->origin().x, camera_ray->origin().y, camera_ray->origin().z);
                //luisa::compute::device_log("camera_weight = {}", camera_weight);
            };


        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("dir = {}, {}, {}", g_ray->direction().x, g_ray->direction().y, g_ray->direction().z);
                //luisa::compute::device_log("origin = {}, {}, {}", g_ray->origin().x, g_ray->origin().y, g_ray->origin().z);
               //luisa::compute::device_log("weight = {}", weight);
        };


        auto ray = g_ray;
        auto g_first = pipeline().geometry()->intersect(g_ray);
        auto pdf_bsdf = def(1e16f);

        $for (depth, node<OwnkernelPathTracing>()->max_depth()) {

            /* $if (radius * 100000000000000000000000000000000000.f > 0.00000000000001f) {
               
                $break;
            };*/

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if (!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };
            positionRecorded = true;
            // hit light
            if (!pipeline().lights().empty()) {
                $outline {
                    $if (it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {

                            //luisa::compute::device_log("hit {}, {}, {}", it->p().x, it->p().y, it->p().z);
                        };
                    };
                };
            }

            $if (!it->shape().has_surface()) { $break; };

            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();

            auto u_rr = def(0.f);
            auto rr_depth = node<OwnkernelPathTracing>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // generate uniform samples
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                // sample one light
                light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);
            };

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);

            $outline {
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                call.execute([&](const Surface::Closure *closure) noexcept {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if (*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if (light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch (surface_sample.event) {
                        $case (Surface::event_enter) { eta_scale = sqr(eta); };
                        $case (Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                });
            };

            beta = zero_if_any_nan(beta);
            $if (beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<OwnkernelPathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if (depth + 1u >= rr_depth) {
                $if (q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        //InKernelFinalize(pixel_id.x , Li, camera_ray, first);
        Float3 sample = spectrum->srgb(swl, Li);
        auto energy = clamp(reduce_sum(sample) * 0.3333333f *  (*lensPdf), 0.f, 200.f);
        //auto rad = (sample.x + sample.y + sample.z) / 3.f;

        /*
        Var<float> gf = 100;
        auto object_to_sensor_ratio = (fd / sensorDistance);
        auto resolution = make_float2(node<OwnkernelPathTracing>()->resolution());
        Float resox = resolution.x;
        Float resoy = resolution.y;
        Float2 pixel_offset = .5f * resolution;
        auto projected_pixel_size =
            
                
                 // portrait mode
                 min((object_to_sensor_ratio * .024f / resox),
                     (object_to_sensor_ratio * .036f / resoy));
        auto pix = make_float2(coord.x.cast<Float>(), coord.y.cast<Float>());
        auto coord_focal = (pix - pixel_offset) * projected_pixel_size;
        auto p_focal = make_float3(coord_focal.x, -coord_focal.y, fd);
        $for (i, g3dCount) {
            auto g3d = g3dBuffer->read(i);
            auto g3dCenter = g3d.xyz();
            auto g3dRadius = g3d.w;
            auto rpos = make_float2(g3dCenter.x, g3dCenter.y);
            auto p_focal2 = make_float2(p_focal.x, p_focal.y);
            auto nearg = rpos - p_focal2;
            auto gd = length(nearg);
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("first = {}, {}, {}", g_first->p().x, g_first->p().y, g_first->p().z);
                luisa::compute::device_log("gd/lR= {} / {}", gd,  lensRadius);
            };
            $if (gd < (lensRadius * (1.f + sensorDistance) ) & gd < gf) {
                countB->write(coord1D, i + 1u);
                gf = gd;
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    
                    luisa::compute::device_log("in");
                };
            };

        };*/
        
        




        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("first = {}, {}, {}", g_first->p().x, g_first->p().y, g_first->p().z);
            luisa::compute::device_log("energy/valid = {} / {}", energy, g_first->valid());
        };

         $if (g_first->valid()) {
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto d = length(ray->origin() - g_first->p());
            auto scale = (d - node<OwnkernelPathTracing>()->focus_distance()) / (d * (node<OwnkernelPathTracing>()->focus_distance() - node<OwnkernelPathTracing>()->focal_length() * float(1e-3)));
            auto diameter = abs(scale * (node<OwnkernelPathTracing>()->focal_length() * float(1e-3) * node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / node<OwnkernelPathTracing>()->aperture()));
            auto radius = diameter * 0.5f;
            auto radiusRatio = constants::pi * sqr(radius) / (node<OwnkernelPathTracing>()->aperture() * node<OwnkernelPathTracing>()->aperture());

            auto screenSpaceRadius = radius / node<OwnkernelPathTracing>()->aperture();

            //auto coord = dispatch_id().xy();
            //auto coord1D = coord.y * size.x + coord.x;
            /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("id = {}", coord1D);
            };*/
            auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(), coord.y.cast<float>() / size.y.cast<float>());

            $if (pixel_id.x < SCREEN_SPACE_RECORD_RES) {
                auto posScreen = GetRasterPosition(g_first->p(), 0, size, scale );
                Float energyWeight = 1.f;
                {
                    energyWeight = exp(-16.f * length(normalizedPos - posScreen));
                }
                posScreen = posScreen * 0.8f + 0.1f;
                auto distance = length(g_first->p() - origin);
                auto pointToBounce = normalize(g_first->p() - camera_ray->origin());
                auto forward = node<OwnkernelPathTracing>()->look_at() - node<OwnkernelPathTracing>()->position() / length(node<OwnkernelPathTracing>()->look_at() - node<OwnkernelPathTracing>()->position());
                auto cosTheta0 = dot(forward, pointToBounce);
                auto width = ((node<OwnkernelPathTracing>()->aperture() / node<OwnkernelPathTracing>()->focal_length() * float(1e-3)) * d * cosTheta0) / float(SCREEN_SPACE_RECORD_RES);

                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("screenRadius = {}, posS = {}, {}", screenSpaceRadius, posScreen.x, posScreen.y);
                };

                $if (screenSpaceRadius > RADIUS_THRESHOLD & all(posScreen > 0.f) & all(posScreen < 1.f)) {
                    auto recordId = (posScreen.x * SCREEN_SPACE_RECORD_RES).cast<uint>() + (posScreen.y * SCREEN_SPACE_RECORD_RES).cast<uint>() * SCREEN_SPACE_RECORD_RES;
                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("record = {}", recordId);
                    };
                    countBuffer->atomic(recordId).x.fetch_add(energyWeight);
                    countBuffer->atomic(recordId).y.fetch_add(energy);
                    Float3 position = g_first->p() ;
                    /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("pos.x = {}, pos.y = {}, pos.z = {}, rad = {}", position.x, position.y, position.z, rad);
                    };*/
                    auto atomicRef = records->atomic(recordId);
                    //atomicRef.accumulation.x.fetch_add(screenSpaceRadius * energyWeight);
                    atomicRef.accumulation.x.fetch_add(screenSpaceRadius * energyWeight);
                    //atomicRef.accumulation.y.fetch_add(energy * energyWeight);
                    atomicRef.accumulation.y.fetch_add(energy * energyWeight);
                    atomicRef.position.x.fetch_add(position.x * energy);
                    atomicRef.position.y.fetch_add(position.y * energy);
                    atomicRef.position.z.fetch_add(position.z * energy);
                    atomicRef.position.w.fetch_add(width * energy);
                    
                    recordCB->atomic(0).fetch_add(1);
                    Li = Li ;
                };
            };
            /* Stream stream = _device.create_stream();
            constexpr float selectionProbabilityLr = 2e-1f;
            Var<uint> BI = BUFFER_ITER;
            $if ( sampleId < BI) {
                CommandList cmds{};
                //std::vector<float> sel_copy{selectionProbabilityLr};
                //selectionProbOptimizer->Step(selectionProbabilityLr);
                cmds << accountPotentialPointsShader().dispatch(SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
                stream << cmds.commit();
                
                stream << labelBuffer.copy_to(hostLabels.data()) << synchronize();
               unsigned componentCount = ConnectedComponents(hostLabels, SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES);
                stream
                    << labelBuffer.copy_from(hostLabels.data())
                    << clearAABBShader().dispatch(componentBoundBuffer.size())
                    << generateAABBShader().dispatch(SCREEN_SPACE_RECORD_RES, SCREEN_SPACE_RECORD_RES)
                    << AABB2G3DShader().dispatch(componentCount)
                    << g3dCountBuffer.view(0, 1).copy_from(&componentCount)
                    << synchronize();
            };*/
            // auto count = countBuffer->read(coord1D);
            //countBuffer->write(coord1D, make_float2(count.x + 1.f, count.y));
            /* #ifdef LIS_EXPERIMENT
        //Experiment
        $if (all(dispatch_id().xy() == TARGET_PIXEL)) {
            sampleCountBuffer->atomic(*radianceCoord).fetch_add(1u);
            testRadianceBuffer->atomic(*radianceCoord).fetch_max(energy);
        };
#endif*/
        };
        
        Var<DoFRecord> ref;
        ref.position = make_float4(first->p(), 0.f);
        auto L = spectrum->srgb(swl, Li);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("L = {}, {}, {}", L.x, L.y, L.z);
            //luisa::compute::device_log("origin = {}, {}, {}", g_ray->origin().x, g_ray->origin().y, g_ray->origin().z);
            //luisa::compute::device_log("weight = {}", weight);
        };
        /* Float valuex = first->p().x;
        Float valuey = first->p().y;
        Float valuez = first->p().z;*/
        //auto coord = dispatch_id().xy();
        //auto coord1D = coord.y * size.x + coord.x;
        
        
        return L;
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    //add
    void Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time)  {

        if (f == 1)
        {
            hostLabels.resize(labelBuffer.size());
            f = 2;
        }

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("Li start");
        };

        uint2 size = node<OwnkernelPathTracing>()->resolution();
        Var<uint2> a = make_uint2(pixel_id);
        Var<uint> coordx = a.x.cast<uint>();
        Var<uint> coordy = pixel_id.y;
        Bool positionRecorded = false;
        Var<uint> sampleId = coordx + coordy * size.x;
        //std::vector<int> hostLabels;
        //hostLabels.resize(labelBuffer.size());
        //Accel init
        /* auto stream = device->create_stream();
        stream << SharedKernels::Manager()->ZeroOut(g3dBuffer)
               << SharedKernels::Manager()->ZeroOut(g3dCountBuffer)
               << SharedKernels::Manager()->ZeroOut(countBuffer);
        //<<SharedKernels::Manager()->ZeroOut(defocusImageBuffer);
        stream << synchronize();

#ifdef LIS_EXPERIMENT
        //Experiment purpose
        testRadianceBuffer = device->create_buffer<float>(dist_res * dist_res);
        sampleCountBuffer = device->create_buffer<unsigned>(dist_res * dist_res);
        stream << SharedKernels::Manager()->ZeroOut(testRadianceBuffer) << SharedKernels::Manager()->ZeroOut(sampleCountBuffer) << synchronize();
        //------------------------------
#endif*/

        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        Bool positionrec = false;
        //records = _device.create_buffer<DoFRecord>(SCREEN_SPACE_RECORD_RES * SCREEN_SPACE_RECORD_RES);


        auto g3dCount = g3dCountBuffer->read(0);
        uint usedg3ds = 0u;

        auto first = pipeline().geometry()->intersect(camera_ray);
        auto d = length(camera_ray->origin() - first->p());
        auto scale = (d - node<OwnkernelPathTracing>()->focus_distance()) / (d * (node<OwnkernelPathTracing>()->focus_distance() - node<OwnkernelPathTracing>()->focal_length() * float(1e-3)));
        auto diameter = abs(scale * (node<OwnkernelPathTracing>()->focal_length() * float(1e-3) * node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / node<OwnkernelPathTracing>()->aperture()));
        auto radius = diameter * 0.5f;

        CommandList;

        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);

        $for (depth, node<OwnkernelPathTracing>()->max_depth()) {

            /* $if (radius * 100000000000000000000000000000000000.f > 0.00000000000001f) {
               
                $break;
            };*/

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if (!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };
            positionRecorded = true;
            // hit light
            if (!pipeline().lights().empty()) {
                $outline {
                    $if (it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                };
            }

            $if (!it->shape().has_surface()) { $break; };

            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();

            auto u_rr = def(0.f);
            auto rr_depth = node<OwnkernelPathTracing>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // generate uniform samples
            auto light_sample = LightSampler::Sample::zero(swl.dimension());
            $outline {
                // sample one light
                light_sample = light_sampler()->sample(
                    *it, u_light_selection, u_light_surface, swl, time);
            };

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);

            $outline {
                PolymorphicCall<Surface::Closure> call;
                pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                call.execute([&](const Surface::Closure *closure) noexcept {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if (*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if (light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.shadow_ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    beta *= w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch (surface_sample.event) {
                        $case (Surface::event_enter) { eta_scale = sqr(eta); };
                        $case (Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                });
            };

            beta = zero_if_any_nan(beta);
            $if (beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<OwnkernelPathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if (depth + 1u >= rr_depth) {
                $if (q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        //InKernelFinalize(pixel_id.x , Li, camera_ray, first);
        Float3 sample = spectrum->srgb(swl, Li);
        auto energy = clamp(reduce_sum(sample) * 0.3333333f, 0.f, 200.f);

        
        $if (true) {
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto d = length(camera_ray->origin() - first->p());
            auto scale = (d - node<OwnkernelPathTracing>()->focus_distance()) / (d * (node<OwnkernelPathTracing>()->focus_distance() - node<OwnkernelPathTracing>()->focal_length() * float(1e-3)));
            auto diameter = abs(scale * (node<OwnkernelPathTracing>()->focal_length() * float(1e-3) * node<OwnkernelPathTracing>()->focal_length() * float(1e-3) / node<OwnkernelPathTracing>()->aperture()));
            auto radius = diameter * 0.5f;
            auto radiusRatio = constants::pi * sqr(radius) / (node<OwnkernelPathTracing>()->aperture() * node<OwnkernelPathTracing>()->aperture());

            auto screenSpaceRadius = radius / node<OwnkernelPathTracing>()->aperture();

            auto coord = dispatch_id().xy();
            auto coord1D = coord.y * size.x + coord.x;
            auto normalizedPos = make_float2(coord.x.cast<float>() / size.x.cast<float>(), coord.y.cast<float>() / size.y.cast<float>());

            $if (pixel_id.x < SCREEN_SPACE_RECORD_RES) {
                auto posScreen = GetRasterPosition(first->p(), 0, size, scale);
                Float energyWeight = 1.f;
                {
                    energyWeight = exp(-16.f * length(normalizedPos - posScreen));
                }
                posScreen = posScreen * 0.8f + 0.1f;
                auto distance = length(first->p() - camera_ray->origin());
                auto pointToBounce = normalize(first->p() - camera_ray->origin());
                auto cosTheta0 = dot(node<OwnkernelPathTracing>()->look_at(), pointToBounce);
                auto width = ((node<OwnkernelPathTracing>()->aperture() / node<OwnkernelPathTracing>()->focal_length() * float(1e-3)) * d * cosTheta0) / float(SCREEN_SPACE_RECORD_RES);

                $if (screenSpaceRadius > RADIUS_THRESHOLD & all(posScreen > 0.f) & all(posScreen < 1.f)) {
                    //auto recordId = (posScreen.x * SCREEN_SPACE_RECORD_RES).cast<uint>() + (posScreen.y * SCREEN_SPACE_RECORD_RES).cast<uint>() * SCREEN_SPACE_RECORD_RES;

                    countBuffer->atomic(sampleId).x.fetch_add(energyWeight);
                    countBuffer->atomic(sampleId).y.fetch_add(energy);
                    Float3 position = first->p() * energy;
                    
                    auto atomicRef = records->atomic(sampleId);
                    atomicRef.accumulation.x.fetch_add(screenSpaceRadius * energyWeight);
                    atomicRef.accumulation.y.fetch_add(energy * energyWeight);
                    atomicRef.position.x.fetch_add(position.x);
                    atomicRef.position.y.fetch_add(position.y);
                    atomicRef.position.z.fetch_add(position.z);
                    atomicRef.position.w.fetch_add(width * energy);
                    auto count = countBuffer->read(coord1D);
                    countBuffer->write(coord1D, make_float2(count.x + 1.f, count.y));

                };
                
                

            };
           
            
            


        };
        
    }

    void show_value2()
    {
        float host_result = 0 ;
        //result_buf.copy_to(&host_result);
        std::cout << "Result: " << host_result << std::endl;
        float2 res;
        countBuffer.copy_to(&res);
        std::cout << "Result: " << res.x << res.y << std::endl;


    }

    void set0(CommandBuffer& command_buffer, uint2 resolution)
    {
        Kernel2D set0kernel = [&]() noexcept {
            auto coord = make_int2(dispatch_x().cast<int>(), dispatch_y().cast<int>());
            //auto size = dispatch_size().xy();
            Int2 size = node<OwnkernelPathTracing>()->resolution();
            auto coord1D = coord.y * size.x + coord.x;
            countB->write(coord1D, 0u);
        };
        auto shader = _device.compile(set0kernel);
        command_buffer << shader().dispatch(resolution)
                       << synchronize();


    }

    void saveMatrixToFile(const std::vector<int> &matrix, const std::string &filename) {
        const int ROWS = SCREEN_SPACE_RECORD_RES;
        const int COLS = SCREEN_SPACE_RECORD_RES;

        if (matrix.size() != ROWS * COLS) {
            //std::cerr << "エラー: ベクターのサイズが256x256ではありません。" << std::endl;
            return;
        }

        std::ofstream outFile(filename);// ← ここでファイルが無ければ作成される
        if (!outFile.is_open()) {
            //std::cerr << "ファイルを開けませんでした: " << filename << std::endl;
            return;
        }

        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                outFile << matrix[i * COLS + j];
                if (j != COLS - 1) {
                    outFile << " ";
                }
            }
            outFile << "\n";
        }
        //outFile << matrix[32896];

        outFile.close();
        std::cout << "ファイルに保存されました: " << filename << std::endl;
    }

    /* void save_binary_image(const std::vector<uint8_t> &binary_data, const std::string &filename) {
        const int width = 256;
        const int height = 256;

        if (binary_data.size() != width * height) {
            throw std::runtime_error("binary_data size must be 256 * 256.");
        }

        std::vector<uint8_t> image(width * height);

        for (int i = 0; i < width * height; ++i) {
            image[i] = binary_data[i] ? 255 : 0;// 1 → 白, 0 → 黒
        }

        // 書き出し（グレースケール 1 チャンネル PNG）
        stbi_write_png(filename.c_str(), width, height, 1, image.data(), width);
    }*/

    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera)  noexcept override {

        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer << compute::synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;

        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
           
            //auto L = Li_COC(camera, frame_index, pixel_id, time);
            //camera->film()->accumulate(pixel_id, shutter_weight * L);

        };
        
         Kernel2D render_kernel2 = [&](UInt frame_index, Float time, Float shutter_weight, BufferVar<DoFRecord> records , BufferVar<float2> countbuffer) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            //out.write(0, pixel_id.y);
            auto L = Li_COC(camera, frame_index, pixel_id, time);
            

            //auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        };

        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto render2 = pipeline().device().compile(render_kernel2);
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering override.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        Buffer<uint> output = _device.create_buffer<uint>(16);
        std::vector<uint> host(16);
        for (auto s : shutter_samples) {
            //std::cout << "out cycle "  << std::endl;
            pipeline().update(command_buffer, s.point.time);
            set0(command_buffer, resolution);
            for (auto i = 0u; i < s.spp; i++) {
                
                command_buffer << render2(sample_id++, s.point.time, s.point.weight, records.view() , countBuffer.view())
                                      .dispatch(resolution)
                               << output.copy_to(host.data());
                command_buffer << synchronize();


                
                postsampling(command_buffer, i);
                
                
                std::cout << "Result: " << host[0] << std::endl;
                dispatch_count++;
                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
                auto dispatches_per_commit = 4u;
                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
        command_buffer << synchronize();
        progress.done();
        //result_buf->write(0, dispatch_size_x);
      
        auto render_time = clock.toc();
        
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }






};


luisa::unique_ptr<Integrator::Instance> OwnkernelPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<OwnkernelPathTracingInstance>(
        pipeline, command_buffer, this);
}


}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::OwnkernelPathTracing)
