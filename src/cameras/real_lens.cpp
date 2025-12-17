
#include <dsl/rtx/ray.h>
#include <util/sampling.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>
#include <iostream>

constexpr auto X = 0;
constexpr auto Y = 50;

struct Lenselement {
    float curvanature;
    float thick;
    float refraction;
    float diameter;
};

LUISA_STRUCT(Lenselement,
             curvanature, thick, refraction, diameter) {};



namespace luisa::render {

using namespace luisa::compute;

class RealLensCamera : public Camera {

private:
    float _aperture;
    float _focal_length;
    float _focus_distance;
    vector<float> _curvanature;
    vector<float> _thick;
    vector<float> _refra_index;
    vector<float> _aperture_diameter;
    int _lens_count;
    float _fov;

public:
    RealLensCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _aperture{desc->property_float_or_default("aperture", 2.f)},
          _focal_length{desc->property_float_or_default("focal_length", 35.f)},
          _curvanature{desc->property_float_list("curvanature")},
          _thick{desc->property_float_list("thick")},
          _refra_index{desc->property_float_list("index_refraction")},
          _aperture_diameter{desc->property_float_list("aperture_diameter")},
          _fov{radians(std::clamp(desc->property_float_or_default("fov", 35.0f), 1e-3f, 180.f - 1e-3f))},
          _lens_count{desc->property_int("lens_count")},
          _focus_distance{desc->property_float_or_default(
              "focus_distance", lazy_construct([desc] {
                  auto target = desc->property_float3("look_at");
                  auto position = desc->property_float3("position");
                  return length(target - position);
              }))} {
        _focus_distance = std::max(std::abs(_focus_distance), 1e-4f);
        
        
        

    }
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return true; }
    [[nodiscard]] auto aperture() const noexcept { return _aperture; }
    [[nodiscard]] auto focal_length() const noexcept { return _focal_length; }
    [[nodiscard]] auto focus_distance() const noexcept { return _focus_distance; }
    [[nodiscard]] auto curvanature() const noexcept { return _curvanature; }
    [[nodiscard]] auto thick() const noexcept { return _thick; }
    [[nodiscard]] auto refra() const noexcept { return _refra_index; }
    [[nodiscard]] auto apdi() const noexcept { return _aperture_diameter; }
    [[nodiscard]] auto lens_count() const noexcept { return _lens_count; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
    
};

struct RealLensCameraData {
    float2 pixel_offset;
    float2 resolution;
    float focus_distance;
    float lens_radius;
    float projected_pixel_size;
    int mode;
    float tan_half_fov;
};



}// namespace luisa::render

LUISA_STRUCT(luisa::render::RealLensCameraData,
             pixel_offset, resolution, focus_distance,
             lens_radius, projected_pixel_size, mode, tan_half_fov) {};

namespace luisa::render {

class RealLensCameraInstance : public Camera::Instance {

private:
    BufferView<RealLensCameraData> _device_data;
    Device &_device = pipeline().device();
    luisa::compute::Buffer<float> curvanature = _device.create_buffer<float>(20);
    luisa::compute::Buffer<float> thick = _device.create_buffer<float>(20);
    luisa::compute::Buffer<float> refraction = _device.create_buffer<float>(20);
    luisa::compute::Buffer<float> diameter = _device.create_buffer<float>(20);

public:
    explicit RealLensCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const RealLensCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _device_data{ppl.arena_buffer<RealLensCameraData>(1u)}
          {
        auto v = camera->focus_distance();
        auto f = camera->focal_length() * 1e-3f;
        auto u = 1.f / (1.f / f - 1.f / v);// 1 / f = 1 / v + 1 / sensor_plane
        auto object_to_sensor_ratio = static_cast<float>(v / u);
        auto lens_radius = static_cast<float>(.5 * f / camera->aperture());
        auto resolution = make_float2(camera->film()->resolution());
        auto pixel_offset = .5f * resolution;
        auto projected_pixel_size =
            resolution.x > resolution.y ?
                // landscape mode
                min(static_cast<float>(object_to_sensor_ratio * .036 / resolution.x),
                    static_cast<float>(object_to_sensor_ratio * .024 / resolution.y)) :
                // portrait mode
                min(static_cast<float>(object_to_sensor_ratio * .024 / resolution.x),
                    static_cast<float>(object_to_sensor_ratio * .036 / resolution.y));
        
        auto mode = 0;
        if (resolution.x > resolution.y) {
            mode = 1;
        } 
       
        auto cuv = camera->curvanature();
        auto tik = node<RealLensCamera>()->thick();
        auto ir = node<RealLensCamera>()->refra();
        auto ad = node<RealLensCamera>()->apdi();
        const auto lc = node<RealLensCamera>()->lens_count();
        
       

        
        
        for (int i = 0; i < lc; ++i) {
            cuv[i] = cuv[i] * 1e-3f;
            tik[i] = tik[i] * 1e-3f;
            ad[i] = ad[i] * 1e-3f;
        }

        
       
        
       
        
        RealLensCameraData host_data{pixel_offset, resolution, v,
                                     lens_radius, projected_pixel_size, mode, tan(camera->fov() * 0.5f)};



        command_buffer << _device_data.copy_from(&host_data)
                       << curvanature.copy_from(cuv.data())
                       << thick.copy_from(tik.data())
                       << refraction.copy_from(ir.data())
                       << diameter.copy_from(ad.data())
                       << commit();
    }

    //get front Z
    Float LensFrontZ( int lenscount) const {
       Float zsum = 0.f;
        0.f;
       $for (i, lenscount) {
           zsum += thick->read(i);
           
           $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
               //luisa::compute::device_log("real_element = {}, {}, {}", i2, zsum, lenscount);
           };
       };
        return zsum;
    };

    //get rear Z
    Float LensRearZ(Var<int> statecount) {
        Int sc = 0;
        $for (i, statecount) {
            sc += 1;
        };
        return thick->read(sc);
    };

    //get rear apurture
    Float LensRearRadius(Var<int> statecount) {
        Int sc = 0;
        $for (i, statecount) {
            sc += 1;
        };
        return diameter->read(sc);
    };

    Var<bool> IntersectSphericalElement(Float radius, Float zCenter, const Var<Ray> &ray, Float *t, Float3 *n) const{

        Var<bool> hit = true;
        Float3 sphy_origin = ray->origin() - make_float3(0.f, 0.f, zCenter);
        Float A = dot(ray->direction(), ray->direction());
        Float B = 2.f * dot(ray->direction(), sphy_origin);
        Float C = dot(sphy_origin, sphy_origin) - (radius * radius);
        Float t0, t1;

        //Quadratic
        Float D = (B * B) - (4.f * A * C);
        Var<bool> hanbetsu = (D < 0.f);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(0,0) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("real_han = {} {} {} {}", zCenter, A, B, C);
        };
        $if (hanbetsu) {
            hit = false;
        }
        $else {
            t0 = (-B - sqrt(D)) / (2.f * A);
            t1 = (-B + sqrt(D)) / (2.f * A);

            Var<bool> useCloserT = (ray->direction().z > 0) ^ (radius < 0);
            //select(false, true, bool)
            *t = select(max(t0, t1), min(t0, t1), useCloserT);
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(0,0) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("solution = {}, {}", t0, t1);
            };
            $if(*t < 0.f) {
                hit = false;
            } 
            $else {
                *n = sphy_origin + (*t) * ray->direction();
                *n = select(-*n, *n, dot(*n, -ray->direction()) < 0.f);
            };
        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(0,0) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("real_hit = {}", hit);
        };
        return hit;
    };

    Var<bool> Refract(const Float3 &wi, const Float3 &n, Float ir, Float3 *wt) const {
        Var<bool> refraction = true;
        Float cosThetaI = dot(normalize(n), normalize(wi));
        Float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
        Float sin2ThetaT = ir * ir * sin2ThetaI;
        Float cosThetaT = 0.f;
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("real_sin2 = {}", sin2ThetaT);
            luisa::compute::device_log("real_wt = {}, {}, {}, {}, {}", ir, wi, cosThetaI, cosThetaT, n);
        };
        $if (sin2ThetaT >= 1) {
            refraction = false;
        } 
        $else {
            cosThetaT = sqrt(1 - sin2ThetaT);
            *wt = ir * -wi + (ir * cosThetaI - cosThetaT) * n;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_wt = {}, {}, {}, {}, {}", ir, wi, cosThetaI, cosThetaT, n);
                luisa::compute::device_log("real_wt2 = {}, {}", (ir * cosThetaI - cosThetaT), (ir * cosThetaI - cosThetaT) * n);
            };
        };

        return refraction;
    }

    Var<float3> sample_cosine_hemisphere_fov(Var<float2> u, Float fov_rad) const noexcept {

        static Callable impl = [fov_rad](Var<float2> u) noexcept {
            // FOV îºäpÇÃ cos
            Var<float> cos_theta_max = cos(fov_rad * 0.5f);

            // z ÇÃåvéZÅiãtCDFÅj
            Var<float> z = sqrt(1.0f - u.y * (1.0f - cos_theta_max * cos_theta_max));

            // îºåa r
            Var<float> r = sqrt(max(1.0f - z * z, 0.0f));

            // É”
            Var<float> phi = 2.0f * 3.14159265359f * u.x;

            // x, y
            Var<float> x = r * cos(phi);
            Var<float> y = r * sin(phi);

            return make_float3(x, y, z);
        };

        return impl(u);
    }




    [[nodiscard]] std::pair<Var<Ray>, Float> _generate_ray_in_camera_space(Expr<float2> pixel,
                                                                           Expr<float2> u_lens,
                                                                           Expr<float> /* time */) const noexcept override {
        
        const auto lc = node<RealLensCamera>()->lens_count();
        Int misspoint = 0;

        auto data = _device_data->read(0u);
        Float SumZ = LensFrontZ(lc);
        auto sceneX = .024f;
        auto sceneY = .024f;
        $if(data.mode == 1) {
            sceneX = .024f;
            sceneY = .024f;
        };

        Float weight = 1.f;
        auto p = (pixel * 2.0f - data.resolution) * (data.tan_half_fov / data.resolution.y);
        auto direction = normalize(make_float3(p.x, -p.y, -1.f));
        //pinhole
        //auto ray = make_ray(make_float3(), direction);
        // //auto first_ray = ray;

        Int hanbetsu = 0;
        Float2 resolution = data.resolution;

        Float coordX = (pixel.x - data.pixel_offset.x) * sceneX / resolution.x;
        Float coordY = (pixel.y - data.pixel_offset.y) * sceneY / resolution.y;
        auto coordScene = make_float3(coordX, -coordY, 0.f);

        Float fov =  pi / 4.f;
        auto coord_d = sample_cosine_hemisphere_fov(u_lens, fov);
        auto scene_d = normalize(make_float3(coord_d.xy(), -coord_d.z));


        uint tes = 1u;
        uint tes2 = 2u;
        auto test = "test";

        uint tes3 = select(tes, tes2, false);

        $if(luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("camera_dir = {}, {}, {}", normalize(p_focal - p_lens).x, normalize(p_focal - p_lens).y, normalize(p_focal - p_lens).z);
            //luisa::compute::device_log("real_element = {}, {}", tik[1], tes3);
            luisa::compute::device_log("real_pro = {},{},{}", coordScene, scene_d, tes3);
        };

        auto test_d = make_float3(0.f, 0.f, -1.f);
        
        //sceme
        auto ray = make_ray(coordScene, scene_d);
        auto first_ray = make_ray(coordScene, scene_d);
        
        //real system
        Float elementZ = 0;

        $for (i, lc) {
            Int index = lc - i - 1;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("start_loop {}", i);
                luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), diameter->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };

            elementZ -= thick->read(index);
            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isStop = (curvanature->read(index) == 0.f);
            $if (isStop) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
            }
            $else {
                Float radius = curvanature->read(index);
                Float zCenter = elementZ + curvanature->read(index);
                
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), hanbetsu);
                
            };


            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > diameter->read(index) * diameter->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            

            Float3 new_origin = phit;
            //update ray

            Float3 w;
            $if (!isStop) {
                Float etaI = refraction->read(index);
                Float etaT = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0) {
                        etaT = refraction->read(index - 1);
                    };
                };
                

                
                $if (!Refract(normalize(-ray->direction()), normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };
            };

           

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("real_w = {}", w);
            };
            w = normalize(w);
            ray = make_ray(new_origin, make_float3(w.xy(), w.z));
/* */
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                luisa::compute::device_log("endloop {}", i);
            };
        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("weight {}", weight);
        };
       
        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            weight = 0.f;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("false_ray = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("true_ray {}", weight);
            };
            ray = make_ray(ray->origin(), make_float3(-ray->direction().xy(), ray->direction().z));
        };
       
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("camera_ray = {}, {}", first_ray->origin(), first_ray->direction());
        };
        
        return std::make_pair(std::move(ray), weight);
    }
};

luisa::unique_ptr<Camera::Instance> RealLensCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<RealLensCameraInstance>(
        pipeline, command_buffer, this);
}

using ClipPlaneRealLensCamera = ClipPlaneCameraWrapper<
    RealLensCamera, RealLensCameraInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ClipPlaneRealLensCamera)