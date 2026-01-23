
#include <dsl/rtx/ray.h>
#include <util/sampling.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>
#include <iostream>

//#define LIS_EXPERIMENT

constexpr auto X = 0;
constexpr auto Y = 0;
constexpr auto CHAIN = 20;
constexpr unsigned RESOLUTION = 256;
constexpr unsigned MAX_I = 32;
constexpr float solver_threshold = 1e-4f;
constexpr float step_scale = 1.f;
constexpr float length_threshold = 1e-4f;
constexpr float angle_threshold = 1e-3f;

struct BB2D {
    luisa::compute::Float2 packed_min;
    luisa::compute::Float2 packed_max;
};
struct Lenselement {
    float curvanature;
    float thick;
    float refraction;
    float radius;
};

LUISA_STRUCT(Lenselement,
             curvanature, thick, refraction, radius) {};


struct ChainVerts {
    //point
    luisa::compute::float3 point;
    //normal
    luisa::compute::float3 n;
    //index
    int index;
    luisa::compute::float3 center;

    //base
    float u;
    float v;

    luisa::compute::float3 dp_du;
    luisa::compute::float3 dp_dv;

    //tangent 
    luisa::compute::float3 s;
    luisa::compute::float3 t;
    luisa::compute::float3 ds_du;
    luisa::compute::float3 ds_dv;
    luisa::compute::float3 dt_du;
    luisa::compute::float3 dt_dv;
     
    
    // Used in multi-bounce version
    luisa::compute::float2 C;
    
    luisa::compute::float2x2 dC_dx_prev;
    luisa::compute::float2x2 dC_dx_cur;
    luisa::compute::float2x2 dC_dx_next;

    //other
    luisa::compute::float2x2 tmp;
    luisa::compute::float2x2 inv_lambda;
    luisa::compute::float2 dx;
    
    
  
};

LUISA_STRUCT(ChainVerts,
             point, n, index, center, u, v, dp_du, dp_dv, s, t, ds_du, ds_dv, dt_du, dt_dv, C, dC_dx_prev, dC_dx_cur, dC_dx_next, tmp, inv_lambda, dx) {};

struct ME_element {
    luisa::compute::float3 first;
    luisa::compute::float3 emit;
    float sign;
    int use;
};

LUISA_STRUCT(ME_element,
             first, emit, sign, use) {};

namespace luisa::render {

using namespace luisa::compute;





class RealLensCamera : public Camera {

private:
    
    vector<float> _curvanature;
    vector<float> _thick;
    vector<float> _refra_index;
    vector<float> _aperture_diameter;
    int _lens_count;
    float _fov;

public:
    RealLensCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _curvanature{desc->property_float_list("curvanature")},
          _thick{desc->property_float_list("thick")},
          _refra_index{desc->property_float_list("index_refraction")},
          _aperture_diameter{desc->property_float_list("aperture_diameter")},
          _fov{radians(std::clamp(desc->property_float_or_default("fov", 35.0f), 1e-3f, 180.f - 1e-3f))},
          _lens_count{desc->property_int("lens_count")}
    { }

    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return true; }
   
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
    luisa::compute::Buffer<float> curvanature = _device.create_buffer<float>(CHAIN);
    luisa::compute::Buffer<float> thick = _device.create_buffer<float>(CHAIN);
    luisa::compute::Buffer<float> refraction = _device.create_buffer<float>(CHAIN);
    luisa::compute::Buffer<float> radius = _device.create_buffer<float>(CHAIN);
    luisa::compute::Buffer<ChainVerts[CHAIN]> vertex = _device.create_buffer<ChainVerts[CHAIN]>(RESOLUTION * RESOLUTION);
    luisa::compute::Buffer<ME_element> element = _device.create_buffer<ME_element>(RESOLUTION * RESOLUTION);

public:
    explicit RealLensCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const RealLensCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _device_data{ppl.arena_buffer<RealLensCameraData>(1u)}
          {
        auto v = 1.f;
        auto object_to_sensor_ratio = 1.f;
        auto lens_radius = 1.f;
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
            ad[i] = ad[i] * 1e-3f * 0.5f;
        }

       //Float fd = Focusdistance();
       //dlt = FocusThickLens(3.f);
        
       
        
        RealLensCameraData host_data{pixel_offset, resolution, v,
                                     lens_radius, projected_pixel_size, mode, tan(camera->fov() * 0.5f)};



        command_buffer << _device_data.copy_from(&host_data)
                       << curvanature.copy_from(cuv.data())
                       << thick.copy_from(tik.data())
                       << refraction.copy_from(ir.data())
                       << radius.copy_from(ad.data())
                       << commit();
    }


    void Clear_path(Var<ChainVerts[CHAIN]> path) const {
        $for (i, CHAIN) {
            path[i].point = make_float3(0.f);
            path[i].n = make_float3(0.f);
            path[i].center = make_float3(0.f);
            path[i].index = 0.f;
            path[i].u = 0.f;
            path[i].v = 0.f;
            path[i].dx = make_float2(0.f);
        };
    }

    //get front Z
    Float LensFrontZ( Var<int> lenscount) const {
       Float zsum = 0.f;
        0.f;
       $for (i, lenscount) {
           zsum += thick->read(i);
           
           $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
               //luisa::compute::device_log("real_element = {}, {}, {}", i2, zsum, lenscount);
           };
       };
        return -zsum;
    };

    //get rear Z
    Float LensRearZ(int statecount) const {
        /* Int sc = 0;
        $for (i, statecount) {
            sc += 1;
        };*/
        return -thick->read(statecount);
    };

    //get rear apurture
    Float LensRearRadius(int statecount) const {
        /* Int sc = 0;
        $for (i, statecount) {
            sc += 1;
        };*/
        return radius->read(statecount);
    };

    Var<bool> IntersectSphericalElement(Float radius, Float zCenter, const Var<Ray> &ray, Float *t, Float3 *n, Bool film) const{

        Var<bool> hit = true;
        Float3 sphy_origin = ray->origin() - make_float3(0.f, 0.f, zCenter);
        Float A = dot(ray->direction(), ray->direction());
        Float B = 2.f * dot(ray->direction(), sphy_origin);
        Float C = dot(sphy_origin, sphy_origin) - (radius * radius);
        Float t0, t1;

        //Quadratic
        Float D = (B * B) - (4.f * A * C);
        Var<bool> hanbetsu = (D < 0.f);
#ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X,Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("real_han = {} {} {} {}", zCenter, A, B, C);
        };
#endif

        
        $if (hanbetsu) {
            hit = false;
        }
        $else {
            t0 = (-B - sqrt(D)) / (2.f * A);
            t1 = (-B + sqrt(D)) / (2.f * A);

            Var<bool> useCloserT = (ray->direction().z > 0) ^ (radius < 0);
            //select(false, true, bool)
            *t = select(max(t0, t1), min(t0, t1), useCloserT);


            $if (film) {
                $if (t0 > 0) {
                    *t = t0;
                }
                $elif (t1 > 0) {
                    *t = t1;
                }
                $else {
                    // Both intersections are behind the ray
                    *t = -1.f;
                };
            }
            $else {
                $if (t1 > 0) {
                    *t = t0;
                }
                $elif (t0 > 0) {
                    *t = t1;
                }
                $else {
                    // Both intersections are behind the ray
                    *t = -1.f;
                };
            };
            
            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X,Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("solution = {}, {}", t0, t1);
            };
            #endif
            $if(*t < 0.f) {
                hit = false;
            } 
            $else {
                *n = sphy_origin + (*t) * ray->direction();
                *n = select(*n, -*n, dot(*n, -ray->direction()) < 0.f);
            };
        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X,Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("real_hit = {}", hit);
        };
        return hit;
    };

    Var<bool> Refract(const Float3 &wi, Float3 *n, Float ir, Float3 *wt) const {

        Float3 nn = *n;
        $if (dot(wi, *n) < 0.f) {
            nn = -nn;
            *n = nn;

        };
        Var<bool> refraction = true;
        Float cosThetaI = dot(normalize(nn), normalize(wi));
        Float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
        Float sin2ThetaT = ir * ir * sin2ThetaI;
        Float cosThetaT = 0.f;

        #ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("real_sin2 = {}", sin2ThetaT);
            luisa::compute::device_log("real_wt = {}, {}, {}, {}, {}", ir, wi, cosThetaI, cosThetaT, n);
        };
        #endif

        $if (sin2ThetaT >= 1) {
            refraction = false;
        } 
        $else {
            cosThetaT = sqrt(1 - sin2ThetaT);
            *wt = ir * -wi + (ir * cosThetaI - cosThetaT) * nn;

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_wt = {}, {}, {}, {}, {}", ir, wi, cosThetaI, cosThetaT, n);
                luisa::compute::device_log("real_wt2 = {}, {}", (ir * cosThetaI - cosThetaT), (ir * cosThetaI - cosThetaT) * n);
            };
            #endif
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

    Var<float3> sample_uniform_disk(Var<float2> u, Float radius, Float z_plane) const noexcept {

        static Callable impl = [](Var<float2> u, Var<float> R, Var<float> z) noexcept {
            Var<float> r = R * sqrt(u.y);// ñ êœàÍól
            Var<float> phi = 2.0f * 3.14159265359f * u.x;

            Var<float> x = r * cos(phi);
            Var<float> y = r * sin(phi);

            return make_float3(x, y, z);
        };

        return impl(u, radius, z_plane);
    }

    Float det(const Float2x2 &A) const {
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }

    Float2x2 inverse(const Float2x2 &A) const {
        Float d = det(A);
        // åƒÇ—èoÇµë§Ç≈ d Ç™ 0 Ç≈Ç»Ç¢Ç±Ç∆Çï€èÿÇ∑ÇÈëOíÒ
        Float inv_d = 1.f / d;
        return make_float2x2(
            A[1][1] * inv_d, -A[0][1] * inv_d,
            -A[1][0] * inv_d, A[0][0] * inv_d);
    }

    auto invert (const Float2x2 &A, Float2x2 &Ainv) const {
        Float determinant = det(A);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            
            //luisa::compute::device_log("det {}", determinant);
        };
        Bool invert = true;
        $if (abs(determinant) == 0) {
            invert = false;
        };
        Ainv = inverse(A);
        return invert;
    };

    Bool TraceLencesFromFilm(const Var<Ray> &rCamera, Var<Ray>* rOut) const{
        Float elementZ = 0.f;
        const Int lc = node<RealLensCamera>()->lens_count();
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;

        #ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("rCamera {}, {}", ray->origin(), ray->direction());
            
        };
        #endif

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("start_loop_function {}", i);
                //luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), radius->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };
            #endif

            
           
            elementZ -= thick->read(index);
            
            $if (refraction->read(index) == 0.f) {
                $continue;
            };

           
            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isPlane = (curvanature->read(index) == 0.f);
            $if (isPlane) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
                normal = make_float3(0.f, 0.f, 1.f);
            }
            $else {
                Float radius = curvanature->read(index);
                zCenter = elementZ + curvanature->read(index);

                #ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("intersect {}", !IntersectSphericalElement(radius, zCenter, ray, &t, &normal));
                    
                };
                #endif

                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), isStop);
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
            #endif

            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(phit - make_float3(0.f, 0.f, zCenter));

            Float3 new_origin = phit;
            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaI = refraction->read(index);
                Float etaT = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0) {
                        etaT = refraction->read(index - 1);
                    };
                };
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("check_ref= {},{}", etaI, etaT);
                };
                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };


                #ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("real_w = {}", w);
                };
                #endif

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("check_ray= {},{}", ray->origin(), ray->direction());
            };
            /* */
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                //luisa::compute::device_log("endloop {}", i);
            };
            

        };

        

        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            trace = false;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("false_from_film = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            trace = true;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("true_from_film = {}, {}", ray->origin(), ray->direction());
            };
            *rOut = make_ray(ray->origin(), make_float3(ray->direction().xy(), ray->direction().z));
        };

        return trace;
    }

    Bool TraceLencesFromSceneD(const Var<Ray> &rCamera, Var<Ray> *rOut, const Bool ignore) const {
        
        const Int lc = node<RealLensCamera>()->lens_count();
        Float elementZ = LensFrontZ(lc);
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("rCamera {}, {}", ray->origin(), ray->direction());
        };

        $for (i, lc) {
            Int index = i;
            Float zCenter = 0.f;


            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("start_loop_function {}", i);
                //luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), radius->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };

            
            $if (ignore) {
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("ignore");
                };
                $if (refraction->read(index) == 0.f) {
                    elementZ += thick->read(index);
                    $continue;
                };
            };
            


            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isPlane = (curvanature->read(index) == 0.f);
            $if (isPlane) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
                normal = make_float3(0.f, 0.f, 1.f);
            }
            $else {
                Float radius = curvanature->read(index);
                zCenter = elementZ + curvanature->read(index);

                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("intersect {}", !IntersectSphericalElement(radius, zCenter, ray, &t, &normal));
                };
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), isStop);
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(phit - make_float3(0.f, 0.f, zCenter));
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("check_normal= {}, {}, {}", normal, phit, zCenter);
            };
            Float3 new_origin = phit;
            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaT = refraction->read(index);
                Float etaI = 1.f;
                $if (index > 0 ) {
                    $if (refraction->read(index - 1) != 0.f) {
                        etaI = refraction->read(index - 1);
                    };   
                };

                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };

                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("real_w = {}", w);
                };
                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                
            };
            
            elementZ += thick->read(index);
            /* */
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                //luisa::compute::device_log("endloop {}", i);
            };
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("check_ray= {},{}", ray->origin(), ray->direction());
            };

        };

        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            trace = false;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy()) & ignore) {
                luisa::compute::device_log("false_from_scene = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy()) & ignore) {
                luisa::compute::device_log("true_from_scene = {}, {}", ray->origin(), ray->direction());
            };
            *rOut = make_ray(ray->origin(), make_float3(ray->direction().xy(), ray->direction().z));
        };
       
        return trace;
    }

    Bool TraceLencesFromScene(const Var<Ray> &rCamera, Var<Ray> *rOut, const Bool ignore) const {

        const Int lc = node<RealLensCamera>()->lens_count();
        Float elementZ = LensFrontZ(lc);
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("rCamera {}, {}", ray->origin(), ray->direction());
        };

        $for (i, lc) {
            Int index = i;
            Float zCenter = 0.f;

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("start_loop_function {}", i);
                //luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), radius->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };

            $if (ignore) {
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("ignore");
                };
                $if (refraction->read(index) == 0.f) {
                    elementZ += thick->read(index);
                    $continue;
                };
            };

            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isPlane = (curvanature->read(index) == 0.f);
            $if (isPlane) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
                normal = make_float3(0.f, 0.f, 1.f);
            }
            $else {
                Float radius = curvanature->read(index);
                zCenter = elementZ + curvanature->read(index);

                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("intersect {}", !IntersectSphericalElement(radius, zCenter, ray, &t, &normal));
                };
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), isStop);
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(phit - make_float3(0.f, 0.f, zCenter));

            Float3 new_origin = phit;
            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaT = refraction->read(index);
                Float etaI = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0.f) {
                        etaI = refraction->read(index - 1);
                    };
                };

                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };

                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    //luisa::compute::device_log("real_w = {}", w);
                };
                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
            };

            elementZ += thick->read(index);
            /* */
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                //luisa::compute::device_log("endloop {}", i);
            };
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("check_ray= {},{}", ray->origin(), ray->direction());
            };
        };

        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            trace = false;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy()) & ignore) {
                //luisa::compute::device_log("false_from_scene = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy()) & ignore) {
                //luisa::compute::device_log("true_from_scene = {}, {}", ray->origin(), ray->direction());
            };
            *rOut = make_ray(ray->origin(), make_float3(ray->direction().xy(), ray->direction().z));
        };

        return trace;
    }

    Bool TraceLences(const Var<Ray> &rCamera, Var<Ray> *rOut, Var<ChainVerts[CHAIN]> &intersect) const {
        Float elementZ = -0.f;
        const Int lc = node<RealLensCamera>()->lens_count();
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        Int num = 0;

        #ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("rCamera {}, {}", ray->origin(), ray->direction());
        };
        #endif

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("start_loop_function {}", i);
                luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), radius->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };
            #endif

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("check_ray= {},{},{},{}", ray->origin(), ray->direction(), index, num);
            };

            elementZ -= thick->read(index);
            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isPlane = (curvanature->read(index) == 0.f);
            $if (isPlane) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
            }
            $else {
                Float radius = curvanature->read(index);
                zCenter = elementZ + curvanature->read(index);

                #ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("intersect {}", !IntersectSphericalElement(radius, zCenter, ray, &t, &normal));
                };
                #endif


                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), normal);
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
            #endif

            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(normal);

            #ifdef LIS_EXPERIMENT
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("hit_normal = {}, {}", phit, normal);
            };
            #endif

             normal = normalize(phit - make_float3(0.f, 0.f, zCenter));

            Float3 new_origin = phit;
            
            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaI = refraction->read(index);
                Float etaT = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0) {
                        etaT = refraction->read(index - 1);
                    };
                };

                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };

                #ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("real_w = {}", w);
                };
                #endif

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                intersect[num].point = new_origin;
                intersect[num].n = normal;
                intersect[num].index = index;
                intersect[num].center = make_float3(0.f, 0.f, zCenter);
                num = num + 1;
                
            };

            /* */

            #ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                luisa::compute::device_log("endloop {}", i);
            };
            #endif

        };

        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            trace = false;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("false_ray = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            trace = true;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("true_ray= {},{}", ray->origin(), ray->direction());
            };
            *rOut = make_ray(make_float3(ray->origin().xy(), ray->origin().z), make_float3(ray->direction().xy(), ray->direction().z));
        };

        return trace;
    }

    Bool TraceLences(const Var<Ray> &rCamera, Var<Ray> *rOut) const {
        Float elementZ = -0.f;
        const Int lc = node<RealLensCamera>()->lens_count();
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        Int num = 0;

#ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("rCamera {}, {}", ray->origin(), ray->direction());
        };
#endif

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

#ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("start_loop_function {}", i);
                luisa::compute::device_log("real_element = {}, {}, {}, {}", curvanature->read(index), thick->read(index), refraction->read(index), radius->read(index));
                //luisa::compute::device_log("real_loop = {}, {}, {}", i, ray->direction(), hanbetsu);
                //luisa::compute::device_log("real_pro = {},{},{}", data.mode, data.pixel_offset, data.projected_pixel_size);
            };
#endif

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("check_ray= {},{},{},{}", ray->origin(), ray->direction(), index, num);
            };

            elementZ -= thick->read(index);
            //compute intersection
            Float t;
            Float3 normal;
            Var<bool> isPlane = (curvanature->read(index) == 0.f);
            $if (isPlane) {
                t = (elementZ - ray->origin().z) / ray->direction().z;
            }
            $else {
                Float radius = curvanature->read(index);
                zCenter = elementZ + curvanature->read(index);

#ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("intersect {}", !IntersectSphericalElement(radius, zCenter, ray, &t, &normal));
                };
#endif

                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

#ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("real_loop = {}, {}, {}", index, ray->origin(), normal);
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("real_next");
            };
#endif

            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(normal);

#ifdef LIS_EXPERIMENT

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("hit_normal = {}, {}", phit, normal);
            };
#endif

            normal = normalize(phit - make_float3(0.f, 0.f, zCenter));

            Float3 new_origin = phit;

            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaI = refraction->read(index);
                Float etaT = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0) {
                        etaT = refraction->read(index - 1);
                    };
                };

                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };

#ifdef LIS_EXPERIMENT
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("real_w = {}", w);
                };
#endif

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                
                num = num + 1;
            };

            /* */

#ifdef LIS_EXPERIMENT
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("next_ray {}, {}", ray->origin(), ray->direction());
                luisa::compute::device_log("endloop {}", i);
            };
#endif
        };

        $if (hanbetsu == 1) {
            //ray = make_ray(make_float3(0.f), make_float3(0.f, 0.f, 0.f));
            trace = false;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("false_ray = {}, {},{}", ray->origin(), ray->direction(), misspoint);
            };
        }
        $else {
            trace = true;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("true_ray= {},{}", ray->origin(), ray->direction());
            };
            *rOut = make_ray(make_float3(ray->origin().xy(), ray->origin().z), make_float3(ray->direction().xy(), ray->direction().z));
        };

        return trace;
    }


    void ComputeCardinalPoints(const Var<Ray> &rIn, const Var<Ray> &rOut, Float *pz, Float *fz) const {
        Var<Ray> rO = rOut;
        Var<Ray> rI = rIn;
        Float tf = -rO->origin().x / rO->direction().x;
        *fz = (rO->origin() + rO->direction() * tf).z;
        Float tp = (rI->origin().x - rO->origin().x) / rO->direction().x;
        *pz = (rO->origin() + rO->direction() * tp).z;
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("o, d = {},{}", rO->origin(), rO->direction());
            //luisa::compute::device_log("pz, fz = {},{}", (rO->origin() + rO->direction() * tp).z, (rO->origin() + rO->direction() * tf).z);
        };
    }

    void ComputeThickLensApproximation(Float2 &pz, Float2 &fz) const {
        Float two = 2.f;
        Float x = 0.024f  * .1f;

       // x = 2.f / luisa::sqrt(3);
        const auto lc = node<RealLensCamera>()->lens_count();

        Float3 scene_o = make_float3(x, 0.f, LensFrontZ(lc) - 1.f);
        Float3 scene_d = make_float3(0.f, 0.f, 1.f);
       // scene_d = normalize(make_float3(-1.f, 0.f, luisa::sqrt(3)));
        Var<Ray> rScene = make_ray(scene_o, scene_d);


        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("from_scene = {},{}", rScene->origin(), rScene->direction());
            
        };

       
        Var<Ray> rFilm;
        TraceLencesFromScene(rScene, &rFilm, true);
        ComputeCardinalPoints(rScene, rFilm, &pz[0], &fz[0]);



        Float3 film_o = make_float3(x, 0.f, 0.f);
        Float3 film_d = make_float3(0.f, 0.f, -1.f);
       // film_d = normalize(make_float3(-1.f, 0.f, -luisa::sqrt(3)));

        rFilm = make_ray(film_o, film_d);

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("from_firm = {},{}", rFilm->origin(), rFilm->direction());
        };
        

        TraceLencesFromFilm(rFilm, &rScene);
        ComputeCardinalPoints(rFilm, rScene, &pz[1], &fz[1]);
    }

    Float FocusThickLens(Float focusDistance) const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("pz, fz = {},{}", pz, fz);
        };
        Float f = fz[0] - pz[0];
        Float z = -focusDistance;
        Float delta = 0.5f * (pz[1] - z + pz[0] - sqrt((pz[1] - z - pz[0]) * (pz[1] - z - 4.f * f - pz[0])));

        return delta;
    }

    Float sensordistance(Float &fl) const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("pz, fz = {},{}", pz, fz);
        };
        Float inv_f = 1.f / (fz[0] - pz[0]) - 1.f / (fz[1] - pz[1]);
        fl = abs(fz[0] - pz[0]);

        return abs(pz[0]);
    }

    Float focusdistance() const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("pz, fz = {},{}", pz, fz);
        };
        Float inv_f = abs(1.f / (fz[0] - pz[0]) - 1.f / (fz[1] - pz[1]));
        

        return 1.f / inv_f;
    }

    BB2D BoundExitPupil(Float pFilmX0, Float pFilmX1) const {
        BB2D pupilBounds{};
        const auto lc = node<RealLensCamera>()->lens_count();
        //sample
        const Int nSamples = 256 * 256;
        Int nExitingRays = 0;

        //Compute BB
        Float rearRadius = LensRearRadius(lc - 1);
        BB2D projRearBounds{};
        projRearBounds.packed_min = {-1.5f * rearRadius, -1.5f * rearRadius};
        projRearBounds.packed_max = {1.5f * rearRadius, 1.5f * rearRadius};

        $for (i, nSamples) {
            Float PFX = lerp(pFilmX0, pFilmX1, (i + 0.5f) / nSamples);
 
        };

        return projRearBounds;
    }
    
    Float get_aper() const{
        Int i = 0;
        Float x = 0.f;
        Float l = 0.01f;
        const auto lc = node<RealLensCamera>()->lens_count();

        Float3 scene_o = make_float3(0.f, 0.f, LensFrontZ(lc) - 1.f);
        Float3 scene_d = make_float3(0.f, 0.f, 1.f);
        Var<Ray> rScene = make_ray(scene_o, scene_d);

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("get_aper = {},{}", rScene->origin(), rScene->direction());
        };

        Var<Ray> rFilm;
        $while (i < 30) {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("get_loop = {}", i);
                //luisa::compute::device_log("get_aper = {},{}", rScene->origin(), rScene->direction());
            };

            $if (TraceLencesFromScene(rScene, &rFilm, false)) {
                x = rScene->origin().x;
                
                i = i + 1;
            }
            $else {
                l = l * .5f;
                i = i + 1;
            };
            
            Float3 new_o = make_float3(x + l, rScene->origin().yz());
            rScene->set_origin(new_o);

        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("get_aper = {},{}", rScene->origin(), rScene->direction());
        };

        return x;
    }

    Float wrap01(Float u) const {
        u = u - floor(u);
        $if (u >= 1.f) {
            u = 0.f;
        };
        return u;
    }

    Float unwrap_u(Float u_wrapped, Float u_ref) const {
        // u_wrapped: 0..1
        Float ret;
        Float a = u_wrapped;
        Float b = u_wrapped + 1.f;
        Float c = u_wrapped - 1.f;
        Float da = abs(a - u_ref);
        Float db = abs(b - u_ref);
        Float dc = abs(c - u_ref);
        ret = a;
        $if (db < da & db < dc) { ret = b; };
        $if (dc < da & dc < db) { ret = c; };
        return ret;
    }


    void sphere_uv_from_xyz(const Float3& p, const Float3& c, Float r, Float& u, Float& v) const{//safe çÃóp
        const Float pi = (Float)3.14159265358979323846;
        const Float two_pi = (Float)6.2831853071795864769;

        Float3 d = (p - c) * (1.f / r);
        // safety normalize (optional)
        d = normalize(d);
        

        Float dz = d.z;
        $if (dz < -1.f) {
            dz = -1.f;
        };
        $if (dz > 1.f) {
            dz = 1.f;
        };
        Float phi = acos(dz);// 0..pi
        v = phi / pi;      // 0..1
        

        Float xy2 = d.x * d.x + d.y * d.y;
        $if (xy2 < 1e-16f) {
            // near pole: u is undefined, choose a convention
            u = 0.f;
            $return();
        };

        Float theta = atan2(d.y, d.x);        // -pi..pi
        $if (theta < 0.f) { 
            theta += two_pi; 
        };// 0..2pi


        u = wrap01(theta / two_pi);// 0..1
        
    }

    void sphere_p_n_dp_uv(const Float3& c, Float r, Float u_in, Float v_in, Float3& p, Float3& n, Float3& dp_du, Float3& dp_dv) const {
        const Float pi = (Float)3.14159265358979323846;
        const Float two_pi = (Float)6.2831853071795864769;

        Float u = wrap01(u_in);
        Float v = clamp(v_in, 0.f, 1.f);

        Float theta = two_pi * u;
        Float phi = pi * v;

        Float sin_th = sin(theta), cos_th = cos(theta);
        Float sin_ph = sin(phi), cos_ph = cos(phi);

        // unit normal
        n = make_float3(sin_ph * cos_th, sin_ph * sin_th, cos_ph);

        // position
        p = c + n * r;

        // Å›p/Å›theta, Å›p/Å›phi
        Float3 dp_dtheta = make_float3( -r * sin_ph * sin_th, r * sin_ph * cos_th, 0.f);
        Float3 dp_dphi = make_float3( r * cos_ph * cos_th, r * cos_ph * sin_th, -r * sin_ph);

        // chain rule: theta=2ÉŒu, phi=ÉŒv
        dp_du = dp_dtheta * two_pi;
        dp_dv = dp_dphi * pi;
    }

    Float3 safe_normalize(const Float3& v) const {
        Float n2 = dot(v, v);
        Float3 ret;
        $if(n2 <= 1e-18f) {
            ret = make_float3(0.f, 0.f, 0.f);
        }
        $else{
            ret = v * (1.f / sqrt(n2));
        };
        return ret;
    }

    void build_tangent_frame_from_dp(const Float3& dp_du, const Float3& dp_dv, Float3& n, Float3& s, Float3& t) const {

        n = safe_normalize(n);

        Float3 s0 = dp_du;
        $if(dot(s0, s0) < 1e-16f) {
            s0 = dp_dv;
        };

        // tangentize
        s0 = s0 - n * dot(n, s0);
        s = safe_normalize(s0);

        Float3 t0 = dp_dv - s * dot(s, dp_dv);
        t0 = t0 - n * dot(n, t0);

        $if(dot(t0, t0) < 1e-16f) {
            t0 = cross(n, s);
        };
        t = safe_normalize(t0);

        // enforce right-handed
        $if(dot(cross(s, t), n) < 0.f) {
            t = t * -1.f;
        };
    }

    void st_and_derivs_from_xyz_fd(Var<ChainVerts>& path, Float eps_u, Float eps_v) const {


        auto dex = path.index;
        auto radius = abs(curvanature->read(dex));

        Float u_wrapped = 0.f;
        Float v_wrapped = 0.f;



        sphere_uv_from_xyz(path.point, path.center, radius, u_wrapped, v_wrapped);

        path.v = v_wrapped;
        path.u = unwrap_u(u_wrapped, path.u);



        Float3 pos_from_uv;
        Float3 nor_from_uv;

        //mark
        sphere_p_n_dp_uv(path.center, radius, path.u, path.v, pos_from_uv, nor_from_uv, path.dp_du, path.dp_dv);
        $if(nor_from_uv.z < 0.f) {
            //nor_from_uv = nor_from_uv * -1.f;
        };

        //mark
        path.n = nor_from_uv;

        build_tangent_frame_from_dp(path.dp_du, path.dp_dv, nor_from_uv, path.s, path.t);

        //u-derivs
        Float u_p = path.u + eps_u;
        Float u_m = path.u - eps_u;

        Float3 pos_from_uv_up;
        Float3 nor_from_uv_up;
        Float3 dpdu_from_uv_up;
        Float3 dpdv_from_uv_up;
        Float3 s_from_uv_up;
        Float3 t_from_uv_up;

        sphere_p_n_dp_uv(path.center, radius, u_p, path.v, pos_from_uv_up, nor_from_uv_up, dpdu_from_uv_up, dpdv_from_uv_up);
        build_tangent_frame_from_dp(dpdu_from_uv_up, dpdv_from_uv_up, nor_from_uv_up, s_from_uv_up, t_from_uv_up);

        Float3 pos_from_uv_um;
        Float3 nor_from_uv_um;
        Float3 dpdu_from_uv_um;
        Float3 dpdv_from_uv_um;
        Float3 s_from_uv_um;
        Float3 t_from_uv_um;

        sphere_p_n_dp_uv(path.center, radius, u_m, path.v, pos_from_uv_um, nor_from_uv_um, dpdu_from_uv_um, dpdv_from_uv_um);
        build_tangent_frame_from_dp(dpdu_from_uv_um, dpdv_from_uv_um, nor_from_uv_um, s_from_uv_um, t_from_uv_um);

        Float inv2eu = 1.f / (2.f * eps_u);
        path.ds_du = (s_from_uv_up - s_from_uv_um) * inv2eu;
        path.dt_du = (t_from_uv_up - t_from_uv_um) * inv2eu;


        //v-derivs
        Float v_p = clamp(path.v + eps_v, 0.f, 1.f);
        Float v_m = clamp(path.v - eps_v, 0.f, 1.f);

        Float dv_p = v_p - path.v;
        Float dv_m = path.v - v_m;

        $if(dv_p > 0.f & dv_m > 0.f){
            Float3 pos_from_uv_vp;
            Float3 nor_from_uv_vp;
            Float3 dpdu_from_uv_vp;
            Float3 dpdv_from_uv_vp;
            Float3 s_from_uv_vp;
            Float3 t_from_uv_vp;

            sphere_p_n_dp_uv(path.center, radius, path.u, v_p, pos_from_uv_vp, nor_from_uv_vp, dpdu_from_uv_vp, dpdv_from_uv_vp);
            build_tangent_frame_from_dp(dpdu_from_uv_vp, dpdv_from_uv_vp, nor_from_uv_vp, s_from_uv_vp, t_from_uv_vp);

            Float3 pos_from_uv_vm;
            Float3 nor_from_uv_vm;
            Float3 dpdu_from_uv_vm;
            Float3 dpdv_from_uv_vm;
            Float3 s_from_uv_vm;
            Float3 t_from_uv_vm;

            sphere_p_n_dp_uv(path.center, radius, path.u, v_m, pos_from_uv_vm, nor_from_uv_vm, dpdu_from_uv_vm, dpdv_from_uv_vm);
            build_tangent_frame_from_dp(dpdu_from_uv_vm, dpdv_from_uv_vm, nor_from_uv_vm, s_from_uv_vm, t_from_uv_vm);

            Float inv2ev = 1.f / (2.f * eps_v);
            path.ds_dv = (s_from_uv_vp - s_from_uv_vm) * inv2ev;
            path.dt_dv = (t_from_uv_vp - t_from_uv_vm) * inv2ev;
        }
        $elif(dv_p > 0.f) {
            Float3 pos_from_uv_vp;
            Float3 nor_from_uv_vp;
            Float3 dpdu_from_uv_vp;
            Float3 dpdv_from_uv_vp;
            Float3 s_from_uv_vp;
            Float3 t_from_uv_vp;

            sphere_p_n_dp_uv(path.center, radius, path.u, v_p, pos_from_uv_vp, nor_from_uv_vp, dpdu_from_uv_vp, dpdv_from_uv_vp);
            build_tangent_frame_from_dp(dpdu_from_uv_vp, dpdv_from_uv_vp, nor_from_uv_vp, s_from_uv_vp, t_from_uv_vp);

            Float inv = 1.f / dv_p;
            path.ds_dv = (s_from_uv_vp - path.s) * inv;
            path.dt_dv = (t_from_uv_vp - path.t) * inv;
        }
        $elif(dv_m > 0.f) {
            Float3 pos_from_uv_vm;
            Float3 nor_from_uv_vm;
            Float3 dpdu_from_uv_vm;
            Float3 dpdv_from_uv_vm;
            Float3 s_from_uv_vm;
            Float3 t_from_uv_vm;

            sphere_p_n_dp_uv(path.center, radius, path.u, v_m, pos_from_uv_vm, nor_from_uv_vm, dpdu_from_uv_vm, dpdv_from_uv_vm);
            build_tangent_frame_from_dp(dpdu_from_uv_vm, dpdv_from_uv_vm, nor_from_uv_vm, s_from_uv_vm, t_from_uv_vm);

            Float inv = 1.f / dv_m;
            path.ds_dv = (path.s - s_from_uv_vm) * inv;
            path.dt_dv = (path.t - t_from_uv_vm) * inv;
        }
        $else {
            path.ds_dv = make_float3(0.f);
            path.dt_dv = make_float3(0.f);
        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("normal = {}, {}", path.n, nor_from_uv);
        };
    }

    //ïœâªó ÇÃåvéZ
    Bool invert_tridiagonal_step(Var<ChainVerts[CHAIN]> &path, Int size) const{
        /*
        Int si = size;
        Bool judge = true;

        $if (si != 0) {
            path[0].tmp = path[0].dC_dx_prev;
            Var<float2x2> m = path[0].dC_dx_cur;

            $if (!(invert(m, path[0].inv_lambda))) {
                judge = false;
            };

            $if (judge) {
                $for (i, si) {
                    path[i].tmp = path[i].dC_dx_prev * path[i - 1].inv_lambda;
                    Float2x2 m = path[i].dC_dx_cur - path[i].tmp * path[i - 1].dC_dx_next;
                    $if (!invert(m, path[i].inv_lambda)) {
                        judge = false;
                        $break;
                    };
                };
            };


            $if (judge) {
                path[0].dx = path[0].C;
                $for (i, si) {
                    path[i].dx = path[i].C - path[i].tmp * path[i - 1].dx;
                };

                path[si - 1].dx = path[si - 1].inv_lambda * path[si - 1].dx;

                $for (i, si - 1) {
                    auto idx = si - 2;
                    path[i].dx = path[i].inv_lambda * (path[i].dx - path[i].dC_dx_next * path[i + 1].dx);
                };
            };


        };*/

        Int si = size;
        Bool judge = true;

        $if (si != 0) {
            path[0].tmp = path[0].dC_dx_prev;
            Var<float2x2> m = path[0].dC_dx_cur;

            $if (!(invert(m, path[0].inv_lambda))) {
                judge = false;
            };

            $if (judge) {
                // gamma0 = inv(B0) * C0
                path[0].tmp = path[0].inv_lambda * path[0].dC_dx_next;// gamma
                // rhs0 = inv(B0) * r0
                path[0].dx = path[0].inv_lambda * path[0].C;
            };

            $if (judge) {
                $for (i, si - 1) {
                    Int k = i + 1;

                    /* Float2x2 A = path[k].dC_dx_prev;
                    Float2x2 B = path[k].dC_dx_cur;

                    Float2x2 denom = B - A * path[k - 1].tmp;
                    Float2x2 inv_denom;
                    $if (!invert(denom, inv_denom)) {
                        judge = false;
                        $break;
                    };
                    path[k].inv_lambda = inv_denom;

                    // gamma_k = inv(denom) * C_k  (for k==n-1, C_k is unused but harmless)
                    path[k].tmp = inv_denom * path[k].dC_dx_next;

                    // rhs_k = inv(denom) * (r_k - A*rhs_{k-1})
                    Float2 r = path[k].C - A * path[k - 1].dx;
                    path[k].dx = inv_denom * r;*/

                    path[k].tmp = path[k].dC_dx_prev * path[k - 1].inv_lambda;
                    Float2x2 m = path[k].dC_dx_cur - path[k].tmp * path[k - 1].dC_dx_next;
                    $if (!invert(m, path[k].inv_lambda)) {
                        judge = false;
                        $break;
                    };
                };
            };

            $if (judge) {
                path[0].dx = path[0].C;
                $for (i, si - 1) {
                    Int k = i + 1;
                    path[k].dx = path[k].C - path[k].tmp * path[k - 1].dx;
                };

                path[si - 1].dx = path[si - 1].inv_lambda * path[si - 1].dx;

                $for (i, si - 1) {
                    auto idx = si - i - 2;
                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        //luisa::compute::device_log("invert = {}", idx);
                    };
                    path[idx].dx = path[idx].inv_lambda * (path[idx].dx - path[idx].dC_dx_next * path[idx + 1].dx);
                };
            };
            /*
            $if (judge) {
                // dx_{n-1} is already rhs_{n-1}
                $for (j, si - 1) {
                    Int k = (si - 2) - j;// n-2 .. 0
                    path[k].dx = path[k].dx - path[k].tmp * path[k + 1].dx;
                };*/
        };

        return judge;
    }


    //frameÇÃéüÇé¿ëï


    Bool compute_der_halfvector(Float3 start, Float3 emit, Var<ChainVerts[20]> &path) const {
        Bool compute = true;
        Int size = 0;
        $for (i, CHAIN) {
            $if (path[i].index == 1) {
                size = i + 1;
                $break;
            };
        };

        $if (curvanature->read(0) != 0.f) {
            size = size + 1;
        };

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("ME_size = {}", size);
        };
        
        $for (i, size) {
            st_and_derivs_from_xyz_fd(path[i], 1e-4f, 1e-4f);
        };

        $for (i, size) {
            //set C
            path[i].C = make_float2(0.f);
            path[i].dC_dx_prev = make_float2x2(0.f);
            path[i].dC_dx_cur = make_float2x2(0.f);
            path[i].dC_dx_next = make_float2x2(0.f);

            auto dex = path[i].index;

            //set point
            Float3 x_prev;
            Float3 x_next;
            Float3 x_cur = path[i].point;

            $if (i == 0) {
                x_prev = start;
            }
            $else {
                x_prev = path[i - 1].point;
            };

            $if (i == size - 1) {
                x_next = emit;
            }
            $else {
                x_next = path[i + 1].point;
            };

            
            //set wo
            Bool end_fixed_direction = ((i == size - 1) & false);
            Float3 wo;
            $if (end_fixed_direction) {
                wo = make_float3(0.f, 0.f, -1.f);
            }
            $else {
                wo = x_next - x_cur;
            };
            $if (length(wo) < 1e-7f) {
                compute = false;
                $break;
            };
            Float ilo = 1.f / length(wo);
            wo = normalize(wo);

            //set wi
            Float3 wi = x_prev - x_cur;
            $if (length(wi) < 1e-7f) {
                compute = false;
                $break;
            };
            Float ili = 1.f / length(wi);
            wi = normalize(wi);

            //set half vector
            Float eta;
            $if(i == 0) {
                
                Float etaI = 1.f;
                Float etaT = refraction->read(dex - 1);
                eta = etaT / etaI; 
            }
            $elif(i == size - 1) {
                
                Float etaI = refraction->read(dex);
                Float etaT = 1.f;
                eta = etaT / etaI; 
            }
            $else {
                
                Float etaI = refraction->read(dex);
                Float etaT = refraction->read(dex - 1);
                eta = etaT / etaI;
            };

            Float3 nn = path[i].n;

            $if (dot(nn, wi) > 0.f) {
                // wi Ç™äOë§Ç…èoÇƒÇ¢Ç≠ï˚å¸Åi=ì¸éÀÇÕì‡ë§Ç©ÇÁóàÇΩÅj
                // incident medium ÇÕì‡ë§
                
                //path[i].n = -nn;// Ågì¸éÀë§Ç…å¸Ç¢ÇΩñ@ê¸ÅhÇ÷ëµÇ¶ÇÈó¨ãVÇ‡ëΩÇ¢
                nn = -nn;
            }
            $else {
                // wi Ç™ì‡ë§Ç÷å¸Ç≠Åi=ì¸éÀÇÕäOë§Ç©ÇÁóàÇΩÅj

                nn = nn;
            };
            Float3 h = wi + eta * wo;
            $if (dot(h, nn) < 0.f) {
                //h = h * -1.f;
            };
            h = h * -1.f;
            Float ilh = 1.f / length(h);
            h = normalize(h);

            ilo = ilo * eta * ilh;
            ili = ili * ilh;

            //prepare u,v

            /*
            Float3 p_i = path[i].point;
            Float3 c_i = path[i].center;
            Float u_i = 0.f;
            Float v_i = 0.f;
            Float r_i = abs(curvanature->read(dex));
            sphere_uv_from_xyz(p_i, c_i, r_i, u_i, v_i);
             
            Float3 normal_i = make_float3(0.f);
            Float3 dp_du_i = make_float3(0.f);
            Float3 dp_dv_i = make_float3(0.f);
            sphere_p_n_dp_uv(c_i, r_i, u_i, v_i, p_i, normal_i, dp_du_i, dp_dv_i);

            Float3 s_i = make_float3(0.f);
            Float3 t_i = make_float3(0.f);
            build_tangent_frame_from_dp(dp_du_i, dp_dv_i, normal_i, s_i, t_i);


            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("u_v = {}, {}", u_i, v_i);
                luisa::compute::device_log("du_dv = {}, {}", dp_du_i, dp_dv_i);
                luisa::compute::device_log("s_t = {}, {}", s_i, t_i);

            };*/


            //st_and_derivs_from_xyz_fd(path[i], 1e-4f, 1e-4f);
            $if (i > 0) {
                //st_and_derivs_from_xyz_fd(path[i - 1], 1e-4f, 1e-4f);
            };
           

            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("one_function");
                //luisa::compute::device_log("u_v = {}, {}", path[i].u, path[i].v);
                //luisa::compute::device_log("du_dv = {}, {}", path[i].dp_du, path[i].dp_dv);
                //luisa::compute::device_log("s_t = {}, {}", path[i].s, path[i].t);
                //luisa::compute::device_log("s_t_du = {}, {}", path[i].ds_du, path[i].dt_du);
                //luisa::compute::device_log("s_t_dv = {}, {}", path[i].ds_dv, path[i].dt_dv);
            };

            
            // Derivative of specular constraint w.r.t. x_{i-1}
            Float3 dh_du;
            Float3 dh_dv;

            $if (i > 0) {
                dh_du = ili * (path[i - 1].dp_du - wi * dot(wi, path[i - 1].dp_du));
                dh_dv = ili * (path[i - 1].dp_dv - wi * dot(wi, path[i - 1].dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path[i].dC_dx_prev = make_float2x2(
                    dot(path[i].s, dh_du), dot(path[i].s, dh_dv),
                    dot(path[i].t, dh_du), dot(path[i].t, dh_dv));
            };

            // Derivative of specular constraint w.r.t. x_{i}
            $if (end_fixed_direction) {
                // When the 'wo' direction is fixed, the derivative here simplifies.
                dh_du = ili * (-path[i].dp_du + wi * dot(wi, path[i].dp_du));
                dh_dv = ili * (-path[i].dp_dv + wi * dot(wi, path[i].dp_dv));
            } 
            $else {
                // Standard case for fixed emitter position
                dh_du = -path[i].dp_du * (ili + ilo) + wi * (dot(wi, path[i].dp_du) * ili) + wo * (dot(wo, path[i].dp_du) * ilo);
                dh_dv = -path[i].dp_dv * (ili + ilo) + wi * (dot(wi, path[i].dp_dv) * ili) + wo * (dot(wo, path[i].dp_dv) * ilo);
            };

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            $if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            };

            path[i].dC_dx_cur = make_float2x2(
                dot(path[i].ds_du, h) + dot(path[i].s, dh_du), dot(path[i].ds_dv, h) + dot(path[i].s, dh_dv),
                dot(path[i].dt_du, h) + dot(path[i].t, dh_du), dot(path[i].dt_dv, h) + dot(path[i].t, dh_dv));

            // Derivative of specular constraint w.r.t. x_{i+1}
            $if (i < size - 1) {
                dh_du = ilo * (path[i + 1].dp_du - wo * dot(wo, path[i + 1].dp_du));
                dh_dv = ilo * (path[i + 1].dp_dv - wo * dot(wo, path[i + 1].dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path[i].dC_dx_next = make_float2x2(
                    dot(path[i].s, dh_du), dot(path[i].s, dh_dv),
                    dot(path[i].t, dh_du), dot(path[i].t, dh_dv));
            };

            // Evaluate specular constraint
            auto H = make_float2(dot(path[i].s, h), dot(path[i].t, h));
            auto n_offset = make_float3(0.f, 0.f, 1.f);
            auto N = make_float2(n_offset[0], n_offset[1]);
            path[i].C = H - N;



        };

        $if (!invert_tridiagonal_step(path, size)) {
            compute = false;
        };


        return compute;
    }

    Bool reproject(const Float3 start, const Float3 emit, Var<ChainVerts[20]> proposed_path, const Int size) const {
        
        Float3 first_point = proposed_path[0].point;
        Float3 direction_to_first = first_point - start;

        Var<Ray> proposed_ray = make_ray(start, direction_to_first);
        Var<Ray> generate_ray;

        Bool success = TraceLences(proposed_ray, &generate_ray);


         $for (i, size) {
            auto id = proposed_path[i].index;
            auto center = proposed_path[i].center;
            auto r = curvanature->read(id);
            auto p_prop = proposed_path[i].point;
            auto p_proj = center + r * normalize(p_prop - center);
            //proposed_path[i].point = p_proj;
        };
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("newton {}", proposed_path[0].point);
            
        };

        $if (success) {
            Float distance = emit.z - generate_ray->origin().z;
            Float t = distance / generate_ray->direction().z;
            Float3 target = generate_ray->origin() + t * generate_ray->direction();
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                
                luisa::compute::device_log("newton {}, {}", proposed_path[size -1].point, generate_ray->origin());
                luisa::compute::device_log("newton_target {},{}", target, start);
            };

            Float check_d = length(generate_ray->origin() - proposed_path[size - 1].point);
            Float angle_d;
            $if (check_d > length_threshold) {
                //success = false;
            }
            $else {
                Float3 target_angle = normalize(emit - proposed_path[size - 1].point);
                angle_d = 1.f - dot(normalize(generate_ray->direction()), target_angle);
                $if (angle_d > angle_threshold) {
                    //success = false;
                };
            };
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {

                luisa::compute::device_log("difference {}, {}", check_d, angle_d);
                //luisa::compute::device_log("newton_target {},{}", target, start);
            };

        };
        


        return success;
    }

    Bool last_check(const Float3 start, const Float3 emit, Var<ChainVerts[20]> proposed_path, const Int size) const {

        Float3 first_point = proposed_path[0].point;
        Float3 direction_to_first = first_point - start;

        Var<Ray> proposed_ray = make_ray(start, direction_to_first);
        Var<Ray> generate_ray;

        Bool success = TraceLences(proposed_ray, &generate_ray);

        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("newton {}", proposed_path[0].point);
        };

        $if (success) {
            Float distance = emit.z - generate_ray->origin().z;
            Float t = distance / generate_ray->direction().z;
            Float3 target = generate_ray->origin() + t * generate_ray->direction();
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {

                //luisa::compute::device_log("newton {}, {}", proposed_path[size - 1].point, generate_ray->origin());
                luisa::compute::device_log("newton_target {},{}", target, start);
            };

            Float check_d = length(generate_ray->origin() - proposed_path[size - 1].point);
            Float angle_d;
            $if (check_d > length_threshold) {
                success = false;
            }
            $else {
                Float3 target_angle = normalize(emit - proposed_path[size - 1].point);
                angle_d = 1.f - dot(normalize(generate_ray->direction()), target_angle);
                $if (angle_d > angle_threshold) {
                    success = false;
                };
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {

                luisa::compute::device_log("difference {}, {}", check_d, angle_d);
                //luisa::compute::device_log("newton_size {}", size);
            };
        };

        return success;
    }


    Bool newton_solver(const Float3 start, const Float3 emit, Var<ChainVerts[CHAIN]>& path) const {
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("newton {}, {}", start, emit);
        };
        
        Bool newton = true;
        Float number = 0.f;

        Bool success = false;
        UInt iterations = 0u;
        Float beta = 1.f;
        
        Var<ChainVerts[CHAIN]> proposed_path = path;

        
        Int size = 0;
        $for (i, CHAIN) {
            $if (path[i].index == 1) {
                size = i + 1;
                $break;
            };
        };

        $if (curvanature->read(0) != 0.f) {
            size = size + 1;
        };
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("size {}", size);
        };

        Bool use_half_vector = true;
        Bool needs_step_update = true;
        $while (iterations < MAX_I) {
            Bool step_success = true;
            $if (needs_step_update) {
                $if (use_half_vector) {
                    // Use standard manifold formulation using half-vector constraints
                    step_success = compute_der_halfvector(start, emit, path);
                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        //luisa::compute::device_log("step_suc {}", step_success);
                    };
                }
                $else {
                    // Use angle-difference constraint formulation
                    //step_success = compute_der_anglediff(si.p, ei);
                };
            };

            $if (!step_success) {
                $break;
            };


            
            // Check for success
            Bool converged = true;
            Float max_C = 0.f;
            $for (i, size) {
                
                $if(max_C < length(path[i].C)) {
                    max_C = length(path[i].C);
                };

                $if (length(path[i].C) > solver_threshold) {
                    converged = false;
                    
                    $break;
                };
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("convege {}", max_C);
                
            };

            $if (converged) {
                $if (last_check(start, emit, path, size)) {
                    success = true;
                }
                $else {
                    success = false;

                };
                success = true;
                $break;
            };

            // Make a proposal
            $for (i, CHAIN) {
                proposed_path[i].point = make_float3(0.f);
            };
            
            $for (i, size) {
                Float3 p_prop = path[i].point - step_scale * beta * (path[i].dp_du * path[i].dx[0] + path[i].dp_dv * path[i].dx[1]);
                proposed_path[i].point = p_prop;
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("dx {}", path[i].dx);
                    
                };
            };

            // Project back to surfaces
            Bool project_success = reproject(start, emit, proposed_path, size);
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("reproject {}", project_success);
            };

            Bool temp = compute_der_halfvector(start, emit, proposed_path);

            Float max_C_p = 0.f;
            $for (i, size) {
                
                $if (max_C_p < length(proposed_path[i].C)) {
                    max_C_p = length(proposed_path[i].C);
                };

                
            };

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("convege_p {}", max_C_p);
                //luisa::compute::device_log("beta {}", beta);
                
            };

            $if (project_success) {
                project_success = (max_C > max_C_p);
            };
            


            $if (!project_success) {
                beta = 0.5f * beta;
                needs_step_update = false;
            } 
            $else {
                beta = min(1.f, 2.f * beta);
                $for (i, size) {
                    path[i].point = proposed_path[i].point;
                };
                needs_step_update = true;
            };

            iterations = iterations + 1;
        };

        $if (!success) {
            newton = false;
        }
        $else {
            
            $for (i, size) {
                //set point
                Float3 x_prev;
                Float3 x_next;
                Float3 x_cur = path[i].point;

                $if (i == 0) {
                    x_prev = start;
                }
                $else {
                    x_prev = path[i - 1].point;
                };

                $if (i == size - 1) {
                    x_next = emit;
                }
                $else {
                    x_next = path[i + 1].point;
                };

                Bool end_fixed_direction = ((i == size - 1) & false);
                Float3 wo;
                $if (end_fixed_direction) {
                    wo = make_float3(0.f, 0.f, -1.f);
                }
                $else {
                    wo = x_next - x_cur;
                };
                Float ilo = 1.f / length(wo);
                wo = normalize(wo);

                //set wi
                Float3 wi = x_prev - x_cur;
                Float ili = 1.f / length(wi);
                wi = normalize(wi);

                Float cos_theta_i = dot(path[i].n, wi);
                Float cos_theta_o = dot(path[i].n, wo);
                Bool refract = cos_theta_i * cos_theta_o < 0.f;
                Bool reflect = !refract;

                Float eta;
                auto dex = path[i].index;
                $if (i == 0) {

                    Float etaI = 1.f;
                    Float etaT = refraction->read(dex - 1);
                    eta = etaT / etaI;
                }
                $elif (i == size - 1) {

                    Float etaI = refraction->read(dex);
                    Float etaT = 1.f;
                    eta = etaT / etaI;
                }
                $else {

                    Float etaI = refraction->read(dex);
                    Float etaT = refraction->read(dex - 1);
                    eta = etaT / etaI;
                };

                $if ((eta == 1.f & !reflect) | (eta != 1.f & !refract)) {
                    newton = false;
                };
            };
        }; 
        
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("bool {}", success);
        };

        return newton;
    }






    [[nodiscard]] std::pair<Var<Ray>, Float> _generate_ray_in_camera_space(Expr<float2> pixel,
                                                                           Expr<float2> u_lens,
                                                                           Expr<float> /* time */) const noexcept override {
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("ganerate_ray_in_camera_start");
        };


        const auto lc = node<RealLensCamera>()->lens_count();
        Int misspoint = 0;

        ArrayFloat3<20> intersect_v;
        float3 ver[20];
        Float fl;
        Float sd = sensordistance(fl);
        auto coord = dispatch_id().xy();
        auto coord1D = coord.y * RESOLUTION + coord.x;

        auto path = vertex->read(coord1D);
        auto elem = element->read(coord1D);

       

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
        auto coordScene = make_float3(coordX, coordY, 0.f);

        Float fov =  pi / 4.f;
        //auto coord_d = sample_cosine_hemisphere_fov(u_lens, fov);
        //auto scene_d = normalize(make_float3(coord_d.xy(), -coord_d.z));


        auto p_lens = sample_uniform_disk(u_lens, LensRearRadius(lc - 1), LensRearZ(lc - 1));
        //p_lens = sample_uniform_disk(u_lens, 0.001f, LensRearZ(lc - 1));
        auto scene_d = normalize(p_lens - coordScene);

        uint tes = 1u;
        uint tes2 = 2u;
        auto test = "test";

        uint tes3 = select(tes, tes2, false);
        
        Float dlt = 0.f;
        //dlt = FocusThickLens(3.f);
        //delta->write(0, dlt);
       
        #ifdef LIS_EXPERIMENT
        $if(luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("camera_dir = {}, {}, {}", normalize(p_focal - p_lens).x, normalize(p_focal - p_lens).y, normalize(p_focal - p_lens).z);
            luisa::compute::device_log("test_Lens = {}, {}", p_lens, coordScene);
            luisa::compute::device_log("real_pro = {},{},{}", coordScene, scene_d, tes3);
        };
        #endif

        auto test_d = make_float3(0.f, 0.f, -1.f);
        
        //sceme
        auto ray = make_ray(coordScene, scene_d);
        auto first_ray = make_ray(coordScene, scene_d);

        //test
        first_ray = make_ray(coordScene, test_d);
        //real system
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("coordScene {}", coordScene);
        };


        Bool trace;
        auto bunki = elem.sign ;
        $if (bunki == 0) {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("LensTrace");
            };
            Clear_path(path);
            trace = TraceLences(first_ray, &ray, path);
            $if (!trace) {
                weight = 0.f;
            }
            $else {
                elem.first = first_ray->origin();
                elem.use = 1;
                element->write(coord1D, elem);
            };
        }
        $elif (bunki == 1) {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("Manifold Exploration start");
            };
            Float3 emit;
            $if(elem.use != 1) {
                trace = false;
            }
            $else {
                Float3 start = elem.first;
                Float3 emit = elem.emit;
                trace = newton_solver(start, emit, path);
            };
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("Manifold Exploration {}", trace);
            };
            $if (!trace) {
                weight = 0.f;
            }
            $else {
                Float3 ray_o = path[lc - 1].point;
                Float3 ray_d = emit - ray_o;
                ray = make_ray(ray_o, ray_d);
                weight = 1.f;
            };

        }
        $else {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("Error");
            };
        };
        //6.4.2
        /**/
       
        
        #ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("check = {}, {}", trace, weight);
            
        };
        #endif

        //6.4.3
        
        Float pz;
        Float fz;
        //ComputeCardinalPoints(first_ray, ray, &pz, &fz);

        BB2D testpupil{};
        testpupil = BoundExitPupil(1.f, 1.f);


        #ifdef LIS_EXPERIMENT
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("test_cardinal = {},{}", pz, fz);
            luisa::compute::device_log("test_delta = {}", dlt);
            luisa::compute::device_log("test_pupil = {}", testpupil.packed_max);
        };
        #endif

        
       //vertex->write(0, );


        $if (trace) {
            
            //chain.v = intersect_v;
            vertex->write(coord1D, path);
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                //luisa::compute::device_log("test_fd = {},{}", pz, fz);
                luisa::compute::device_log("test_vertex = {},{}", path[0].point, path[1].point);
                luisa::compute::device_log("test_vertex = {},{}", path[0].n, path[1].n);
                luisa::compute::device_log("test_vertex = {},{}", path[0].index, path[1].index);
                //luisa::compute::device_log("test_pupil = {}", testpupil.packed_max);
            };
        };
        
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            //luisa::compute::device_log("test_sd = {}", sd);
        };

        Float cosTheta = normalize(first_ray->direction()).z;
        Float cos4Theta = (cosTheta * cosTheta) * (cosTheta * cosTheta);
        //weight = weight * cos4Theta;


        /* Float3 start = make_float3(0.f);
        Float3 emit = make_float3(0.f, 0.f, -5.f);
        Bool NS = newton_solver(start, emit, path);
        //test_ME
        
        

      // Bool ME = compute_der_halfvector(start, emit, path);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("test_ME = {}", NS);
            luisa::compute::device_log("test_ME = {}", path[0].point);
        };*/

        
/**/



        return std::make_pair(std::move(ray), weight);
    }

    [[nodiscard]] Bool _get_camera_param(Float &aper, Float &fl, Float &fd, Float &sd) const noexcept override { //aper = entrance pupil, fl = EFL, fd = compute, sd = pz
        Bool get_p = true;
        sd = sensordistance(fl);
        aper = get_aper() * 2.f;
        fd = abs (1.f / (1.f / fl - 1.f / sd));
        //fd = focusdistance();
        return get_p;
    }

    [[nodiscard]] std::pair<Var<Ray>, Float> _get_ray_Manifold(Expr<float2> pixel, Float3 emit, Float sign) const noexcept override {
        Float get_manifold_weight = 1.f;

        auto coord = dispatch_id().xy();
        auto coord1D = coord.y * RESOLUTION + coord.x;

        //Var<ChainVerts[CHAIN]> path = vertex->read(coord1D);

        auto get = element->read(coord1D);
        get.emit = emit;
        get.sign = sign;
        element->write(coord1D, get);


        auto data = _device_data->read(0u);
        Float2 resolution = data.resolution;
        auto sceneX = .024f;
        auto sceneY = .024f;
        Float coordX = (pixel.x - data.pixel_offset.x) * sceneX / resolution.x;
        Float coordY = (pixel.y - data.pixel_offset.y) * sceneY / resolution.y;
        Float3 coordScene = make_float3(coordX, coordY, 0.f);

        Var<Ray> ret_ray = make_ray(make_float3(0.f), make_float3(0.f));



;/*
        Int size = 0;
        $for (i, CHAIN) {
            $if (path[i].index == 1) {
                size = i + 1;
                $break;
            };
        };
        
         
        $if(size != 0) {
            $if (newton_solver(coordScene, coordScene, path)) {
                Float3 propose_origin = path[size - 1].point;
                Float3 propose_direction = emit - path[size - 1].point;

                //ret_ray = make_ray(propose_origin, propose_direction);
            }
            $else {
                get_manifold_weight = 0.f;
            };
        }
        $else {
            get_manifold_weight = 0.f;
        };*/
        
        return std::make_pair(std::move(ret_ray), get_manifold_weight);

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