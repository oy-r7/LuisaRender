
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
constexpr float solver_threshold = 5e-4f;
constexpr float step_scale = .1f;
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
    luisa::compute::Buffer<ChainVerts> vertex = _device.create_buffer<ChainVerts>(RESOLUTION * RESOLUTION * CHAIN);
    luisa::compute::Buffer<ChainVerts> propose_vertex = _device.create_buffer<ChainVerts>(RESOLUTION * RESOLUTION * CHAIN);
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


    void Clear_path(Var<ChainVerts> path) const {
       
        path.point = make_float3(0.f);
        path.n = make_float3(0.f);
        path.center = make_float3(0.f);
        path.index = 0.f;
        path.u = 0.f;
        path.v = 0.f;
        path.dx = make_float2(0.f);
        path.C = make_float2(0.f);
        
    }

    //get front Z
    Float LensFrontZ( Var<int> lenscount) const {
       Float zsum = 0.f;
        0.f;
       $for (i, lenscount) {
           zsum += thick->read(i);
           
           
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
                    *t = t1;
                }
                $elif (t0 > 0) {
                    *t = t0;
                }
                $else {
                    // Both intersections are behind the ray
                    *t = -1.f;
                };
            };/**/
            
            
            $if(*t < 0.f) {
                hit = false;
            } 
            $else {
                *n = sphy_origin + (*t) * ray->direction();
                *n = select(*n, -*n, dot(*n, -ray->direction()) < 0.f);
            };
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

       

        $if (sin2ThetaT >= 1) {
            refraction = false;
        } 
        $else {
            cosThetaT = sqrt(1 - sin2ThetaT);
            *wt = ir * -wi + (ir * cosThetaI - cosThetaT) * nn;

            
        };

        return refraction;
    }

    Var<float3> sample_cosine_hemisphere_fov(Var<float2> u, Float fov_rad) const noexcept {

        static Callable impl = [fov_rad](Var<float2> u) noexcept {
            // FOV 半角の cos
            Var<float> cos_theta_max = cos(fov_rad * 0.5f);

            // z の計算（逆CDF）
            Var<float> z = sqrt(1.0f - u.y * (1.0f - cos_theta_max * cos_theta_max));

            // 半径 r
            Var<float> r = sqrt(max(1.0f - z * z, 0.0f));

            // φ
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
            Var<float> r = R * sqrt(u.y);// 面積一様
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
        // 呼び出し側で d が 0 でないことを保証する前提
        Float inv_d = 1.f / d;
        return make_float2x2(
            A[1][1] * inv_d, -A[0][1] * inv_d,
            -A[1][0] * inv_d, A[0][0] * inv_d);
    }

    auto invert (const Float2x2 &A, Float2x2 &Ainv) const {
        Float determinant = det(A);
        
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

        

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

            

            
           
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

                

                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
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


               

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
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

                
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
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

               
                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                
            };
            
            elementZ += thick->read(index);
            /* */
            

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
       

        $for (i, lc) {
            Int index = i;
            Float zCenter = 0.f;

            

            $if (ignore) {
                
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

               
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
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

               
                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
            };

            elementZ += thick->read(index);
            /* */
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy()) & ignore) {
                luisa::compute::device_log("from_scene = {}", new_origin);
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

    Bool TraceFrontLencesFromScene(const Var<Ray> &rCamera, Var<Ray> *rOut, const Bool ignore) const {

        const Int lc = node<RealLensCamera>()->lens_count();
        Float elementZ = LensFrontZ(lc);
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        

        $for (i, lc) {
            Int index = i;
            Float zCenter = 0.f;

            

            $if (ignore) {
               
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

               
                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
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
            ray->set_origin(new_origin);

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

                
                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
            }
            $else {
                $break;
            };

            elementZ += thick->read(index);
            /* */
            
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

    Bool TraceLencesSave(const Var<Ray> &rCamera, Var<Ray> *rOut) const {
        Float elementZ = -0.f;
        const auto lc = node<RealLensCamera>()->lens_count();
        Var<Ray> ray = rCamera;
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        Int num = 0;
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
        ArrayFloat3<CHAIN> point_s;
        ArrayFloat3<CHAIN> normal_s;
        ArrayFloat3<CHAIN> center_s;
        ArrayFloat<CHAIN> index_s;

        

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

            

            
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

               


                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

           

            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(normal);

            
          

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

               

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                
                
                point_s[i] = new_origin;
                normal_s[i] = normal;
                index_s[i] = index;
                center_s[i] = make_float3(0.f, 0.f, zCenter);
                
                num = num + 1;
                
            };

            /* */

         

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
            $for (i, num) {
                auto id = coord1D + i;
                auto intersect = vertex->read(id);
                intersect.point = point_s[i];
                intersect.n = normal_s[i];
                intersect.index = index_s[i];
                intersect.center = center_s[i];
                //vertex->write(id, intersect);
            };
            
        };

        return trace;
    }

    Bool make_seed(Float3 emit, Float3 start) const {
        Float elementZ = -0.f;
        const auto lc = node<RealLensCamera>()->lens_count();
        Float3 ray_o = start;
        Float3 ray_d = (emit - start);
        //ray_d = make_float3(0.f, 0.f, -1.f);

        Var<Ray> ray = make_ray(ray_o, ray_d);
        Int hanbetsu = 0;
        Int misspoint = 0;
        Var<bool> trace = true;
        Int num = 0;
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
        ArrayFloat3<CHAIN> point_s;
        ArrayFloat3<CHAIN> normal_s;
        ArrayFloat3<CHAIN> center_s;
        ArrayFloat<CHAIN> index_s;

        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;

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

                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };

            

            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(normal);

            normal = normalize(phit - make_float3(0.f, 0.f, zCenter));

            Float3 new_origin = phit;

            //update ray

            Float3 w;
            $if (!(refraction->read(index) == 0.f)) {
                Float etaI = 1.f;
                Float etaT = 1.f;
                /*
                Float etaI = refraction->read(index);
                Float etaT = 1.f;
                $if (index > 0) {
                    $if (refraction->read(index - 1) != 0) {
                        etaT = refraction->read(index - 1);
                    };
                };*/

                $if (!Refract(normalize(-ray->direction()), &normal, etaI / etaT, &w)) {
                    hanbetsu = 1;
                    misspoint = 3;
                    $break;
                };

                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));

                point_s[i] = new_origin;
                normal_s[i] = normal;
                index_s[i] = index;
                center_s[i] = make_float3(0.f, 0.f, zCenter);

                num = num + 1;
            };
          /*   $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("seed_point= {}", new_origin);
            };
            */
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
            
            $for (i, num) {
                auto id = coord1D + i;
                auto intersect = vertex->read(id);
                intersect.point = point_s[i];
                intersect.n = normal_s[i];
                intersect.index = index_s[i];
                intersect.center = center_s[i];
                vertex->write(id, intersect);
                
            };
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



        $for (i, lc) {
            Int index = lc - i - 1;
            Float zCenter = 0.f;


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



                $if (!IntersectSphericalElement(radius, zCenter, ray, &t, &normal, true)) {
                    hanbetsu = 1;
                    misspoint = 1;
                    $break;
                };
            };


            //test intersection
            Float3 phit = ray->origin() + t * ray->direction();
            Float r2 = phit.x * phit.x + phit.y * phit.y;

            $if (r2 > radius->read(index) * radius->read(index)) {
                hanbetsu = 1;
                misspoint = 2;
                $break;
            };

            normal = normalize(normal);



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



                w = normalize(w);
                ray = make_ray(new_origin, make_float3(w.xy(), w.z));
                
                num = num + 1;
            };

            /* */


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
            luisa::compute::device_log("pz, fz = {},{}", (rO->origin() + rO->direction() * tp).z, (rO->origin() + rO->direction() * tf).z);
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
        TraceLencesFromScene(rScene, &rFilm, false);
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

    void ComputeThickLensApproximation(Float2 &pz, Float2 &fz, Float3 &fp, Float3 &epc, Float &per) const {
        Float two = 2.f;
        Float x = 0.024f * .1f;

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

        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;

        Float eps = 1e-4f;

        $if(refraction->read(0) == 0.f) {
            epc = make_float3(0.f, 0.f, LensFrontZ(lc));
        }
        $else {
            Var<Ray> rA = make_ray(make_float3(0.f, 0.f, LensFrontZ(lc) - 1.f), make_float3(0.f, 0.f, 1.f));
            Var<Ray> outA;
            TraceFrontLencesFromScene(rA, &outA, false);

            Var<Ray> rB = make_ray(make_float3(eps, 0.f, LensFrontZ(lc) - 1.f), make_float3(0.f, 0.f, 1.f));
            Var<Ray> outB;
            TraceFrontLencesFromScene(rB, &outB, false);

            Float sA = outA->direction().x / outA->direction().z;
            Float sB = outB->direction().x / outB->direction().z;

            // x(z) = x0 + s*(z - z0)
            // 2本のレイが同じ z で持つ x の差は
            // Δx(z) = (xB0 - xA0) + (sB - sA)*(z - z0)
            // 光軸上の pupilCenter は Δx(z)=0 となる z で近似できる（近軸で有効）
            Float dx0 = outB->origin().x - outA->origin().x;
            Float ds = sB - sA;
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("camera_outA = {}", outA);
                luisa::compute::device_log("camera_outB = {}", outB);
            };
            $if (abs(ds) < 1e-8f) {
                ; /* ほぼ平行 → 計算不能/無限遠 */
            };

            Float z0 = outA->origin().z;// 同じzStartを使ったならどちらでもよい
            Float pupilZ = z0 - dx0 / ds;// ← ここで rA を使っている
            epc = make_float3(0.f, 0.f, pupilZ);


            Float tA = (LensFrontZ(lc) - outA->origin().z) / outA->direction().z;
            Float tB = (LensFrontZ(lc) - outB->origin().z) / outB->direction().z;
            Float3 outAtA = outA->origin() + tA * outA->direction();
            Float xA_stop = outAtA.x;// ほぼ0
            Float3 outBtB = outB->origin() + tB * outB->direction();
            Float xB_stop = outBtB.x;

            Float m = (xB_stop - xA_stop) / eps;// stop面での倍率（近軸）
            per = abs(m);
        };
        



        film_o = fp;
        film_d = epc - fp;
        /*
        rFilm = make_ray(film_o, film_d);
        Var<Ray> rK;

        TraceLencesFromFilm(rFilm, &rScene);
        Float tHp = (pz[0] - rScene->origin().z) / rScene->direction().z;
        fp = rScene->origin() + tHp * rScene->direction();
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("camera_epc = {}, {}", pz[0], epc);
        };*/

    }

    Float FocusThickLens(Float focusDistance) const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz);
        
        Float f = fz[0] - pz[0];
        Float z = -focusDistance;
        Float delta = 0.5f * (pz[1] - z + pz[0] - sqrt((pz[1] - z - pz[0]) * (pz[1] - z - 4.f * f - pz[0])));

        return delta;
    }

    Float get_param(Float &fl, Float3 &fp, Float3 &epc, Float &per) const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz, fp, epc, per);
       
        Float inv_f = 1.f / (fz[0] - pz[0]) - 1.f / (fz[1] - pz[1]);
        fl = abs(fz[0] - pz[0]);

        Float filmZ = 0.f;
        Float sd = abs(filmZ - pz[1]);
        fp = make_float3(0.f, 0.f, pz[0]);
        return sd;
    }

    Float focusdistance() const {
        Float2 pz = make_float2(0.f);
        Float2 fz = make_float2(0.f);
        ComputeThickLensApproximation(pz, fz);
        
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

       

        Var<Ray> rFilm;
        $while (i < 30) {
            

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


    void sphere_uv_from_xyz(const Float3& p, const Float3& c, Float r, Float& u, Float& v) const{//safe 採用
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
        $if (xy2 < 1e-10f) {
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

        // ∂p/∂theta, ∂p/∂phi
        Float3 dp_dtheta = make_float3( -r * sin_ph * sin_th, r * sin_ph * cos_th, 0.f);
        Float3 dp_dphi = make_float3( r * cos_ph * cos_th, r * cos_ph * sin_th, -r * sin_ph);

        // chain rule: theta=2πu, phi=πv
        dp_du = dp_dtheta * two_pi;
        dp_dv = dp_dphi * pi;
    }

    Float3 safe_normalize(const Float3& v) const {
        Float n2 = dot(v, v);
        Float3 ret;
        $if(n2 <= 1e-10f) {
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
        $if(dot(s0, s0) < 1e-10f) {
            s0 = dp_dv;
        };

        // tangentize
        s0 = s0 - n * dot(n, s0);
        s = safe_normalize(s0);

        Float3 t0 = dp_dv - s * dot(s, dp_dv);
        t0 = t0 - n * dot(n, t0);

        $if(dot(t0, t0) < 1e-10f) {
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

       
    }


    void reorient_frame(Float3 &s, Float3 &t, Float3 s_ref, Float3 t_ref) const {
        $if (dot(s, s_ref) < 0.f) {
            s = -s;
            t = -t;
        };
        $if (dot(t, t_ref) < 0.f) { t = -t; };
    }

    void build_tangent_frame_from_n(Float3 n, Float3 &s, Float3 &t) const {
        Float3 a;
        $if (abs(n.z) < 0.999f) { a = make_float3(0.f, 0.f, 1.f); }
        $else { a = make_float3(1.f, 0.f, 0.f); };

        s = normalize(cross(a, n));
        t = cross(n, s);
    }

    Float3 project_to_sphere(Float3 p, Float3 C, Float R) const {
        Float3 v = p - C;
        Float len2 = dot(v, v);
        Float3 ret;
        $if (len2 < 1e-20f) {
            ret =  C + make_float3(0.f, 0.f, R); 
        }
        $else {
            Float inv_len = rsqrt(len2);
            ret = C + v * (R * inv_len);
        };
        
        return ret;
    }


    void st_and_derivs_from_xyz_local2d_fd(
        Var<ChainVerts> &path,
        Float eps_a_in, Float eps_b_in) const {

        auto dex = path.index;
        Float radius = abs(curvanature->read(dex));

        // eps を半径スケールに合わせる（任意だが推奨）
        Float eps_a = max(eps_a_in * radius, 1e-7f);
        Float eps_b = max(eps_b_in * radius, 1e-7f);

        // 1) point を球面へ正規化
        path.point = project_to_sphere(path.point, path.center, radius);

        // 2) 法線
        Float3 n0 = normalize(path.point - path.center);
        path.n = n0;

        // 3) s,t を法線から安定に生成
        Float3 s0, t0;
        build_tangent_frame_from_n(n0, s0, t0);
        path.s = s0;
        path.t = t0;

        // 4) dp/da, dp/db（既存名を流用）
        path.dp_du = s0;
        path.dp_dv = t0;

        // ---- a方向 FD ----
        Float3 p_ap = project_to_sphere(path.point + s0 * eps_a, path.center, radius);
        Float3 n_ap = normalize(p_ap - path.center);
        Float3 s_ap, t_ap;
        build_tangent_frame_from_n(n_ap, s_ap, t_ap);
        reorient_frame(s_ap, t_ap, s0, t0);

        Float3 p_am = project_to_sphere(path.point - s0 * eps_a, path.center, radius);
        Float3 n_am = normalize(p_am - path.center);
        Float3 s_am, t_am;
        build_tangent_frame_from_n(n_am, s_am, t_am);
        reorient_frame(s_am, t_am, s0, t0);

        Float inv2ea = 1.f / (2.f * eps_a);
        path.ds_du = (s_ap - s_am) * inv2ea;
        path.dt_du = (t_ap - t_am) * inv2ea;

        // ---- b方向 FD ----
        Float3 p_bp = project_to_sphere(path.point + t0 * eps_b, path.center, radius);
        Float3 n_bp = normalize(p_bp - path.center);
        Float3 s_bp, t_bp;
        build_tangent_frame_from_n(n_bp, s_bp, t_bp);
        reorient_frame(s_bp, t_bp, s0, t0);

        Float3 p_bm = project_to_sphere(path.point - t0 * eps_b, path.center, radius);
        Float3 n_bm = normalize(p_bm - path.center);
        Float3 s_bm, t_bm;
        build_tangent_frame_from_n(n_bm, s_bm, t_bm);
        reorient_frame(s_bm, t_bm, s0, t0);

        Float inv2eb = 1.f / (2.f * eps_b);
        path.ds_dv = (s_bp - s_bm) * inv2eb;
        path.dt_dv = (t_bp - t_bm) * inv2eb;
    }



    //変化量の計算
    Bool invert_tridiagonal_step( Int size) const{
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
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
        auto id = coord1D;
        Bool judge = true;

        $if (si != 0) {
            auto path_f = vertex->read(id);
            path_f.tmp = path_f.dC_dx_prev;
            Var<float2x2> m = path_f.dC_dx_cur;

            $if (!(invert(m, path_f.inv_lambda))) {
                judge = false;
            };

            $if (judge) {
                // gamma0 = inv(B0) * C0
                path_f.tmp = path_f.inv_lambda * path_f.dC_dx_next;// gamma
                // rhs0 = inv(B0) * r0
                path_f.dx = path_f.inv_lambda * path_f.C;
            };

            vertex->write(id, path_f);

            $if (judge) {
                $for (i, si - 1) {
                    Int k = i + 1;
                    auto path_k = vertex->read(id + k);
                    auto path_k_p = vertex->read(id + i);

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

                    path_k.tmp = path_k.dC_dx_prev * path_k_p.inv_lambda;
                    Float2x2 m = path_k.dC_dx_cur - path_k.tmp * path_k_p.dC_dx_next;
                    $if (!invert(m, path_k.inv_lambda)) {
                        judge = false;
                        $break;
                    };
                    vertex->write(id + k, path_k);
                    

                };
            };

            $if (judge) {
                auto path_f = vertex->read(id);
                path_f.dx = path_f.C;
                vertex->write(id, path_f);

                $for (i, si - 1) {
                    Int k = i + 1;
                    auto path_k = vertex->read(id + k);
                    auto path_k_p = vertex->read(id + i);
                    path_k.dx = path_k.C - path_k.tmp * path_k_p.dx;
                    vertex->write(id + k, path_k);
                };

                auto path_l = vertex->read(id + si - 1);
                path_l.dx = path_l.inv_lambda * path_l.dx;
                vertex->write(id + si - 1, path_l);

                $for (i, si - 1) {
                    auto idx = si - i - 2;
                    auto path_idx = vertex->read(id + idx);
                    auto path_idx_n = vertex->read(id + idx + 1);
                    path_idx.dx = path_idx.inv_lambda * (path_idx.dx - path_idx.dC_dx_next * path_idx_n.dx);
                    vertex->write(id + idx, path_idx);
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

    Bool invert_tridiagonal_step_propose(Int size) const {
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
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
        auto id = coord1D;
        Bool judge = true;

        $if (si != 0) {
            auto path_f = propose_vertex->read(id);
            path_f.tmp = path_f.dC_dx_prev;
            Var<float2x2> m = path_f.dC_dx_cur;

            $if (!(invert(m, path_f.inv_lambda))) {
                judge = false;
            };

            $if (judge) {
                // gamma0 = inv(B0) * C0
                path_f.tmp = path_f.inv_lambda * path_f.dC_dx_next;// gamma
                // rhs0 = inv(B0) * r0
                path_f.dx = path_f.inv_lambda * path_f.C;
            };

            propose_vertex->write(id, path_f);

            $if (judge) {
                $for (i, si - 1) {
                    Int k = i + 1;
                    auto path_k = propose_vertex->read(id + k);
                    auto path_k_p = propose_vertex->read(id + i);

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

                    path_k.tmp = path_k.dC_dx_prev * path_k_p.inv_lambda;
                    Float2x2 m = path_k.dC_dx_cur - path_k.tmp * path_k_p.dC_dx_next;
                    $if (!invert(m, path_k.inv_lambda)) {
                        judge = false;
                        $break;
                    };
                    propose_vertex->write(id + k, path_k);
                };
            };

            $if (judge) {
                auto path_f = propose_vertex->read(id);
                path_f.dx = path_f.C;
                propose_vertex->write(id, path_f);

                $for (i, si - 1) {
                    Int k = i + 1;
                    auto path_k = propose_vertex->read(id + k);
                    auto path_k_p = propose_vertex->read(id + i);
                    path_k.dx = path_k.C - path_k.tmp * path_k_p.dx;
                    propose_vertex->write(id + k, path_k);
                };

                auto path_l = propose_vertex->read(id + si - 1);
                path_l.dx = path_l.inv_lambda * path_l.dx;
                propose_vertex->write(id + si - 1, path_l);

                $for (i, si - 1) {
                    auto idx = si - i - 2;
                    auto path_idx = propose_vertex->read(id + idx);
                    auto path_idx_n = propose_vertex->read(id + idx + 1);
                    path_idx.dx = path_idx.inv_lambda * (path_idx.dx - path_idx.dC_dx_next * path_idx_n.dx);
                    propose_vertex->write(id + idx, path_idx);
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


    //frameの次を実装


    Float compute_Max_Cp(Float3 start, Float3 emit, ArrayFloat3<CHAIN> path, Int Size) const {
        
        Bool compute = true;
        Int size = Size;
        
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
        Float max_C_p = 0.f;

        
        
        $for (i, size) {
            auto elem_path = vertex->read(coord1D + i);
            
            elem_path.point = path[i];
            auto id = coord1D + i;
            //st_and_derivs_from_xyz_fd(elem_path, 1e-4f, 1e-4f);
            st_and_derivs_from_xyz_local2d_fd(elem_path, 1e-4f, 1e-4f);
            propose_vertex->write(id, elem_path);
        };

        $for (i, size) {

            auto id = coord1D + i;
            auto path = propose_vertex->read(id);
            auto path_i_p = propose_vertex->read(id);
            auto path_i_n = propose_vertex->read(id);

            //set C
            path.C = make_float2(0.f);
            path.dC_dx_prev = make_float2x2(0.f);
            path.dC_dx_cur = make_float2x2(0.f);
            path.dC_dx_next = make_float2x2(0.f);

            auto dex = path.index;

            //set point
            Float3 x_prev;
            Float3 x_next;
            Float3 x_cur = path.point;

            $if (i == 0) {
                x_prev = start;
            }
            $else {
                path_i_p = propose_vertex->read(id - 1);
                x_prev = path_i_p.point;
            };

            $if (i == size - 1) {
                x_next = emit;
            }
            $else {
                path_i_n = propose_vertex->read(id + 1);
                x_next = path_i_n.point;
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

            Float3 nn = path.n;

            $if (dot(nn, wi) > 0.f) {
                // wi が外側に出ていく方向（=入射は内側から来た）
                // incident medium は内側

                //path[i].n = -nn;// “入射側に向いた法線”へ揃える流儀も多い
                nn = -nn;
            }
            $else {
                // wi が内側へ向く（=入射は外側から来た）

                nn = nn;
            };
            Float3 h = wi + eta * wo;
            $if (dot(h, nn) < 0.f) {
                //h = h * -1.f;
            };

            Float h_l = dot(h, h);
            $if (h_l < 1e-7f) {
                compute = false;
                $break;
            };


            h = h * -1.f;
            Float ilh = 1.f / length(h);
            h = normalize(h);

            ilo = ilo * eta * ilh;
            ili = ili * ilh;

            //prepare u,v

           

            //st_and_derivs_from_xyz_fd(path[i], 1e-4f, 1e-4f);
            $if (i > 0) {
                //st_and_derivs_from_xyz_fd(path[i - 1], 1e-4f, 1e-4f);
            };

            // Derivative of specular constraint w.r.t. x_{i-1}
            Float3 dh_du;
            Float3 dh_dv;

            $if (i > 0) {
                dh_du = ili * (path_i_p.dp_du - wi * dot(wi, path_i_p.dp_du));
                dh_dv = ili * (path_i_p.dp_dv - wi * dot(wi, path_i_p.dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path.dC_dx_prev = make_float2x2(
                    dot(path.s, dh_du), dot(path.s, dh_dv),
                    dot(path.t, dh_du), dot(path.t, dh_dv));
            };

            // Derivative of specular constraint w.r.t. x_{i}
            $if (end_fixed_direction) {
                // When the 'wo' direction is fixed, the derivative here simplifies.
                dh_du = ili * (-path.dp_du + wi * dot(wi, path.dp_du));
                dh_dv = ili * (-path.dp_dv + wi * dot(wi, path.dp_dv));
            }
            $else {
                // Standard case for fixed emitter position
                dh_du = -path.dp_du * (ili + ilo) + wi * (dot(wi, path.dp_du) * ili) + wo * (dot(wo, path.dp_du) * ilo);
                dh_dv = -path.dp_dv * (ili + ilo) + wi * (dot(wi, path.dp_dv) * ili) + wo * (dot(wo, path.dp_dv) * ilo);
            };

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            $if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            };

            path.dC_dx_cur = make_float2x2(
                dot(path.ds_du, h) + dot(path.s, dh_du), dot(path.ds_dv, h) + dot(path.s, dh_dv),
                dot(path.dt_du, h) + dot(path.t, dh_du), dot(path.dt_dv, h) + dot(path.t, dh_dv));

            // Derivative of specular constraint w.r.t. x_{i+1}
            $if (i < size - 1) {
                dh_du = ilo * (path_i_n.dp_du - wo * dot(wo, path_i_n.dp_du));
                dh_dv = ilo * (path_i_n.dp_dv - wo * dot(wo, path_i_n.dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path.dC_dx_next = make_float2x2(
                    dot(path.s, dh_du), dot(path.s, dh_dv),
                    dot(path.t, dh_du), dot(path.t, dh_dv));
            };

            // Evaluate specular constraint
            auto H = make_float2(dot(path.s, h), dot(path.t, h));
            auto n_offset = make_float3(0.f, 0.f, 1.f);
            auto N = make_float2(n_offset[0], n_offset[1]);
            path.C = H - N;
            propose_vertex->write(id, path);
        };

        $if (!invert_tridiagonal_step_propose(size)) {
            compute = false;
        };

       

        $if (compute) {

            $for (i, size) {
                auto p_path = propose_vertex->read(coord1D + i);
                $if (max_C_p < length(p_path.C)) {
                    max_C_p = length(p_path.C);
                };
            };
        }
        $else {
            max_C_p = 1000.f;
        };


        return max_C_p;
    }

    Bool compute_der_halfvector(Float3 start, Float3 emit, Int Size) const {
       
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;

        
        //auto path = vertex->read(coord1D);
        Bool compute = true;
        Int size = Size;
        

        

        

        $for (i, size) {
            auto id = coord1D + i;
            auto path = vertex->read(id);
            //st_and_derivs_from_xyz_fd(path, 1e-4f, 1e-4f);
            st_and_derivs_from_xyz_local2d_fd(path, 1e-4f, 1e-4f);
            vertex->write(id, path);
            
        };

        $for (i, size) {

            auto id = coord1D + i;
            auto path = vertex->read(id);
            auto path_i_p = vertex->read(id);
            auto path_i_n = vertex->read(id);

            //set C
            path.C = make_float2(0.f);
            path.dC_dx_prev = make_float2x2(0.f);
            path.dC_dx_cur = make_float2x2(0.f);
            path.dC_dx_next = make_float2x2(0.f);

            auto dex = path.index;

            //set point
            Float3 x_prev;
            Float3 x_next;
            Float3 x_cur = path.point;

            $if (i == 0) {
                x_prev = start;
            }
            $else {
                path_i_p = vertex->read(id - 1);
                x_prev = path_i_p.point;
            };

            $if (i == size - 1) {
                x_next = emit;
            }
            $else {
                path_i_n = vertex->read(id + 1);
                x_next = path_i_n.point;
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

            Float3 nn = path.n;

            $if (dot(nn, wi) > 0.f) {
                // wi が外側に出ていく方向（=入射は内側から来た）
                // incident medium は内側

                //path[i].n = -nn;// “入射側に向いた法線”へ揃える流儀も多い
                nn = -nn;
            }
            $else {
                // wi が内側へ向く（=入射は外側から来た）

                nn = nn;
            };
            Float3 h = wi + eta * wo;
            $if (dot(h, nn) < 0.f) {
                //h = h * -1.f;
            };

            Float h_l = dot(h, h);
            $if (h_l < 1e-7f) {
                compute = false;
                $break;
            };

            h = h * -1.f;
            Float ilh = 1.f / length(h);
            h = normalize(h);

            ilo = ilo * eta * ilh;
            ili = ili * ilh;

            //prepare u,v

           

            //st_and_derivs_from_xyz_fd(path[i], 1e-4f, 1e-4f);
            $if (i > 0) {
                //st_and_derivs_from_xyz_fd(path[i - 1], 1e-4f, 1e-4f);
            };

           

            // Derivative of specular constraint w.r.t. x_{i-1}
            Float3 dh_du;
            Float3 dh_dv;

            $if (i > 0) {
                dh_du = ili * (path_i_p.dp_du - wi * dot(wi, path_i_p.dp_du));
                dh_dv = ili * (path_i_p.dp_dv - wi * dot(wi, path_i_p.dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path.dC_dx_prev = make_float2x2(
                    dot(path.s, dh_du), dot(path.s, dh_dv),
                    dot(path.t, dh_du), dot(path.t, dh_dv));
            };

            // Derivative of specular constraint w.r.t. x_{i}
            $if (end_fixed_direction) {
                // When the 'wo' direction is fixed, the derivative here simplifies.
                dh_du = ili * (-path.dp_du + wi * dot(wi, path.dp_du));
                dh_dv = ili * (-path.dp_dv + wi * dot(wi, path.dp_dv));
            }
            $else {
                // Standard case for fixed emitter position
                dh_du = -path.dp_du * (ili + ilo) + wi * (dot(wi, path.dp_du) * ili) + wo * (dot(wo, path.dp_du) * ilo);
                dh_dv = -path.dp_dv * (ili + ilo) + wi * (dot(wi, path.dp_dv) * ili) + wo * (dot(wo, path.dp_dv) * ilo);
            };

            dh_du -= h * dot(dh_du, h);
            dh_dv -= h * dot(dh_dv, h);

            $if (eta != 1.f) {
                dh_du *= -1.f;
                dh_dv *= -1.f;
            };

            path.dC_dx_cur = make_float2x2(
                dot(path.ds_du, h) + dot(path.s, dh_du), dot(path.ds_dv, h) + dot(path.s, dh_dv),
                dot(path.dt_du, h) + dot(path.t, dh_du), dot(path.dt_dv, h) + dot(path.t, dh_dv));

            // Derivative of specular constraint w.r.t. x_{i+1}
            $if (i < size - 1) {
                dh_du = ilo * (path_i_n.dp_du - wo * dot(wo, path_i_n.dp_du));
                dh_dv = ilo * (path_i_n.dp_dv - wo * dot(wo, path_i_n.dp_dv));

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                $if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                };

                path.dC_dx_next = make_float2x2(
                    dot(path.s, dh_du), dot(path.s, dh_dv),
                    dot(path.t, dh_du), dot(path.t, dh_dv));
            };

            // Evaluate specular constraint
            auto H = make_float2(dot(path.s, h), dot(path.t, h));
            auto n_offset = make_float3(0.f, 0.f, 1.f);
            auto N = make_float2(n_offset[0], n_offset[1]);
            path.C = H - N;
            vertex->write(id, path);
        };
        
        

        $if (!invert_tridiagonal_step(size)) {
            compute = false;
        };
        //vertex->write(coord1D, path);


        return compute;
    }

    Bool reproject(const Float3 start, const Float3 emit, ArrayFloat3<20> proposed_path, const Int size) const {
        
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;
       

        $for (i, size) {
            auto k = i;
            auto propath = vertex->read(coord1D + k);
            auto gene_center = propath.center;
            auto pro_idx = propath.index;
            

            proposed_path[k] = gene_center + abs(curvanature->read(pro_idx)) * normalize(proposed_path[k] - gene_center);
        };

        Float3 first_point = proposed_path[0];
        Float3 direction_to_first = first_point - start;

        Var<Ray> proposed_ray = make_ray(start, direction_to_first);
        Var<Ray> generate_ray;

        Bool success = TraceLences(proposed_ray, &generate_ray);

        /*
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("repro {},{}", generate_ray->origin(),proposed_path[1]);
        };
         */
        
        

        
        


        return success;
    }

    Bool last_check(const Float3 start, const Float3 emit, const Int size) const {

        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;

        $for (i, size) {
            auto k = i;
            auto path = vertex->read(coord1D + k);
            auto gene_center = path.center;
            auto pro_idx = path.index;

            path.point = gene_center + abs(curvanature->read(pro_idx)) * normalize(path.point - gene_center);
            vertex->write(coord1D + k, path);
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("newton_point= {}", path.point);
            };
        };

        auto first_v = vertex->read(coord1D);
        Float3 first_point = first_v.point;
        Float3 direction_to_first = first_point - start;

        Var<Ray> proposed_ray = make_ray(start, direction_to_first);
        Var<Ray> generate_ray;

        Bool success = TraceLences(proposed_ray, &generate_ray);

         
        $if (success) {
            Float distance = -4.95f - generate_ray->origin().z;
            Float t = distance / generate_ray->direction().z;
            Float3 target = generate_ray->origin() + t * generate_ray->direction();
            

            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("check {}", target);
            };

            auto last_v = vertex->read(coord1D + size - 1);
            Float check_d = length(generate_ray->origin() - last_v.point);
            Float angle_d;
            $if (check_d > length_threshold) {
                success = false;
            }
            $else {
                Float3 target_angle = normalize(emit - last_v.point);
                angle_d = 1.f - dot(normalize(generate_ray->direction()), target_angle);
                $if (angle_d > angle_threshold) {
                    success = false;
                }
                $else {
                    $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                        luisa::compute::device_log("check {},{}", last_v.point, target_angle);
                    };
                };
            };

            
        };

        return success;
    }


    Bool newton_solver(const Float3 start, const Float3 emit ) const {
        
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x) * CHAIN;

        
        
       
        Bool newton = true;
        Float number = 0.f;

        Bool success = false;
        UInt iterations = 0u;
        Float beta = 1.f;
        


        ArrayFloat3<CHAIN> proposed_path;
        const auto lc = node<RealLensCamera>()->lens_count();
        
        Int size = lc - 1;
       
        
        
         
        Bool use_half_vector = true;
        Bool needs_step_update = true;
        $for (it, MAX_I) {
            Bool step_success = true;
            
            $if (needs_step_update) {
                $if (use_half_vector) {
                    // Use standard manifold formulation using half-vector constraints
                   step_success = compute_der_halfvector(start, emit, size);
                   
                }
                $else {
                    // Use angle-difference constraint formulation
                    //step_success = compute_der_anglediff(si.p, ei);
                };
            };
            
            //auto coord = dispatch_id().xy();
            //auto coord1D = coord.y * RESOLUTION + coord.x;
            //auto path = vertex->read(coord1D);
            
            $if (!step_success) {
                $break;
            };

            $for (i, size) {
                auto id = coord1D + i;
                auto path = vertex->read(id);
                /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("check_p {}", path.v);
                };*/
            };
            
            // Check for success
            Bool converged = true;
            Float max_C = 0.f;
            $for (i, size) {
                auto id = coord1D + i;
                auto path = vertex->read(id);
                $if(max_C < length(path.C)) {
                    max_C = length(path.C);
                };

                $if (length(path.C) > solver_threshold) {
                    converged = false;
                    
                    
                };
            };

           

            $if (converged) {
               
                $if (last_check(start, emit, size)) {
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
                proposed_path[i] = make_float3(0.f);
            };
            
            $for (i, size) {
                auto path = vertex->read(coord1D + i);
                Float3 p_prop = path.point - step_scale * beta * (path.dp_du * path.dx[0] + path.dp_dv * path.dx[1]);
                proposed_path[i] = p_prop;
                
            };

            // Project back to surfaces
            Bool project_success = reproject(start, emit, proposed_path, size);
           
           

            
            
            

            
            
            

            $if (project_success) {
                Float max_C_p =  compute_Max_Cp(start, emit, proposed_path, size);
                project_success = (max_C  > max_C_p);
               /* $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("check {},{}", max_C, max_C_p);
                }; */
            };
            


            $if (!project_success) {
                beta = 0.5f * beta;
                needs_step_update = false;
            } 
            $else {
                beta = min(1.f, 2.f * beta);
                $for (i, size) {
                    auto path = vertex->read(coord1D + i);
                    path.point = proposed_path[i];
                    vertex->write(coord1D + i, path);
                };
                needs_step_update = true;
            };

            $if (beta < 1e-6f) {
                //$break;
            };

            //vertex->write(coord1D, path);
            iterations = iterations + 1;
        };

        $if (!success) {
            newton = false;
        }
        $else {
            
            $for (i, size) {
                //set point
                auto path = vertex->read(coord1D + i);
                Float3 x_prev;
                Float3 x_next;
                Float3 x_cur = path.point;

                $if (i == 0) {
                    x_prev = start;
                }
                $else {
                    auto path_p = vertex->read(coord1D + i - 1);
                    x_prev = path_p.point;
                };

                $if (i == size - 1) {
                    x_next = emit;
                }
                $else {
                    auto path_n = vertex->read(coord1D + i + 1);
                    x_next = path_n.point;
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

                Float cos_theta_i = dot(path.n, wi);
                Float cos_theta_o = dot(path.n, wo);
                Bool refract = cos_theta_i * cos_theta_o < 0.f;
                Bool reflect = !refract;

                Float eta;
                auto dex = path.index;
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

        
       
       
        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x);

        
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


        auto p_lens = sample_uniform_disk(u_lens, LensRearRadius(lc - 1), LensRearZ(lc - 1) * 1.2f);
        //p_lens = sample_uniform_disk(u_lens, 0.001f, LensRearZ(lc - 1));
        auto scene_d = normalize(p_lens - coordScene);

        uint tes = 1u;
        uint tes2 = 2u;
        auto test = "test";

        uint tes3 = select(tes, tes2, false);
        
        Float dlt = 0.f;
        //dlt = FocusThickLens(3.f);
        //delta->write(0, dlt);
       
       

        auto test_d = make_float3(0.f, 0.f, -1.f);
        
        //sceme
        auto ray = make_ray(coordScene, scene_d);
        auto first_ray = make_ray(coordScene, scene_d);

        //test
        //first_ray = make_ray(coordScene, test_d);
        //real system
        


        Bool trace;
        auto bunki = elem.sign ;
        $if (bunki == 0) {
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("LensTrace {}", first_ray->origin());
            };
           
            trace = TraceLencesSave(first_ray, &ray);
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
                luisa::compute::device_log("Manifold Exploration start {}, {}, {}", elem.use, elem.emit, elem.first);
            };
            Float3 emit;
            $if(elem.use != 1) {
                trace = false;
            }
            $else {
                Float3 start = elem.first;
                emit = elem.emit;
                trace = newton_solver(start, emit);
            };
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("Manifold Exploration {}", trace);
            };
            $if (!trace) {
                weight = 0.f;
            }
            $else {
                auto coord1D_chain = (coord.y * RESOLUTION + coord.x) * CHAIN;
                auto path = vertex->read(coord1D_chain + lc - 2);
                Float3 ray_o = path.point;
                Float3 ray_d = normalize(emit - ray_o);
                $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                    luisa::compute::device_log("check_ray {},{}", ray_o, ray_d);
                };
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
       
        
       

        //6.4.3
        
        Float pz;
        Float fz;
        //ComputeCardinalPoints(first_ray, ray, &pz, &fz);

        BB2D testpupil{};
        testpupil = BoundExitPupil(1.f, 1.f);


       
        
       //vertex->write(0, );


        
        

        Float cosTheta = normalize(first_ray->direction()).z;
        Float cos4Theta = (cosTheta * cosTheta) * (cosTheta * cosTheta);
        weight = weight * cos4Theta;


        /* Float3 start = make_float3(0.f);
        Float3 emit = make_float3(0.f, 0.f, -5.f);
        Bool NS = newton_solver(start, emit, path);
        //test_ME
        
        

      // Bool ME = compute_der_halfvector(start, emit, path);
        */

        
/**/



        return std::make_pair(std::move(ray), weight);
    }

    [[nodiscard]] Bool _get_camera_param(Float &aper, Float &fl, Float &fd, Float &sd, Float3 &fp) const noexcept override { //aper = entrance pupil, fl = EFL, fd = compute, sd = pz
        Bool get_p = true;

        auto data = _device_data->read(0u);
        
        auto sceneX = .024f;
        auto sceneY = .024f;
        Float2 resolution = data.resolution;

        Float coordX = (fp.x - data.pixel_offset.x) * sceneX / resolution.x;
        Float coordY = (fp.y - data.pixel_offset.y) * sceneY / resolution.y;
        fp = make_float3(coordX, coordY, 0.f);
        Float3 epc;
        Float pt;
        sd = get_param(fl, fp, epc, pt);
        aper = get_aper() * 2.f;
        fd =  (1.f / (1.f / fl - 1.f / sd));
        //fd = focusdistance();
        fp = fp + fd * make_float3(0.f, 0.f, -1.f);

        return get_p;
    }

     [[nodiscard]] Bool _get_camera_param(Float &aper, Float &fl, Float &fd, Float &sd, Float3 &fp, Float3 &sp, Float3 &epc) const noexcept override {//aper = entrance pupil, fl = EFL, fd = compute, sd = pz
        Bool get_p = true;

        auto data = _device_data->read(0u);

        auto sceneX = .024f;
        auto sceneY = .024f;
        Float2 resolution = data.resolution;

        Float coordX = (fp.x - data.pixel_offset.x) * sceneX / resolution.x;
        Float coordY = (fp.y - data.pixel_offset.y) * sceneY / resolution.y;
        fp = make_float3(coordX, coordY, 0.f);
        sp = make_float3(coordX, coordY, 0.f);
        Float percentage;
        sd = get_param(fl, fp, epc, percentage);

        $if (refraction->read(0) == 0.f) {
           aper = radius->read(0) * 2.f;
        }
        $else {
           aper = radius->read(0) * 2.f * percentage;
        };
        
        //aper = get_aper() * 2.f;


        fd = (1.f / (1.f / fl - 1.f / sd));
        //fd = 0.6f;
        //fd = focusdistance();

        Float3 n = make_float3(0.f, 0.f, -1.f);
        Float3 fp0 = fp + n * fd;

        Float3 dir = normalize(epc - sp);
        $if (dir.z > 0) {
            dir = -dir;
        };
        Var<Ray> rFilm = make_ray(sp, dir);
        Var<Ray> rScene;
        Bool han = TraceLences(rFilm, &rScene);
        
        Float denom = dot(rScene->direction(), n);

        Float t = dot(fp0 - rScene->origin(), n) / denom;
        fp = rScene->origin() + t * rScene->direction();

        Float s = abs(5.f - sd);
        Float sprime = (1.f / (1.f / fl - 1.f / s));
        //fp = fp + fd * make_float3(0.f, 0.f, -1.f);
        $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
            luisa::compute::device_log("test_fp = {}, {}", epc, rScene->direction());
        };
        return get_p;
    }

    [[nodiscard]] std::pair<Var<Ray>, Float> _get_ray_Manifold(Expr<float2> pixel, Float3 emit, Float sign) const noexcept override {
        Float get_manifold_weight = 1.f;

        auto coord = dispatch_id().xy();
        auto coord1D = (coord.y * RESOLUTION + coord.x);

        //Var<ChainVerts[CHAIN]> path = vertex->read(coord1D);

        auto get = element->read(coord1D);
        get.emit = emit;
        get.sign = sign;
        
        $if (sign == 0) {
            auto data = _device_data->read(0u);
            Float2 resolution = data.resolution;
            auto sceneX = .024f;
            auto sceneY = .024f;
            Float coordX = (pixel.x - data.pixel_offset.x) * sceneX / resolution.x;
            Float coordY = (pixel.y - data.pixel_offset.y) * sceneY / resolution.y;
            Float3 coordScene = make_float3(coordX, coordY, 0.f);

            Bool seed = make_seed(emit, coordScene);

            $if (seed) {
                get.use = 1;
            }
            $else {
                get.use = 0;
            };
            
            $if (luisa::compute::all((luisa::compute::dispatch_size().xy() / 2u) + make_uint2(X, Y) == luisa::compute::dispatch_id().xy())) {
                luisa::compute::device_log("test_me = {}, {}", coordScene, emit);
            };


        };

       
        //get.first = coordScene;

        

        element->write(coord1D, get);
        Var<Ray> ret_ray = make_ray(make_float3(0.f), make_float3(0.f));

        

;
        
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