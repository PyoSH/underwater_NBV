"""
Warp GPU kernels for ImagingSonar sensor (single-env, pointcloud annotator).

Convention:
  - Per-point kernels:  dim = (M,)      — M: number of points from pointcloud annotator
  - Per-bin kernels:    dim = (R, A)    — R: range bins, A: azimuth bins
  - r / azi meshgrid shape: (R, A)
"""
import warp as wp


@wp.func
def cartesian_to_spherical(cart: wp.vec3) -> wp.vec3:
    r = wp.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])
    if r < wp.float32(1e-6):
        return wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    cos_polar = wp.clamp(cart[2] / r, wp.float32(-1.0), wp.float32(1.0))
    return wp.vec3(r, wp.atan2(cart[1], cart[0]), wp.acos(cos_polar))


# ---------------------------------------------------------------------------
# Per-point kernels   dim = (M,)
# ---------------------------------------------------------------------------

@wp.kernel
def compute_intensity(
        pcl:          wp.array(ndim=2, dtype=wp.float32),  # (M, 3) world frame
        normals:      wp.array(ndim=2, dtype=wp.float32),  # (M, 3) world frame
        viewTransform: wp.mat44,
        refl_per_pt:  wp.array(ndim=1, dtype=wp.float32),  # (M,) per-point reflectivity
        attenuation:  float,
        intensity:    wp.array(dtype=wp.float32)            # (M,) [out]
):
    tid = wp.tid()
    pcl_vec    = wp.vec3(pcl[tid, 0], pcl[tid, 1], pcl[tid, 2])
    normal_vec = wp.vec3(normals[tid, 0], normals[tid, 1], normals[tid, 2])
    R = wp.mat33(viewTransform[0, 0], viewTransform[0, 1], viewTransform[0, 2],
                 viewTransform[1, 0], viewTransform[1, 1], viewTransform[1, 2],
                 viewTransform[2, 0], viewTransform[2, 1], viewTransform[2, 2])
    T = wp.vec3(viewTransform[0, 3], viewTransform[1, 3], viewTransform[2, 3])
    sensor_loc    = -(wp.transpose(R) @ T)
    incidence     = pcl_vec - sensor_loc
    dist          = wp.sqrt(incidence[0]*incidence[0] +
                            incidence[1]*incidence[1] +
                            incidence[2]*incidence[2])
    unit_directs  = wp.normalize(pcl_vec - sensor_loc)
    cos_theta     = wp.dot(-unit_directs, normal_vec)
    intensity[tid] = refl_per_pt[tid] * cos_theta * wp.exp(-attenuation * dist)


@wp.kernel
def world2local(
        viewTransform:   wp.mat44,
        pcl_world:       wp.array(ndim=2, dtype=wp.float32),  # (M, 3)
        pcl_local:       wp.array(dtype=wp.vec3),              # (M,) [out]
        pcl_local_spher: wp.array(dtype=wp.vec3)               # (M,) [out]
):
    tid = wp.tid()
    pcl_world_h = wp.vec4(pcl_world[tid, 0], pcl_world[tid, 1],
                          pcl_world[tid, 2], wp.float32(1.0))
    p = viewTransform @ pcl_world_h
    p_local = wp.vec3(p[0], -p[2], p[1])   # y-forward axis swap (oceansim 동일)
    pcl_local[tid]       = p_local
    pcl_local_spher[tid] = cartesian_to_spherical(p_local)


# ---------------------------------------------------------------------------
# Binning kernel   dim = (M,)
# ---------------------------------------------------------------------------

@wp.kernel
def bin_intensity(
        pcl_spher: wp.array(dtype=wp.vec3),
        intensity: wp.array(dtype=wp.float32),
        x_offset: wp.float32, y_offset: wp.float32,
        x_res:    wp.float32, y_res:    wp.float32,
        bin_sum:   wp.array(ndim=2, dtype=wp.float32),  # (R, A) [out]
        bin_count: wp.array(ndim=2, dtype=wp.int32)     # (R, A) [out]
):
    tid = wp.tid()
    x = pcl_spher[tid][0]
    y = pcl_spher[tid][1]
    x_bin = wp.int32((x - x_offset) / x_res)
    y_bin = wp.int32((y - y_offset) / y_res)
    if x_bin < 0 or x_bin >= bin_sum.shape[0]:
        return
    if y_bin < 0 or y_bin >= bin_sum.shape[1]:
        return
    wp.atomic_add(bin_sum,   x_bin, y_bin, intensity[tid])
    wp.atomic_add(bin_count, x_bin, y_bin, 1)


# ---------------------------------------------------------------------------
# Reduction / averaging kernels   dim = (R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def average(
        sum:   wp.array(ndim=2, dtype=wp.float32),
        count: wp.array(ndim=2, dtype=wp.int32),
        avg:   wp.array(ndim=2, dtype=wp.float32)
):
    i, j = wp.tid()
    if count[i, j] > 0:
        avg[i, j] = sum[i, j] / wp.float32(count[i, j])


@wp.kernel
def all_max(
        array:     wp.array(ndim=2, dtype=wp.float32),
        max_value: wp.array(dtype=wp.float32)           # (1,) [out]
):
    i, j = wp.tid()
    wp.atomic_max(max_value, 0, array[i, j])


@wp.kernel
def range_max(
        array:     wp.array(ndim=2, dtype=wp.float32),
        max_value: wp.array(dtype=wp.float32)           # (R,) [out]
):
    i, j = wp.tid()
    wp.atomic_max(max_value, i, array[i, j])


# ---------------------------------------------------------------------------
# Noise kernels   dim = (R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def normal_2d(
        seed:   int,
        mean:   float,
        std:    float,
        output: wp.array(ndim=2, dtype=wp.float32)
):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)
    output[i, j] = mean + std * wp.randn(state)


@wp.kernel
def range_dependent_rayleigh_2d(
        seed:           int,
        r:              wp.array(ndim=2, dtype=wp.float32),
        azi:            wp.array(ndim=2, dtype=wp.float32),
        max_range:      float,
        rayleigh_scale: float,
        central_peak:   float,
        central_std:    float,
        output:         wp.array(ndim=2, dtype=wp.float32)
):
    i, j = wp.tid()
    state = wp.rand_init(seed, i * output.shape[1] + j)
    n1 = wp.randn(state)
    n2 = wp.randn(state)
    rayleigh = rayleigh_scale * wp.sqrt(n1*n1 + n2*n2)
    output[i, j] = wp.pow(r[i, j] / max_range, 2.0) * \
        (1.0 + central_peak * wp.exp(
            -wp.pow(azi[i, j] - wp.PI / 2.0, 2.0) / central_std)) * rayleigh


# ---------------------------------------------------------------------------
# Compositing kernels   dim = (R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def make_sonar_map_all(
        r:               wp.array(ndim=2, dtype=wp.float32),
        azi:             wp.array(ndim=2, dtype=wp.float32),
        intensity:       wp.array(ndim=2, dtype=wp.float32),  # in-place
        max_intensity:   wp.array(dtype=wp.float32),           # (1,)
        gau_noise:       wp.array(ndim=2, dtype=wp.float32),
        range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
        offset:          wp.float32,
        gain:            wp.float32,
        result:          wp.array(ndim=2, dtype=wp.vec3)       # (R, A) [out]
):
    i, j = wp.tid()
    intensity[i, j]  = intensity[i, j] / max_intensity[0]
    intensity[i, j] += offset
    intensity[i, j] *= gain
    intensity[i, j] *= (0.5 + gau_noise[i, j])
    intensity[i, j] += range_ray_noise[i, j]
    intensity[i, j]  = wp.clamp(intensity[i, j], wp.float32(0.0), wp.float32(1.0))
    result[i, j] = wp.vec3(r[i, j] * wp.cos(azi[i, j]),
                            r[i, j] * wp.sin(azi[i, j]),
                            intensity[i, j])


@wp.kernel
def make_sonar_map_range(
        r:               wp.array(ndim=2, dtype=wp.float32),
        azi:             wp.array(ndim=2, dtype=wp.float32),
        intensity:       wp.array(ndim=2, dtype=wp.float32),  # in-place
        max_intensity:   wp.array(dtype=wp.float32),           # (R,)
        gau_noise:       wp.array(ndim=2, dtype=wp.float32),
        range_ray_noise: wp.array(ndim=2, dtype=wp.float32),
        offset:          wp.float32,
        gain:            wp.float32,
        result:          wp.array(ndim=2, dtype=wp.vec3)       # (R, A) [out]
):
    i, j = wp.tid()
    if max_intensity[i] != 0.0:
        intensity[i, j] = intensity[i, j] / max_intensity[i]
    intensity[i, j] *= (0.5 + gau_noise[i, j])
    intensity[i, j] += range_ray_noise[i, j]
    intensity[i, j] += offset
    intensity[i, j] *= gain
    intensity[i, j]  = wp.clamp(intensity[i, j], wp.float32(0.0), wp.float32(1.0))
    result[i, j] = wp.vec3(r[i, j] * wp.cos(azi[i, j]),
                            r[i, j] * wp.sin(azi[i, j]),
                            intensity[i, j])


@wp.kernel
def make_sonar_image(
        sonar_data:  wp.array(ndim=2, dtype=wp.vec3),   # (R, A)
        sonar_image: wp.array(ndim=3, dtype=wp.uint8)   # (R, A+1, 4) [out]
):
    i, j = wp.tid()
    width     = sonar_data.shape[1]
    sonar_rgb = wp.uint8(sonar_data[i, j][2] * wp.float32(255))
    sonar_image[i, width - j, 0] = sonar_rgb
    sonar_image[i, width - j, 1] = sonar_rgb
    sonar_image[i, width - j, 2] = sonar_rgb
    sonar_image[i, width - j, 3] = wp.uint8(255)
