"""
Warp GPU kernels for ImagingSonar sensor.

Batch convention (matching Isaac Lab Camera output):
  - Per-point kernels:  dim = (N_env, H*W)  →  n, tid = wp.tid()
  - Per-bin kernels:    dim = (N_env, R, A)  →  n, i, j = wp.tid()
  - r / azi meshgrid is shared across envs: shape (R, A), not (N, R, A)
"""
import warp as wp


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@wp.func
def cartesian_to_spherical(cart: wp.vec3) -> wp.vec3:
    """Cartesian (x,y,z) → spherical (r, azimuth, polar)."""
    r = wp.sqrt(cart[0]*cart[0] + cart[1]*cart[1] + cart[2]*cart[2])
    return wp.vec3(r,
                   wp.atan2(cart[1], cart[0]),
                   wp.acos(cart[2] / r))


# ---------------------------------------------------------------------------
# Per-point kernels   dim = (N_env, H*W)
# ---------------------------------------------------------------------------

@wp.kernel
def compute_intensity(
        pcl:            wp.array(ndim=3, dtype=wp.float32),  # (N, H*W, 3)
        normals:        wp.array(ndim=3, dtype=wp.float32),  # (N, H*W, 3)
        viewTransforms: wp.array(ndim=3, dtype=wp.float32),  # (N, 4, 4)
        semantics:      wp.array(ndim=2, dtype=wp.uint32),   # (N, H*W)
        indexToRefl:    wp.array(ndim=2, dtype=wp.float32),  # (N, max_id+1)
        attenuation:    float,
        intensity:      wp.array(ndim=2, dtype=wp.float32)   # (N, H*W) [output]
):
    """reflectivity × cos(θ) × exp(-att × dist) per point per env."""
    n, tid = wp.tid()

    pcl_vec    = wp.vec3(pcl[n, tid, 0], pcl[n, tid, 1], pcl[n, tid, 2])
    normal_vec = wp.vec3(normals[n, tid, 0], normals[n, tid, 1], normals[n, tid, 2])

    R = wp.mat33(
        viewTransforms[n, 0, 0], viewTransforms[n, 0, 1], viewTransforms[n, 0, 2],
        viewTransforms[n, 1, 0], viewTransforms[n, 1, 1], viewTransforms[n, 1, 2],
        viewTransforms[n, 2, 0], viewTransforms[n, 2, 1], viewTransforms[n, 2, 2],
    )
    T = wp.vec3(viewTransforms[n, 0, 3], viewTransforms[n, 1, 3], viewTransforms[n, 2, 3])

    sensor_loc   = -(wp.transpose(R) @ T)
    incidence    = pcl_vec - sensor_loc
    dist         = wp.sqrt(incidence[0]*incidence[0] +
                           incidence[1]*incidence[1] +
                           incidence[2]*incidence[2])
    unit_directs = wp.normalize(pcl_vec - sensor_loc)
    cos_theta    = wp.dot(-unit_directs, normal_vec)
    reflectivity = indexToRefl[n, semantics[n, tid]]

    intensity[n, tid] = reflectivity * cos_theta * wp.exp(-attenuation * dist)


@wp.kernel
def world2local(
        viewTransforms:  wp.array(ndim=3, dtype=wp.float32),  # (N, 4, 4)
        pcl_world:       wp.array(ndim=3, dtype=wp.float32),  # (N, H*W, 3)
        pcl_local:       wp.array(ndim=2, dtype=wp.vec3),     # (N, H*W) [output]
        pcl_local_spher: wp.array(ndim=2, dtype=wp.vec3)      # (N, H*W) [output]
):
    """World → sensor-local (y-forward axis swap) → spherical, per env."""
    n, tid = wp.tid()

    # Manual mat44 × vec4 (Warp의 wp.mat44는 @연산자 미지원)
    x = pcl_world[n, tid, 0]
    y = pcl_world[n, tid, 1]
    z = pcl_world[n, tid, 2]

    r0 = viewTransforms[n, 0, 0]*x + viewTransforms[n, 0, 1]*y + viewTransforms[n, 0, 2]*z + viewTransforms[n, 0, 3]
    r1 = viewTransforms[n, 1, 0]*x + viewTransforms[n, 1, 1]*y + viewTransforms[n, 1, 2]*z + viewTransforms[n, 1, 3]
    r2 = viewTransforms[n, 2, 0]*x + viewTransforms[n, 2, 1]*y + viewTransforms[n, 2, 2]*z + viewTransforms[n, 2, 3]

    p_local = wp.vec3(r0, -r2, r1)  # y-forward axis swap

    pcl_local[n, tid]       = p_local
    pcl_local_spher[n, tid] = cartesian_to_spherical(p_local)


# ---------------------------------------------------------------------------
# Binning kernel   dim = (N_env, H*W)
# ---------------------------------------------------------------------------

@wp.kernel
def bin_intensity(
        pcl_spher: wp.array(ndim=2, dtype=wp.vec3),    # (N, H*W)
        intensity: wp.array(ndim=2, dtype=wp.float32), # (N, H*W)
        x_offset:  wp.float32,
        y_offset:  wp.float32,
        x_res:     wp.float32,
        y_res:     wp.float32,
        bin_sum:   wp.array(ndim=3, dtype=wp.float32), # (N, R, A) [output]
        bin_count: wp.array(ndim=3, dtype=wp.int32)    # (N, R, A) [output]
):
    n, tid = wp.tid()

    x = pcl_spher[n, tid][0]
    y = pcl_spher[n, tid][1]

    x_bin_idx = wp.int32((x - x_offset) / x_res)
    y_bin_idx = wp.int32((y - y_offset) / y_res)

    wp.atomic_add(bin_sum,   n, x_bin_idx, y_bin_idx, intensity[n, tid])
    wp.atomic_add(bin_count, n, x_bin_idx, y_bin_idx, 1)


# ---------------------------------------------------------------------------
# Reduction / averaging kernels   dim = (N_env, R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def average(
        sum:   wp.array(ndim=3, dtype=wp.float32),  # (N, R, A)
        count: wp.array(ndim=3, dtype=wp.int32),
        avg:   wp.array(ndim=3, dtype=wp.float32)   # (N, R, A) [output]
):
    n, i, j = wp.tid()
    if count[n, i, j] > 0:
        avg[n, i, j] = sum[n, i, j] / wp.float32(count[n, i, j])


@wp.kernel
def all_max(
        array:     wp.array(ndim=3, dtype=wp.float32),  # (N, R, A)
        max_value: wp.array(ndim=1, dtype=wp.float32)   # (N,)     [output]
):
    """Global maximum per env."""
    n, i, j = wp.tid()
    wp.atomic_max(max_value, n, array[n, i, j])


@wp.kernel
def range_max(
        array:     wp.array(ndim=3, dtype=wp.float32),  # (N, R, A)
        max_value: wp.array(ndim=2, dtype=wp.float32)   # (N, R)   [output]
):
    """Per-range-row maximum per env."""
    n, i, j = wp.tid()
    wp.atomic_max(max_value, n, i, array[n, i, j])


# ---------------------------------------------------------------------------
# Noise kernels   dim = (N_env, R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def normal_2d(
        seed:   int,
        mean:   float,
        std:    float,
        output: wp.array(ndim=3, dtype=wp.float32)  # (N, R, A) [output]
):
    n, i, j = wp.tid()
    R = output.shape[1]
    A = output.shape[2]
    state = wp.rand_init(seed, n * R * A + i * A + j)
    output[n, i, j] = mean + std * wp.randn(state)


@wp.kernel
def range_dependent_rayleigh_2d(
        seed:           int,
        r:              wp.array(ndim=2, dtype=wp.float32),  # (R, A)  shared grid
        azi:            wp.array(ndim=2, dtype=wp.float32),  # (R, A)
        max_range:      float,
        rayleigh_scale: float,
        central_peak:   float,
        central_std:    float,
        output:         wp.array(ndim=3, dtype=wp.float32)   # (N, R, A) [output]
):
    n, i, j = wp.tid()
    R_bins = output.shape[1]
    A_bins = output.shape[2]
    state = wp.rand_init(seed, n * R_bins * A_bins + i * A_bins + j)

    n1 = wp.randn(state)
    n2 = wp.randn(state)
    rayleigh = rayleigh_scale * wp.sqrt(n1*n1 + n2*n2)

    range_factor   = wp.pow(r[i, j] / max_range, 2.0)
    central_factor = 1.0 + central_peak * wp.exp(
        -wp.pow(azi[i, j] - wp.PI / 2.0, 2.0) / central_std)

    output[n, i, j] = range_factor * central_factor * rayleigh


# ---------------------------------------------------------------------------
# Compositing kernels   dim = (N_env, R, A)
# ---------------------------------------------------------------------------

@wp.kernel
def make_sonar_map_all(
        r:               wp.array(ndim=2, dtype=wp.float32),  # (R, A)  shared
        azi:             wp.array(ndim=2, dtype=wp.float32),
        intensity:       wp.array(ndim=3, dtype=wp.float32),  # (N, R, A) in-place
        max_intensity:   wp.array(ndim=1, dtype=wp.float32),  # (N,)
        gau_noise:       wp.array(ndim=3, dtype=wp.float32),  # (N, R, A)
        range_ray_noise: wp.array(ndim=3, dtype=wp.float32),
        offset:          wp.float32,
        gain:            wp.float32,
        result:          wp.array(ndim=3, dtype=wp.vec3)       # (N, R, A) [output]
):
    n, i, j = wp.tid()
    intensity[n, i, j]  = intensity[n, i, j] / max_intensity[n]
    intensity[n, i, j] += offset
    intensity[n, i, j] *= gain
    intensity[n, i, j] *= (0.5 + gau_noise[n, i, j])
    intensity[n, i, j] += range_ray_noise[n, i, j]
    intensity[n, i, j]  = wp.clamp(intensity[n, i, j], wp.float32(0.0), wp.float32(1.0))

    result[n, i, j] = wp.vec3(r[i, j] * wp.cos(azi[i, j]),
                               r[i, j] * wp.sin(azi[i, j]),
                               intensity[n, i, j])


@wp.kernel
def make_sonar_map_range(
        r:               wp.array(ndim=2, dtype=wp.float32),  # (R, A)  shared
        azi:             wp.array(ndim=2, dtype=wp.float32),
        intensity:       wp.array(ndim=3, dtype=wp.float32),  # (N, R, A) in-place
        max_intensity:   wp.array(ndim=2, dtype=wp.float32),  # (N, R)
        gau_noise:       wp.array(ndim=3, dtype=wp.float32),
        range_ray_noise: wp.array(ndim=3, dtype=wp.float32),
        offset:          wp.float32,
        gain:            wp.float32,
        result:          wp.array(ndim=3, dtype=wp.vec3)       # (N, R, A) [output]
):
    n, i, j = wp.tid()
    if max_intensity[n, i] != 0.0:
        intensity[n, i, j] = intensity[n, i, j] / max_intensity[n, i]
    intensity[n, i, j] *= (0.5 + gau_noise[n, i, j])
    intensity[n, i, j] += range_ray_noise[n, i, j]
    intensity[n, i, j] += offset
    intensity[n, i, j] *= gain
    intensity[n, i, j]  = wp.clamp(intensity[n, i, j], wp.float32(0.0), wp.float32(1.0))

    result[n, i, j] = wp.vec3(r[i, j] * wp.cos(azi[i, j]),
                               r[i, j] * wp.sin(azi[i, j]),
                               intensity[n, i, j])


@wp.kernel
def make_sonar_image(
        sonar_data:  wp.array(ndim=3, dtype=wp.vec3),   # (N, R, A)
        sonar_image: wp.array(ndim=4, dtype=wp.uint8)   # (N, R, A+1, 4) [output]
):
    n, i, j = wp.tid()
    width     = sonar_data.shape[2]
    sonar_rgb = wp.uint8(sonar_data[n, i, j][2] * wp.float32(255))
    sonar_image[n, i, width - j, 0] = sonar_rgb
    sonar_image[n, i, width - j, 1] = sonar_rgb
    sonar_image[n, i, width - j, 2] = sonar_rgb
    sonar_image[n, i, width - j, 3] = wp.uint8(255)