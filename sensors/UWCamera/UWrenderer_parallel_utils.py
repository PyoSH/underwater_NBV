import warp as wp


@wp.func
def vec3_exp(exponent: wp.vec3):
    return wp.vec3f(wp.exp(exponent[0]), wp.exp(exponent[1]), wp.exp(exponent[2]))

@wp.func
def vec3_mul(vec_1: wp.vec3,
            vec_2: wp.vec3):
    return wp.vec3f(vec_1[0] * vec_2[0], vec_1[1] * vec_2[1], vec_1[2] * vec_2[2])

@wp.kernel
def UW_render_batch(
    raw_image:          wp.array(ndim=4, dtype=wp.uint8),    # (N, H, W, 4)
    depth_image:        wp.array(ndim=4, dtype=wp.float32),  # (N, H, W, 1)
    backscatter_value:  wp.array(ndim=2, dtype=wp.float32),  # (N, 3) per-env
    atten_coeff:        wp.array(ndim=2, dtype=wp.float32),  # (N, 3) per-env
    backscatter_coeff:  wp.array(ndim=2, dtype=wp.float32),  # (N, 3) per-env
    uw_image:           wp.array(ndim=4, dtype=wp.uint8)
    ):
    n, i, j = wp.tid()
    raw_RGB = wp.vec3f(wp.float32(raw_image[n, i, j, 0]),
                       wp.float32(raw_image[n, i, j, 1]),
                       wp.float32(raw_image[n, i, j, 2]))
    depth = depth_image[n, i, j, 0]

    ac = wp.vec3f(atten_coeff[n, 0],       atten_coeff[n, 1],       atten_coeff[n, 2])
    bv = wp.vec3f(backscatter_value[n, 0], backscatter_value[n, 1], backscatter_value[n, 2])
    bc = wp.vec3f(backscatter_coeff[n, 0], backscatter_coeff[n, 1], backscatter_coeff[n, 2])

    exp_atten = vec3_exp(- depth * ac)
    exp_back  = vec3_exp(- depth * bc)
    UW_RGB = vec3_mul(raw_RGB, exp_atten) + vec3_mul(bv * wp.float32(255), (wp.vec3f(1.0, 1.0, 1.0) - exp_back))

    uw_image[n, i, j, 0] = wp.uint8(wp.clamp(UW_RGB[0], wp.float32(0), wp.float32(255)))
    uw_image[n, i, j, 1] = wp.uint8(wp.clamp(UW_RGB[1], wp.float32(0), wp.float32(255)))
    uw_image[n, i, j, 2] = wp.uint8(wp.clamp(UW_RGB[2], wp.float32(0), wp.float32(255)))
    uw_image[n, i, j, 3] = raw_image[n, i, j, 3]
