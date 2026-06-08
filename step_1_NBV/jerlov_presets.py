"""Jerlov water type presets for OceanSim UWCamera.
depth_d=3.25m (sceneCfg.floorDepth)  z_rep=1.0m (psi_min)
Generated via calculator.py (Akkaynak & Treibitz CVPR 2018).
Order: R, G, B
"""

JERLOV_PRESETS = {
    "IB": dict(
        atten_coeff       = (0.325835, 0.196346, 0.177762),
        backscatter_coeff = (0.279616, 0.186807, 0.176059),
        backscatter_value = (0.181711, 0.495286, 0.647559),
    ),
    "II": dict(
        atten_coeff       = (0.386476, 0.262386, 0.257074),
        backscatter_coeff = (0.352879, 0.255191, 0.255578),
        backscatter_value = (0.194231, 0.487382, 0.582686),
    ),
    "III": dict(
        atten_coeff       = (0.512426, 0.395839, 0.411934),
        backscatter_coeff = (0.490660, 0.391060, 0.410257),
        backscatter_value = (0.198886, 0.449882, 0.471129),
    ),
    "1C": dict(
        atten_coeff       = (0.664638, 0.551640, 0.579858),
        backscatter_coeff = (0.648422, 0.547907, 0.577633),
        backscatter_value = (0.212554, 0.447515, 0.359715),
    ),
    "3C": dict(
        atten_coeff       = (0.904291, 0.803651, 0.846336),
        backscatter_coeff = (0.893365, 0.800837, 0.844162),
        backscatter_value = (0.202987, 0.365099, 0.251567),
    ),
    "5C": dict(
        atten_coeff       = (1.466923, 1.395696, 1.488586),
        backscatter_coeff = (1.460575, 1.393686, 1.484593),
        backscatter_value = (0.172095, 0.266036, 0.160790),
    ),
}
