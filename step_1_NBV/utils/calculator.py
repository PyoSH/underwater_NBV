import numpy as np
import pandas as pd
import cv2
import os

if not hasattr(np, 'trapezoid'):   # NumPy < 2.0 호환
    np.trapezoid = np.trapz

# ---------------------------------------------------------------------------
# Reference: Akkaynak & Treibitz, "A Revised Underwater Image Formation Model"
#            IEEE CVPR 2018
#
# Forward model (Eq. 20):
#   Ic = Jc * exp(-βD * z) + B∞c * (1 - exp(-βB * z))
#
# Wavelength range: 300-700 nm (Kd data only valid to ~720 nm; user-specified 700 nm)
# ---------------------------------------------------------------------------

class UnderwaterImageSimulator:
    """
    Applies Jerlov water-type optics to a terrestrial image to simulate
    underwater appearance using the revised image formation model.

    Available water types: 'IB', 'II', 'III', '1C', '3C', '5C'
    """


    def __init__(self, jerlov_path, flir_path, cie_d65_path, water_type='IB'):
        self.wavelengths = np.arange(300, 716, dtype=float)  # 300-715 nm, 1 nm step
        self.water_type  = water_type

        # Inherent optical properties (IOPs)
        self.a, self.b, self.kd = self._load_jerlov(jerlov_path, water_type)
        self.beta = self.a + self.b  # beam attenuation β(λ) = a(λ) + b(λ)

        # Surface illuminant E(λ): CIE D65, normalised to peak = 1
        self.E_surface = self._load_d65(cie_d65_path)

        # Camera spectral response Sc(λ) for R, G, B channels (FLIR), normalised
        self.S = self._load_flir_spectral(flir_path)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_jerlov(self, path, water_type):
        """
        Loads a(λ), b(λ), Kd(λ) from the dedicated sheets in jerlov_IOPs.xlsx.

        Sheet layout (no header row; wavelength is implicit):
          row i  → λ = 300 + i  nm
          col 0  → Jerlov IB
          col 1  → Jerlov II
          col 2  → Jerlov III
          col 3  → Jerlov 1C
          col 4  → Jerlov 3C
          col 5  → Jerlov 5C

        'a' and 'b' sheets: 501 rows (300–800 nm)
        'kd' sheet        : 416 rows (300–715 nm)
        """
        TYPES = ['IB', 'II', 'III', '1C', '3C', '5C']
        if water_type not in TYPES:
            raise ValueError(f"Unknown water type '{water_type}'. Choose from: {TYPES}")
        idx = TYPES.index(water_type)

        df_a  = pd.read_excel(path, sheet_name='a',  header=None)
        df_b  = pd.read_excel(path, sheet_name='b',  header=None)
        df_kd = pd.read_excel(path, sheet_name='kd', header=None)

        wl_ab = np.arange(300, 300 + len(df_a))   # 300–800 nm
        wl_kd = np.arange(300, 300 + len(df_kd))  # 300–715 nm

        a  = np.interp(self.wavelengths, wl_ab, df_a.iloc[:, idx].values.astype(float))
        b  = np.interp(self.wavelengths, wl_ab, df_b.iloc[:, idx].values.astype(float))
        kd = np.interp(self.wavelengths, wl_kd, df_kd.iloc[:, idx].values.astype(float))

        return a, b, kd

    def _load_d65(self, path):
        """CIE D65 illuminant, normalised to max = 1."""
        df = pd.read_csv(path, header=None, names=['wl', 'val'])
        E  = np.interp(
            self.wavelengths,
            df['wl'].values.astype(float),
            df['val'].values.astype(float),
        )
        return E / E.max()

    def _load_flir_spectral(self, flir_path):
        """
        FLIR camera spectral response from Flir_Spectral_curve.xlsx.

        Layout (no header; wavelength is implicit):
          row i  → λ = 300 + i  nm   (501 rows: 300–800 nm)
          col 0  → R channel
          col 1  → G channel
          col 2  → B channel

        Returns S of shape (3, N_wl): row 0=R, 1=G, 2=B, normalised to max=1.
        """
        df = pd.read_excel(flir_path, header=None)
        wl = np.arange(300, 300 + len(df))
        S  = np.stack([
            np.interp(self.wavelengths, wl, df.iloc[:, i].values.astype(float))
            for i in range(3)
        ])                          # (3, N_wl)
        S  = np.clip(S, 0.0, None)
        S /= S.max(axis=1, keepdims=True)
        return S

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def calculate_wideband_coeffs(self, depth_d, dist_z):
        """
        Compute per-channel wideband attenuation coefficients and veiling light.

        Parameters
        ----------
        depth_d : float   Vertical water depth [m].
        dist_z  : float   Camera-to-subject distance along LOS [m].

        Returns
        -------
        beta_D : ndarray (3,)  – βD  direct-signal attenuation [m⁻¹], order R/G/B
        beta_B : ndarray (3,)  – βB  backscatter attenuation   [m⁻¹], order R/G/B
        B_inf  : ndarray (3,)  – B∞c wideband veiling light    [0-1], order R/G/B
        """
        wl = self.wavelengths   # (N_wl,)

        # Ambient light at depth d: E(d,λ) = E_surface(λ) · exp(-Kd(λ) · d)
        E_d = self.E_surface * np.exp(-self.kd * depth_d)  # (N_wl,)

        # self.S: (3, N_wl)  — broadcasts against (N_wl,) along last axis

        # κ = ∫ Sc(λ) E_surface(λ) dλ   [normalisation: white surface = 1]
        kappa = np.trapezoid(self.S * self.E_surface, wl, axis=1)  # (3,)

        # Spectral veiling-light function (Eq. 7): B∞(λ) = b(λ) E(d,λ) / β(λ)
        B_inf_lambda = self.b * E_d / self.beta  # (N_wl,)

        # Wideband veiling light B∞c (Eq. 11): (1/κ) ∫ Sc B∞(λ) dλ
        B_inf = np.trapezoid(self.S * B_inf_lambda, wl, axis=1) / kappa  # (3,)

        # Backscatter Bc(z) (Eq. 15): (1/κ) ∫ Sc B∞(λ) (1 - e^{-βz}) dλ
        Bc_z = np.trapezoid(
            self.S * B_inf_lambda * (1.0 - np.exp(-self.beta * dist_z)),
            wl, axis=1
        ) / kappa  # (3,)

        # βBc (Eq. 17): -ln(1 - Bc/B∞c) / z
        ratio  = np.clip(Bc_z / B_inf, 0.0, 1.0 - 1e-9)
        beta_B = -np.log(1.0 - ratio) / dist_z  # (3,)

        # Direct signal with flat reflectance ρ=1 (Eq. 9, 10)
        Jc_flat = np.trapezoid(self.S * E_d,                              wl, axis=1)  # (3,)
        Dc_z    = np.trapezoid(self.S * E_d * np.exp(-self.beta * dist_z), wl, axis=1)  # (3,)

        # βDc (Eq. 13): -ln(Dc(z)/Jc) / z
        beta_D = -np.log(Dc_z / Jc_flat) / dist_z  # (3,)

        return beta_D, beta_B, B_inf

    def build_coefficient_lut(self, depth_d, z_max, n_z=500):
        """
        S6: precompute coefficients with beta_D as a LUT over z (flat rho=1).
        beta_B is nearly z-invariant (paper Fig.7a); B_inf does not depend on z.

        Parameters
        ----------
        depth_d : float   Vertical water depth [m].
        z_max   : float   Maximum camera-to-subject distance to cover [m].
        n_z     : int     Number of LUT points (default 500).

        Returns
        -------
        dict with keys:
            z_lut       : (n_z,)   z sample points [m]
            beta_D_lut  : (n_z, 3) βD per z, order R/G/B
            beta_B      : (3,)     βB scalar (z-invariant approximation)
            B_inf       : (3,)     B∞c wideband veiling light [0-1]
            depth_d     : float
            water_type  : str
        """
        wl  = self.wavelengths
        E_d = self.E_surface * np.exp(-self.kd * depth_d)    # (N_wl,)

        kappa        = np.trapezoid(self.S * self.E_surface, wl, axis=1)        # (3,)
        B_inf_lambda = self.b * E_d / self.beta                                  # (N_wl,)
        B_inf        = np.trapezoid(self.S * B_inf_lambda, wl, axis=1) / kappa  # (3,)

        # beta_B at mid-range z (nearly constant — paper Fig.7a)
        ref_z  = z_max / 2.0
        Bc_ref = np.trapezoid(
            self.S * B_inf_lambda * (1.0 - np.exp(-self.beta * ref_z)), wl, axis=1
        ) / kappa
        ratio  = np.clip(Bc_ref / B_inf, 0.0, 1.0 - 1e-9)
        beta_B = -np.log(1.0 - ratio) / ref_z                                   # (3,)

        # beta_D LUT: vectorized over z (flat rho=1)
        z_lut     = np.linspace(0.05, z_max, n_z)                               # (n_z,)
        Jc_flat   = np.trapezoid(self.S * E_d, wl, axis=1)                      # (3,)
        exp_term  = np.exp(-self.beta[np.newaxis, :] * z_lut[:, np.newaxis])    # (n_z, N_wl)
        integrand = (self.S * E_d)[np.newaxis] * exp_term[:, np.newaxis]        # (n_z, 3, N_wl)
        Dc_z      = np.trapezoid(integrand, wl, axis=-1)                        # (n_z, 3)
        beta_D_lut = -np.log(Dc_z / Jc_flat[np.newaxis]) / z_lut[:, np.newaxis]  # (n_z, 3)

        return dict(z_lut=z_lut, beta_D_lut=beta_D_lut,
                    beta_B=beta_B, B_inf=B_inf,
                    depth_d=np.float64(depth_d), water_type=self.water_type)

    @staticmethod
    def save_coefficients(path, coeffs):
        np.savez(path, **{k: np.array(v) for k, v in coeffs.items()})

    @staticmethod
    def load_coefficients(path):
        data   = np.load(path, allow_pickle=True)
        coeffs = {k: data[k] for k in data.files}
        coeffs['water_type'] = str(coeffs['water_type'])
        return coeffs

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------

    def simulate_underwater_image(self, image_path, depth_map, coeffs):
        """
        Apply the revised underwater image formation model (Eq. 20) to a
        terrestrial image using a per-pixel depth map.

        Parameters
        ----------
        image_path : str
            Path to the terrestrial input image (J_c).
        depth_map  : ndarray (H, W)
            Per-pixel camera-to-subject distance z [m] along the line of sight.
        coeffs     : dict
            Output of build_coefficient_lut() or load_coefficients().
            beta_D is applied per pixel via LUT interpolation (S6).

        Model: Ic = Jc · exp(-βD(z) · z) + B∞c · (1 - exp(-βB · z))
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        z_lut      = coeffs['z_lut']
        beta_D_lut = coeffs['beta_D_lut']   # (n_z, 3)
        beta_B     = coeffs['beta_B']        # (3,)
        B_inf      = coeffs['B_inf']         # (3,)

        # Per-pixel beta_D via LUT interpolation: (H, W, 3)
        beta_D_map = np.stack([
            np.interp(depth_map, z_lut, beta_D_lut[:, c])
            for c in range(3)
        ], axis=-1)

        z   = depth_map[:, :, np.newaxis]
        out = (img_rgb * np.exp(-beta_D_map * z)
               + B_inf * (1.0 - np.exp(-beta_B * z)))

        out = np.clip(out, 0.0, 1.0)
        return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    _DIR         = os.path.dirname(os.path.abspath(__file__))
    JERLOV_FILE  = os.path.join(_DIR, 'jerlov_IOPs.xlsx')
    FLIR_FILE    = os.path.join(_DIR, 'Flir_Spectral_curve.xlsx')
    D65_FILE     = os.path.join(_DIR, 'CIE_std_illum_D65.csv')
    WATER_TYPE   = '5C'   # choose: IB | II | III | 1C | 3C | 5C
    DEPTH_D      = 1.0    # vertical water depth [m]
    Z_MAX        = 30.0   # max camera-to-subject distance to cover [m]
    COEFF_FILE   = f'coeffs_{WATER_TYPE}_d{DEPTH_D}.npz'
    INPUT_IMAGE  = '/home/kriso/cyk/dataset/image/1776141486.127162.png'
    DEPTH_IMAGE  = '/home/kriso/cyk/utils/v4/PromptDA_depth/1776141486.127162.png'

    sim = UnderwaterImageSimulator(
        jerlov_path  = JERLOV_FILE,
        flir_path    = FLIR_FILE,
        cie_d65_path = D65_FILE,
        water_type   = WATER_TYPE,
    )

    # Build and save coefficient LUT (S6: beta_D per z, flat rho)
    coeffs = sim.build_coefficient_lut(DEPTH_D, z_max=Z_MAX)
    UnderwaterImageSimulator.save_coefficients(COEFF_FILE, coeffs)
    print(f"Saved coefficients → '{COEFF_FILE}'")

    # Print beta_D at a few representative distances
    sample_z = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0, 15.0, 20.0]
    print(f"\nJerlov {WATER_TYPE}  |  depth_d = {DEPTH_D} m  (beta_B and B_inf are z-invariant)")
    print(f"\n{'ch':>3}  " + "  ".join(f"{'βD(z='+str(z)+'m)':>12}" for z in sample_z)
          + f"  {'βB':>10}  {'B∞c':>10}")
    print("-" * (3 + 14 * len(sample_z) + 24))
    for i, ch in enumerate(['R', 'G', 'B']):
        beta_D_samples = [float(np.interp(z, coeffs['z_lut'], coeffs['beta_D_lut'][:, i]))
                          for z in sample_z]
        row = "  ".join(f"{v:>12.5f}" for v in beta_D_samples)
        print(f"{ch:>3}  {row}  {coeffs['beta_B'][i]:>10.5f}  {coeffs['B_inf'][i]:>10.6f}")

    if os.path.exists(INPUT_IMAGE) and os.path.exists(DEPTH_IMAGE):
        # Load coefficients and simulate (demonstrates save/load workflow)
        coeffs    = UnderwaterImageSimulator.load_coefficients(COEFF_FILE)
        depth_map = cv2.imread(DEPTH_IMAGE, cv2.IMREAD_UNCHANGED).astype(np.float32)
        out_img   = sim.simulate_underwater_image(INPUT_IMAGE, depth_map / 1000.0, coeffs)
        cv2.imwrite('simulated_underwater.jpg', out_img)
        print(f"\nSaved 'simulated_underwater.jpg'")
    else:
        print(f"\n('{INPUT_IMAGE}' or '{DEPTH_IMAGE}' not found — skipping simulation)")
