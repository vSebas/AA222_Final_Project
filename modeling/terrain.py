"""

The SCP problem (from the status update) is

    min  integral_0^Tf ( c_T * sum_i T_i + c_r * R(L_G) + J_P ) dt
    s.t. p(0) = p0, p(Tf) = pg,
         |x_B_dot| <= x_B_dot_max,
         |x_B_ddot| <= x_B_ddot_max, |y_B_ddot| <= y_B_ddot_max,
         P_cons(t) <= P_avail(t).

This module produces R(L_G) (the continuous accumulated terrain-risk term)
and the hard slope constraint set |theta| <= 20 deg via a boolean
traversability mask.

USE_REAL_LOLA flag
------------------
USE_REAL_LOLA = False (default): a physically realistic synthetic 30 km
crater cutout is generated. The whole downstream pipeline (slope,
roughness, mask, risk, SCP-facing API) runs on it.

USE_REAL_LOLA = True: the moon package's crater_cutout / read_warped_window
is used to extract the real DEM from
Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif (which must live in moon-master/data/).
If moon.io cannot be imported (it depends on legacy GDAL bindings), a direct
rasterio-only fallback reader is used.

South-pole / Artemis note
-------------------------
The named-crater shortcut crater_cutout(...) is fed by the IAU gazetteer;
most well-known craters are equatorial / near-side. The Artemis south-pole
scenario the proposal targets requires craters at lat < -80 deg
(Shackleton, de Gerlache, Faustini, Cabeus, Nobile, Shoemaker, Haworth,
Sverdrup, Amundsen, Malapert, ...). The global 118 m/px LOLA mosaic is in
an equirectangular projection and degenerates at the poles; for production
work at the pole, drop in a polar product such as SLDEM2015
(LDEM_85S_*_FLOAT.IMG) and feed it through the same compute_*/build_terrain
code path. The terrain pipeline below is region-agnostic.

"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

# ----------------------------------------------------------------------
# Master switch
# ----------------------------------------------------------------------
USE_REAL_LOLA = False   # flip True when the 8 GB tif is on disk

# The nominal LOLA DEM ground sampling: 118 m / pixel at the equator.
LOLA_PIXEL_SCALE_M = 118.0

# Moon radius used by the LOLA product (matches moon.config.Constants).
LOLA_MOON_RADIUS_M = 1_737_400.0

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
TABLE_DIR = os.path.join(THIS_DIR, "tables")
TIF_FNAME = "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"


# ======================================================================
# Stage 0:  DEM acquisition (real or synthetic)
# ======================================================================

def _real_lola_via_moon_pkg(crater_name=None, lon=None, lat=None,
                            side_deg=None) -> np.ndarray:
    """Try the moon package first; this is the spec'd path."""
    from moon.io import crater_cutout, read_warped_window  # local import
    if crater_name is not None:
        return np.asarray(crater_cutout(crater_name))
    if lon is None or lat is None or side_deg is None:
        raise ValueError("Provide crater_name OR (lon, lat, side_deg).")
    return np.asarray(read_warped_window(lon, lat, side_deg,
                                         convert_km_to_deg=False))


def _real_lola_via_rasterio(lon=None, lat=None, side_deg=None) -> np.ndarray:
    """Pure-rasterio fallback when moon.io can't import (no GDAL bindings).

    Reads a simple unwarped equirectangular window from the LOLA tif.
    No reprojection: at moderate latitudes the longitude pixel size is
    1/cos(lat) of the latitude pixel size, which is fine for slope/roughness
    near the equator but distorts toward the pole. Good enough as an
    interim path until a proper polar LOLA product is plugged in.
    """
    import rasterio
    from rasterio.windows import from_bounds
    if lon is None or lat is None or side_deg is None:
        raise ValueError("rasterio fallback needs lon, lat, side_deg.")
    tif_path = os.path.join(DATA_DIR, TIF_FNAME)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"{TIF_FNAME} not found in {DATA_DIR}. Download from "
            "http://planetarymaps.usgs.gov/mosaic/" + TIF_FNAME)
    with rasterio.open(tif_path) as src:
        win = from_bounds(lon - side_deg / 2, lat - side_deg / 2,
                          lon + side_deg / 2, lat + side_deg / 2,
                          transform=src.transform)
        arr = src.read(1, window=win)
    # LOLA scaling factor: elevation_m = pixel_value * 0.5
    return arr.astype(np.float64) * 0.5


def _real_lola_dem(crater_name=None, lon=None, lat=None,
                   side_deg=None) -> Tuple[np.ndarray, float]:
    """Returns (dem_m, pixel_scale_m) from the real LOLA mosaic."""
    try:
        dem = _real_lola_via_moon_pkg(crater_name=crater_name, lon=lon,
                                      lat=lat, side_deg=side_deg)
    except (ImportError, ModuleNotFoundError) as e:
        # moon.io imports legacy `gdal`; fall back to rasterio.
        print(f"[terrain] moon.io unavailable ({e}); using rasterio fallback.")
        if crater_name is not None:
            # need lon/lat/diameter from the IAU table to do this fallback
            lon, lat, side_deg = _lookup_crater(crater_name)
        dem = _real_lola_via_rasterio(lon=lon, lat=lat, side_deg=side_deg)

    # In an orthographic warp (moon.io path) the pixel is metric and ~118 m.
    # In the rasterio equirectangular fallback the latitude axis is metric
    # at 118 m/px; longitude axis stretches by 1/cos(lat). The dominant scale
    # is still ~118 m/px and we use it as the isotropic estimate. For a
    # production polar pipeline, switch to a polar stereographic product.
    return dem.astype(np.float64), float(LOLA_PIXEL_SCALE_M)


def _lookup_crater(name: str) -> Tuple[float, float, float]:
    """Read the IAU CSV directly to avoid importing moon.features."""
    path = os.path.join(TABLE_DIR, "iau_approved_craters.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Feature_Name"].strip().lower() == name.strip().lower():
                lat = float(r["Center_Latitude"])
                lon = float(r["Center_Longitude"])
                d_km = float(r["Diameter"])
                # convert diameter (km) to side (deg) with 30% padding
                side_deg = 1.3 * d_km / (LOLA_MOON_RADIUS_M / 1000.0) * 180.0 / math.pi
                return lon, lat, side_deg
    raise KeyError(f"crater {name!r} not found in IAU table")


def _synthetic_crater(
    cutout_size_km: float = 30.0,
    pixel_scale_m: float = LOLA_PIXEL_SCALE_M,
    crater_diameter_km: float = 22.0,
    floor_depth_m: float = 700.0,
    rim_height_m: float = 230.0,
    ejecta_height_m: float = 60.0,
    floor_fraction: float = 0.62,           # r_floor / r_crater
    rim_sigma_fraction: float = 0.05,       # Gaussian rim std / r_crater
    ejecta_decay_fraction: float = 0.55,    # exp decay length / r_crater
    plain_roughness_m: float = 0.20,
    wall_roughness_m: float = 1.5,
    floor_roughness_m: float = 0.50,
    noise_correlation_px: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Physically motivated synthetic complex crater.

    Components (axisymmetric, then perturbed with correlated noise):

      * flat floor at -floor_depth_m,
      * cosine-ramp wall from floor to plain (smooth, zero-derivative at both
        ends, so neither floor-wall nor wall-rim junction is a cliff),
      * Gaussian rim bump centred at the crater radius,
      * exponential ejecta blanket outside the rim,
      * smooth plains at 0 m baseline.

    The cosine wall is the key: a naive parabolic bowl makes the entire
    interior a steep cliff. Real complex craters of this size class are
    flat-floored with gently sloped walls.
    """
    n = int(round(cutout_size_km * 1000.0 / pixel_scale_m))
    if n % 2 == 0:
        n += 1                                  # odd -> exact centre pixel
    half = (n - 1) // 2
    x = (np.arange(n) - half) * pixel_scale_m
    y = (np.arange(n) - half) * pixel_scale_m
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)

    r_c = 1000.0 * crater_diameter_km / 2.0
    r_f = floor_fraction * r_c
    sigma_r = rim_sigma_fraction * r_c
    e_scale = ejecta_decay_fraction * r_c

    # --- smooth axisymmetric backbone ----------------------------------
    # cosine wall: -floor_depth at r_f, 0 at r_c, slope=0 at both ends.
    t = np.clip((R - r_f) / max(r_c - r_f, 1e-9), 0.0, 1.0)
    wall = -floor_depth_m * 0.5 * (1.0 + np.cos(np.pi * t))
    z_base = np.where(R <= r_f, -floor_depth_m,
                      np.where(R <= r_c, wall, 0.0))

    rim = rim_height_m * np.exp(-((R - r_c) / sigma_r) ** 2)
    ejecta = np.where(R > r_c,
                      ejecta_height_m * np.exp(-(R - r_c) / e_scale),
                      0.0)

    z = z_base + rim + ejecta

    # --- spatially varying micro-roughness ----------------------------
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, n))
    corr = gaussian_filter(raw, sigma=noise_correlation_px)
    corr /= corr.std()                          # unit-variance noise field

    # roughness amplitude:
    #   * very low on plains and far ejecta
    #   * moderate on the floor (regolith-fill)
    #   * highest in the wall annulus (rubble / slumping)
    wall_centre_r = 0.5 * (r_f + r_c)
    wall_band = np.exp(-((R - wall_centre_r) / (0.5 * (r_c - r_f) + 1e-9)) ** 2)
    floor_band = np.where(R <= r_f, 1.0, 0.0)

    sigma_field = (plain_roughness_m
                   + (wall_roughness_m - plain_roughness_m) * wall_band
                   + (floor_roughness_m - plain_roughness_m) * floor_band)
    z += corr * sigma_field

    return z.astype(np.float64), float(pixel_scale_m)


def get_dem(crater_name: Optional[str] = None,
            lon: Optional[float] = None,
            lat: Optional[float] = None,
            side_deg: Optional[float] = None,
            *,
            cutout_size_km: float = 30.0,
            crater_diameter_km: float = 22.0,
            seed: int = 42) -> Tuple[np.ndarray, float]:
    """Return (dem_metres, pixel_scale_m).

    USE_REAL_LOLA=True  -> real cutout via moon.io (or rasterio fallback).
    USE_REAL_LOLA=False -> synthetic crater; ignores lon/lat/side_deg.
    """
    if USE_REAL_LOLA:
        return _real_lola_dem(crater_name=crater_name, lon=lon, lat=lat,
                              side_deg=side_deg)
    return _synthetic_crater(cutout_size_km=cutout_size_km,
                             crater_diameter_km=crater_diameter_km,
                             seed=seed)


# ======================================================================
# Stage 1:  slope + roughness
# ======================================================================

def compute_slope_deg(dem: np.ndarray, pixel_scale_m: float) -> np.ndarray:
    """Slope in degrees from |grad z|. Central differences, metric spacing."""
    gy, gx = np.gradient(dem, pixel_scale_m, pixel_scale_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def compute_roughness_m(dem: np.ndarray, window: int = 5) -> np.ndarray:
    """Local elevation std over an N x N window, via Var = E[z^2] - E[z]^2.

    Note: this naive metric conflates a smooth-but-steep slope with true
    micro-roughness, since a 15 deg ramp over a 5-pixel window already
    produces tens of metres of std. That's fine for our use because:
      * slope itself is the dominant hard gate (>20 deg => no-go),
      * roughness here primarily acts as a soft cost in R(x,y) that adds
        extra penalty for cells with high local elevation variation
        (which includes both rubble and steep-and-sharp transitions).
    Detrending with a wider kernel introduces ringing at the rim (sharp
    rim peak / ejecta drop) that would spuriously inflate roughness in
    the ejecta plain and break the planner downstream.
    """
    m1 = uniform_filter(dem, size=window)
    m2 = uniform_filter(dem * dem, size=window)
    var = np.clip(m2 - m1 * m1, 0.0, None)
    return np.sqrt(var)


# ======================================================================
# Stage 2:  traversability mask + continuous risk field
# ======================================================================

def compute_traversability(slope_deg: np.ndarray,
                           roughness_m: np.ndarray,
                           slope_limit_deg: float = 20.0,
                           rough_limit_m: float = 2.0) -> np.ndarray:
    """Boolean drivable mask (True = drivable).

    Hard no-go where slope > slope_limit_deg OR roughness > rough_limit_m.
    This is the obstacle set O used by the global planner and as a hard
    constraint in SCP.
    """
    return (slope_deg <= slope_limit_deg) & (roughness_m <= rough_limit_m)


def compute_risk_field(slope_deg: np.ndarray,
                       roughness_m: np.ndarray,
                       traversable: np.ndarray,
                       soft_slope_deg: float = 12.0,
                       slope_limit_deg: float = 20.0,
                       rough_ref_m: float = 5.0,
                       no_go_penalty: float = 50.0,
                       smooth_sigma_px: float = 0.6) -> np.ndarray:
    """Continuous, smooth risk field R(x,y).

    Components (all non-negative):
      * slope term:     0 below soft_slope_deg, rises quadratically above
                        (clamped at slope_limit_deg = 1.0).
      * roughness term: (roughness / rough_ref_m) ** 2, weight 0.5.
      * no-go penalty:  no_go_penalty on cells not in the traversable mask.

    The sum is finally Gaussian-blurred so its finite-difference gradient is
    well-behaved inside SCP (no Heaviside-step at the mask boundary).
    """
    span = max(slope_limit_deg - soft_slope_deg, 1e-9)
    r_slope = (np.maximum(slope_deg - soft_slope_deg, 0.0) / span) ** 2
    r_rough = (roughness_m / rough_ref_m) ** 2
    r_nogo = no_go_penalty * (~traversable).astype(np.float64)
    risk = r_slope + 0.5 * r_rough + r_nogo
    if smooth_sigma_px > 0:
        risk = gaussian_filter(risk, sigma=smooth_sigma_px)
    return risk


# ======================================================================
# TerrainModel:  SCP-facing API
# ======================================================================

@dataclass
class TerrainModel:
    """Bundles all layers + continuous-metric-coordinate accessors.

    Convention:
      * dem[row, col]; row = 0 at y = y0_m (south-most), col = 0 at x = x0_m.
      * pixel_scale_m is isotropic (square pixels in metres).
      * matplotlib: imshow(..., extent=extent_m, origin='lower').
    """
    dem_m: np.ndarray
    slope_deg: np.ndarray
    roughness_m: np.ndarray
    traversable: np.ndarray
    risk: np.ndarray
    pixel_scale_m: float
    x0_m: float
    y0_m: float

    # --------- properties --------------------------------------------
    @property
    def shape(self):
        return self.dem_m.shape

    @property
    def extent_m(self):
        h, w = self.dem_m.shape
        return (self.x0_m, self.x0_m + w * self.pixel_scale_m,
                self.y0_m, self.y0_m + h * self.pixel_scale_m)

    # --------- coord conversion --------------------------------------
    def _xy_to_rowcol(self, x_m, y_m):
        col = (np.asarray(x_m, dtype=np.float64) - self.x0_m) / self.pixel_scale_m
        row = (np.asarray(y_m, dtype=np.float64) - self.y0_m) / self.pixel_scale_m
        return row, col

    # --------- SCP-facing samplers -----------------------------------
    def risk_at(self, x_m, y_m):
        """Bilinear sample of the risk field at metric (x, y)."""
        row, col = self._xy_to_rowcol(x_m, y_m)
        coords = np.vstack([np.atleast_1d(row).ravel(),
                            np.atleast_1d(col).ravel()])
        out = map_coordinates(self.risk, coords, order=1,
                              mode="nearest").reshape(np.shape(x_m))
        return out

    def risk_grad(self, x_m, y_m, h_m: Optional[float] = None):
        """Central-difference gradient (dR/dx, dR/dy) at metric (x, y)."""
        if h_m is None:
            h_m = self.pixel_scale_m
        dRdx = (self.risk_at(np.add(x_m, h_m), y_m)
                - self.risk_at(np.subtract(x_m, h_m), y_m)) / (2.0 * h_m)
        dRdy = (self.risk_at(x_m, np.add(y_m, h_m))
                - self.risk_at(x_m, np.subtract(y_m, h_m))) / (2.0 * h_m)
        return dRdx, dRdy

    def path_risk(self, xs_m, ys_m) -> float:
        """Trapezoid-rule integral of R along the polyline (xs, ys).

        This is the scalar that goes into the c_r * R(L_G) objective term.
        """
        xs = np.asarray(xs_m, dtype=np.float64)
        ys = np.asarray(ys_m, dtype=np.float64)
        if xs.shape != ys.shape or xs.ndim != 1:
            raise ValueError("xs_m and ys_m must be 1-D arrays of equal length")
        ds = np.hypot(np.diff(xs), np.diff(ys))
        Rs = self.risk_at(xs, ys)
        return float(np.sum(0.5 * (Rs[:-1] + Rs[1:]) * ds))


# ======================================================================
# End-to-end driver
# ======================================================================

def build_terrain(
    crater_name: Optional[str] = None,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    side_deg: Optional[float] = None,
    *,
    cutout_size_km: float = 30.0,
    crater_diameter_km: float = 22.0,
    slope_limit_deg: float = 20.0,
    rough_limit_m: float = 30.0,
    soft_slope_deg: float = 12.0,
    rough_ref_m: float = 5.0,
    no_go_penalty: float = 50.0,
    smooth_sigma_px: float = 0.6,
    seed: int = 42,
) -> TerrainModel:
    """Build everything: DEM -> slope -> roughness -> mask -> risk -> model."""
    dem, dx = get_dem(crater_name=crater_name, lon=lon, lat=lat,
                      side_deg=side_deg, cutout_size_km=cutout_size_km,
                      crater_diameter_km=crater_diameter_km, seed=seed)
    slope = compute_slope_deg(dem, dx)
    rough = compute_roughness_m(dem, window=5)
    trav = compute_traversability(slope, rough, slope_limit_deg, rough_limit_m)
    risk = compute_risk_field(slope, rough, trav,
                              soft_slope_deg=soft_slope_deg,
                              slope_limit_deg=slope_limit_deg,
                              rough_ref_m=rough_ref_m,
                              no_go_penalty=no_go_penalty,
                              smooth_sigma_px=smooth_sigma_px)
    h, w = dem.shape
    x0 = -0.5 * w * dx
    y0 = -0.5 * h * dx
    return TerrainModel(dem_m=dem, slope_deg=slope, roughness_m=rough,
                        traversable=trav, risk=risk, pixel_scale_m=dx,
                        x0_m=x0, y0_m=y0)


# ======================================================================
# Helpers used by the __main__ validation block
# ======================================================================

def print_south_pole_craters(min_diameter_km: float = 20.0,
                             lat_max: float = -80.0) -> list:
    """Print Artemis-relevant south-pole craters from the real IAU CSV."""
    path = os.path.join(TABLE_DIR, "iau_approved_craters.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lat = float(r["Center_Latitude"])
                d_km = float(r["Diameter"])
            except (ValueError, KeyError):
                continue
            if (lat < lat_max
                    and d_km > min_diameter_km
                    and r.get("Approval_Status", "").strip() == "Approved"):
                rows.append((r["Feature_Name"], lat,
                             float(r["Center_Longitude"]), d_km))
    rows.sort(key=lambda x: x[1])
    print(f"\nSouth-pole craters (lat < {lat_max} deg, diameter > "
          f"{min_diameter_km} km, approved):")
    print(f"  {'name':<14} {'lat':>8} {'lon':>9} {'diam (km)':>10}")
    for name, lat, lon, d in rows:
        # ASCII-safe printing on Windows consoles
        safe = name.encode("ascii", "replace").decode("ascii")
        print(f"  {safe:<14} {lat:8.2f} {lon:9.2f} {d:10.2f}")
    print(f"  ({len(rows)} craters)\n")
    return rows


# ======================================================================
# Validation block
# ======================================================================

def _validate():
    print("=" * 70)
    print(" terrain.py validation")
    print(f"   USE_REAL_LOLA = {USE_REAL_LOLA}")
    print("=" * 70)

    print_south_pole_craters()

    # Build the model
    tm = build_terrain()

    # 1. Geometry / unit sanity
    h, w = tm.dem_m.shape
    print(f"DEM shape           : {tm.dem_m.shape}")
    print(f"pixel scale (m)     : {tm.pixel_scale_m:.3f}")
    extent_km = tuple(e / 1000.0 for e in tm.extent_m)
    print(f"extent (km)         : x [{extent_km[0]:.2f}, {extent_km[1]:.2f}], "
          f"y [{extent_km[2]:.2f}, {extent_km[3]:.2f}]")
    print(f"elevation range (m) : "
          f"{tm.dem_m.min():.1f} .. {tm.dem_m.max():.1f}")
    print(f"slope range (deg)   : "
          f"{tm.slope_deg.min():.3f} .. {tm.slope_deg.max():.3f}")
    print(f"roughness range (m) : "
          f"{tm.roughness_m.min():.3f} .. {tm.roughness_m.max():.3f}")
    trav_frac = tm.traversable.mean()
    print(f"traversable fraction: {trav_frac:.2%}")

    # 2. Pointwise risk probes
    #    Pick a "deep plain" point — far from the crater, near the edge.
    plain_xy = (-13_000.0, 13_000.0)
    center_xy = (0.0, 0.0)
    wall_xy = (9_500.0, 0.0)         # inside the wall annulus

    r_center = float(tm.risk_at(*center_xy))
    r_plain = float(tm.risk_at(*plain_xy))
    r_wall = float(tm.risk_at(*wall_xy))

    print()
    print(f"risk @ crater centre / floor ({center_xy[0]/1e3:+.1f},"
          f"{center_xy[1]/1e3:+.1f}) km : {r_center:.4f}")
    print(f"risk @ flat plain corner     ({plain_xy[0]/1e3:+.1f},"
          f"{plain_xy[1]/1e3:+.1f}) km : {r_plain:.4f}")
    print(f"risk @ wall                  ({wall_xy[0]/1e3:+.1f},"
          f"{wall_xy[1]/1e3:+.1f}) km : {r_wall:.4f}")

    assert r_plain < 0.5, f"flat-plain risk should be ~0, got {r_plain}"
    assert r_plain * 100 < r_wall, (
        f"plain risk ({r_plain}) must be << wall risk ({r_wall})")

    # 3. Path comparison: straight-through vs detour
    n_pts = 600
    # straight: along y = 0, from x=-13 km to x=+13 km
    xs_s = np.linspace(-13_000.0, 13_000.0, n_pts)
    ys_s = np.zeros(n_pts)
    R_straight = tm.path_risk(xs_s, ys_s)
    L_straight = float(np.sum(np.hypot(np.diff(xs_s), np.diff(ys_s))))

    # detour: semicircle of radius 13 km centred at origin, passing
    # through (0, +13 km). Shares endpoints with the straight path.
    theta = np.linspace(np.pi, 0.0, n_pts)
    r_arc = 13_000.0
    xs_d = r_arc * np.cos(theta)
    ys_d = r_arc * np.sin(theta)
    R_detour = tm.path_risk(xs_d, ys_d)
    L_detour = float(np.sum(np.hypot(np.diff(xs_d), np.diff(ys_d))))

    print()
    print(f"straight-through path: length {L_straight/1e3:6.2f} km, "
          f"path_risk = {R_straight:12.2f}")
    print(f"detour (semicircle) :  length {L_detour/1e3:6.2f} km, "
          f"path_risk = {R_detour:12.2f}")
    assert R_detour < R_straight, (
        f"detour ({R_detour:.2f}) should be cheaper than "
        f"straight ({R_straight:.2f})")
    print(f"-> detour is {R_straight / max(R_detour, 1e-9):.1f}x "
          "cheaper than straight-through.  PASS.")

    # 4. risk_grad smoke test
    dRdx, dRdy = tm.risk_grad(*wall_xy)
    print()
    print(f"risk_grad @ wall: dR/dx = {float(dRdx):+.4e},  "
          f"dR/dy = {float(dRdy):+.4e}")

    # 5. Figure
    out_png = os.path.join(THIS_DIR, "terrain_validation.png")
    _save_validation_figure(tm, xs_s, ys_s, xs_d, ys_d, out_png)
    print(f"\nsaved figure -> {out_png}")
    print("\nALL VALIDATION CHECKS PASSED.")


def _save_validation_figure(tm: TerrainModel, xs_s, ys_s, xs_d, ys_d,
                            out_png: str):
    ext_km = [e / 1000.0 for e in tm.extent_m]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(tm.dem_m, extent=ext_km, origin="lower",
                   cmap="terrain")
    fig.colorbar(im, ax=ax, label="elevation [m]")
    ax.set_title("Elevation (synthetic complex crater)"
                 if not USE_REAL_LOLA else "Elevation (LOLA cutout)")

    ax = axes[0, 1]
    im = ax.imshow(tm.slope_deg, extent=ext_km, origin="lower",
                   cmap="magma", vmin=0, vmax=30)
    fig.colorbar(im, ax=ax, label="slope [deg]")
    ax.set_title("Slope")

    ax = axes[1, 0]
    im = ax.imshow(tm.traversable.astype(np.uint8), extent=ext_km,
                   origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="0 = no-go, 1 = drivable")
    ax.set_title(f"Traversability mask "
                 f"({tm.traversable.mean()*100:.1f}% drivable)")

    ax = axes[1, 1]
    risk_show = np.clip(tm.risk, 0, np.percentile(tm.risk, 99.5))
    im = ax.imshow(risk_show, extent=ext_km, origin="lower",
                   cmap="inferno")
    fig.colorbar(im, ax=ax, label="risk R(x,y)")
    ax.plot(np.asarray(xs_s) / 1e3, np.asarray(ys_s) / 1e3,
            color="cyan", lw=2.0, ls="--", label="straight-through")
    ax.plot(np.asarray(xs_d) / 1e3, np.asarray(ys_d) / 1e3,
            color="white", lw=2.0, label="detour")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.7)
    ax.set_title("Risk field with candidate paths")

    for ax in axes.flat:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    _validate()
