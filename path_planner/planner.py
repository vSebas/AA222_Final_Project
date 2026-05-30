"""

Pipeline role
-------------
    Stage 0  LOLA DEM cutout
    Stage 1  slope + roughness
    Stage 2  traversability mask + risk field R(x,y)   <-- terrain.py
    Stage 3  global planner (A*)                       <-- THIS FILE
    Stage 4  SCP trajectory optimizer

This module produces the *warm start* that the Stage-4 SCP optimizer
linearizes around. SCP needs a feasible-ish initial trajectory; plain
8-connected grid A* on the traversability mask produces exactly that.

It is built against the terrain.py `TerrainModel` API and does not modify
terrain.py or the moon package.

A* edge cost
------------
    cost(u, v) = d(u, v) * (1 + w_risk * R_avg(u, v))

with d the Euclidean step distance in metres (pixel_scale orthogonal,
pixel_scale*sqrt(2) diagonal) and R_avg the mean cell-centre risk of the
two cells. The heuristic is the pure straight-line metric distance to the
goal — risk is deliberately kept OUT of the heuristic so A* stays
admissible (every edge cost is >= its geometric length).

time_parameterize note
----------------------
The rover state in the status update is (x_G, y_G, psi_G) — heading is a
core state variable. So time_parameterize returns a 4-tuple
(xs, ys, ts, psis): the constant-velocity (x,y,t) guess SCP needs, plus a
heading guess derived from the path tangent. The first three entries are
the (xs, ys, ts) contract from the task spec.

"""

from __future__ import annotations

import heapq
import itertools
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from terrain import TerrainModel, build_terrain

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

Cell = Tuple[int, int]            # (row, col)
Point = Tuple[float, float]       # (x_m, y_m)


# ======================================================================
# Coordinate helpers (TerrainModel convention: cell (row,col) centre is at
# metric (x0 + col*dx, y0 + row*dx); risk[row, col]; origin='lower').
# ======================================================================

def _metric_to_cell(tm: TerrainModel, x_m: float, y_m: float) -> Cell:
    col = int(round((x_m - tm.x0_m) / tm.pixel_scale_m))
    row = int(round((y_m - tm.y0_m) / tm.pixel_scale_m))
    return row, col


def _cell_to_metric(tm: TerrainModel, row: int, col: int) -> Point:
    x = tm.x0_m + col * tm.pixel_scale_m
    y = tm.y0_m + row * tm.pixel_scale_m
    return float(x), float(y)


def _in_bounds(tm: TerrainModel, row: int, col: int) -> bool:
    h, w = tm.traversable.shape
    return 0 <= row < h and 0 <= col < w


def _cell_centre_risk(tm: TerrainModel) -> np.ndarray:
    """Risk sampled at every cell centre, as an (h, w) grid.

    Equivalent to tm.risk, but obtained through the same tm.risk_at the
    SCP path integral uses, so the A* edge cost and tm.path_risk agree
    exactly on cell-centre polylines.
    """
    h, w = tm.shape
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    xc = tm.x0_m + cols * tm.pixel_scale_m
    yc = tm.y0_m + rows * tm.pixel_scale_m
    xx, yy = np.meshgrid(xc, yc)
    return np.asarray(tm.risk_at(xx, yy), dtype=np.float64)


def _nearest_traversable_cell(tm: TerrainModel, row: int, col: int,
                              max_radius: int = 30) -> Optional[Cell]:
    """Return (row,col) if drivable, else the closest drivable cell."""
    trav = tm.traversable
    h, w = trav.shape
    if 0 <= row < h and 0 <= col < w and trav[row, col]:
        return row, col
    for rad in range(1, max_radius + 1):
        best = None
        best_d2 = None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r, c = row + dr, col + dc
                if 0 <= r < h and 0 <= c < w and trav[r, c]:
                    d2 = dr * dr + dc * dc
                    if best_d2 is None or d2 < best_d2:
                        best, best_d2 = (r, c), d2
        if best is not None:
            return best
    return None


# ======================================================================
# Stage 3:  8-connected grid A*
# ======================================================================

def astar_on_terrain(tm: TerrainModel,
                     start_m: Point,
                     goal_m: Point,
                     w_risk: float = 10.0) -> Optional[List[Point]]:
    """8-connected grid A* on the traversability mask.

    Returns a list of (x_m, y_m) waypoints from start to goal (cell
    centres), or None if start/goal are invalid or no path exists.

    Edge cost  : d(u,v) * (1 + w_risk * R_avg(u,v))
    Heuristic  : straight-line metric distance to goal (admissible).
    """
    trav = tm.traversable
    h, w = trav.shape
    dx = tm.pixel_scale_m

    start = _metric_to_cell(tm, start_m[0], start_m[1])
    goal = _metric_to_cell(tm, goal_m[0], goal_m[1])

    # Reject out-of-bounds or non-traversable endpoints.
    for cell in (start, goal):
        if not _in_bounds(tm, *cell):
            return None
        if not trav[cell]:
            return None

    risk_grid = _cell_centre_risk(tm)
    gx, gy = _cell_to_metric(tm, *goal)

    def heuristic(cell: Cell) -> float:
        cx, cy = _cell_to_metric(tm, *cell)
        return math.hypot(cx - gx, cy - gy)

    # 8-connected steps with their geometric lengths.
    steps = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            steps.append((dr, dc, dx * math.hypot(dr, dc)))

    counter = itertools.count()                 # heap tiebreaker
    open_heap = [(heuristic(start), next(counter), start)]
    g_score = {start: 0.0}
    came_from: dict = {}
    closed: set = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [_cell_to_metric(tm, r, c) for r, c in cells]
        closed.add(current)

        r0, c0 = current
        g_cur = g_score[current]
        for dr, dc, step in steps:
            r1, c1 = r0 + dr, c0 + dc
            if not (0 <= r1 < h and 0 <= c1 < w):
                continue
            if not trav[r1, c1]:
                continue
            nb = (r1, c1)
            if nb in closed:
                continue
            r_avg = 0.5 * (risk_grid[r0, c0] + risk_grid[r1, c1])
            tentative = g_cur + step * (1.0 + w_risk * r_avg)
            if tentative < g_score.get(nb, math.inf):
                g_score[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_heap,
                               (tentative + heuristic(nb), next(counter), nb))

    return None


# ======================================================================
# Path post-processing for the SCP warm start
# ======================================================================

def smooth_path(waypoints: List[Point],
                n_resample: int = 80,
                smoothing_passes: int = 3) -> List[Point]:
    """Resample to n_resample arc-length-even points, then moving-average
    smooth. Endpoints are preserved exactly."""
    pts = np.asarray(waypoints, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return [tuple(map(float, p)) for p in pts]

    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:                                  # degenerate
        return [tuple(map(float, pts[0]))] * n_resample

    s_new = np.linspace(0.0, total, n_resample)
    xs = np.interp(s_new, s, pts[:, 0])
    ys = np.interp(s_new, s, pts[:, 1])

    # 3-tap moving average; interior only, so endpoints never move.
    for _ in range(smoothing_passes):
        xn, yn = xs.copy(), ys.copy()
        xn[1:-1] = (xs[:-2] + xs[1:-1] + xs[2:]) / 3.0
        yn[1:-1] = (ys[:-2] + ys[1:-1] + ys[2:]) / 3.0
        xs, ys = xn, yn

    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    return list(zip(xs.tolist(), ys.tolist()))


def time_parameterize(waypoints: List[Point], v0: float):
    """Constant-velocity time parameterization of a waypoint list.

    Returns (xs, ys, ts, psis) as numpy arrays:
      xs, ys : waypoint coordinates (m)
      ts     : timestamps (s), ts[0]=0, ts[-1]=T_f = L / v0
      psis   : heading guess (rad, unwrapped) from the path tangent
               -- the SCP rover model carries heading psi_G as state.
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    xs = pts[:, 0].copy()
    ys = pts[:, 1].copy()

    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ts = s / max(v0, 1e-9)                            # t = arc_length / v0

    psis = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    return xs, ys, ts, psis


# ======================================================================
# Stretch: clearance-aware A*
# ======================================================================

def astar_with_clearance(tm: TerrainModel,
                         start_m: Point,
                         goal_m: Point,
                         w_risk: float = 10.0,
                         clearance_m: float = 236.0) -> Optional[List[Point]]:
    """A* on a traversability mask morphologically eroded by clearance_m,
    so the warm start keeps a margin from obstacle edges. Thin wrapper
    around astar_on_terrain — astar_on_terrain itself is unchanged."""
    n_px = max(int(round(clearance_m / tm.pixel_scale_m)), 1)
    eroded = binary_erosion(tm.traversable, iterations=n_px)
    tm_eroded = replace(tm, traversable=eroded)
    return astar_on_terrain(tm_eroded, start_m, goal_m, w_risk=w_risk)


# ======================================================================
# Validation
# ======================================================================

def _save_planner_figure(tm, xs_l, ys_l, raw_path, smooth_pts,
                         start_m, goal_m, out_png):
    ext_km = [e / 1e3 for e in tm.extent_m]
    fig, ax = plt.subplots(figsize=(9, 8))

    risk_show = np.clip(tm.risk, 0, np.percentile(tm.risk, 99.5))
    im = ax.imshow(risk_show, extent=ext_km, origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, label="risk R(x,y)")

    ax.plot(np.asarray(xs_l) / 1e3, np.asarray(ys_l) / 1e3,
            color="cyan", lw=2.0, ls="--", label="straight-line baseline")
    rx = np.array([p[0] for p in raw_path]) / 1e3
    ry = np.array([p[1] for p in raw_path]) / 1e3
    ax.plot(rx, ry, color="white", lw=1.4, alpha=0.9,
            label="A* raw (w_risk=10)")
    sx = np.array([p[0] for p in smooth_pts]) / 1e3
    sy = np.array([p[1] for p in smooth_pts]) / 1e3
    ax.plot(sx, sy, color="lime", lw=2.3, label="smoothed warm start")

    ax.plot(start_m[0] / 1e3, start_m[1] / 1e3, "o", color="deepskyblue",
            ms=12, mec="black", label="start")
    ax.plot(goal_m[0] / 1e3, goal_m[1] / 1e3, "*", color="gold",
            ms=20, mec="black", label="goal")

    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title("Stage-3 planner: A* warm start on the risk field")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _validate():
    print("=" * 70)
    print(" planner.py validation  (Stage-3 A* warm start)")
    print("=" * 70)

    tm = build_terrain()
    ext_km = tuple(e / 1e3 for e in tm.extent_m)
    print(f"DEM shape         : {tm.shape}")
    print(f"pixel scale (m)   : {tm.pixel_scale_m:.3f}")
    print(f"extent (km)       : x [{ext_km[0]:.2f}, {ext_km[1]:.2f}], "
          f"y [{ext_km[2]:.2f}, {ext_km[3]:.2f}]")
    print(f"traversable frac  : {tm.traversable.mean():.2%}")

    # --- start / goal on opposite sides, snapped to traversable plain ---
    start_req = (-13_000.0, 0.0)
    goal_req = (13_000.0, 0.0)
    s_cell = _nearest_traversable_cell(tm, *_metric_to_cell(tm, *start_req))
    g_cell = _nearest_traversable_cell(tm, *_metric_to_cell(tm, *goal_req))
    assert s_cell is not None and g_cell is not None, "no drivable endpoint"
    start_m = _cell_to_metric(tm, *s_cell)
    goal_m = _cell_to_metric(tm, *g_cell)
    print(f"\nstart  : req {start_req} -> cell {s_cell} -> "
          f"({start_m[0]:.0f}, {start_m[1]:.0f}) m  "
          f"traversable={bool(tm.traversable[s_cell])}")
    print(f"goal   : req {goal_req} -> cell {g_cell} -> "
          f"({goal_m[0]:.0f}, {goal_m[1]:.0f}) m  "
          f"traversable={bool(tm.traversable[g_cell])}")

    # --- A* at w_risk = 10 ----------------------------------------------
    path = astar_on_terrain(tm, start_m, goal_m, w_risk=10.0)
    assert path is not None, "A* found no path at w_risk=10"
    print(f"\nA* (w_risk=10)    : {len(path)} waypoints")
    assert _metric_to_cell(tm, *path[0]) == s_cell, "path does not start at start"
    assert _metric_to_cell(tm, *path[-1]) == g_cell, "path does not end at goal"
    print("endpoint check    : starts at start cell, ends at goal cell   OK")

    # --- A* vs straight-line baseline under tm.path_risk ----------------
    xs_a = np.array([p[0] for p in path])
    ys_a = np.array([p[1] for p in path])
    R_astar = tm.path_risk(xs_a, ys_a)
    L_astar = float(np.sum(np.hypot(np.diff(xs_a), np.diff(ys_a))))

    n_line = 600
    xs_l = np.linspace(start_m[0], goal_m[0], n_line)
    ys_l = np.linspace(start_m[1], goal_m[1], n_line)
    R_line = tm.path_risk(xs_l, ys_l)
    L_line = float(np.sum(np.hypot(np.diff(xs_l), np.diff(ys_l))))

    print(f"\nstraight line     : length {L_line/1e3:7.2f} km   "
          f"path_risk {R_line:16.2f}")
    print(f"A* (w_risk=10)    : length {L_astar/1e3:7.2f} km   "
          f"path_risk {R_astar:16.2f}")
    assert R_astar < R_line, (
        f"A* path_risk ({R_astar}) must be < straight line ({R_line})")
    print(f"-> A* path_risk is {R_line/max(R_astar,1e-9):.0f}x lower "
          "than straight-through.   OK")

    # --- w_risk sweep: distance-vs-risk tradeoff ------------------------
    print("\nw_risk sweep (distance vs risk tradeoff):")
    print(f"  {'w_risk':>8}  {'length (km)':>12}  {'path_risk':>16}")
    sweep = {}
    for w in (0.0, 1.0, 10.0, 100.0):
        p = astar_on_terrain(tm, start_m, goal_m, w_risk=w)
        assert p is not None, f"A* found no path at w_risk={w}"
        xs = np.array([q[0] for q in p])
        ys = np.array([q[1] for q in p])
        L = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        R = tm.path_risk(xs, ys)
        sweep[w] = (L, R, p)
        print(f"  {w:8.1f}  {L/1e3:12.2f}  {R:16.2f}")

    ws = [0.0, 1.0, 10.0, 100.0]
    Ls = [sweep[w][0] for w in ws]
    Rs = [sweep[w][1] for w in ws]
    assert all(Ls[i] <= Ls[i + 1] + 1e-6 for i in range(len(ws) - 1)), \
        f"length not monotonically non-decreasing: {Ls}"
    assert all(Rs[i] >= Rs[i + 1] - 1e-6 for i in range(len(ws) - 1)), \
        f"path_risk not monotonically non-increasing: {Rs}"
    print("-> length non-decreasing, path_risk non-increasing in w_risk.  OK")

    # --- smooth + time-parameterize the w_risk=10 path ------------------
    raw10 = sweep[10.0][2]
    smooth_pts = smooth_path(raw10, n_resample=80, smoothing_passes=3)
    assert math.isclose(smooth_pts[0][0], raw10[0][0]) and \
        math.isclose(smooth_pts[0][1], raw10[0][1]), "start not preserved"
    assert math.isclose(smooth_pts[-1][0], raw10[-1][0]) and \
        math.isclose(smooth_pts[-1][1], raw10[-1][1]), "goal not preserved"
    print(f"\nsmooth_path       : {len(raw10)} -> {len(smooth_pts)} waypoints, "
          "endpoints preserved   OK")

    v0 = 0.25                                         # nominal rover speed m/s
    xs_t, ys_t, ts_t, psis_t = time_parameterize(smooth_pts, v0)
    L_sm = float(np.sum(np.hypot(np.diff(xs_t), np.diff(ys_t))))
    Tf = float(ts_t[-1])
    assert abs(float(ts_t[0])) < 1e-9, "ts[0] must be 0"
    assert math.isclose(Tf, L_sm / v0, rel_tol=1e-6), "T_f != L / v0"
    print(f"time_parameterize : v0={v0} m/s  L={L_sm/1e3:.2f} km  "
          f"T_f={Tf:.0f} s ({Tf/3600:.2f} h)")
    print(f"                    ts[0]={ts_t[0]:.3f}  ts[-1]=T_f  "
          f"heading samples={len(psis_t)}   OK")

    # --- figure ---------------------------------------------------------
    out_png = os.path.join(THIS_DIR, "planner_validation.png")
    _save_planner_figure(tm, xs_l, ys_l, raw10, smooth_pts,
                         start_m, goal_m, out_png)
    print(f"\nsaved figure -> {out_png}")

    print("\n" + "=" * 70)
    print(" ALL VALIDATION CHECKS PASSED.")
    print("=" * 70)

    # --- stretch: clearance-aware A* ------------------------------------
    # Demonstrated at w_risk=0: with no risk weighting the shortest path
    # hugs the no-go ring, so eroding the mask visibly pushes it clear
    # (at w_risk=10 the risk term already keeps a margin, so erosion is a
    # no-op there).
    print("\n--- stretch: clearance-aware A* (demonstrated at w_risk=0) ---")
    print(f"  {'clearance':>12}  {'wpts':>5}  {'length (km)':>12}  "
          f"{'path_risk':>14}")
    base = astar_on_terrain(tm, start_m, goal_m, w_risk=0.0)
    xb = np.array([p[0] for p in base])
    yb = np.array([p[1] for p in base])
    Lb = float(np.sum(np.hypot(np.diff(xb), np.diff(yb))))
    print(f"  {'0 m':>12}  {len(base):5d}  {Lb/1e3:12.2f}  "
          f"{tm.path_risk(xb, yb):14.2f}")
    for clr in (236.0, 472.0, 708.0):
        pc = astar_with_clearance(tm, start_m, goal_m,
                                  w_risk=0.0, clearance_m=clr)
        if pc is None:
            print(f"  {f'{clr:.0f} m':>12}  no path (mask eroded away)")
            continue
        xs = np.array([p[0] for p in pc])
        ys = np.array([p[1] for p in pc])
        L = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        R = tm.path_risk(xs, ys)
        print(f"  {f'{clr:.0f} m':>12}  {len(pc):5d}  {L/1e3:12.2f}  "
              f"{R:14.2f}")


if __name__ == "__main__":
    _validate()
