//! The live turn of a selection, fused (the real voxels turn
//! during the drag, so every step rebuilds the grid; in numpy that was ten
//! passes over the whole grid, one after another).
//!
//! `live_turn` builds the grid as it looks at one angle: the press voxels with
//! the selection cleared and its turned box pasted back, and the new selection
//! mask, in one parallel pass. `apply_grid` writes such a grid into the model:
//! every cell that differs changes, never a cut or a locked one, and the dirty
//! chunks come back; `changed_chunks` says which chunks that write would touch,
//! so the undo can snapshot them first. Both do what rok.core.grid does in
//! numpy (set_grid, changed_chunks), cell for cell.

use rayon::prelude::*;

use crate::bake;

/// (want, new mask) for the selection under `mask` inside the inclusive box
/// [lo, hi] of a grid of `shape`, turned by `degrees` about `axis` and pasted
/// at `position` (the turned box's cell (0, 0, 0), in grid cells, as
/// free_rotate.turned_place puts it). Cells of the turned box outside the grid
/// are lost.
#[allow(clippy::too_many_arguments)]
pub fn live_turn(
    press: &[u8], mask: &[bool], shape: [usize; 3], lo: [usize; 3], hi: [usize; 3],
    axis: usize, degrees: f64, position: [i64; 3],
) -> (Vec<u8>, Vec<bool>) {
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let bshape = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let bstrides = [bshape[1] * bshape[2], bshape[2], 1usize];
    // The selected voxels in their box.
    let mut block = vec![0u8; bshape[0] * bshape[1] * bshape[2]];
    block.par_chunks_mut(bstrides[0]).enumerate().for_each(|(bx, slab)| {
        let x = lo[0] + bx;
        for by in 0..bshape[1] {
            let y = lo[1] + by;
            let row = x * strides[0] + y * strides[1] + lo[2];
            let out = &mut slab[by * bstrides[1]..(by + 1) * bstrides[1]];
            for (bz, cell) in out.iter_mut().enumerate() {
                let i = row + bz;
                if mask[i] {
                    *cell = press[i];
                }
            }
        }
    });
    let (turned, tshape) = bake::rotate_voxels(&block, bshape, axis, degrees);
    let tstrides = [tshape[1] * tshape[2], tshape[2], 1usize];
    // The grid at this angle: the press, the selection cleared, the turned box on top.
    let mut want = vec![0u8; press.len()];
    let mut new_mask = vec![false; press.len()];
    want.par_chunks_mut(strides[0]).zip(new_mask.par_chunks_mut(strides[0])).enumerate().for_each(
        |(x, (wslab, mslab))| {
            let tx = x as i64 - position[0];
            let in_x = tx >= 0 && (tx as usize) < tshape[0];
            for y in 0..shape[1] {
                let ty = y as i64 - position[1];
                let in_y = in_x && ty >= 0 && (ty as usize) < tshape[1];
                let base = x * strides[0] + y * strides[1];
                for z in 0..shape[2] {
                    let i = base + z;
                    let mut v = if mask[i] { 0 } else { press[i] };
                    let mut sel = false;
                    if in_y {
                        let tz = z as i64 - position[2];
                        if tz >= 0 && (tz as usize) < tshape[2] {
                            let t = turned[tx as usize * tstrides[0] + ty as usize * tstrides[1] + tz as usize];
                            if t != 0 {
                                v = t;
                                sel = true;
                            }
                        }
                    }
                    wslab[y * strides[1] + z] = v;
                    mslab[y * strides[1] + z] = sel;
                }
            }
        },
    );
    (want, new_mask)
}

// --- the cage: free-form deformation -----------------------
//
// The numeric kernel of rok/core/lattice.py, operation for operation: the same basis
// recurrence, the same loop order over a span's control points, the same multiply order,
// the same adjugate solve, the same start, four Newton steps with no early exit, the
// same accept test and edge rule. All f64; Rust does not fuse a multiply and an add, so
// the rounding is numpy's. Change one side and change the other.

const MAX_DEGREE: usize = 3;
const MAX_TERMS: usize = 6; // basis functions on a span: degree 5 (a Bernstein span of 6 points) plus one
const NEWTON_STEPS: usize = 4;
const DET_FRACTION: f64 = 1e-3;
const DAMP: f64 = 0.5;
const ACCEPT_RESIDUAL: f64 = 0.5;

struct Cage<'a> {
    lo: [f64; 3],
    size: [f64; 3],
    hi: [f64; 3],
    counts: [usize; 3],
    degree: [usize; 3],
    spans: [usize; 3],
    knots: [Vec<f64>; 3],
    points: &'a [f64], // (nx, ny, nz, 3), C order
    det_min: f64,
}

/// The p + 1 nonzero basis functions of degree p on knot span `s` at `t`.
fn funs(u: &[f64], s: usize, t: f64, p: usize) -> [f64; MAX_TERMS] {
    let mut n = [0.0f64; MAX_TERMS];
    n[0] = 1.0;
    let mut left = [0.0f64; MAX_TERMS];
    let mut right = [0.0f64; MAX_TERMS];
    for j in 1..=p {
        left[j] = t - u[s + 1 - j];
        right[j] = u[s + j] - t;
        let mut saved = 0.0f64;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

fn span_first(spans: usize, t: f64) -> usize {
    let f = (t * spans as f64).floor();
    let f = if f.is_finite() { f } else { 0.0 };
    f.max(0.0).min((spans - 1) as f64) as usize
}

impl<'a> Cage<'a> {
    fn new(lo: [i64; 3], size: [i64; 3], counts: [usize; 3], points: &'a [f64], bernstein: bool) -> Self {
        let mut degree = [0usize; 3];
        let mut spans = [0usize; 3];
        let mut knots: [Vec<f64>; 3] = [vec![], vec![], vec![]];
        for a in 0..3 {
            let p = if bernstein { counts[a] - 1 } else { MAX_DEGREE.min(counts[a] - 1) };
            degree[a] = p;
            spans[a] = counts[a] - p;
            let mut k = vec![0.0f64; p + 1];
            for j in 1..spans[a] {
                k.push(j as f64 / spans[a] as f64);
            }
            k.extend(std::iter::repeat_n(1.0f64, p + 1));
            knots[a] = k;
        }
        let lof = [lo[0] as f64, lo[1] as f64, lo[2] as f64];
        let sizef = [size[0] as f64, size[1] as f64, size[2] as f64];
        let hi = [lof[0] + sizef[0] - 1.0, lof[1] + sizef[1] - 1.0, lof[2] + sizef[2] - 1.0];
        let det_min = DET_FRACTION * sizef[0] * sizef[1] * sizef[2];
        Cage { lo: lof, size: sizef, hi, counts, degree, spans, knots, points, det_min }
    }

    /// (first point, values, derivatives) of one axis' basis at `t`.
    fn basis(&self, axis: usize, t: f64) -> (usize, [f64; MAX_TERMS], [f64; MAX_TERMS]) {
        let p = self.degree[axis];
        let u = &self.knots[axis];
        let first = span_first(self.spans[axis], t);
        let s = first + p;
        let top = funs(u, s, t, p);
        let low = funs(u, s, t, p - 1);
        let mut d = [0.0f64; MAX_TERMS];
        for a in 0..=p {
            let i = s - p + a;
            let t1 = if a >= 1 { low[a - 1] / (u[i + p] - u[i]) } else { 0.0 };
            let t2 = if a + 1 <= p { low[a] / (u[i + p + 1] - u[i + 1]) } else { 0.0 };
            d[a] = p as f64 * (t1 - t2);
        }
        (first, top, d)
    }

    /// F(u), and J[c][b] = dF_c / du_b when `jac`.
    fn eval(&self, u: [f64; 3], jac: bool) -> ([f64; 3], [[f64; 3]; 3]) {
        let (fx, vx, dx) = self.basis(0, u[0]);
        let (fy, vy, dy) = self.basis(1, u[1]);
        let (fz, vz, dz) = self.basis(2, u[2]);
        let mut f = [0.0f64; 3];
        let mut j = [[0.0f64; 3]; 3];
        let (ny, nz) = (self.counts[1], self.counts[2]);
        for a in 0..=self.degree[0] {
            for b in 0..=self.degree[1] {
                let w_ab = vx[a] * vy[b];
                let dx_ab = dx[a] * vy[b];
                let dy_ab = vx[a] * dy[b];
                for c in 0..=self.degree[2] {
                    let at = (((fx + a) * ny + (fy + b)) * nz + (fz + c)) * 3;
                    let p = [self.points[at], self.points[at + 1], self.points[at + 2]];
                    let w = w_ab * vz[c];
                    for k in 0..3 {
                        f[k] += w * p[k];
                    }
                    if jac {
                        let wx = dx_ab * vz[c];
                        let wy = dy_ab * vz[c];
                        let wz = w_ab * dz[c];
                        for k in 0..3 {
                            j[k][0] += wx * p[k];
                            j[k][1] += wy * p[k];
                            j[k][2] += wz * p[k];
                        }
                    }
                }
            }
        }
        (f, j)
    }

    /// J^-1 r by the adjugate, damped when J is near singular.
    fn solve(&self, m: &[[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        let adj = [
            [c00, m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]],
            [c01, m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]],
            [c02, m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]],
        ];
        let small = det.abs() < self.det_min;
        let det_used = if small { self.det_min.copysign(det) } else { det };
        let factor = if small { DAMP } else { 1.0 };
        let mut step = [0.0f64; 3];
        for i in 0..3 {
            step[i] = (((adj[i][0] * r[0] + adj[i][1] * r[1]) + adj[i][2] * r[2]) / det_used) * factor;
        }
        step
    }

    fn newton(&self, x: [f64; 3], u0: [f64; 3]) -> [f64; 3] {
        let mut u = u0;
        for _ in 0..NEWTON_STEPS {
            let (f, j) = self.eval(u, true);
            let step = self.solve(&j, [f[0] - x[0], f[1] - x[1], f[2] - x[2]]);
            u = [u[0] - step[0], u[1] - step[1], u[2] - step[2]];
        }
        u
    }

    fn accepted(&self, u: [f64; 3], x: [f64; 3]) -> bool {
        let (f, _) = self.eval(u, false);
        let inside = (0..3).all(|a| u[a] >= 0.0 && u[a] <= 1.0);
        let near = (0..3).map(|a| (f[a] - x[a]).abs()).fold(f64::NEG_INFINITY, f64::max) <= ACCEPT_RESIDUAL;
        inside && near
    }

    /// The rest cell (per axis) the target cell `cell` takes its colour from, or None.
    fn inverse_cell(&self, cell: [i64; 3]) -> Option<[usize; 3]> {
        let x = [cell[0] as f64 + 0.5, cell[1] as f64 + 0.5, cell[2] as f64 + 0.5];
        let u0 = [
            (x[0] - self.lo[0]) / self.size[0],
            (x[1] - self.lo[1]) / self.size[1],
            (x[2] - self.lo[2]) / self.size[2],
        ];
        let mut u = self.newton(x, u0);
        if !self.accepted(u, x) {
            let start = [
                (span_first(self.spans[0], u0[0]) as f64 + 0.5) / self.spans[0] as f64,
                (span_first(self.spans[1], u0[1]) as f64 + 0.5) / self.spans[1] as f64,
                (span_first(self.spans[2], u0[2]) as f64 + 0.5) / self.spans[2] as f64,
            ];
            u = self.newton(x, start);
            if !self.accepted(u, x) {
                return None;
            }
        }
        let mut rc = [0usize; 3];
        for a in 0..3 {
            let q = (self.lo[a] + u[a] * self.size[a]).floor();
            let q = if q.is_finite() { q } else { self.lo[a] };
            rc[a] = q.max(self.lo[a]).min(self.hi[a]) as usize;
        }
        Some(rc)
    }
}

/// (want, new mask) of a cage deformation (rok/core/lattice.py `deform_grid`): the
/// rest voxels with the rest selection cleared, then every cell inside one of `boxes`
/// (inclusive lo xyz, hi xyz, clipped to the grid by the caller) whose inverse lands on
/// a selected, filled rest cell takes that cell's colour. The rest box is `lo ..
/// lo + size`, `points` is (nx, ny, nz, 3) in C order.
#[allow(clippy::too_many_arguments)]
pub fn live_deform(
    rest: &[u8], mask: &[bool], shape: [usize; 3], lo: [i64; 3], size: [i64; 3], counts: [usize; 3],
    points: &[f64], boxes: &[[i64; 6]], bernstein: bool,
) -> (Vec<u8>, Vec<bool>) {
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let cage = Cage::new(lo, size, counts, points, bernstein);
    let mut want = vec![0u8; rest.len()];
    let mut new_mask = vec![false; rest.len()];
    want.par_chunks_mut(strides[0]).zip(new_mask.par_chunks_mut(strides[0])).enumerate().for_each(
        |(x, (wslab, mslab))| {
            let base = x * strides[0];
            for (i, w) in wslab.iter_mut().enumerate() {
                *w = if mask[base + i] { 0 } else { rest[base + i] };
            }
            let here: Vec<&[i64; 6]> = boxes.iter().filter(|b| b[0] <= x as i64 && x as i64 <= b[3]).collect();
            if here.is_empty() {
                return;
            }
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let (yi, zi) = (y as i64, z as i64);
                    if !here.iter().any(|b| b[1] <= yi && yi <= b[4] && b[2] <= zi && zi <= b[5]) {
                        continue;
                    }
                    if let Some(rc) = cage.inverse_cell([x as i64, yi, zi]) {
                        let r = rc[0] * strides[0] + rc[1] * strides[1] + rc[2];
                        if mask[r] && rest[r] != 0 {
                            wslab[y * strides[1] + z] = rest[r];
                            mslab[y * strides[1] + z] = true;
                        }
                    }
                }
            }
        },
    );
    (want, new_mask)
}

// --- the fast cage: the inverse by interpolation, checked --------------------------------
//
// A dense 128 x 128 x 128 selection has 1.9 million target cells; four Newton steps a cell
// (`live_deform`) take 0.6 s. Here the exact solve runs only at the corners of 8-cell
// blocks, and every cell takes its rest place from the trilinear blend of its block's
// corners. The blend is checked: at the centre of each block the exact solve must agree
// with it to FAST_TOL of a cell, and every corner and centre must have converged. A block
// that fails (a fold, a sharp bend, a corner Newton lost) is solved cell by cell, exactly,
// as `live_deform` does. Accuracy: a rest place is right to about FAST_TOL of a cell, so a
// cell can pick a neighbour of the exact rest cell only when it lies that close to a rest
// cell's edge (tests/test_native_paths.py counts how many).

const BLOCK: i64 = 8;
const FAST_TOL: f64 = 0.02; // cells
const SAMPLE_OK: f64 = 1e-6; // cells: residual of a converged exact solve

impl<'a> Cage<'a> {
    fn residual(&self, u: [f64; 3], x: [f64; 3]) -> f64 {
        let (f, _) = self.eval(u, false);
        (0..3).map(|a| (f[a] - x[a]).abs()).fold(f64::NEG_INFINITY, f64::max)
    }

    /// The exact rest-box coordinate of position `x`, with the retry of `inverse_cell`, and
    /// whether the solve converged (residual within SAMPLE_OK). Not clipped to the rest box.
    fn solve_at(&self, x: [f64; 3]) -> ([f64; 3], bool) {
        let u0 = [
            (x[0] - self.lo[0]) / self.size[0],
            (x[1] - self.lo[1]) / self.size[1],
            (x[2] - self.lo[2]) / self.size[2],
        ];
        let mut u = self.newton(x, u0);
        let mut res = self.residual(u, x);
        if !(res <= SAMPLE_OK) {
            let start = [
                (span_first(self.spans[0], u0[0]) as f64 + 0.5) / self.spans[0] as f64,
                (span_first(self.spans[1], u0[1]) as f64 + 0.5) / self.spans[1] as f64,
                (span_first(self.spans[2], u0[2]) as f64 + 0.5) / self.spans[2] as f64,
            ];
            let u2 = self.newton(x, start);
            let r2 = self.residual(u2, x);
            if r2 < res || res.is_nan() {
                u = u2;
                res = r2;
            }
        }
        (u, res <= SAMPLE_OK)
    }

    /// The rest cell of a rest-box coordinate (the edge rule of `inverse_cell`).
    fn rest_cell(&self, u: [f64; 3]) -> [usize; 3] {
        let mut rc = [0usize; 3];
        for a in 0..3 {
            let q = (self.lo[a] + u[a] * self.size[a]).floor();
            let q = if q.is_finite() { q } else { self.lo[a] };
            rc[a] = q.max(self.lo[a]).min(self.hi[a]) as usize;
        }
        rc
    }
}

/// `live_deform` with the inverse by checked interpolation (see above). Same arguments
/// and result; the answer differs from the exact one only where a rest place lies within
/// FAST_TOL of a rest cell's edge.
#[allow(clippy::too_many_arguments)]
pub fn live_deform_fast(
    rest: &[u8], mask: &[bool], shape: [usize; 3], lo: [i64; 3], size: [i64; 3], counts: [usize; 3],
    points: &[f64], boxes: &[[i64; 6]], bernstein: bool,
) -> (Vec<u8>, Vec<bool>) {
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let cage = Cage::new(lo, size, counts, points, bernstein);
    let mut want = vec![0u8; rest.len()];
    let mut new_mask = vec![false; rest.len()];
    want.par_chunks_mut(strides[0]).zip(new_mask.par_chunks_mut(strides[0])).enumerate().for_each(
        |(x, (wslab, _))| {
            let base = x * strides[0];
            for (i, w) in wslab.iter_mut().enumerate() {
                *w = if mask[base + i] { 0 } else { rest[base + i] };
            }
        },
    );
    if boxes.is_empty() {
        return (want, new_mask);
    }
    // The region the blocks tile: the box around every target box.
    let mut blo = [i64::MAX; 3];
    let mut bhi = [i64::MIN; 3];
    for b in boxes {
        for a in 0..3 {
            blo[a] = blo[a].min(b[a]);
            bhi[a] = bhi[a].max(b[a + 3]);
        }
    }
    let nb = [
        ((bhi[0] - blo[0] + 1 + BLOCK - 1) / BLOCK) as usize,
        ((bhi[1] - blo[1] + 1 + BLOCK - 1) / BLOCK) as usize,
        ((bhi[2] - blo[2] + 1 + BLOCK - 1) / BLOCK) as usize,
    ];
    let ns = [nb[0] + 1, nb[1] + 1, nb[2] + 1];
    let pos = |a: usize, i: usize| (blo[a] + i as i64 * BLOCK) as f64 + 0.5;
    // The exact solve at every block corner.
    let samples: Vec<([f64; 3], bool)> = (0..ns[0] * ns[1] * ns[2])
        .into_par_iter()
        .map(|n| {
            let (i, j, k) = (n / (ns[1] * ns[2]), (n / ns[2]) % ns[1], n % ns[2]);
            cage.solve_at([pos(0, i), pos(1, j), pos(2, k)])
        })
        .collect();
    let sample = |i: usize, j: usize, k: usize| samples[(i * ns[1] + j) * ns[2] + k];
    // A block is interpolated when its corners and its centre converged and agree.
    let good: Vec<bool> = (0..nb[0] * nb[1] * nb[2])
        .into_par_iter()
        .map(|n| {
            let (i, j, k) = (n / (nb[1] * nb[2]), (n / nb[2]) % nb[1], n % nb[2]);
            let mut mean = [0.0f64; 3];
            for (di, dj, dk) in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)] {
                let (u, ok) = sample(i + di, j + dj, k + dk);
                if !ok {
                    return false;
                }
                for a in 0..3 {
                    mean[a] += u[a] * 0.125;
                }
            }
            let half = BLOCK as f64 / 2.0;
            let (uc, ok) = cage.solve_at([pos(0, i) + half, pos(1, j) + half, pos(2, k) + half]);
            ok && (0..3).all(|a| (uc[a] - mean[a]).abs() * cage.size[a] <= FAST_TOL)
        })
        .collect();
    want.par_chunks_mut(strides[0]).zip(new_mask.par_chunks_mut(strides[0])).enumerate().for_each(
        |(x, (wslab, mslab))| {
            let xi = x as i64;
            let here: Vec<&[i64; 6]> = boxes.iter().filter(|b| b[0] <= xi && xi <= b[3]).collect();
            if here.is_empty() {
                return;
            }
            let bi = ((xi - blo[0]) / BLOCK) as usize;
            let fx = ((xi - blo[0]) % BLOCK) as f64 / BLOCK as f64;
            for bj in 0..nb[1] {
                let y0 = blo[1] + bj as i64 * BLOCK;
                let y1 = (y0 + BLOCK - 1).min(shape[1] as i64 - 1);
                for bk in 0..nb[2] {
                    let z0 = blo[2] + bk as i64 * BLOCK;
                    let z1 = (z0 + BLOCK - 1).min(shape[2] as i64 - 1);
                    if y0 > y1 || z0 > z1 {
                        continue;
                    }
                    let whole = here.iter().any(|b| b[1] <= y0 && y1 <= b[4] && b[2] <= z0 && z1 <= b[5]);
                    let is_good = good[(bi * nb[1] + bj) * nb[2] + bk];
                    // x-blend of the four edge corners of this plane
                    let mut e = [[[0.0f64; 3]; 2]; 2];
                    if is_good {
                        for dj in 0..2 {
                            for dk in 0..2 {
                                let (u0, _) = sample(bi, bj + dj, bk + dk);
                                let (u1, _) = sample(bi + 1, bj + dj, bk + dk);
                                for a in 0..3 {
                                    e[dj][dk][a] = u0[a] + (u1[a] - u0[a]) * fx;
                                }
                            }
                        }
                    }
                    for y in y0..=y1 {
                        let fy = (y - y0) as f64 / BLOCK as f64;
                        // y-blend: the two ends of the z run
                        let mut ends = [[0.0f64; 3]; 2];
                        if is_good {
                            for dk in 0..2 {
                                for a in 0..3 {
                                    ends[dk][a] = e[0][dk][a] + (e[1][dk][a] - e[0][dk][a]) * fy;
                                }
                            }
                        }
                        for z in z0..=z1 {
                            if !whole && !here.iter().any(|b| b[1] <= y && y <= b[4] && b[2] <= z && z <= b[5]) {
                                continue;
                            }
                            let rc = if is_good {
                                let fz = (z - z0) as f64 / BLOCK as f64;
                                let u = [
                                    ends[0][0] + (ends[1][0] - ends[0][0]) * fz,
                                    ends[0][1] + (ends[1][1] - ends[0][1]) * fz,
                                    ends[0][2] + (ends[1][2] - ends[0][2]) * fz,
                                ];
                                if !(u[0] >= 0.0 && u[0] <= 1.0 && u[1] >= 0.0 && u[1] <= 1.0 && u[2] >= 0.0 && u[2] <= 1.0) {
                                    continue;
                                }
                                Some(cage.rest_cell(u))
                            } else {
                                cage.inverse_cell([xi, y, z])
                            };
                            if let Some(rc) = rc {
                                let r = rc[0] * strides[0] + rc[1] * strides[1] + rc[2];
                                if mask[r] && rest[r] != 0 {
                                    wslab[y as usize * strides[1] + z as usize] = rest[r];
                                    mslab[y as usize * strides[1] + z as usize] = true;
                                }
                            }
                        }
                    }
                }
            }
        },
    );
    (want, new_mask)
}

fn chunk_counts(shape: [usize; 3], chunk: usize) -> [usize; 3] {
    [shape[0].div_ceil(chunk), shape[1].div_ceil(chunk), shape[2].div_ceil(chunk)]
}

/// Where a write of `want` changes a voxel: it differs, and the cell is not
/// cut (unless `ignore_cut`) and not locked. One bool per cell.
fn changed_mask(
    voxels: &[u8], want: &[u8], lock: Option<&[bool]>, cut: Option<&[bool]>, ignore_cut: bool, shape: [usize; 3],
) -> Vec<bool> {
    let row = shape[1] * shape[2];
    let mut changed = vec![false; voxels.len()];
    changed.par_chunks_mut(row).enumerate().for_each(|(x, slab)| {
        let base = x * row;
        for (k, c) in slab.iter_mut().enumerate() {
            let i = base + k;
            let mut d = voxels[i] != want[i];
            if let (Some(cut), false) = (cut, ignore_cut) {
                d &= !cut[i];
            }
            if let Some(lock) = lock {
                d &= !lock[i];
            }
            *c = d;
        }
    });
    changed
}

/// The chunks of the changed cells and of the chunks beyond a changed cell on
/// a chunk's face: what grid._mark_dirty_cells marks.
fn chunks_of(changed: &[bool], shape: [usize; 3], chunk: usize) -> Vec<(usize, usize, usize)> {
    let counts = chunk_counts(shape, chunk);
    let n = counts[0] * counts[1] * counts[2];
    let row = shape[1] * shape[2];
    let flags = changed
        .par_chunks(row)
        .enumerate()
        .map(|(x, slab)| {
            let mut local = vec![false; n];
            let cx = x / chunk;
            for (k, &c) in slab.iter().enumerate() {
                if !c {
                    continue;
                }
                let y = k / shape[2];
                let z = k % shape[2];
                let (cy, cz) = (y / chunk, z / chunk);
                let mut mark = |a: usize, b: usize, c: usize| local[(a * counts[1] + b) * counts[2] + c] = true;
                mark(cx, cy, cz);
                if x % chunk == 0 && cx > 0 {
                    mark(cx - 1, cy, cz);
                }
                if (x + 1) % chunk == 0 && cx + 1 < counts[0] {
                    mark(cx + 1, cy, cz);
                }
                if y.is_multiple_of(chunk) && cy > 0 {
                    mark(cx, cy - 1, cz);
                }
                if (y + 1).is_multiple_of(chunk) && cy + 1 < counts[1] {
                    mark(cx, cy + 1, cz);
                }
                if z.is_multiple_of(chunk) && cz > 0 {
                    mark(cx, cy, cz - 1);
                }
                if (z + 1).is_multiple_of(chunk) && cz + 1 < counts[2] {
                    mark(cx, cy, cz + 1);
                }
            }
            local
        })
        .reduce(|| vec![false; n], |mut a, b| {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x |= *y;
            }
            a
        });
    let mut out = Vec::new();
    for (i, &f) in flags.iter().enumerate() {
        if f {
            out.push((i / (counts[1] * counts[2]), (i / counts[2]) % counts[1], i % counts[2]));
        }
    }
    out
}

/// The chunks a write of `want` would mark dirty.
pub fn changed_chunks(
    voxels: &[u8], want: &[u8], lock: Option<&[bool]>, cut: Option<&[bool]>, ignore_cut: bool, shape: [usize; 3],
    chunk: usize,
) -> Vec<(usize, usize, usize)> {
    chunks_of(&changed_mask(voxels, want, lock, cut, ignore_cut, shape), shape, chunk)
}

/// Write `want` into `voxels` (and the visible copy, which never shows a cut
/// cell): (changed count, writes the lock stopped, dirty chunks). `cut` is
/// the section planes' mask whenever there are planes; `ignore_cut` writes
/// the cut cells too (a palette swap), into `voxels` only.
#[allow(clippy::too_many_arguments)]
pub fn apply_grid(
    voxels: &mut [u8], visible: Option<&mut [u8]>, want: &[u8], lock: Option<&[bool]>, cut: Option<&[bool]>,
    ignore_cut: bool, shape: [usize; 3], chunk: usize,
) -> (usize, usize, Vec<(usize, usize, usize)>) {
    let changed = changed_mask(voxels, want, lock, cut, ignore_cut, shape);
    // The writes the lock stopped: among the cells that differ and would be written.
    let stopped = match lock {
        None => 0,
        Some(lock) => (0..voxels.len())
            .into_par_iter()
            .filter(|&i| voxels[i] != want[i] && lock[i] && (ignore_cut || cut.is_none_or(|c| !c[i])))
            .count(),
    };
    let row = shape[1] * shape[2];
    let count = voxels
        .par_chunks_mut(row)
        .enumerate()
        .map(|(x, slab)| {
            let base = x * row;
            let mut n = 0usize;
            for (k, v) in slab.iter_mut().enumerate() {
                if changed[base + k] {
                    *v = want[base + k];
                    n += 1;
                }
            }
            n
        })
        .sum();
    if let Some(vis) = visible {
        vis.par_chunks_mut(row).enumerate().for_each(|(x, slab)| {
            let base = x * row;
            for (k, v) in slab.iter_mut().enumerate() {
                let i = base + k;
                if changed[i] && cut.is_none_or(|c| !c[i]) {
                    *v = want[i];
                }
            }
        });
    }
    (count, stopped, chunks_of(&changed, shape, chunk))
}

/// How many cells hold each of the 256 indices: per thread 256 bins, summed.
/// numpy's bincount widens 16 million bytes to int64 first (36 ms on a
/// 256-cube; the palette strip asks after every stroke).
pub fn count_colours(voxels: &[u8]) -> [i64; 256] {
    voxels
        .par_chunks(1 << 16)
        .map(|part| {
            let mut bins = [0i64; 256];
            for &v in part {
                bins[v as usize] += 1;
            }
            bins
        })
        .reduce(|| [0i64; 256], |mut a, b| {
            for i in 0..256 {
                a[i] += b[i];
            }
            a
        })
}

/// The brush: one box of the grid, `mask` (the sphere, C order over `dims`)
/// says which cells the brush covers. Mode 0 fills empty cells with `colour`
/// (Attach), 1 clears filled ones (Erase), 2 recolours filled cells of
/// another colour (Paint). A locked cell is skipped and counted in
/// `stopped`; `cut` (box-sized) marks cells beyond a section plane, which are
/// skipped. `visible` follows `voxels`; `mark` (grid-sized) gets a true for
/// every written cell (the stroke's own record of what it took or placed).
/// Returns (changed count, stopped, dirty chunks). No cell list: a brush of
/// size 50 covers a million cells (strokes must be fast).
#[allow(clippy::too_many_arguments)]
pub fn brush_write(
    voxels: &mut [u8], visible: Option<&mut [u8]>, lock: Option<&[bool]>, cut: Option<&[bool]>, lo: [usize; 3],
    dims: [usize; 3], mask: &[bool], mode: u8, colour: u8, mark: Option<&mut [bool]>, shape: [usize; 3], chunk: usize,
) -> (usize, usize, Vec<(usize, usize, usize)>) {
    let row = shape[1] * shape[2];
    let brow = dims[1] * dims[2];
    // Pass 1: which cells of the box change (box-sized flags), per x-slab in parallel.
    let mut hit = vec![false; dims[0] * brow];
    let stopped: usize = hit
        .par_chunks_mut(brow)
        .enumerate()
        .map(|(bx, slab)| {
            let x = lo[0] + bx;
            let mut stopped = 0usize;
            for by in 0..dims[1] {
                let y = lo[1] + by;
                let gbase = x * row + y * shape[2] + lo[2];
                let bbase = bx * brow + by * dims[2];
                for bz in 0..dims[2] {
                    let b = bbase + bz;
                    if !mask[b] || cut.is_some_and(|c| c[b]) {
                        continue;
                    }
                    let g = gbase + bz;
                    let cur = voxels[g];
                    let wants = match mode {
                        0 => cur == 0,
                        1 => cur != 0,
                        _ => cur != 0 && cur != colour,
                    };
                    if !wants {
                        continue;
                    }
                    if lock.is_some_and(|l| l[g]) {
                        stopped += 1;
                        continue;
                    }
                    slab[by * dims[2] + bz] = true;
                }
            }
            stopped
        })
        .sum();
    let value = if mode == 1 { 0 } else { colour };
    // Pass 2: the writes, per x-slab of the box.
    let write = |target: &mut [u8]| {
        let base = lo[0] * row;
        let span = dims[0] * row;
        target[base..base + span].par_chunks_mut(row).enumerate().for_each(|(bx, slab)| {
            for by in 0..dims[1] {
                let y = lo[1] + by;
                let bbase = bx * brow + by * dims[2];
                for bz in 0..dims[2] {
                    if hit[bbase + bz] {
                        slab[y * shape[2] + lo[2] + bz] = value;
                    }
                }
            }
        });
    };
    write(voxels);
    if let Some(vis) = visible {
        write(vis);
    }
    if let Some(mark) = mark {
        let base = lo[0] * row;
        let span = dims[0] * row;
        mark[base..base + span].par_chunks_mut(row).enumerate().for_each(|(bx, slab)| {
            for by in 0..dims[1] {
                let y = lo[1] + by;
                let bbase = bx * brow + by * dims[2];
                for bz in 0..dims[2] {
                    if hit[bbase + bz] {
                        slab[y * shape[2] + lo[2] + bz] = true;
                    }
                }
            }
        });
    }
    // Pass 3: the count and the dirty chunks (a cell on a chunk's face dirties the neighbour too).
    let count = hit.par_iter().filter(|&&h| h).count();
    let counts = [shape[0].div_ceil(chunk), shape[1].div_ceil(chunk), shape[2].div_ceil(chunk)];
    let mut flags = vec![false; counts[0] * counts[1] * counts[2]];
    let mut set = |c: [isize; 3]| {
        if (0..3).all(|k| c[k] >= 0 && (c[k] as usize) < counts[k]) {
            flags[(c[0] as usize * counts[1] + c[1] as usize) * counts[2] + c[2] as usize] = true;
        }
    };
    for bx in 0..dims[0] {
        for by in 0..dims[1] {
            let bbase = bx * brow + by * dims[2];
            for bz in 0..dims[2] {
                if !hit[bbase + bz] {
                    continue;
                }
                let p = [lo[0] + bx, lo[1] + by, lo[2] + bz];
                let c = [(p[0] / chunk) as isize, (p[1] / chunk) as isize, (p[2] / chunk) as isize];
                set(c);
                for k in 0..3 {
                    if p[k].is_multiple_of(chunk) {
                        let mut n = c;
                        n[k] -= 1;
                        set(n);
                    }
                    if (p[k] + 1).is_multiple_of(chunk) {
                        let mut n = c;
                        n[k] += 1;
                        set(n);
                    }
                }
            }
        }
    }
    let mut dirty = Vec::new();
    for (i, &f) in flags.iter().enumerate() {
        if f {
            dirty.push((i / (counts[1] * counts[2]), i / counts[2] % counts[1], i % counts[2]));
        }
    }
    (count, stopped, dirty)
}

/// The cells of the box `lo` + `dims` within the capsule from `a` to `b`:
/// squared distance to the segment at most `limit` (edit.capsule_mask,
/// A fast cut's floor must be one straight line, not a row of
/// spheres). Flat bool box, x slowest.
pub fn capsule_mask(a: [f64; 3], b: [f64; 3], limit: f64, lo: [i64; 3], dims: [usize; 3]) -> Vec<bool> {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let brow = dims[1] * dims[2];
    let mut out = vec![false; dims[0] * brow];
    out.par_chunks_mut(brow.max(1)).enumerate().for_each(|(bx, slab)| {
        let px = (lo[0] + bx as i64) as f64 - a[0];
        for by in 0..dims[1] {
            let py = (lo[1] + by as i64) as f64 - a[1];
            for bz in 0..dims[2] {
                let pz = (lo[2] + bz as i64) as f64 - a[2];
                let t = if len2 > 0.0 { ((px * ab[0] + py * ab[1] + pz * ab[2]) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let dx = px - t * ab[0];
                let dy = py - t * ab[1];
                let dz = pz - t * ab[2];
                slab[by * dims[2] + bz] = dx * dx + dy * dy + dz * dz <= limit;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_write_changes_only_what_differs_and_names_the_chunks() {
        let shape = [40, 8, 8];
        let mut voxels = vec![0u8; 40 * 64];
        let mut want = voxels.clone();
        want[0] = 1; // cell (0, 0, 0): chunk 0
        want[33 * 64 + 7 * 8 + 7] = 2; // (33, 7, 7): chunk 1
        want[31 * 64] = 3; // (31, 0, 0): on chunk 0's face toward chunk 1
        let (count, stopped, dirty) = apply_grid(&mut voxels, None, &want, None, None, false, shape, 32);
        assert_eq!((count, stopped), (3, 0));
        assert_eq!(voxels, want);
        assert_eq!(dirty, vec![(0, 0, 0), (1, 0, 0)]);
        let lock = (0..voxels.len()).map(|i| i == 0).collect::<Vec<_>>();
        let mut again = vec![0u8; 40 * 64];
        let (count, stopped, _) = apply_grid(&mut again, None, &want, Some(&lock), None, false, shape, 32);
        assert_eq!((count, stopped), (2, 1));
        assert_eq!(again[0], 0);
    }

    #[test]
    fn a_turn_clears_the_selection_and_pastes_its_turned_box() {
        let shape = [8, 8, 1];
        let mut press = vec![0u8; 64];
        let mut mask = vec![false; 64];
        for x in 1..6 {
            press[x * 8 + 4] = 5; // a bar along x at y = 4
            mask[x * 8 + 4] = true;
        }
        press[7 * 8 + 7] = 9; // not selected: stays
        let (want, new_mask) = live_turn(&press, &mask, shape, [1, 4, 0], [5, 4, 0], 2, 90.0, [3, 2, 0]);
        assert_eq!(want[7 * 8 + 7], 9);
        assert_eq!(new_mask.iter().filter(|&&m| m).count(), 5);
        for x in 1..6 {
            assert!(!new_mask[x * 8 + 4] || want[x * 8 + 4] == 5);
        }
        assert_eq!(want.iter().filter(|&&v| v == 5).count(), 5);
    }
}
