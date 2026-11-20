//! The bake: a nearest-neighbour inverse lookup that matches
//! `rok.core.free_rotate` cell for cell. The new grid is as big as the turned
//! box. Every new cell takes the colour of the old cell found through the
//! inverse turn about the centre.

use rayon::prelude::*;

pub const STEP: f64 = 0.1;

/// `degrees` rounded to tenths and wrapped into (-180, 180].
pub fn snap(degrees: f64) -> f64 {
    let mut k = (degrees / STEP).round_ties_even() as i64;
    k = (k + 1800).rem_euclid(3600) - 1800;
    if k <= -1800 {
        k += 3600;
    }
    k as f64 / 10.0
}

/// Number of quarter turns when `degrees` is a multiple of 90, else None.
pub fn quarter_turns(degrees: f64) -> Option<i64> {
    let d = snap(degrees);
    let q = d / 90.0;
    if (q - q.round_ties_even()).abs() < 1e-9 {
        Some((q.round_ties_even() as i64).rem_euclid(4))
    } else {
        None
    }
}

/// The size of the grid that holds `size` turned by `degrees` about `axis`.
pub fn turned_size(size: [usize; 3], axis: usize, degrees: f64) -> [usize; 3] {
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    let mut out = size;
    if let Some(q) = quarter_turns(degrees) {
        if q % 2 == 1 {
            out.swap(a, b);
        }
        return out;
    }
    let rad = degrees.to_radians();
    let c = rad.cos().abs();
    let s = rad.sin().abs();
    let (ga, gb) = (size[a] as f64, size[b] as f64);
    out[a] = ((c * ga + s * gb - 1e-6).ceil() as usize).max(1);
    out[b] = ((s * ga + c * gb - 1e-6).ceil() as usize).max(1);
    out
}

/// Turn `voxels` (x, y, z, C order) by `degrees` about `axis`.
/// Returns the new voxels and their size.
pub fn rotate_voxels(voxels: &[u8], size: [usize; 3], axis: usize, degrees: f64) -> (Vec<u8>, [usize; 3]) {
    let degrees = snap(degrees);
    let new = turned_size(size, axis, degrees);
    let (a, b, c) = ((axis + 1) % 3, (axis + 2) % 3, axis);
    let (sa, sb) = (size[a], size[b]);
    let (na, nb) = (new[a], new[b]);
    let rad = degrees.to_radians();
    let (cs, sn) = (rad.cos(), rad.sin());
    let st = [size[1] * size[2], size[2], 1usize];
    let os = [new[1] * new[2], new[2], 1usize];

    // Source offset for every (ia, ib) of the turning plane; -1 outside.
    let mut map = vec![-1i64; na * nb];
    map.par_chunks_mut(nb).enumerate().for_each(|(ia, row)| {
        let ii = ia as f64 - (na as f64 - 1.0) / 2.0;
        for (ib, slot) in row.iter_mut().enumerate() {
            let jj = ib as f64 - (nb as f64 - 1.0) / 2.0;
            let si = (cs * ii + sn * jj + (sa as f64 - 1.0) / 2.0).round_ties_even() as i64;
            let sj = (-sn * ii + cs * jj + (sb as f64 - 1.0) / 2.0).round_ties_even() as i64;
            if si >= 0 && (si as usize) < sa && sj >= 0 && (sj as usize) < sb {
                *slot = si * st[a] as i64 + sj * st[b] as i64;
            }
        }
    });

    let mut out = vec![0u8; new[0] * new[1] * new[2]];
    let nc = new[c];
    out.par_chunks_mut(os[0]).enumerate().for_each(|(o0, slab)| {
        for o1 in 0..new[1] {
            for o2 in 0..new[2] {
                let pos = [o0, o1, o2];
                let off = map[pos[a] * nb + pos[b]];
                if off >= 0 {
                    let ic = pos[c];
                    debug_assert!(ic < nc);
                    slab[o1 * os[1] + o2] = voxels[off as usize + ic * st[c]];
                }
            }
        }
    });
    (out, new)
}

/// 8^3 brick occupancy: 1 where the brick holds any voxel.
pub const BRICK: usize = 8;

pub fn brick_dims(size: [usize; 3]) -> [usize; 3] {
    [size[0].div_ceil(BRICK), size[1].div_ceil(BRICK), size[2].div_ceil(BRICK)]
}

pub fn build_bricks(voxels: &[u8], size: [usize; 3]) -> (Vec<u8>, [usize; 3]) {
    let bd = brick_dims(size);
    let mut out = vec![0u8; bd[0] * bd[1] * bd[2]];
    out.par_chunks_mut(bd[1] * bd[2]).enumerate().for_each(|(bx, slab)| {
        for by in 0..bd[1] {
            for bz in 0..bd[2] {
                let mut any = false;
                'scan: for x in bx * BRICK..((bx + 1) * BRICK).min(size[0]) {
                    for y in by * BRICK..((by + 1) * BRICK).min(size[1]) {
                        let base = (x * size[1] + y) * size[2];
                        let z0 = bz * BRICK;
                        let z1 = ((bz + 1) * BRICK).min(size[2]);
                        if voxels[base + z0..base + z1].iter().any(|&v| v != 0) {
                            any = true;
                            break 'scan;
                        }
                    }
                }
                slab[by * bd[2] + bz] = any as u8;
            }
        }
    });
    (out, bd)
}
