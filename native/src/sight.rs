//! Which cells a camera sees (rok.core.select.seen_cells), in parallel: a
//! cell is seen when one of its open faces that points toward the camera has
//! a clear line from its centre to the camera, sampled every quarter voxel,
//! exactly as the numpy path samples it (the same float64 steps, so the two
//! agree cell for cell). In Python a Marquee pick with "Visible only"
//! on took seconds for a big model.

use rayon::prelude::*;

const STEP: f64 = 0.25;

/// `voxels` is a C-order grid of `shape`; `cells` holds N cells as (x, y, z)
/// triples. `eye` is the camera in grid coordinates (perspective), or None
/// with `toward` the direction to a camera infinitely far (orthographic).
pub fn seen_cells(
    voxels: &[u8], shape: [usize; 3], cells: &[i64], eye: Option<[f64; 3]>, toward: Option<[f64; 3]>,
) -> Vec<bool> {
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let size = [shape[0] as i64, shape[1] as i64, shape[2] as i64];
    let reach = ((shape[0] * shape[0] + shape[1] * shape[1] + shape[2] * shape[2]) as f64).sqrt() + 2.0;
    let steps = (reach / STEP).ceil() as usize; // np.arange(0, reach, STEP)
    let filled = |at: [i64; 3]| -> bool {
        voxels[at[0] as usize * strides[0] + at[1] as usize * strides[1] + at[2] as usize] != 0
    };
    let inside = |at: [i64; 3]| -> bool {
        (0..3).all(|i| at[i] >= 0 && at[i] < size[i])
    };
    // The orthographic direction, once.
    let ortho = toward.map(|d| {
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
        [d[0] / n, d[1] / n, d[2] / n]
    });
    (0..cells.len() / 3)
        .into_par_iter()
        .map(|n| {
            let cell = [cells[3 * n], cells[3 * n + 1], cells[3 * n + 2]];
            let centre = [cell[0] as f64 + 0.5, cell[1] as f64 + 0.5, cell[2] as f64 + 0.5];
            let to_cam = match (eye, toward) {
                (Some(e), _) => [e[0] - centre[0], e[1] - centre[1], e[2] - centre[2]],
                (None, Some(d)) => d,
                (None, None) => return false,
            };
            for axis in 0..3 {
                let sign: i64 = if to_cam[axis] > 0.0 { 1 } else if to_cam[axis] < 0.0 { -1 } else { 0 };
                if sign == 0 {
                    continue;
                }
                let mut nb = cell;
                nb[axis] += sign;
                if inside(nb) && filled(nb) {
                    continue; // the face is covered
                }
                let mut point = centre;
                point[axis] += 0.5 * (sign as f64) * (1.0 + 1e-6);
                let (direction, length) = match eye {
                    Some(e) => {
                        let line = [e[0] - point[0], e[1] - point[1], e[2] - point[2]];
                        let len = (line[0] * line[0] + line[1] * line[1] + line[2] * line[2]).sqrt();
                        let d = len.max(1e-9);
                        ([line[0] / d, line[1] / d, line[2] / d], len)
                    }
                    None => (ortho.unwrap(), f64::INFINITY),
                };
                let mut blocked = false;
                for k in 0..steps {
                    let t = k as f64 * STEP;
                    if t >= length {
                        break;
                    }
                    let at = [
                        (point[0] + direction[0] * t).floor() as i64,
                        (point[1] + direction[1] * t).floor() as i64,
                        (point[2] + direction[2] * t).floor() as i64,
                    ];
                    if !inside(at) {
                        break; // a straight line never comes back into the grid
                    }
                    if filled(at) {
                        blocked = true;
                        break;
                    }
                }
                if !blocked {
                    return true;
                }
            }
            false
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_covered_cell_is_not_seen_from_above_and_an_open_one_is() {
        let shape = [3, 3, 3];
        let mut voxels = vec![0u8; 27];
        voxels[1 * 9 + 1 * 3 + 0] = 1; // (1, 1, 0)
        voxels[1 * 9 + 1 * 3 + 1] = 1; // (1, 1, 1) on top of it
        let seen = seen_cells(&voxels, shape, &[1, 1, 0, 1, 1, 1], None, Some([0.0, 0.0, 1.0]));
        assert_eq!(seen, vec![false, true]);
        let seen = seen_cells(&voxels, shape, &[1, 1, 0, 1, 1, 1], Some([1.5, 1.5, 10.0]), None);
        assert_eq!(seen, vec![false, true]);
    }
}
