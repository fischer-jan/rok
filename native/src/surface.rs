//! The exposed faces of a bool mask, as rok.core.hud.mask_surface builds
//! them: the same quads in the same order, so the Python and the native
//! path draw the same picture (the Box preview and a big
//! selection's skin spent 10 to 15 ms a step here in numpy).

use rayon::prelude::*;

/// Corner offsets of a cell's face, one row per face direction:
/// +X, -X, +Y, -Y, +Z, -Z (rok.core.mesh._FACE_CORNERS).
const FACE_CORNERS: [[[f32; 3]; 4]; 6] = [
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
];
const NORMALS: [[i64; 3]; 6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const QUAD_TRIS: [usize; 6] = [0, 1, 2, 0, 2, 3];
const QUAD_LINES: [usize; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

/// `mask` is a C-order bool array of `shape`. The faces of the cells inside the
/// inclusive box [lo, hi] (clipped to the grid) whose neighbour is empty. With
/// `neighbours`, a cell at the box's edge looks at the mask just outside it;
/// without, the outside counts as empty. Every quad is pushed out by `offset`.
/// Returns (tris: Q * 6 points, lines: Q * 8 points), each point three f32.
pub fn mask_surface(
    mask: &[bool], shape: [usize; 3], lo: [i64; 3], hi: [i64; 3], offset: f32, neighbours: bool,
) -> (Vec<f32>, Vec<f32>) {
    let shape_i = [shape[0] as i64, shape[1] as i64, shape[2] as i64];
    let mut lo_c = [0i64; 3];
    let mut hi_c = [0i64; 3];
    for a in 0..3 {
        lo_c[a] = lo[a].max(0);
        hi_c[a] = hi[a].min(shape_i[a] - 1);
        if hi_c[a] < lo_c[a] {
            return (Vec::new(), Vec::new());
        }
    }
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let at = |x: i64, y: i64, z: i64| mask[x as usize * strides[0] + y as usize * strides[1] + z as usize];
    let filled = |x: i64, y: i64, z: i64| -> bool {
        let inside_box = x >= lo_c[0] && x <= hi_c[0] && y >= lo_c[1] && y <= hi_c[1] && z >= lo_c[2] && z <= hi_c[2];
        if inside_box {
            return at(x, y, z);
        }
        if !neighbours {
            return false;
        }
        let inside_grid = x >= 0 && x < shape_i[0] && y >= 0 && y < shape_i[1] && z >= 0 && z < shape_i[2];
        inside_grid && at(x, y, z)
    };
    let origin = [lo_c[0] as f32, lo_c[1] as f32, lo_c[2] as f32];
    // One face direction per task; the rows join in order, as numpy concatenates them.
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..6usize)
        .into_par_iter()
        .map(|row| {
            let n = NORMALS[row];
            let push = [n[0] as f32 * offset, n[1] as f32 * offset, n[2] as f32 * offset];
            let mut tris: Vec<f32> = Vec::new();
            let mut lines: Vec<f32> = Vec::new();
            for x in lo_c[0]..=hi_c[0] {
                for y in lo_c[1]..=hi_c[1] {
                    for z in lo_c[2]..=hi_c[2] {
                        if !at(x, y, z) || filled(x + n[0], y + n[1], z + n[2]) {
                            continue;
                        }
                        // As numpy: (float32(cell - lo) + float32(lo)) + corner + push.
                        let c = [
                            (x - lo_c[0]) as f32 + origin[0],
                            (y - lo_c[1]) as f32 + origin[1],
                            (z - lo_c[2]) as f32 + origin[2],
                        ];
                        let mut corners = [[0f32; 3]; 4];
                        for (k, corner) in FACE_CORNERS[row].iter().enumerate() {
                            for a in 0..3 {
                                corners[k][a] = (c[a] + corner[a]) + push[a];
                            }
                        }
                        for &k in QUAD_TRIS.iter() {
                            tris.extend_from_slice(&corners[k]);
                        }
                        for &k in QUAD_LINES.iter() {
                            lines.extend_from_slice(&corners[k]);
                        }
                    }
                }
            }
            (tris, lines)
        })
        .collect();
    let mut tris = Vec::with_capacity(rows.iter().map(|r| r.0.len()).sum());
    let mut lines = Vec::with_capacity(rows.iter().map(|r| r.1.len()).sum());
    for (t, l) in rows {
        tris.extend_from_slice(&t);
        lines.extend_from_slice(&l);
    }
    (tris, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_cell_has_six_faces_and_two_touching_cells_ten() {
        let shape = [3, 1, 1];
        let mut mask = vec![false; 3];
        mask[0] = true;
        let (tris, lines) = mask_surface(&mask, shape, [0, 0, 0], [2, 0, 0], 0.0, false);
        assert_eq!(tris.len(), 6 * 6 * 3);
        assert_eq!(lines.len(), 6 * 8 * 3);
        mask[1] = true;
        let (tris, _) = mask_surface(&mask, shape, [0, 0, 0], [2, 0, 0], 0.0, false);
        assert_eq!(tris.len(), 10 * 6 * 3);
        // The window ends after the first cell: with neighbours the shared face is hidden.
        let (tris, _) = mask_surface(&mask, shape, [0, 0, 0], [0, 0, 0], 0.0, true);
        assert_eq!(tris.len(), 5 * 6 * 3);
        let (tris, _) = mask_surface(&mask, shape, [0, 0, 0], [0, 0, 0], 0.0, false);
        assert_eq!(tris.len(), 6 * 6 * 3);
    }
}
