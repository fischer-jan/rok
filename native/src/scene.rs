//! CPU side of the scene: models, camera and the pick (same DDA as the shader).

use crate::bake;
use glam::{DMat4, DVec3};

/// A box of voxels the window shows as if they were in the model (the hover's
/// result: what a click would do): `data` in C order, a 0 keeps the model's own voxel. An erase
/// ghost blanks the model's voxel where `data` is set instead. The model's
/// own voxels and bricks stay the truth; only the GPU copy carries the ghost.
#[derive(Clone)]
pub struct Ghost {
    pub origin: [usize; 3],
    pub dims: [usize; 3],
    pub data: Vec<u8>,
    pub erase: bool,
}

impl Ghost {
    pub fn fits(&self, size: [usize; 3]) -> bool {
        (0..3).all(|k| self.origin[k] + self.dims[k] <= size[k]) && self.data.len() == self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// The ghost's value at a model cell, or 0.
    fn at(&self, x: usize, y: usize, z: usize) -> u8 {
        let (o, d) = (self.origin, self.dims);
        if x < o[0] || y < o[1] || z < o[2] || x >= o[0] + d[0] || y >= o[1] + d[1] || z >= o[2] + d[2] {
            return 0;
        }
        self.data[((x - o[0]) * d[1] + (y - o[1])) * d[2] + (z - o[2])]
    }

    fn any_in(&self, x0: usize, x1: usize, y0: usize, y1: usize, z0: usize, z1: usize) -> bool {
        let (o, d) = (self.origin, self.dims);
        let xa = x0.max(o[0]);
        let xb = x1.min(o[0] + d[0]);
        let ya = y0.max(o[1]);
        let yb = y1.min(o[1] + d[1]);
        let za = z0.max(o[2]);
        let zb = z1.min(o[2] + d[2]);
        if xa >= xb || ya >= yb || za >= zb {
            return false;
        }
        for x in xa..xb {
            for y in ya..yb {
                let base = ((x - o[0]) * d[1] + (y - o[1])) * d[2];
                if self.data[base + za - o[2]..base + zb - o[2]].iter().any(|&v| v != 0) {
                    return true;
                }
            }
        }
        false
    }
}

pub struct Model {
    pub size: [usize; 3],
    pub voxels: Vec<u8>,
    pub bricks: Vec<u8>,
    pub bdim: [usize; 3],
    /// Ghost boxes the GPU copy shows on top of the voxels (none most of the time).
    pub ghost: Vec<Ghost>,
    /// World position of the middle of the model's box, in cells.
    pub center: DVec3,
    /// A turn shown during the drag: (axis, degrees). Only the matrix changes.
    pub preview: Option<(usize, f64)>,
    /// 0 = full colour, 1 = fully faded toward the background (the other models inside a model).
    pub fade: f32,
    pub hidden: bool,
    /// A turn in grid units about the rest place (turn_preview): world = T(position) * turn * cell.
    pub turn: Option<DMat4>,
    /// A CAD field is uploaded for this grid. A voxel edit clears it until the next field.
    pub cad: bool,
    /// A brick model's seams byte per cell (C order, like `voxels`): bits 0..5
    /// say "the neighbour on +x, -x, +y, -y, +z, -z is the same brick". None
    /// for a voxel model, which shows no seams.
    pub seams: Option<Vec<u8>>,
}

impl Model {
    pub fn new(voxels: Vec<u8>, size: [usize; 3], center: DVec3) -> Model {
        let (bricks, bdim) = bake::build_bricks(&voxels, size);
        Model { size, voxels, bricks, bdim, ghost: Vec::new(), center, preview: None, fade: 0.0, hidden: false, turn: None,
                cad: false, seams: None }
    }

    /// Copy a box of the seams field (origin and dims in cells, `data` in C
    /// order) into the model; a voxel model gets a field of zeros first.
    /// Returns true when the field is new (the GPU copy needs a new texture).
    pub fn write_seams(&mut self, origin: [usize; 3], dims: [usize; 3], data: &[u8]) -> bool {
        let s = self.size;
        let fresh = self.seams.is_none();
        let field = self.seams.get_or_insert_with(|| vec![0u8; s[0] * s[1] * s[2]]);
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let src = (x * dims[1] + y) * dims[2];
                let dst = ((origin[0] + x) * s[1] + origin[1] + y) * s[2] + origin[2];
                field[dst..dst + dims[2]].copy_from_slice(&data[src..src + dims[2]]);
            }
        }
        fresh
    }

    /// Copy a changed box (origin and dims in cells, `data` in C order) into the
    /// voxels and recompute the bricks it touches. Returns the brick box to upload.
    pub fn write_region(&mut self, origin: [usize; 3], dims: [usize; 3], data: &[u8]) -> ([usize; 3], [usize; 3], Vec<u8>) {
        let s = self.size;
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let src = (x * dims[1] + y) * dims[2];
                let dst = ((origin[0] + x) * s[1] + origin[1] + y) * s[2] + origin[2];
                self.voxels[dst..dst + dims[2]].copy_from_slice(&data[src..src + dims[2]]);
            }
        }
        self.cad = false; // the field was fitted to the old voxels
        self.region_bricks(origin, dims, true)
    }

    /// The box of the model's own voxels at `origin`, in C order.
    pub fn region(&self, origin: [usize; 3], dims: [usize; 3]) -> Vec<u8> {
        let s = self.size;
        let mut out = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let src = ((origin[0] + x) * s[1] + origin[1] + y) * s[2] + origin[2];
                out.extend_from_slice(&self.voxels[src..src + dims[2]]);
            }
        }
        out
    }

    /// What the GPU shows for a box: the model's own voxels with the ghosts on top.
    pub fn shown_region(&self, origin: [usize; 3], dims: [usize; 3]) -> Vec<u8> {
        let mut out = self.region(origin, dims);
        for g in &self.ghost {
            if !g.any_in(origin[0], origin[0] + dims[0], origin[1], origin[1] + dims[1], origin[2], origin[2] + dims[2]) {
                continue;
            }
            for x in 0..dims[0] {
                for y in 0..dims[1] {
                    for z in 0..dims[2] {
                        let v = g.at(origin[0] + x, origin[1] + y, origin[2] + z);
                        if v != 0 {
                            out[(x * dims[1] + y) * dims[2] + z] = if g.erase { 0 } else { v };
                        }
                    }
                }
            }
        }
        out
    }

    /// The bricks that touch a box, from the model's own voxels (stored when
    /// `store`), with the ghosts' cells counted in the copy returned for the GPU.
    pub fn region_bricks(&mut self, origin: [usize; 3], dims: [usize; 3], store: bool) -> ([usize; 3], [usize; 3], Vec<u8>) {
        let s = self.size;
        let b = bake::BRICK;
        let lo = [origin[0] / b, origin[1] / b, origin[2] / b];
        let hi = [
            (origin[0] + dims[0] - 1) / b + 1,
            (origin[1] + dims[1] - 1) / b + 1,
            (origin[2] + dims[2] - 1) / b + 1,
        ];
        let bd = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut out = Vec::with_capacity(bd[0] * bd[1] * bd[2]);
        for bx in lo[0]..hi[0] {
            for by in lo[1]..hi[1] {
                for bz in lo[2]..hi[2] {
                    let mut any = false;
                    'scan: for x in bx * b..((bx + 1) * b).min(s[0]) {
                        for y in by * b..((by + 1) * b).min(s[1]) {
                            let base = (x * s[1] + y) * s[2];
                            if self.voxels[base + bz * b..base + ((bz + 1) * b).min(s[2])].iter().any(|&v| v != 0) {
                                any = true;
                                break 'scan;
                            }
                        }
                    }
                    let flat = (bx * self.bdim[1] + by) * self.bdim[2] + bz;
                    if store {
                        self.bricks[flat] = any as u8;
                    }
                    let ghost = !any
                        && self.ghost.iter().any(|g| {
                            !g.erase && g.any_in(bx * b, ((bx + 1) * b).min(s[0]), by * b, ((by + 1) * b).min(s[1]), bz * b, ((bz + 1) * b).min(s[2]))
                        });
                    out.push((any || ghost) as u8);
                }
            }
        }
        (lo, bd, out)
    }

    /// Swap the ghosts for new ones. Returns the boxes whose GPU copy must be
    /// written again: the old ghosts' boxes (back to the model's own voxels)
    /// and the new ones. Boxes outside the model are dropped.
    pub fn set_ghost(&mut self, ghost: Vec<Ghost>) -> Vec<([usize; 3], [usize; 3])> {
        let size = self.size;
        let mut boxes: Vec<([usize; 3], [usize; 3])> = self.ghost.iter().map(|g| (g.origin, g.dims)).collect();
        self.ghost = ghost.into_iter().filter(|g| g.fits(size)).collect();
        boxes.extend(self.ghost.iter().map(|g| (g.origin, g.dims)));
        boxes
    }

    pub fn half(&self) -> DVec3 {
        DVec3::new(self.size[0] as f64, self.size[1] as f64, self.size[2] as f64) * 0.5
    }

    /// Model (cell units, box from 0 to size) to world.
    pub fn matrix(&self) -> DMat4 {
        if let Some(turn) = self.turn {
            return DMat4::from_translation(self.center - self.half()) * turn;
        }
        let mut m = DMat4::from_translation(self.center);
        if let Some((axis, deg)) = self.preview {
            let a = match axis {
                0 => DVec3::X,
                1 => DVec3::Y,
                _ => DVec3::Z,
            };
            m *= DMat4::from_axis_angle(a, deg.to_radians());
        }
        m * DMat4::from_translation(-self.half())
    }

    pub fn step_cap(&self, margin: usize) -> u32 {
        (self.size[0] + self.size[1] + self.size[2] + margin) as u32
    }
}
