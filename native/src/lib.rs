mod bake;
mod gpu;
mod host;
mod overlay;
mod postfx;
mod xs;
mod scene;
mod trace;
mod surface;
mod live;
mod sight;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Turn a voxel array; returns the new (x, y, z) uint8 array.
#[pyfunction]
fn bake_rotate<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray3<'py, u8>,
    axis: usize,
    degrees: f64,
) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    let shape = voxels.shape();
    let size = [shape[0], shape[1], shape[2]];
    let data: Vec<u8> = voxels.as_array().iter().copied().collect();
    let (out, new) = py.detach(|| bake::rotate_voxels(&data, size, axis, degrees));
    PyArray1::from_vec(py, out).reshape([new[0], new[1], new[2]])
}

/// The exposed faces of a bool mask inside the inclusive box [lo, hi], as
/// rok.core.hud.mask_surface builds them: (tris (Q * 6, 3), lines (Q * 8, 3)) float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mask_surface<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray3<'py, bool>,
    lo: (i64, i64, i64),
    hi: (i64, i64, i64),
    offset: f32,
    neighbours: bool,
) -> PyResult<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, numpy::PyArray2<f32>>)> {
    let shape = mask.shape();
    let size = [shape[0], shape[1], shape[2]];
    let view = mask.as_array();
    let owned;
    let data: &[bool] = match view.as_slice() {
        Some(s) => s,
        None => {
            owned = view.iter().copied().collect::<Vec<bool>>();
            &owned
        }
    };
    let (tris, lines) = py.detach(|| {
        surface::mask_surface(data, size, [lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2], offset, neighbours)
    });
    let nt = tris.len() / 3;
    let nl = lines.len() / 3;
    Ok((
        PyArray1::from_vec(py, tris).reshape([nt, 3])?,
        PyArray1::from_vec(py, lines).reshape([nl, 3])?,
    ))
}

/// mask_surface for many boxes of one mask at once, in parallel: a skin's
/// dirty blocks (a live turn rebuilt 27 blocks a step, one
/// call and one copy of the mask each). `boxes` holds (lo, hi) pairs; the
/// result holds (tris, lines) in the same order.
#[pyfunction]
fn mask_surface_blocks<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray3<'py, bool>,
    boxes: Vec<((i64, i64, i64), (i64, i64, i64))>,
    offset: f32,
    neighbours: bool,
) -> PyResult<Vec<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, numpy::PyArray2<f32>>)>> {
    use rayon::prelude::*;
    let shape = mask.shape();
    let size = [shape[0], shape[1], shape[2]];
    let view = mask.as_array();
    let owned;
    let data: &[bool] = match view.as_slice() {
        Some(s) => s,
        None => {
            owned = view.iter().copied().collect::<Vec<bool>>();
            &owned
        }
    };
    let built: Vec<(Vec<f32>, Vec<f32>)> = py.detach(|| {
        boxes
            .par_iter()
            .map(|(lo, hi)| surface::mask_surface(data, size, [lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2], offset, neighbours))
            .collect()
    });
    let mut out = Vec::with_capacity(built.len());
    for (tris, lines) in built {
        let nt = tris.len() / 3;
        let nl = lines.len() / 3;
        out.push((
            PyArray1::from_vec(py, tris).reshape([nt, 3])?,
            PyArray1::from_vec(py, lines).reshape([nl, 3])?,
        ));
    }
    Ok(out)
}

fn slice3<'a, T: numpy::Element>(arr: &'a numpy::PyReadonlyArray3<'a, T>) -> PyResult<&'a [T]> {
    arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("the array must be C-contiguous"))
}

/// The grid of a selection's live turn at one angle, and its new selection
/// mask (live.rs): (want (x, y, z) uint8, mask (x, y, z) bool).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn live_turn<'py>(
    py: Python<'py>,
    press: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray3<'py, bool>,
    lo: (usize, usize, usize),
    hi: (usize, usize, usize),
    axis: usize,
    degrees: f64,
    position: (i64, i64, i64),
) -> PyResult<(Bound<'py, numpy::PyArray3<u8>>, Bound<'py, numpy::PyArray3<bool>>)> {
    let shape = press.shape();
    let size = [shape[0], shape[1], shape[2]];
    let p = slice3(&press)?;
    let m = slice3(&mask)?;
    let (want, new_mask) = py.detach(|| {
        live::live_turn(p, m, size, [lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2], axis, degrees,
                        [position.0, position.1, position.2])
    });
    Ok((
        PyArray1::from_vec(py, want).reshape(size)?,
        PyArray1::from_vec(py, new_mask).reshape(size)?,
    ))
}

/// The grid of a cage deformation (live.rs `live_deform` or, with `fast`, `live_deform_fast`;
/// rok/core/lattice.py `deform_grid`): (want (x, y, z) uint8, mask (x, y, z) bool). `boxes`
/// is (K, 6) int64, inclusive lo xyz and hi xyz of the cells to visit; `points` is
/// (nx, ny, nz, 3) float64; `bernstein` picks one full-degree span per axis.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn deform<'py>(
    py: Python<'py>,
    fast: bool,
    rest: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray3<'py, bool>,
    lo: (i64, i64, i64),
    size: (i64, i64, i64),
    counts: (usize, usize, usize),
    points: numpy::PyReadonlyArray4<'py, f64>,
    boxes: PyReadonlyArray2<'py, i64>,
    bernstein: bool,
) -> PyResult<(Bound<'py, numpy::PyArray3<u8>>, Bound<'py, numpy::PyArray3<bool>>)> {
    let shape = rest.shape();
    let dims = [shape[0], shape[1], shape[2]];
    let r = slice3(&rest)?;
    let m = slice3(&mask)?;
    let bad = |what: &str| pyo3::exceptions::PyValueError::new_err(what.to_string());
    if counts.0 < 2 || counts.1 < 2 || counts.2 < 2 {
        return Err(bad("the cage needs at least two points per axis"));
    }
    if counts.0 > 6 || counts.1 > 6 || counts.2 > 6 {
        return Err(bad("the cage takes at most six points per axis"));
    }
    if points.shape() != [counts.0, counts.1, counts.2, 3] {
        return Err(bad("points must be (nx, ny, nz, 3)"));
    }
    if boxes.shape().len() != 2 || boxes.shape()[1] != 6 {
        return Err(bad("boxes must be (K, 6)"));
    }
    let pts = points.as_slice().map_err(|_| bad("the points must be C-contiguous"))?;
    let bx = boxes.as_slice().map_err(|_| bad("the boxes must be C-contiguous"))?;
    let boxes: Vec<[i64; 6]> = bx.chunks_exact(6).map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]]).collect();
    if boxes.iter().any(|b| b[0] < 0 || b[1] < 0 || b[2] < 0 || b[3] >= dims[0] as i64
        || b[4] >= dims[1] as i64 || b[5] >= dims[2] as i64) {
        return Err(bad("a box reaches outside the grid"));
    }
    let (want, new_mask) = py.detach(|| {
        let (lo, size, counts) = ([lo.0, lo.1, lo.2], [size.0, size.1, size.2], [counts.0, counts.1, counts.2]);
        if fast {
            live::live_deform_fast(r, m, dims, lo, size, counts, pts, &boxes, bernstein)
        } else {
            live::live_deform(r, m, dims, lo, size, counts, pts, &boxes, bernstein)
        }
    });
    Ok((
        PyArray1::from_vec(py, want).reshape(dims)?,
        PyArray1::from_vec(py, new_mask).reshape(dims)?,
    ))
}

/// The exact cage deformation: four Newton steps in every target cell (live.rs `live_deform`).
#[pyfunction]
#[pyo3(signature = (rest, mask, lo, size, counts, points, boxes, bernstein=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn live_deform<'py>(
    py: Python<'py>,
    rest: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray3<'py, bool>,
    lo: (i64, i64, i64),
    size: (i64, i64, i64),
    counts: (usize, usize, usize),
    points: numpy::PyReadonlyArray4<'py, f64>,
    boxes: PyReadonlyArray2<'py, i64>,
    bernstein: bool,
) -> PyResult<(Bound<'py, numpy::PyArray3<u8>>, Bound<'py, numpy::PyArray3<bool>>)> {
    deform(py, false, rest, mask, lo, size, counts, points, boxes, bernstein)
}

/// The cage deformation with the inverse by checked interpolation (live.rs `live_deform_fast`).
#[pyfunction]
#[pyo3(signature = (rest, mask, lo, size, counts, points, boxes, bernstein=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn live_deform_fast<'py>(
    py: Python<'py>,
    rest: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray3<'py, bool>,
    lo: (i64, i64, i64),
    size: (i64, i64, i64),
    counts: (usize, usize, usize),
    points: numpy::PyReadonlyArray4<'py, f64>,
    boxes: PyReadonlyArray2<'py, i64>,
    bernstein: bool,
) -> PyResult<(Bound<'py, numpy::PyArray3<u8>>, Bound<'py, numpy::PyArray3<bool>>)> {
    deform(py, true, rest, mask, lo, size, counts, points, boxes, bernstein)
}

/// The chunks a write of `want` into `voxels` would mark dirty (grid.changed_chunks).
#[pyfunction]
fn changed_chunks<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray3<'py, u8>,
    want: PyReadonlyArray3<'py, u8>,
    lock: Option<PyReadonlyArray3<'py, bool>>,
    cut: Option<PyReadonlyArray3<'py, bool>>,
    ignore_cut: bool,
    chunk: usize,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let shape = voxels.shape();
    let size = [shape[0], shape[1], shape[2]];
    let v = slice3(&voxels)?;
    let w = slice3(&want)?;
    let l = match &lock { Some(a) => Some(slice3(a)?), None => None };
    let c = match &cut { Some(a) => Some(slice3(a)?), None => None };
    Ok(py.detach(|| live::changed_chunks(v, w, l, c, ignore_cut, size, chunk)))
}

/// Write `want` into `voxels` and the visible copy, as grid.set_grid does:
/// (changed count, writes the lock stopped, dirty chunks).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn apply_grid<'py>(
    py: Python<'py>,
    mut voxels: numpy::PyReadwriteArray3<'py, u8>,
    visible: Option<numpy::PyReadwriteArray3<'py, u8>>,
    want: PyReadonlyArray3<'py, u8>,
    lock: Option<PyReadonlyArray3<'py, bool>>,
    cut: Option<PyReadonlyArray3<'py, bool>>,
    ignore_cut: bool,
    chunk: usize,
) -> PyResult<(usize, usize, Vec<(usize, usize, usize)>)> {
    let shape = voxels.shape();
    let size = [shape[0], shape[1], shape[2]];
    let err = || pyo3::exceptions::PyValueError::new_err("the array must be C-contiguous");
    let w = slice3(&want)?;
    let l = match &lock { Some(a) => Some(slice3(a)?), None => None };
    let c = match &cut { Some(a) => Some(slice3(a)?), None => None };
    let v = voxels.as_slice_mut().map_err(|_| err())?;
    let mut visible = visible;
    let vis = match visible.as_mut() {
        Some(a) => Some(a.as_slice_mut().map_err(|_| err())?),
        None => None,
    };
    Ok(py.detach(|| live::apply_grid(v, vis, w, l, c, ignore_cut, size, chunk)))
}

/// The brush's box write (grid.write_brush): (changed count, writes the lock
/// stopped, dirty chunks). `mask` and `cut` are box-sized (the dims of `mask`),
/// `lock` and `mark` grid-sized.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn brush_write<'py>(
    py: Python<'py>,
    mut voxels: numpy::PyReadwriteArray3<'py, u8>,
    visible: Option<numpy::PyReadwriteArray3<'py, u8>>,
    lock: Option<PyReadonlyArray3<'py, bool>>,
    cut: Option<PyReadonlyArray3<'py, bool>>,
    lo: (usize, usize, usize),
    mask: PyReadonlyArray3<'py, bool>,
    mode: u8,
    colour: u8,
    mark: Option<numpy::PyReadwriteArray3<'py, bool>>,
    chunk: usize,
) -> PyResult<(usize, usize, Vec<(usize, usize, usize)>)> {
    let shape = voxels.shape();
    let size = [shape[0], shape[1], shape[2]];
    let d = mask.shape();
    let dims = [d[0], d[1], d[2]];
    let lo = [lo.0, lo.1, lo.2];
    if (0..3).any(|k| lo[k] + dims[k] > size[k]) {
        return Err(pyo3::exceptions::PyValueError::new_err("the brush box reaches outside the grid"));
    }
    if cut.as_ref().is_some_and(|c| c.shape() != d) {
        return Err(pyo3::exceptions::PyValueError::new_err("cut must have the mask's shape"));
    }
    let err = || pyo3::exceptions::PyValueError::new_err("the array must be C-contiguous");
    let m = slice3(&mask)?;
    let l = match &lock { Some(a) => Some(slice3(a)?), None => None };
    let c = match &cut { Some(a) => Some(slice3(a)?), None => None };
    let v = voxels.as_slice_mut().map_err(|_| err())?;
    let mut visible = visible;
    let vis = match visible.as_mut() {
        Some(a) => Some(a.as_slice_mut().map_err(|_| err())?),
        None => None,
    };
    let mut mark = mark;
    let mk = match mark.as_mut() {
        Some(a) => Some(a.as_slice_mut().map_err(|_| err())?),
        None => None,
    };
    Ok(py.detach(|| live::brush_write(v, vis, l, c, lo, dims, m, mode, colour, mk, size, chunk)))
}

/// The capsule from `a` to `b` in the box `lo` + `dims` (edit.capsule_mask):
/// a flat bool array, x slowest; the caller reshapes it to `dims`.
#[pyfunction]
fn capsule_mask<'py>(
    py: Python<'py>,
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    limit: f64,
    lo: (i64, i64, i64),
    dims: (usize, usize, usize),
) -> Bound<'py, PyArray1<bool>> {
    let out = py.detach(|| {
        live::capsule_mask([a.0, a.1, a.2], [b.0, b.1, b.2], limit, [lo.0, lo.1, lo.2], [dims.0, dims.1, dims.2])
    });
    PyArray1::from_vec(py, out)
}

/// (256,) int64: how many cells hold each palette index (scene.used_counts).
#[pyfunction]
fn count_colours<'py>(py: Python<'py>, voxels: PyReadonlyArray3<'py, u8>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let v = slice3(&voxels)?;
    let bins = py.detach(|| live::count_colours(v));
    Ok(PyArray1::from_vec(py, bins.to_vec()))
}

/// Which of `cells` the camera sees (select.seen_cells): (N,) bool.
#[pyfunction]
fn seen_cells<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray3<'py, u8>,
    cells: numpy::PyReadonlyArray2<'py, i64>,
    eye: Option<(f64, f64, f64)>,
    toward: Option<(f64, f64, f64)>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let shape = voxels.shape();
    let size = [shape[0], shape[1], shape[2]];
    let v = slice3(&voxels)?;
    let c = cells.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("cells must be C-contiguous"))?;
    let e = eye.map(|e| [e.0, e.1, e.2]);
    let t = toward.map(|t| [t.0, t.1, t.2]);
    let out = py.detach(|| sight::seen_cells(v, size, c, e, t));
    Ok(PyArray1::from_vec(py, out))
}

/// Decode PNG bytes to (height, width, RGBA (h, w, 4) uint8): every colour type
/// and depth, 16-bit samples keep their high byte (as rok.core.png does).
/// Raises ValueError for a broken file or one over `max_pixels`.
#[pyfunction]
fn decode_png<'py>(py: Python<'py>, data: &[u8], max_pixels: usize) -> PyResult<(usize, usize, Bound<'py, numpy::PyArray3<u8>>)> {
    let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
    let owned = data.to_vec();
    let (w, h, rgba) = py
        .detach(move || -> Result<(usize, usize, Vec<u8>), String> {
            let mut decoder = png::Decoder::new(std::io::Cursor::new(owned));
            decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
            let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
            let (w, h) = (reader.info().width as usize, reader.info().height as usize);
            if w * h > max_pixels {
                return Err("PNG too large".into());
            }
            let size = reader.output_buffer_size().ok_or("PNG too large")?;
            let mut buf = vec![0u8; size];
            let frame = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
            let (ct, _depth) = reader.output_color_type();
            let px = &buf[..frame.buffer_size()];
            let mut out = Vec::with_capacity(w * h * 4);
            match ct {
                png::ColorType::Rgba => out.extend_from_slice(px),
                png::ColorType::Rgb => px.chunks_exact(3).for_each(|c| out.extend_from_slice(&[c[0], c[1], c[2], 255])),
                png::ColorType::GrayscaleAlpha => px.chunks_exact(2).for_each(|c| out.extend_from_slice(&[c[0], c[0], c[0], c[1]])),
                png::ColorType::Grayscale => px.iter().for_each(|&g| out.extend_from_slice(&[g, g, g, 255])),
                png::ColorType::Indexed => return Err("palette not expanded".into()),
            }
            Ok((w, h, out))
        })
        .map_err(bad)?;
    let arr = PyArray1::from_vec(py, rgba).reshape([h, w, 4])?;
    Ok((h, w, arr))
}

#[pymodule]
fn rok_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bake_rotate, m)?)?;
    m.add_function(wrap_pyfunction!(mask_surface, m)?)?;
    m.add_function(wrap_pyfunction!(mask_surface_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(live_turn, m)?)?;
    m.add_function(wrap_pyfunction!(live_deform, m)?)?;
    m.add_function(wrap_pyfunction!(live_deform_fast, m)?)?;
    m.add_function(wrap_pyfunction!(changed_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(apply_grid, m)?)?;
    m.add_function(wrap_pyfunction!(brush_write, m)?)?;
    m.add_function(wrap_pyfunction!(count_colours, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_mask, m)?)?;
    m.add_function(wrap_pyfunction!(seen_cells, m)?)?;
    m.add_function(wrap_pyfunction!(decode_png, m)?)?;
    m.add_class::<host::Host>()?;
    m.add_class::<host::Waker>()?;
    m.add_class::<xs::ShaderRunner>()?;
    Ok(())
}
