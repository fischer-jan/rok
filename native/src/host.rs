//! The app host: Rust owns the window, the event loop and the GPU; Python drives.
//!
//! Every event goes to one Python callback as `callback(kind, data)`. During
//! the callback Python calls `Host` methods, which only queue commands. The
//! commands are applied after the callback returns, so Python never runs
//! inside a Rust borrow and Rust never calls Python from inside a command.
//! Positions are logical pixels with the origin at the top left; Rust converts
//! to physical pixels once, with the window's scale factor.

use crate::gpu::{model_uniform, FrameU, Gpu, Renderer, ViewFrame};
use crate::overlay::{self, DrawList, IconRun, Overlay, TextRun};
use crate::postfx::PostFx;
use crate::scene::{Ghost, Model};
use crate::trace::{Lamp, Settings as TraceSettings, Tracer, MAT_COLS, MAX_LAMPS};
use glam::{DMat4, DVec3, DVec4};
use glyphon::FontSystem;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, PhysicalKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{CursorGrabMode, Window, WindowId};

/// One view of the quad view: its rectangle (logical pixels, origin top left),
/// its camera, and its pixel size in world units when orthographic (else 0).
#[derive(Clone, Copy)]
struct HostView {
    rect: [f64; 4],
    view_proj: DMat4,
    eye: DVec3,
    pixel: f64,
    // Agent pictures: the depth range of the scene along the view (near, length);
    // farther surfaces get darker. (0, 0) for none.
    cue: [f64; 2],
    // Agent pictures: a light, flat background of this linear grey (0: the editor's dark one).
    bg: f64,
}

enum Cmd {
    Relative(bool),
    Palette(Vec<u8>),
    SetModel { idx: usize, voxels: Vec<u8>, size: [usize; 3], position: DVec3 },
    ModelPosition { idx: usize, position: DVec3 },
    RemoveModel(usize),
    ModelStyle { idx: usize, fade: f32, hidden: bool },
    ModelTurn { idx: usize, turn: Option<DMat4> },
    Region { idx: usize, origin: [usize; 3], dims: [usize; 3], data: Vec<u8> },
    Ghost { idx: usize, boxes: Vec<Ghost> },
    Seams { idx: usize, origin: [usize; 3], dims: [usize; 3], data: Vec<u8> },
    ClearSeams(usize),
    Camera { view_proj: DMat4, eye: DVec3 },
    Occlusion([f32; 2]),
    Views(Vec<HostView>),
    Draw { layer: String, z: i32, list: DrawList },
    DrawOffset { layer: String, offset: [f32; 3] },
    ClearLayer(String),
    Image { name: String, w: u32, h: u32, data: Vec<u8> },
    Redraw,
    Tick(f64),
    Menu(Vec<(String, Vec<Option<MenuEntry>>)>),
    Dialog { token: String, save: bool, title: String, filters: Vec<(String, Vec<String>)>, name: Option<String>, directory: Option<String> },
    Ask { token: String, title: String, text: String, buttons: Vec<String> },
    Title(String),
    Edited(bool),
    Screenshot(String),
    Shot { token: String, w: u32, h: u32, views: Vec<HostView>, keep_look: bool },
    Smooth(bool),
    Look([[f32; 4]; 3]),
    Cad(bool),
    CadHidden(bool),
    CadField { idx: usize, data: Vec<u8>, size: [usize; 3] },
    Grid(bool),
    Render(TraceSettings),
    Picture { token: String, w: u32, h: u32, view_proj: DMat4, eye: DVec3, settings: TraceSettings },
    Materials(Vec<f32>),
    Warp(f64, f64),
    Cursor(String),
    Quit,
}

#[derive(Default)]
struct Shared {
    cmds: Vec<Cmd>,
}

enum Val {
    F(f64),
    B(bool),
    S(String),
    Bytes(Vec<u8>),
    None,
}

type Event = (&'static str, Vec<(&'static str, Val)>);

#[derive(Debug)]
enum UserEv {
    Menu(String),
    Wake,
}

/// Wakes the event loop from another thread: the app then gets a "wake"
/// event. The one thing of the host that other threads may use.
#[pyclass]
pub struct Waker {
    proxy: Arc<Mutex<EventLoopProxy<UserEv>>>,
}

#[pymethods]
impl Waker {
    fn wake(&self) -> bool {
        match self.proxy.lock() {
            Ok(p) => p.send_event(UserEv::Wake).is_ok(),
            Err(_) => false,
        }
    }
}

/// Carries a value into `Python::detach` on the same thread.
struct AssertSend<T>(T);
// SAFETY: `detach` runs the closure on the calling thread; nothing crosses threads.
unsafe impl<T> Send for AssertSend<T> {}

/// The Python side of the host.
#[pyclass(unsendable)]
pub struct Host {
    shared: Rc<RefCell<Shared>>,
    fonts: Rc<RefCell<FontSystem>>,
    title: String,
    icon: Option<String>,
    // The app of a headless run (start_headless): no window, Python drives it.
    headless: RefCell<Option<HostApp>>,
    // The event loop's proxy while `run` runs, for `waker()`.
    proxy: RefCell<Option<Arc<Mutex<EventLoopProxy<UserEv>>>>>,
}

impl Host {
    fn push(&self, cmd: Cmd) {
        self.shared.borrow_mut().cmds.push(cmd);
    }

    fn with_headless<R: Default>(&self, f: impl FnOnce(&mut HostApp) -> R) -> PyResult<R> {
        let Ok(mut guard) = self.headless.try_borrow_mut() else {
            return Ok(R::default()); // called from inside the callback
        };
        match guard.as_mut() {
            Some(app) => Ok(f(app)),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err("call start_headless first")),
        }
    }
}

fn position(p: (f64, f64, f64)) -> DVec3 {
    DVec3::new(p.0, p.1, p.2)
}

/// A 4 x 4 numpy matrix (rows) as a DMat4.
fn mat4_of(m: &PyReadonlyArray2<f64>) -> PyResult<DMat4> {
    if m.shape() != [4, 4] {
        return Err(pyo3::exceptions::PyValueError::new_err("view_proj must be 4 x 4"));
    }
    let rows: Vec<f64> = m.as_array().iter().copied().collect();
    Ok(DMat4::from_cols_array(&rows.try_into().unwrap()).transpose())
}

#[allow(clippy::too_many_arguments)]
fn trace_settings(
    on: bool,
    max_samples: u32,
    bounces: u32,
    lamps: Option<Vec<(u32, f32, f32, f32, f32, f32, f32, f32)>>,
    sky_top: (f32, f32, f32),
    sky_horizon: (f32, f32, f32),
    backdrop: Option<(f32, f32, f32)>,
    ground: Option<(f32, f32, f32)>,
    exposure: f32,
    tone: u32,
    bloom: f32,
    outline: f32,
    voxel_grid: f32,
    emission: f32,
    transparent: bool,
) -> TraceSettings {
        let mut st = TraceSettings::default();
        if let Some(list) = lamps {
            st.lamp_count = list.len().min(MAX_LAMPS) as u32;
            for (k, l) in list.into_iter().take(MAX_LAMPS).enumerate() {
                st.lamps[k] = Lamp { kind: l.0, v: [l.1, l.2, l.3], colour: [l.4, l.5, l.6], size: l.7 };
            }
        }
        let rgb = |c: (f32, f32, f32)| [c.0, c.1, c.2];
        st.on = on;
        st.max_samples = max_samples.max(1);
        st.bounces = bounces;
        st.sky_top = rgb(sky_top);
        st.sky_horizon = rgb(sky_horizon);
        st.backdrop = backdrop.map(rgb);
        st.ground = ground.map(rgb);
        st.exposure = exposure;
        st.tone = tone;
        st.bloom = bloom.max(0.0);
        st.outline = outline.clamp(0.0, 3.0);
        st.voxel_grid = voxel_grid.clamp(0.0, 2.0);
        st.emission = emission;
        st.transparent = transparent;
    st
}

#[pymethods]
impl Host {
    #[new]
    #[pyo3(signature = (title="Rok", icon=None))]
    fn new(title: &str, icon: Option<String>) -> Host {
        Host {
            shared: Rc::new(RefCell::new(Shared::default())),
            fonts: Rc::new(RefCell::new(FontSystem::new())),
            title: title.to_string(),
            icon,
            headless: RefCell::new(None),
            proxy: RefCell::new(None),
        }
    }

    /// A `Waker` for another thread; None before `run` has started (and in a
    /// headless run, which Python drives itself).
    fn waker(&self) -> Option<Waker> {
        self.proxy.borrow().as_ref().map(|p| Waker { proxy: p.clone() })
    }

    /// Start without a window: the GPU renders off-screen at `width` x
    /// `height` logical pixels. Python drives: it calls the app itself, then
    /// `process()` to apply the queued commands and deliver the events they
    /// cause (a "screenshot" event, a cancelled dialog). The "start" event
    /// goes to `callback` before this returns.
    #[pyo3(signature = (callback, width=1000.0, height=1000.0, scale=1.0))]
    fn start_headless(&self, callback: Py<PyAny>, width: f64, height: f64, scale: f64) -> PyResult<()> {
        if self.headless.borrow().is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("already started"));
        }
        let mut app = HostApp::new(self.shared.clone(), self.fonts.clone(), callback, self.title.clone(), self.icon.clone(), None);
        app.headless = true;
        app.start_offscreen(width, height, scale);
        *self.headless.borrow_mut() = Some(app);
        self.with_headless(|app| app.emit(None, "start", vec![("width", Val::F(width)), ("height", Val::F(height)), ("scale", Val::F(scale))]))
    }

    /// The GPU of a headless run: {"name", "backend", "software"}. "software"
    /// is true for a CPU renderer such as lavapipe.
    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.with_headless(|app| app.gfx.as_ref().map(|g| g.gpu.adapter.get_info()))?;
        let d = PyDict::new(py);
        if let Some(i) = info {
            d.set_item("name", i.name)?;
            d.set_item("backend", format!("{:?}", i.backend))?;
            d.set_item("software", i.device_type == wgpu::DeviceType::Cpu)?;
        }
        Ok(d)
    }

    /// Apply the queued commands and deliver the events they cause, then a
    /// due "tick". Returns False once the app asked to quit. A call from
    /// inside the callback does nothing: the outer call delivers.
    fn process(&self) -> PyResult<bool> {
        self.with_headless(|app| {
            app.pump(None);
            app.apply(None);
            app.pump(None);
            if let Some(at) = app.tick_at {
                if Instant::now() >= at {
                    app.tick_at = None;
                    app.emit(None, "tick", vec![]);
                }
            }
            !app.quit
        })
    }

    /// The off-screen size of a headless run, in logical pixels. The caller
    /// sends the app its "resize" event.
    #[pyo3(signature = (width, height, scale=1.0))]
    fn resize_headless(&self, width: f64, height: f64, scale: f64) -> PyResult<()> {
        self.with_headless(|app| {
            app.scale = scale;
            if let Some(g) = &mut app.gfx {
                g.config.width = (width * scale).round().max(1.0) as u32;
                g.config.height = (height * scale).round().max(1.0) as u32;
            }
        })
    }

    /// Open the window and run the event loop until `quit()`. Blocks.
    fn run(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let el = EventLoop::<UserEv>::with_user_event().build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        el.set_control_flow(ControlFlow::Wait);
        let app = HostApp::new(self.shared.clone(), self.fonts.clone(), callback, self.title.clone(), self.icon.clone(), Some(el.create_proxy()));
        *self.proxy.borrow_mut() = app.proxy.clone();
        // Let go of the GIL while the loop waits, so Python threads (the agent
        // socket) can run; every callback takes it again (Python::attach).
        // The loop stays on this thread: AssertSend only carries it into detach.
        let job = AssertSend((el, app));
        let result = py.detach(move || {
            let job = job; // the whole wrapper: closures would capture its fields one by one
            let AssertSend((el, mut app)) = job;
            AssertSend(el.run_app(&mut app).map_err(|e| e.to_string()))
        });
        result.0.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// The palette: a (256, 4) uint8 array of RGBA colours.
    fn set_palette(&self, palette: PyReadonlyArray2<u8>) -> PyResult<()> {
        if palette.shape() != [256, 4] {
            return Err(pyo3::exceptions::PyValueError::new_err("the palette must be (256, 4)"));
        }
        self.push(Cmd::Palette(palette.as_array().iter().copied().collect()));
        Ok(())
    }

    /// Add or replace model `idx`: a (x, y, z) uint8 array; `position` is the
    /// world position of cell (0, 0, 0).
    fn set_model(&self, idx: usize, voxels: PyReadonlyArray3<u8>, position: (f64, f64, f64)) {
        let s = voxels.shape();
        self.push(Cmd::SetModel {
            idx,
            voxels: voxels.as_array().iter().copied().collect(),
            size: [s[0], s[1], s[2]],
            position: self::position(position),
        });
    }

    fn set_model_position(&self, idx: usize, position: (f64, f64, f64)) {
        self.push(Cmd::ModelPosition { idx, position: self::position(position) });
    }

    /// Fade (0..1, toward the background) and hide one model.
    fn set_model_style(&self, idx: usize, fade: f32, hidden: bool) {
        self.push(Cmd::ModelStyle { idx, fade, hidden });
    }

    /// A 4 x 4 turn (row-major, grid units, about cell (0, 0, 0) of the model at
    /// its position), or None for none: the rotate preview.
    #[pyo3(signature = (idx, turn=None))]
    fn set_model_turn(&self, idx: usize, turn: Option<PyReadonlyArray2<f64>>) -> PyResult<()> {
        let turn = match turn {
            None => None,
            Some(t) => {
                if t.shape() != [4, 4] {
                    return Err(pyo3::exceptions::PyValueError::new_err("turn must be 4 x 4"));
                }
                let rows: Vec<f64> = t.as_array().iter().copied().collect();
                Some(DMat4::from_cols_array(&rows.try_into().unwrap()).transpose())
            }
        };
        self.push(Cmd::ModelTurn { idx, turn });
        Ok(())
    }

    fn remove_model(&self, idx: usize) {
        self.push(Cmd::RemoveModel(idx));
    }

    /// Replace a box of model `idx`: `voxels` is the new content of the box at `origin`.
    fn update_region(&self, idx: usize, origin: (usize, usize, usize), voxels: PyReadonlyArray3<u8>) {
        let s = voxels.shape();
        if s.contains(&0) {
            return;
        }
        self.push(Cmd::Region {
            idx,
            origin: [origin.0, origin.1, origin.2],
            dims: [s[0], s[1], s[2]],
            data: voxels.as_array().iter().copied().collect(),
        });
    }

    /// A box of model `idx`'s seams field: one byte per cell, the same-brick
    /// bit of each side (bits 0..5: +x, -x, +y, -y, +z, -z).
    /// The first call makes the model a brick model: it shows seams.
    fn set_seams(&self, idx: usize, origin: (usize, usize, usize), field: PyReadonlyArray3<u8>) {
        let s = field.shape();
        if s.contains(&0) {
            return;
        }
        self.push(Cmd::Seams {
            idx,
            origin: [origin.0, origin.1, origin.2],
            dims: [s[0], s[1], s[2]],
            data: field.as_array().iter().copied().collect(),
        });
    }

    /// Model `idx` is a voxel model again: no seams.
    fn clear_seams(&self, idx: usize) {
        self.push(Cmd::ClearSeams(idx));
    }

    /// Ghost voxels on model `idx`: boxes the window shows as if the voxels were
    /// in the model, without changing it (the hover shows the result of a
    /// click). `boxes` is a list of (origin, (x, y, z) uint8 array); a 0 keeps
    /// the model's own voxel. With `erase`, a set cell blanks the model's voxel
    /// instead (the Erase hover). An empty list takes the ghosts away.
    #[pyo3(signature = (idx, boxes, erase=false))]
    fn set_ghost(&self, idx: usize, boxes: Vec<((usize, usize, usize), PyReadonlyArray3<u8>)>, erase: bool) {
        let boxes = boxes
            .into_iter()
            .map(|(origin, voxels)| {
                let s = voxels.shape();
                Ghost {
                    origin: [origin.0, origin.1, origin.2],
                    dims: [s[0], s[1], s[2]],
                    data: voxels.as_array().iter().copied().collect(),
                    erase,
                }
            })
            .filter(|g| g.dims.iter().all(|&d| d > 0))
            .collect();
        self.push(Cmd::Ghost { idx, boxes });
    }

    /// The camera: a 4 x 4 view-projection matrix (row-major, as numpy holds it,
    /// depth 0..1) and the eye position.
    fn set_camera(&self, view_proj: PyReadonlyArray2<f64>, eye: (f64, f64, f64)) -> PyResult<()> {
        if view_proj.shape() != [4, 4] {
            return Err(pyo3::exceptions::PyValueError::new_err("view_proj must be 4 x 4"));
        }
        let rows: Vec<f64> = view_proj.as_array().iter().copied().collect();
        let m = DMat4::from_cols_array(&rows.try_into().unwrap()).transpose();
        self.push(Cmd::Camera { view_proj: m, eye: self::position(eye) });
        Ok(())
    }

    /// Several views side by side (the quad view): (x, y, w, h, view_proj, eye,
    /// pixel) each, rectangles in logical pixels with the origin at the top
    /// left, `pixel` the world size of one pixel for an orthographic view (0 for
    /// perspective). An empty list goes back to one view over the window
    /// (the last `set_camera`).
    fn set_views(&self, views: Vec<(f64, f64, f64, f64, PyReadonlyArray2<f64>, (f64, f64, f64), f64)>) -> PyResult<()> {
        let mut out = Vec::new();
        for (x, y, w, h, vp, eye, pixel) in views.into_iter().take(crate::gpu::MAX_VIEWS) {
            if vp.shape() != [4, 4] {
                return Err(pyo3::exceptions::PyValueError::new_err("view_proj must be 4 x 4"));
            }
            let rows: Vec<f64> = vp.as_array().iter().copied().collect();
            let m = DMat4::from_cols_array(&rows.try_into().unwrap()).transpose();
            out.push(HostView { rect: [x, y, w, h], view_proj: m, eye: self::position(eye), pixel, cue: [0.0, 0.0], bg: 0.0 });
        }
        self.push(Cmd::Views(out));
        Ok(())
    }

    /// Replace one draw layer. Layers stay until replaced or cleared and draw
    /// in order of (z, layer). Lines, rects and triangles are flat float32 arrays
    /// (see overlay.rs for the fields); icons are (name, x, y, w, h, (r, g, b, a),
    /// image) tuples; texts are (x, y, size, (r, g, b, a), text) tuples. Colours
    /// are sRGB 0..1 floats, except the text colours, which are 0..255.
    #[pyo3(signature = (layer="main", z=0, lines3_depth=None, lines3_top=None, tris3_depth=None, tris3_top=None, tris2=None, rects=None, lines2=None, icons=None, texts=None))]
    fn set_draw(
        &self,
        layer: &str,
        z: i32,
        lines3_depth: Option<PyReadonlyArray1<f32>>,
        lines3_top: Option<PyReadonlyArray1<f32>>,
        tris3_depth: Option<PyReadonlyArray1<f32>>,
        tris3_top: Option<PyReadonlyArray1<f32>>,
        tris2: Option<PyReadonlyArray1<f32>>,
        rects: Option<PyReadonlyArray1<f32>>,
        lines2: Option<PyReadonlyArray1<f32>>,
        icons: Option<Vec<(String, f32, f32, f32, f32, (f32, f32, f32, f32), bool)>>,
        texts: Option<Vec<(f32, f32, f32, (u8, u8, u8, u8), String)>>,
    ) {
        let v = |a: Option<PyReadonlyArray1<f32>>| a.map(|a| a.as_array().to_vec()).unwrap_or_default();
        let list = DrawList {
            lines3_depth: v(lines3_depth),
            lines3_top: v(lines3_top),
            tris3_depth: v(tris3_depth),
            tris3_top: v(tris3_top),
            tris2: v(tris2),
            rects: v(rects),
            lines2: v(lines2),
            icons: icons
                .unwrap_or_default()
                .into_iter()
                .map(|(name, x, y, w, h, t, image)| IconRun { name, rect: [x, y, w, h], tint: [t.0, t.1, t.2, t.3], image })
                .collect(),
            texts: texts
                .unwrap_or_default()
                .into_iter()
                .map(|(x, y, size, c, text)| TextRun { x, y, size, color: [c.0, c.1, c.2, c.3], text })
                .collect(),
        };
        self.push(Cmd::Draw { layer: layer.to_string(), z, list });
    }

    fn clear_draw(&self, layer: String) {
        self.push(Cmd::ClearLayer(layer));
    }

    /// Shift a layer's 3D lines and faces by (x, y, z) world units on the GPU,
    /// without sending its content again.
    fn set_draw_offset(&self, layer: String, x: f32, y: f32, z: f32) {
        self.push(Cmd::DrawOffset { layer, offset: [x, y, z] });
    }

    /// An icon or picture for later `icons` runs: a (h, w, 4) uint8 RGBA array.
    fn add_image(&self, name: String, rgba: PyReadonlyArray3<u8>) -> PyResult<()> {
        let s = rgba.shape();
        if s[2] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("images must be (h, w, 4) RGBA"));
        }
        self.push(Cmd::Image { name, w: s[1] as u32, h: s[0] as u32, data: rgba.as_array().iter().copied().collect() });
        Ok(())
    }

    /// Width and height of `text` at `size` logical pixels.
    fn measure_text(&self, text: &str, size: f32) -> (f32, f32) {
        overlay::measure(&mut self.fonts.borrow_mut(), text, size)
    }

    /// `text` at `size` pixels as an alpha mask: (width, height, bytes), rows
    /// from the top.
    fn text_mask<'py>(&self, py: Python<'py>, text: &str, size: f32) -> (u32, u32, Bound<'py, pyo3::types::PyBytes>) {
        let (w, h, data) = overlay::text_mask(&mut self.fonts.borrow_mut(), text, size);
        (w, h, pyo3::types::PyBytes::new(py, &data))
    }

    fn request_redraw(&self) {
        self.push(Cmd::Redraw);
    }

    /// A "tick" event after `seconds` (the earliest pending one wins).
    fn tick(&self, seconds: f64) {
        self.push(Cmd::Tick(seconds));
    }

    /// The macOS menu bar: [(title, [(id, label, accelerator, check) or None for a line])].
    /// `check` is -1 for a plain item, 0 or 1 for an item with a check mark.
    /// An item can also be a submenu: (label, [items]).
    /// The first menu is the app menu. A menu item sends a "menu" event with its id.
    fn set_menu(&self, menus: Vec<(String, Vec<Option<MenuEntry>>)>) {
        self.push(Cmd::Menu(menus));
    }

    /// A native file dialog. Its answer comes as a "dialog" event with `token`
    /// and `path` (None when cancelled).
    #[pyo3(signature = (token, save, title, filters, name=None, directory=None))]
    fn file_dialog(&self, token: String, save: bool, title: String, filters: Vec<(String, Vec<String>)>, name: Option<String>, directory: Option<String>) {
        self.push(Cmd::Dialog { token, save, title, filters, name, directory });
    }

    /// A native message box with one to three buttons (the first is the default).
    /// Its answer comes as a "dialog" event with `token` and `answer`, the label
    /// of the button pressed.
    fn ask(&self, token: String, title: String, text: String, buttons: Vec<String>) -> PyResult<()> {
        if buttons.is_empty() || buttons.len() > 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("ask takes one to three buttons"));
        }
        self.push(Cmd::Ask { token, title, text, buttons });
        Ok(())
    }

    fn set_title(&self, title: String) {
        self.push(Cmd::Title(title));
    }

    /// The dot in the close button that says "unsaved changes".
    fn set_edited(&self, edited: bool) {
        self.push(Cmd::Edited(edited));
    }

    /// Copy the next frame back: a "screenshot" event brings `token`, `width`,
    /// `height` and `rgba` (bytes, top row first).
    fn screenshot(&self, token: String) {
        self.push(Cmd::Screenshot(token));
    }

    /// A picture for an agent, rendered where this command sits in the queue:
    /// `width` x `height` pixels, the views given as for `set_views`, and of the
    /// draw layers only those named "agent:...". A "screenshot" event brings it
    /// back with `token`. The window's own views and layers stay as they are.
    #[pyo3(signature = (token, width, height, views, background=0.0, keep_look=false))]
    fn render_views(&self, token: String, width: u32, height: u32, views: Vec<(f64, f64, f64, f64, PyReadonlyArray2<f64>, (f64, f64, f64), f64, f64, f64)>, background: f64, keep_look: bool) -> PyResult<()> {
        let mut out = Vec::new();
        for (x, y, w, h, vp, eye, pixel, near, depth) in views.into_iter().take(crate::gpu::MAX_VIEWS) {
            if vp.shape() != [4, 4] {
                return Err(pyo3::exceptions::PyValueError::new_err("view_proj must be 4 x 4"));
            }
            let rows: Vec<f64> = vp.as_array().iter().copied().collect();
            let m = DMat4::from_cols_array(&rows.try_into().unwrap()).transpose();
            out.push(HostView { rect: [x, y, w, h], view_proj: m, eye: self::position(eye), pixel, cue: [near, depth], bg: background });
        }
        if out.is_empty() || width == 0 || height == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("render_views needs a size and at least one view"));
        }
        self.push(Cmd::Shot { token, w: width, h: height, views: out, keep_look });
        Ok(())
    }

    /// Show the edges of the voxels (View > Grid).
    fn set_voxel_grid(&self, on: bool) {
        self.push(Cmd::Grid(on));
    }

    /// Ambient occlusion in the voxels (View > Ambient Occlusion): `strength`
    /// 0 is off, 1 darkens a fully occluded point to black; `reach` is how
    /// many cells a voxel may be away to count.
    fn set_occlusion(&self, strength: f32, reach: f32) {
        self.push(Cmd::Occlusion([strength.clamp(0.0, 1.0), reach.max(1.0)]));
    }

    /// The rendered view (trace.rs): a path traced picture in the first view,
    /// lit as a render preset says (rok/core/render_presets.trace_settings).
    /// Colours are linear RGB. `lamps`: up to four (kind, x, y, z, r, g, b,
    /// size) — kind 0 a sun (x, y, z toward it, the colour its irradiance,
    /// size its width in degrees), kind 1 a glowing sphere (x, y, z its offset
    /// from the scene's middle, the colour its radiant intensity, size its
    /// radius). `backdrop`: what the camera sees where nothing is, else the
    /// sky. `exposure` in stops. `tone`: 0 neutral, 1 clipped, 2 film, 3 film
    /// punchy, 4 film high contrast. `transparent` leaves the background out.
    #[pyo3(signature = (on, max_samples=512, bounces=4, lamps=None, sky_top=(0.42, 0.55, 0.78),
                        sky_horizon=(0.86, 0.88, 0.9), backdrop=None, ground=Some((0.42, 0.42, 0.42)), exposure=0.0,
                        tone=0, bloom=0.0, outline=0.0, voxel_grid=0.0, emission=1.0, transparent=false))]
    fn set_render(
        &self,
        on: bool,
        max_samples: u32,
        bounces: u32,
        lamps: Option<Vec<(u32, f32, f32, f32, f32, f32, f32, f32)>>,
        sky_top: (f32, f32, f32),
        sky_horizon: (f32, f32, f32),
        backdrop: Option<(f32, f32, f32)>,
        ground: Option<(f32, f32, f32)>,
        exposure: f32,
        tone: u32,
        bloom: f32,
        outline: f32,
        voxel_grid: f32,
        emission: f32,
        transparent: bool,
    ) {
        let st = trace_settings(on, max_samples, bounces, lamps, sky_top, sky_horizon, backdrop, ground, exposure, tone, bloom,
                                outline, voxel_grid, emission, transparent);
        self.push(Cmd::Render(st));
    }

    /// A finished picture (rok/app/render_picture.py): the scene path traced
    /// from `view_proj` at width x height pixels, with all `max_samples`, lit
    /// as `set_render` takes it. It comes back as a "screenshot" event with
    /// `token`; a transparent background has alpha 0. The rendered view starts
    /// its samples over after.
    #[pyo3(signature = (token, width, height, view_proj, eye, max_samples=128, bounces=4, lamps=None,
                        sky_top=(0.42, 0.55, 0.78), sky_horizon=(0.86, 0.88, 0.9), backdrop=None,
                        ground=Some((0.42, 0.42, 0.42)), exposure=0.0, tone=0, bloom=0.0, outline=0.0, voxel_grid=0.0,
                        emission=1.0, transparent=false))]
    fn render_picture(
        &self,
        token: String,
        width: u32,
        height: u32,
        view_proj: PyReadonlyArray2<f64>,
        eye: (f64, f64, f64),
        max_samples: u32,
        bounces: u32,
        lamps: Option<Vec<(u32, f32, f32, f32, f32, f32, f32, f32)>>,
        sky_top: (f32, f32, f32),
        sky_horizon: (f32, f32, f32),
        backdrop: Option<(f32, f32, f32)>,
        ground: Option<(f32, f32, f32)>,
        exposure: f32,
        tone: u32,
        bloom: f32,
        outline: f32,
        voxel_grid: f32,
        emission: f32,
        transparent: bool,
    ) -> PyResult<()> {
        let vp = mat4_of(&view_proj)?;
        let st = trace_settings(true, max_samples, bounces, lamps, sky_top, sky_horizon, backdrop, ground, exposure, tone,
                                bloom, outline, voxel_grid, emission, transparent);
        self.push(Cmd::Picture { token, w: width.max(1), h: height.max(1), view_proj: vp, eye: position(eye), settings: st });
        Ok(())
    }

    /// The palette slots' materials for the rendered view: a (256, 8) float32
    /// array of metallic, roughness, transmission, emission, ior, specular, type, 0
    /// (MaterialTable.channels and type_codes).
    fn set_materials(&self, rows: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = rows.shape();
        if shape != [256, MAT_COLS] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("materials must be (256, {MAT_COLS}), not {shape:?}")));
        }
        self.push(Cmd::Materials(rows.as_array().iter().copied().collect()));
        Ok(())
    }

    /// How many samples the rendered view has added up.
    fn render_samples(&self) -> u32 {
        self.with_headless(|app| app.gfx.as_ref().map_or(0, |g| g.tracer.samples())).unwrap_or(0)
    }

    /// How the voxels look. `light`: 0 lit, 1 flat (unlit), 2 toon bands, 3 iso (three
    /// face tones), 4 soft (sky and floor), 5 night (Light colours glow), 6 ceramic
    /// (a glaze); `ao`, `ink`, `rim`, `shadow`: strengths 0..1 of voxel occlusion, edge
    /// lines, bright convex edges and sun shadows; `clay`: 1 for one neutral material;
    /// `ink_light`: 0 dark lines, 1 light ones; `background`: an sRGB colour, or None
    /// for the dark default; `ink_width`: the lines' width in UI pixels.
    #[pyo3(signature = (light=0, ao=0.0, ink=0.0, rim=0.0, shadow=0.0, clay=0.0, ink_light=0.0, background=None, ink_width=1.0))]
    fn set_look(&self, light: u32, ao: f32, ink: f32, rim: f32, shadow: f32, clay: f32, ink_light: f32, background: Option<(f32, f32, f32)>, ink_width: f32) {
        let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let bg = match background {
            Some((r, g, b)) => [lin(r), lin(g), lin(b), 1.0],
            None => [0.0; 4],
        };
        let flags = (if clay > 0.5 { 1.0 } else { 0.0 }) + (if ink_light > 0.5 { 2.0 } else { 0.0 });
        self.push(Cmd::Look([[light as f32, ao, ink, rim], [shadow, flags, ink_width, 0.0], bg]));
    }

    /// The CAD view. The look stays as it is; the flag is applied again each frame,
    /// so a look change does not leave the view.
    fn set_cad(&self, on: bool) {
        self.push(Cmd::Cad(on));
    }

    /// See-through CAD view: the faces wash out and hidden edges can be drawn on top.
    /// Off with the CAD view, so the editing view stays as it was.
    fn set_cad_hidden(&self, on: bool) {
        self.push(Cmd::CadHidden(on));
    }

    /// Fitted-face normals and corner bits for model `idx`: (x, y, z, 4) uint8.
    /// rgb is the normal, 1..255 around 128; a is the corner mask. The size must
    /// match the model's grid.
    fn set_cad_field(&self, idx: usize, field: numpy::PyReadonlyArray4<u8>) -> PyResult<()> {
        let s = field.shape();
        self.push(Cmd::CadField {
            idx,
            data: field.as_array().iter().copied().collect(),
            size: [s[0], s[1], s[2]],
        });
        Ok(())
    }

    /// Smooth the edges of the 3D view (FXAA). On by default.
    fn set_smooth_edges(&self, on: bool) {
        self.push(Cmd::Smooth(on));
    }

    /// Move the mouse pointer (logical pixels, origin top left): for endless drags.
    fn warp_cursor(&self, x: f64, y: f64) {
        self.push(Cmd::Warp(x, y));
    }

    /// The pointer's shape: "default", "crosshair", "move", "grab", "grabbing",
    /// "text", "ew-resize", "ns-resize", "pointer", "wait", "none".
    fn set_cursor(&self, name: String) {
        self.push(Cmd::Cursor(name));
    }

    /// Relative mouse (a number field's drag): the pointer hides and stays
    /// where it is, and the raw moves come as "mouse_delta" events (dx, dy in
    /// points, y down), so a drag goes on at the screen's edge (the
    /// Size field sits at the window's left edge and could only shrink by one).
    fn set_relative(&self, on: bool) {
        self.push(Cmd::Relative(on));
    }

    fn quit(&self) {
        self.push(Cmd::Quit);
    }
}

/// ROK_TIMING=1: sums of the stages of the last frames, printed every 60
/// frames. `py` is the Python callback, `apply` the commands after it,
/// `acquire` the wait for a surface texture, `draw` encode and submit,
/// `present` the present call, `input` the time from the newest mouse
/// move's arrival to its frame's present.
#[derive(Default)]
struct Timing {
    frames: u32,
    events: u32,
    py: f64,
    py_max: f64,
    apply: f64,
    acquire: f64,
    acquire_max: f64,
    draw: f64,
    draw_max: f64,
    present: f64,
    input: f64,
    input_max: f64,
    input_n: u32,
    input_at: Option<Instant>,
    last_print: Option<Instant>,
    no_texture: u32, // frames the surface gave no texture for (a hidden window, a timeout)
    // The loop's turns: a turn runs from its first event to the end of about_to_wait.
    turns: u32,
    turn_start: Option<Instant>,
    turn: f64,
    turn_max: f64,
    slept_at: Option<Instant>, // when about_to_wait ended (the loop then waits for events)
    idle: f64,
    moves_in: u32, // CursorMoved events the system delivered
    turn_py: f64,  // Python time inside the current turn (for the slow-turn print)
}

impl Timing {
    fn report(&mut self) {
        if self.frames < 60 {
            return;
        }
        self.flush();
    }

    /// Print whatever was counted (also the last, short stretch at quit).
    fn flush(&mut self) {
        if self.frames == 0 && self.events == 0 {
            return;
        }
        let f = self.frames as f64;
        let e = self.events.max(1) as f64;
        let since = self.last_print.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        eprintln!(
            "rok timing: {} frames in {:.2} s ({:.0} fps), {} without a surface texture | {} turns ({} moves in, {} events to Python), turn {:.1} ms (max {:.1}), idle between turns {:.1} ms/turn | python {:.2} ms/event (max {:.1}) | apply {:.2} | acquire {:.2} (max {:.1}) | draw {:.2} (max {:.1}) | present {:.2} | input->present {:.1} ms (max {:.1}, n {})",
            self.frames, since, f / since.max(1e-6), self.no_texture, self.turns, self.moves_in, self.events,
            self.turn / self.turns.max(1) as f64 * 1e3, self.turn_max * 1e3, self.idle / self.turns.max(1) as f64 * 1e3,
            self.py / e * 1e3, self.py_max * 1e3, self.apply / f.max(1.0) * 1e3,
            self.acquire / f.max(1.0) * 1e3, self.acquire_max * 1e3, self.draw / f.max(1.0) * 1e3, self.draw_max * 1e3, self.present / f.max(1.0) * 1e3,
            self.input / self.input_n.max(1) as f64 * 1e3, self.input_max * 1e3, self.input_n
        );
        let keep = (self.turn_start, self.slept_at);
        *self = Timing { last_print: Some(Instant::now()), turn_start: keep.0, slept_at: keep.1, ..Default::default() };
    }

    /// An event arrived: the turn begins (once), and the wait before it is idle time.
    fn event_in(&mut self) {
        if self.turn_start.is_none() {
            let now = Instant::now();
            if let Some(slept) = self.slept_at.take() {
                self.idle += (now - slept).as_secs_f64();
            }
            self.turn_start = Some(now);
            self.turn_py = 0.0;
        }
    }

    /// about_to_wait is done: the turn ends. A slow turn prints itself.
    fn turn_end(&mut self, frame_ms: f64) {
        let now = Instant::now();
        if let Some(start) = self.turn_start.take() {
            let dt = (now - start).as_secs_f64();
            self.turns += 1;
            self.turn += dt;
            self.turn_max = self.turn_max.max(dt);
            if dt > 0.03 {
                eprintln!("rok timing: slow turn {:.1} ms: python {:.1} ms, frame {:.1} ms", dt * 1e3, self.turn_py * 1e3, frame_ms);
            }
        }
        self.slept_at = Some(now);
    }
}

struct Gfx {
    gpu: Gpu,
    renderer: Renderer,
    tracer: Tracer,
    overlay: Overlay,
    postfx: PostFx,
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
}

/// How long a trackpad or Ctrl orbit keeps the low-delay mode after its last event.
const GESTURE_HOLD: Duration = Duration::from_millis(250);

struct HostApp {
    shared: Rc<RefCell<Shared>>,
    fonts: Rc<RefCell<FontSystem>>,
    callback: Py<PyAny>,
    title: String,
    icon: Option<String>,
    proxy: Option<Arc<Mutex<EventLoopProxy<UserEv>>>>,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    models: Vec<Model>,
    palette: Vec<u8>,
    view_proj: DMat4,
    eye: DVec3,
    views: Vec<HostView>,
    scale: f64,
    tick_at: Option<Instant>,
    #[cfg(feature = "gui")]
    menu: Option<muda::Menu>,
    queue: VecDeque<Event>,
    pumping: bool,
    // A mouse button is down: frames render in the loop turn that handled the
    // move, and the surface presents at once (no vsync queue), for the least
    // delay between the hand and the picture.
    interacting: bool,
    buttons: u32, // mouse buttons down
    // A trackpad orbit (scroll, pinch, rotate) or a Ctrl+move orbit counts as
    // interacting until this moment: it has no button that goes up
    // (orbiting ran at 30 fps, every frame waited ~23 ms for vsync).
    gesture_until: Option<Instant>,
    ctrl: bool,
    relative: bool, // Host.set_relative: raw mouse moves go to Python as mouse_delta
    immediate_ok: bool, // the surface offers PresentMode::Immediate
    want_frame: bool,   // a frame to render in this loop turn (while interacting)
    timing: Option<Timing>, // ROK_TIMING=1: where a frame's time goes, on stderr
    shots: Vec<String>,
    smooth: bool,
    grid: bool,
    // Ambient occlusion in the voxels: [strength, reach in cells] (Host.set_occlusion).
    occlusion: [f32; 2],
    // The viewport look (Host.set_look), sent to the shader as three vec4s.
    look: [[f32; 4]; 3],
    // CAD view: one grey per face, corner lines, white background. Independent of `look`.
    cad: bool,
    // See-through CAD view. The flag is sent only while `cad` is on.
    cad_hidden: bool,
    headless: bool,
    quit: bool,
    // The rendered view (Host.set_render, Host.set_materials), kept for a new GPU.
    trace_settings: TraceSettings,
    materials: Vec<f32>,
}

impl HostApp {
    fn new(shared: Rc<RefCell<Shared>>, fonts: Rc<RefCell<FontSystem>>, callback: Py<PyAny>, title: String, icon: Option<String>, proxy: Option<EventLoopProxy<UserEv>>) -> HostApp {
        HostApp {
            shared,
            fonts,
            callback,
            title,
            icon,
            proxy: proxy.map(|p| Arc::new(Mutex::new(p))),
            window: None,
            gfx: None,
            models: Vec::new(),
            palette: vec![128; 1024],
            view_proj: DMat4::IDENTITY,
            eye: DVec3::new(0.0, -100.0, 50.0),
            views: Vec::new(),
            scale: 1.0,
            tick_at: None,
            #[cfg(feature = "gui")]
            menu: None,
            queue: VecDeque::new(),
            pumping: false,
            interacting: false,
            buttons: 0,
            gesture_until: None,
            ctrl: false,
            relative: false,
            immediate_ok: false,
            want_frame: false,
            timing: std::env::var_os("ROK_TIMING").map(|_| Timing::default()),
            shots: Vec::new(),
            smooth: true,
            grid: false,
            occlusion: [0.0, 8.0],
            look: [[0.0; 4]; 3],
            cad: false,
            cad_hidden: false,
            headless: false,
            quit: false,
            trace_settings: TraceSettings::default(),
            materials: Vec::new(),
        }
    }

    fn logical(&self) -> (f64, f64) {
        match &self.gfx {
            Some(g) => (g.config.width as f64 / self.scale, g.config.height as f64 / self.scale),
            None => (0.0, 0.0),
        }
    }

    fn emit(&mut self, el: Option<&ActiveEventLoop>, kind: &'static str, fields: Vec<(&'static str, Val)>) {
        self.queue.push_back((kind, fields));
        self.pump(el);
    }

    /// Deliver queued events one by one, applying the commands after each.
    /// A nested call (an event that arrives while a command runs, for example
    /// during a modal dialog) only queues; the outer loop delivers it.
    fn pump(&mut self, el: Option<&ActiveEventLoop>) {
        if self.pumping {
            return;
        }
        self.pumping = true;
        while let Some((kind, fields)) = self.queue.pop_front() {
            Python::attach(|py| {
                let d = PyDict::new(py);
                for (k, v) in fields {
                    let r = match v {
                        Val::F(x) => d.set_item(k, x),
                        Val::B(b) => d.set_item(k, b),
                        Val::S(s) => d.set_item(k, s),
                        Val::Bytes(b) => d.set_item(k, pyo3::types::PyBytes::new(py, &b)),
                        Val::None => d.set_item(k, py.None()),
                    };
                    let _ = r;
                }
                let t0 = Instant::now();
                if let Err(err) = self.callback.call1(py, (kind, d)) {
                    // An exception must not end the app: print it and go on.
                    eprintln!("rok: the Python callback failed on a {kind} event:");
                    err.print(py);
                }
                if let Some(t) = &mut self.timing {
                    let dt = t0.elapsed().as_secs_f64();
                    if dt > 0.015 {
                        eprintln!("rok timing: slow event {kind}: python {:.1} ms", dt * 1e3);
                    }
                    t.events += 1;
                    t.py += dt;
                    t.py_max = t.py_max.max(dt);
                    t.turn_py += dt;
                }
            });
            self.apply(el);
        }
        self.pumping = false;
    }

    fn apply(&mut self, el: Option<&ActiveEventLoop>) {
        let t_apply = Instant::now();
        let cmds = std::mem::take(&mut self.shared.borrow_mut().cmds);
        let mut redraw = false;
        for cmd in cmds {
            // The rendered view builds its scene again after a change to the
            // models; a changed box of voxels and a new palette it takes in place.
            if let Some(g) = &mut self.gfx {
                match &cmd {
                    Cmd::SetModel { .. } | Cmd::ModelPosition { .. } | Cmd::RemoveModel(_) | Cmd::ModelStyle { .. }
                    | Cmd::ModelTurn { .. } => g.tracer.dirty = true,
                    Cmd::Region { idx, origin, dims, .. } => g.tracer.region(*idx, *origin, *dims),
                    Cmd::Palette(_) => g.tracer.materials_changed(),
                    _ => {}
                }
            }
            match cmd {
                Cmd::Render(settings) => {
                    if let Some(g) = &mut self.gfx {
                        if g.tracer.settings != settings {
                            g.tracer.settings = settings;
                            g.tracer.restart();
                        }
                    }
                    self.trace_settings = settings;
                    redraw = true;
                }
                Cmd::Picture { token, w, h, view_proj, eye, settings } => {
                    self.shoot_picture(token, w, h, view_proj, eye, settings);
                    redraw = true;
                }
                Cmd::Materials(rows) => {
                    if let Some(g) = &mut self.gfx {
                        g.tracer.set_materials(rows.clone());
                        g.tracer.materials_changed();
                        g.renderer.set_materials(&g.gpu, &rows);
                    }
                    self.materials = rows;
                    redraw = true;
                }
                Cmd::Palette(p) => {
                    self.palette = p;
                    if let Some(g) = &self.gfx {
                        g.renderer.set_palette(&g.gpu, &self.palette);
                    }
                    redraw = true;
                }
                Cmd::SetModel { idx, voxels, size, position } => {
                    let half = DVec3::new(size[0] as f64, size[1] as f64, size[2] as f64) * 0.5;
                    let mut m = Model::new(voxels, size, position + half);
                    if let Some(old) = self.models.get(idx) {
                        m.fade = old.fade;
                        m.hidden = old.hidden;
                        m.ghost = old.ghost.iter().filter(|g| g.fits(size)).cloned().collect();
                    }
                    if let Some(g) = &mut self.gfx {
                        let at = idx.min(self.models.len());
                        g.renderer.upload_model(&g.gpu, at, &m);
                        for (origin, dims) in m.ghost.iter().map(|gh| (gh.origin, gh.dims)).collect::<Vec<_>>() {
                            let shown = m.shown_region(origin, dims);
                            let bricks = m.region_bricks(origin, dims, false);
                            g.renderer.write_region(&g.gpu, at, origin, dims, &shown, bricks);
                        }
                    }
                    if idx < self.models.len() {
                        self.models[idx] = m;
                    } else {
                        self.models.push(m);
                    }
                    redraw = true;
                }
                Cmd::ModelPosition { idx, position } => {
                    if let Some(m) = self.models.get_mut(idx) {
                        m.center = position + m.half();
                    }
                    redraw = true;
                }
                Cmd::ModelStyle { idx, fade, hidden } => {
                    if let Some(m) = self.models.get_mut(idx) {
                        m.fade = fade.clamp(0.0, 1.0);
                        m.hidden = hidden;
                    }
                    redraw = true;
                }
                Cmd::ModelTurn { idx, turn } => {
                    if let Some(m) = self.models.get_mut(idx) {
                        m.turn = turn;
                    }
                    redraw = true;
                }
                Cmd::RemoveModel(idx) => {
                    if idx < self.models.len() {
                        self.models.remove(idx);
                        if let Some(g) = &mut self.gfx {
                            g.renderer.remove_model(idx);
                        }
                    }
                    redraw = true;
                }
                Cmd::Region { idx, origin, dims, data } => {
                    let Some(m) = self.models.get_mut(idx) else { continue };
                    let fits = (0..3).all(|k| origin[k] + dims[k] <= m.size[k]);
                    if !fits {
                        eprintln!("rok: update_region outside model {idx}: {origin:?} + {dims:?} > {:?}", m.size);
                        continue;
                    }
                    let bricks = m.write_region(origin, dims, &data);
                    if let Some(g) = &self.gfx {
                        if m.ghost.is_empty() {
                            g.renderer.write_region(&g.gpu, idx, origin, dims, &data, bricks);
                        } else {
                            let shown = m.shown_region(origin, dims);
                            g.renderer.write_region(&g.gpu, idx, origin, dims, &shown, bricks);
                        }
                    }
                    redraw = true;
                }
                Cmd::Seams { idx, origin, dims, data } => {
                    let Some(m) = self.models.get_mut(idx) else { continue };
                    if !(0..3).all(|k| origin[k] + dims[k] <= m.size[k]) {
                        eprintln!("rok: set_seams outside model {idx}: {origin:?} + {dims:?} > {:?}", m.size);
                        continue;
                    }
                    let fresh = m.write_seams(origin, dims, &data);
                    if let Some(g) = &mut self.gfx {
                        if fresh {
                            g.renderer.set_seams_field(&g.gpu, idx, m.seams.as_deref(), m.size);
                        } else {
                            g.renderer.write_seams(&g.gpu, idx, origin, dims, &data);
                        }
                    }
                    redraw = true;
                }
                Cmd::ClearSeams(idx) => {
                    let Some(m) = self.models.get_mut(idx) else { continue };
                    if m.seams.take().is_some() {
                        if let Some(g) = &mut self.gfx {
                            g.renderer.set_seams_field(&g.gpu, idx, None, m.size);
                        }
                    }
                    redraw = true;
                }
                Cmd::Ghost { idx, boxes } => {
                    let Some(m) = self.models.get_mut(idx) else { continue };
                    let touched = m.set_ghost(boxes);
                    if let Some(g) = &self.gfx {
                        for (origin, dims) in touched {
                            let shown = m.shown_region(origin, dims);
                            let bricks = m.region_bricks(origin, dims, false);
                            g.renderer.write_region(&g.gpu, idx, origin, dims, &shown, bricks);
                        }
                    }
                    redraw = true;
                }
                Cmd::Camera { view_proj, eye } => {
                    self.view_proj = view_proj;
                    self.eye = eye;
                    redraw = true;
                }
                Cmd::Views(views) => {
                    self.views = views;
                    redraw = true;
                }
                Cmd::Draw { layer, z, list } => {
                    if let Some(g) = &mut self.gfx {
                        g.overlay.set_layer(&g.gpu, &layer, z, list);
                    }
                    redraw = true;
                }
                Cmd::DrawOffset { layer, offset } => {
                    if let Some(g) = &mut self.gfx {
                        g.overlay.set_layer_offset(&layer, offset);
                    }
                    redraw = true;
                }
                Cmd::ClearLayer(layer) => {
                    if let Some(g) = &mut self.gfx {
                        g.overlay.clear_layer(&layer);
                    }
                    redraw = true;
                }
                Cmd::Image { name, w, h, data } => {
                    if let Some(g) = &mut self.gfx {
                        g.overlay.add_image(&g.gpu, &name, w, h, &data);
                    }
                    redraw = true;
                }
                Cmd::Redraw => redraw = true,
                Cmd::Tick(s) => {
                    let at = Instant::now() + Duration::from_secs_f64(s.max(0.0));
                    self.tick_at = Some(self.tick_at.map_or(at, |t| t.min(at)));
                }
                Cmd::Menu(spec) => {
                    if !self.headless {
                        #[cfg(feature = "gui")]
                        self.build_menu(spec);
                        #[cfg(not(feature = "gui"))]
                        drop(spec);
                    }
                }
                #[cfg(not(feature = "gui"))]
                Cmd::Dialog { token, .. } => {
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("path", Val::None)]));
                }
                Cmd::Dialog { token, .. } if self.headless => {
                    // No one to ask: a headless file dialog is cancelled.
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("path", Val::None)]));
                }
                #[cfg(feature = "gui")]
                Cmd::Dialog { token, save, title, filters, name, directory } => {
                    let mut d = rfd::FileDialog::new().set_title(&title);
                    for (label, exts) in &filters {
                        d = d.add_filter(label, exts);
                    }
                    if let Some(n) = &name {
                        d = d.set_file_name(n);
                    }
                    if let Some(dir) = &directory {
                        d = d.set_directory(dir);
                    }
                    let path = if save { d.save_file() } else { d.pick_file() };
                    let path = match path {
                        Some(p) => Val::S(p.to_string_lossy().into_owned()),
                        None => Val::None,
                    };
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("path", path)]));
                }
                #[cfg(not(feature = "gui"))]
                Cmd::Ask { token, buttons, .. } => {
                    let answer = buttons.last().cloned().unwrap_or_default();
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("answer", Val::S(answer))]));
                }
                Cmd::Ask { token, buttons, .. } if self.headless => {
                    // No one to ask: the last button, which is Cancel where there is one.
                    let answer = buttons.last().cloned().unwrap_or_default();
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("answer", Val::S(answer))]));
                }
                #[cfg(feature = "gui")]
                Cmd::Ask { token, title, text, buttons } => {
                    use rfd::{MessageButtons, MessageDialogResult};
                    let b = |i: usize| buttons[i].clone();
                    let kind = match buttons.len() {
                        1 => MessageButtons::OkCustom(b(0)),
                        2 => MessageButtons::OkCancelCustom(b(0), b(1)),
                        _ => MessageButtons::YesNoCancelCustom(b(0), b(1), b(2)),
                    };
                    let result = rfd::MessageDialog::new()
                        .set_level(rfd::MessageLevel::Warning)
                        .set_title(&title)
                        .set_description(&text)
                        .set_buttons(kind)
                        .show();
                    let answer = match result {
                        MessageDialogResult::Custom(label) => label,
                        MessageDialogResult::Yes | MessageDialogResult::Ok => b(0),
                        MessageDialogResult::No => b(1.min(buttons.len() - 1)),
                        MessageDialogResult::Cancel => b(buttons.len() - 1),
                    };
                    self.queue.push_back(("dialog", vec![("token", Val::S(token)), ("answer", Val::S(answer))]));
                }
                Cmd::Title(t) => {
                    if let Some(w) = &self.window {
                        w.set_title(&t);
                    }
                }
                Cmd::Edited(e) => {
                    #[cfg(target_os = "macos")]
                    if let Some(w) = &self.window {
                        use winit::platform::macos::WindowExtMacOS;
                        w.set_document_edited(e);
                    }
                    #[cfg(not(target_os = "macos"))]
                    let _ = e;
                }
                Cmd::Screenshot(token) => self.shots.push(token),
                Cmd::Shot { token, w, h, views, keep_look } => self.shoot_views(token, w, h, views, keep_look),
                Cmd::Grid(on) => {
                    self.grid = on;
                    redraw = true;
                }
                Cmd::Occlusion(o) => {
                    self.occlusion = o;
                    redraw = true;
                }
                Cmd::Look(look) => {
                    self.look = look;
                    redraw = true;
                }
                Cmd::Cad(on) => {
                    self.cad = on;
                    redraw = true;
                }
                Cmd::CadHidden(on) => {
                    self.cad_hidden = on;
                    redraw = true;
                }
                Cmd::CadField { idx, data, size } => {
                    let Some(m) = self.models.get_mut(idx) else { continue };
                    let want = size[0].saturating_mul(size[1]).saturating_mul(size[2]).saturating_mul(4);
                    if size != m.size || data.len() != want || want == 0 {
                        eprintln!("rok: cad field for model {idx} does not match the grid");
                        continue;
                    }
                    m.cad = true;
                    if let Some(g) = &mut self.gfx {
                        g.renderer.set_cad_field(&g.gpu, idx, &data, size);
                    }
                    redraw = true;
                }
                Cmd::Smooth(on) => {
                    self.smooth = on;
                    redraw = true;
                }
                Cmd::Relative(on) => {
                    self.relative = on;
                    if let Some(w) = &self.window {
                        if on {
                            w.set_cursor_visible(false);
                            if w.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                                let _ = w.set_cursor_grab(CursorGrabMode::Confined);
                            }
                        } else {
                            let _ = w.set_cursor_grab(CursorGrabMode::None);
                            w.set_cursor_visible(true);
                        }
                    }
                }
                Cmd::Warp(x, y) => {
                    if let Some(w) = &self.window {
                        let _ = w.set_cursor_position(winit::dpi::LogicalPosition::new(x, y));
                    }
                }
                Cmd::Cursor(name) => {
                    if let Some(w) = &self.window {
                        use winit::window::CursorIcon as C;
                        if name == "none" {
                            w.set_cursor_visible(false);
                        } else {
                            w.set_cursor_visible(true);
                            let icon = match name.as_str() {
                                "crosshair" => C::Crosshair,
                                "move" => C::Move,
                                "grab" => C::Grab,
                                "grabbing" => C::Grabbing,
                                "text" => C::Text,
                                "ew-resize" => C::EwResize,
                                "ns-resize" => C::NsResize,
                                "pointer" => C::Pointer,
                                "wait" => C::Wait,
                                _ => C::Default,
                            };
                            w.set_cursor(icon);
                        }
                    }
                }
                Cmd::Quit => {
                    if let Some(t) = &mut self.timing {
                        t.flush();
                    }
                    match el {
                        Some(el) => el.exit(),
                        None => self.quit = true,
                    }
                }
            }
        }
        self.shoot();
        if let Some(t) = &mut self.timing {
            t.apply += t_apply.elapsed().as_secs_f64();
        }
        if redraw {
            // Rendered in about_to_wait, in this very turn of the loop: a
            // request_redraw() comes back as RedrawRequested on a later turn,
            // and on macOS not at once, so the picture ran behind the hand. RedrawRequested still serves what the system asks
            // for (a resize, an exposed window).
            self.want_frame = true;
        }
    }

    /// An orbit gesture moved the view: interacting until GESTURE_HOLD after it.
    fn gesture(&mut self) {
        self.gesture_until = Some(Instant::now() + GESTURE_HOLD);
        self.set_interacting(true);
    }

    /// Back to vsync once no button is down and the last gesture is over.
    fn settle(&mut self) {
        if let Some(until) = self.gesture_until {
            if Instant::now() < until {
                return;
            }
            self.gesture_until = None;
        }
        if self.buttons == 0 {
            self.set_interacting(false);
        }
    }

    /// Queue a gesture event; one waiting of the same kind takes its deltas,
    /// so the loop delivers one per turn, as it does with mouse moves.
    fn queue_gesture(&mut self, kind: &'static str, fields: Vec<(&'static str, Val)>) {
        self.gesture();
        if let Some(t) = &mut self.timing {
            if t.input_at.is_none() {
                t.input_at = Some(Instant::now());
            }
        }
        if let Some(last) = self.queue.back_mut() {
            if last.0 == kind {
                for ((_, old), (_, new)) in last.1.iter_mut().zip(fields.iter()) {
                    if let (Val::F(a), Val::F(b)) = (old, new) {
                        *a += *b;
                    }
                }
                return;
            }
        }
        self.queue.push_back((kind, fields));
    }

    /// A mouse button went down or up. While it is down the surface presents
    /// at once (PresentMode::Immediate, where the GPU offers it) and frames
    /// render in the turn that handled the move; at rest it goes back to
    /// vsync (Fifo), which never tears.
    fn set_interacting(&mut self, on: bool) {
        if on == self.interacting {
            return;
        }
        self.interacting = on;
        if let Some(g) = &mut self.gfx {
            if let Some(surface) = &g.surface {
                if self.immediate_ok {
                    g.config.present_mode = if on { wgpu::PresentMode::Immediate } else { wgpu::PresentMode::Fifo };
                    surface.configure(&g.gpu.device, &g.config);
                }
            }
        }
    }

    #[cfg(feature = "gui")]
    fn build_menu(&mut self, spec: Vec<(String, Vec<Option<MenuEntry>>)>) {
        #[cfg(target_os = "macos")]
        hide_apple_edit_items();
        let menu = muda::Menu::new();
        for (title, items) in spec {
            // macOS adds Writing Tools, AutoFill, Start Dictation and Emoji &
            // Symbols to a menu it finds by the title "Edit" (they do
            // nothing in Rok). A zero-width space after the title
            // hides the menu from that search and looks the same.
            let title = if cfg!(target_os = "macos") && title == "Edit" { format!("{title}\u{200B}") } else { title };
            let sub = muda::Submenu::new(&title, true);
            append_entries(&sub, items);
            let _ = menu.append(&sub);
        }
        #[cfg(target_os = "macos")]
        menu.init_for_nsapp();
        let Some(proxy) = self.proxy.clone() else { return };
        muda::MenuEvent::set_event_handler(Some(move |e: muda::MenuEvent| {
            if let Ok(p) = proxy.lock() {
                let _ = p.send_event(UserEv::Menu(e.id.0.clone()));
            }
        }));
        self.menu = Some(menu);
    }

    fn configure(&mut self) {
        let (Some(g), Some(w)) = (&mut self.gfx, &self.window) else { return };
        let s = w.inner_size();
        g.config.width = s.width.max(1);
        g.config.height = s.height.max(1);
        if let Some(surface) = &g.surface {
            surface.configure(&g.gpu.device, &g.config);
        }
        self.scale = w.scale_factor();
    }

    fn frame_uniform(&self, vp: DMat4, eye: DVec3, w: u32, h: u32, origin: (u32, u32), pixel: f64, cue: [f64; 2], bg: f64) -> FrameU {
        let inv = vp.inverse();
        let far = inv * DVec4::new(0.0, 0.0, 1.0, 1.0);
        let far = far.truncate() / far.w;
        let near = inv * DVec4::new(0.0, 0.0, 0.0, 1.0);
        let near = near.truncate() / near.w;
        let view_dir = (far - near).normalize_or(DVec3::Y);
        // Lights fixed to the camera: a key from upper left, a fill from the right.
        let right = view_dir.cross(DVec3::Z).normalize_or(DVec3::X);
        let up = right.cross(view_dir).normalize_or(DVec3::Z);
        let key = (-view_dir * 0.6 - right * 0.5 + up * 0.7).normalize();
        let fill = (-view_dir * 0.5 + right * 0.8 - up * 0.1).normalize();
        let m = |m: DMat4| m.as_mat4().to_cols_array_2d();
        FrameU {
            view_proj: m(vp),
            inv_view_proj: m(inv),
            // cam_pos.w and fill_dir.w carry the ambient occlusion's reach and strength.
            cam_pos: [eye.x as f32, eye.y as f32, eye.z as f32, self.occlusion[1]],
            key_dir: [key.x as f32, key.y as f32, key.z as f32, bg as f32],
            fill_dir: [fill.x as f32, fill.y as f32, fill.z as f32, self.occlusion[0]],
            viewport: [w as f32, h as f32, if self.grid { 1.0 } else { 0.0 }, (pixel / self.scale) as f32],
            origin: [origin.0 as f32, origin.1 as f32, cue[0] as f32, cue[1] as f32],
            // look1.w: the UI scale; the ink width is in UI pixels.
            look: {
                let mut look = self.look;
                look[1][3] = self.scale as f32;
                // Bit 4: the CAD view. Bit 8: see-through. Set here so Host.set_look cannot clear them.
                if self.cad {
                    look[1][1] += 4.0;
                    if self.cad_hidden {
                        look[1][1] += 8.0;
                    }
                }
                look
            },
        }
    }

    /// The views of this frame in physical pixels: the quad view's, or one over the window.
    fn view_frames(&self, w: u32, h: u32) -> Vec<(ViewFrame, [[f32; 4]; 4])> {
        let one = [HostView { rect: [0.0, 0.0, w as f64 / self.scale, h as f64 / self.scale], view_proj: self.view_proj, eye: self.eye, pixel: 0.0, cue: [0.0, 0.0], bg: 0.0 }];
        let views: &[HostView] = if self.views.is_empty() { &one } else { &self.views };
        views
            .iter()
            .map(|v| {
                let s = self.scale;
                let x = (v.rect[0] * s).round().max(0.0) as u32;
                let y = (v.rect[1] * s).round().max(0.0) as u32;
                let vw = ((v.rect[2] * s).round() as u32).min(w.saturating_sub(x)).max(1);
                let vh = ((v.rect[3] * s).round() as u32).min(h.saturating_sub(y)).max(1);
                let frame = self.frame_uniform(v.view_proj, v.eye, vw, vh, (x, y), v.pixel, v.cue, v.bg);
                (ViewFrame { frame, rect: [x, y, vw, vh] }, v.view_proj.as_mat4().to_cols_array_2d())
            })
            .collect()
    }

    /// Draw everything into `view` (w x h physical pixels).
    fn draw_into(&mut self, view: &wgpu::TextureView, w: u32, h: u32) {
        let frames = self.view_frames(w, h);
        // Overlay 3D lines and faces move toward the eye by about two pixels' worth of
        // world: enough to beat the voxel faces they lie on, too little to show what
        // lies behind them (a fixed depth step showed hidden box edges near corners).
        let overlay_views: Vec<_> = frames
            .iter()
            .map(|(f, vp)| {
                let px = f.frame.viewport[3]; // world per physical pixel in an orthographic view, else 0
                let per_dist = 2.0 * 2.0 * (35.0f32.to_radians() * 0.5).tan() / f.rect[3].max(1) as f32;
                let pull = if px > 0.0 { [0.0, 2.0 * px, 0.0, 0.0] } else { [per_dist, 0.0, 0.0, 0.0] };
                (*vp, f.rect[2], f.rect[3], f.frame.cam_pos, pull)
            })
            .collect();
        let frames: Vec<ViewFrame> = frames.into_iter().map(|(f, _)| f).collect();
        let uniforms: Vec<_> = self.models.iter().map(|m| model_uniform(m, true, 16)).collect();
        let scale = self.scale as f32;
        let smooth = self.smooth;
        // The rendered view traces the first view (not for agent pictures).
        let (vp0, eye0) = self.views.first().map_or((self.view_proj, self.eye), |v| (v.view_proj, v.eye));
        let rect0 = frames.first().map(|f| f.rect);
        let Some(g) = &mut self.gfx else { return };
        let mut traced = false;
        if g.tracer.settings.on && !g.overlay.agent {
            if let Some(rect) = rect0 {
                let mut enc = g.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trace") });
                traced = g.tracer.encode(&g.gpu, &mut enc, &self.models, &self.palette, vp0, eye0, rect);
                g.gpu.queue.submit([enc.finish()]);
            }
        }
        g.overlay.prepare(&g.gpu, &mut self.fonts.borrow_mut(), &overlay_views, w, h, scale);
        let overlay = &g.overlay;
        let fx = if smooth { Some(&mut g.postfx) } else { None };
        let tracer = if traced { Some(&g.tracer) } else { None };
        g.renderer.render_with(&g.gpu, view, w, h, &frames, &uniforms, fx, tracer, |pass, i| overlay.draw_3d(pass, i), |pass| overlay.draw_2d(pass));
        g.overlay.trim();
    }

    fn render(&mut self) {
        let Some(g) = &mut self.gfx else { return };
        let Some(surface) = &g.surface else { return };
        let t_acquire = Instant::now();
        let tex = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&g.gpu.device, &g.config);
                if let Some(t) = &mut self.timing {
                    t.no_texture += 1;
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }
            other => {
                if let Some(t) = &mut self.timing {
                    t.no_texture += 1;
                    if t.no_texture <= 3 {
                        eprintln!("rok timing: no surface texture: {other:?}");
                    }
                }
                return;
            }
        };
        let view = tex.texture.create_view(&Default::default());
        let (w, h) = (g.config.width, g.config.height);
        let t_draw = Instant::now();
        self.draw_into(&view, w, h);
        let t_present = Instant::now();
        if let Some(g) = &self.gfx {
            g.gpu.queue.present(tex);
            if let Some(t) = &mut self.timing {
                let (acq, drw, prs) = ((t_draw - t_acquire).as_secs_f64(), (t_present - t_draw).as_secs_f64(), t_present.elapsed().as_secs_f64());
                t.frames += 1;
                t.acquire += acq;
                t.acquire_max = t.acquire_max.max(acq);
                t.draw += drw;
                t.draw_max = t.draw_max.max(drw);
                t.present += prs;
                if let Some(at) = t.input_at.take() {
                    let dt = at.elapsed().as_secs_f64();
                    t.input += dt;
                    t.input_max = t.input_max.max(dt);
                    t.input_n += 1;
                }
                if t.last_print.is_none() {
                    t.last_print = Some(Instant::now());
                }
                t.report();
            }
            // The rendered view adds a sample a frame until it has enough.
            if g.tracer.busy() {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
        }
    }

    /// Screenshots render off-screen, so they work while the window is hidden
    /// or the display sleeps (macOS then hands out no window frames).
    fn shoot(&mut self) {
        if self.shots.is_empty() {
            return;
        }
        let Some(g) = &self.gfx else { return };
        let (w, h, format) = (g.config.width, g.config.height, g.config.format);
        let tex = g.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        if let Some(g) = &mut self.gfx {
            g.tracer.settle();
        }
        self.draw_into(&view, w, h);
        // A screenshot of the rendered view waits for all its samples.
        while self.gfx.as_ref().is_some_and(|g| g.tracer.busy()) {
            self.draw_into(&view, w, h);
        }
        let Some(g) = &self.gfx else { return };
        let rgba = read_back(&g.gpu, &tex, w, h);
        for token in std::mem::take(&mut self.shots) {
            self.queue.push_back(("screenshot", vec![
                ("token", Val::S(token)),
                ("width", Val::F(w as f64)),
                ("height", Val::F(h as f64)),
                ("rgba", Val::Bytes(rgba.clone())),
            ]));
        }
    }

    fn new_tracer(&self, gpu: &Gpu, format: wgpu::TextureFormat) -> Tracer {
        let mut t = Tracer::new(gpu, format);
        t.settings = self.trace_settings;
        t.set_materials(self.materials.clone());
        t
    }

    /// The GPU with no window and no surface: every frame is a screenshot.
    fn start_offscreen(&mut self, width: f64, height: f64, scale: f64) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let gpu = Gpu::new(instance, None);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: (width * scale).round().max(1.0) as u32,
            height: (height * scale).round().max(1.0) as u32,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = Renderer::new(&gpu, format, &self.palette);
        if !self.materials.is_empty() {
            renderer.set_materials(&gpu, &self.materials);
        }
        let overlay = Overlay::new(&gpu, format);
        let postfx = PostFx::new(&gpu, format);
        let tracer = self.new_tracer(&gpu, format);
        self.scale = scale;
        self.gfx = Some(Gfx { gpu, renderer, tracer, overlay, postfx, surface: None, config });
    }

    /// A finished picture (Host.render_picture): every sample, then the
    /// display pass into a texture of its own, read back as a screenshot event.
    fn shoot_picture(&mut self, token: String, w: u32, h: u32, vp: DMat4, eye: DVec3, settings: TraceSettings) {
        let Some(g) = &mut self.gfx else { return };
        let format = g.config.format;
        let view_settings = g.tracer.settings;
        g.tracer.settings = settings;
        g.tracer.restart();
        g.tracer.settle();
        let rect = [0, 0, w, h];
        let mut rounds = 0u32;
        loop {
            let mut enc = g.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("picture") });
            let ok = g.tracer.encode(&g.gpu, &mut enc, &self.models, &self.palette, vp, eye, rect);
            g.gpu.queue.submit([enc.finish()]);
            rounds += 1;
            if rounds.is_multiple_of(16) {
                let _ = g.gpu.device.poll(wgpu::PollType::wait_indefinitely()); // keep the queue short
            }
            if !ok || g.tracer.samples() >= settings.max_samples || rounds > 64 * settings.max_samples {
                break;
            }
        }
        let tex = g.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("picture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = g.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("picture depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::gpu::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let dview = depth.create_view(&Default::default());
        let mut enc = g.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("picture show") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("picture show"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &dview,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            g.tracer.draw(&mut pass);
        }
        g.gpu.queue.submit([enc.finish()]);
        let rgba = read_back(&g.gpu, &tex, w, h);
        // The view's own settings again; its samples start over.
        g.tracer.settings = view_settings;
        g.tracer.restart();
        self.queue.push_back(("screenshot", vec![
            ("token", Val::S(token)),
            ("width", Val::F(w as f64)),
            ("height", Val::F(h as f64)),
            ("rgba", Val::Bytes(rgba)),
        ]));
    }

    /// An agent picture (Host.render_views): its own size and views, scale 1,
    /// only the agent layers. Everything is put back after.
    fn shoot_views(&mut self, token: String, w: u32, h: u32, views: Vec<HostView>, keep_look: bool) {
        let Some(g) = &self.gfx else { return };
        let format = g.config.format;
        let tex = g.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("agent picture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let saved_views = std::mem::replace(&mut self.views, views);
        let saved_scale = std::mem::replace(&mut self.scale, 1.0);
        // Agent pictures keep one look whatever the window shows: the plain one,
        // unless the caller asks for the current look (a script's comparison sheet).
        let plain = if keep_look { self.look } else { [[0.0; 4]; 3] };
        let saved_look = std::mem::replace(&mut self.look, plain);
        if let Some(g) = &mut self.gfx {
            g.overlay.agent = true;
        }
        self.draw_into(&view, w, h);
        if let Some(g) = &mut self.gfx {
            g.overlay.agent = false;
        }
        self.views = saved_views;
        self.scale = saved_scale;
        self.look = saved_look;
        let Some(g) = &self.gfx else { return };
        let rgba = read_back(&g.gpu, &tex, w, h);
        self.queue.push_back(("screenshot", vec![
            ("token", Val::S(token)),
            ("width", Val::F(w as f64)),
            ("height", Val::F(h as f64)),
            ("rgba", Val::Bytes(rgba)),
        ]));
    }

    fn start(&mut self, el: &ActiveEventLoop) {
        // No window tabs: macOS then leaves Show Tab Bar and Show All Tabs out
        // of the View menu (Saved Views sits there now).
        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::ActiveEventLoopExtMacOS;
            el.set_allows_automatic_window_tabbing(false);
        }
        // Zoomed at start: the screen's visible area, the menu
        // bar and the Dock stay; 1440 x 900 is the size when it is zoomed back.
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(1440.0, 900.0))
            .with_maximized(true);
        let window = Arc::new(el.create_window(attrs).expect("window"));
        set_icon(self.icon.as_deref());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(el.owned_display_handle())));
        let surface = instance.create_surface(window.clone()).expect("surface");
        let gpu = Gpu::new(instance, Some(&surface));
        let s = window.inner_size();
        let mut config = surface.get_default_config(&gpu.adapter, s.width.max(1), s.height.max(1)).expect("surface config");
        let caps = surface.get_capabilities(&gpu.adapter);
        config.format = caps.formats.iter().copied().find(|f| *f == wgpu::TextureFormat::Bgra8UnormSrgb).unwrap_or(config.format);
        config.present_mode = wgpu::PresentMode::Fifo;
        // One frame in flight, not two: a frame less between the hand and the picture.
        config.desired_maximum_frame_latency = 1;
        self.immediate_ok = caps.present_modes.contains(&wgpu::PresentMode::Immediate);
        if self.timing.is_some() {
            eprintln!("rok timing: present modes {:?}, immediate while a button is down: {}", caps.present_modes, self.immediate_ok);
        }
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        surface.configure(&gpu.device, &config);
        let renderer = Renderer::new(&gpu, config.format, &self.palette);
        if !self.materials.is_empty() {
            renderer.set_materials(&gpu, &self.materials);
        }
        let overlay = Overlay::new(&gpu, config.format);
        let postfx = PostFx::new(&gpu, config.format);
        let tracer = self.new_tracer(&gpu, config.format);
        self.scale = window.scale_factor();
        self.gfx = Some(Gfx { gpu, renderer, tracer, overlay, postfx, surface: Some(surface), config });
        self.window = Some(window);
        let (lw, lh) = self.logical();
        let scale = self.scale;
        self.emit(Some(el), "start", vec![("width", Val::F(lw)), ("height", Val::F(lh)), ("scale", Val::F(scale))]);
    }
}

fn button_name(b: MouseButton) -> String {
    match b {
        MouseButton::Left => "left".into(),
        MouseButton::Right => "right".into(),
        MouseButton::Middle => "middle".into(),
        MouseButton::Back => "back".into(),
        MouseButton::Forward => "forward".into(),
        MouseButton::Other(n) => format!("button{n}"),
    }
}

fn key_name(k: &Key) -> String {
    match k {
        Key::Character(c) => c.to_lowercase(),
        Key::Named(n) => format!("{n:?}"),
        _ => "Unidentified".into(),
    }
}

impl ApplicationHandler<UserEv> for HostApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_none() {
            self.start(el);
        }
    }

    fn user_event(&mut self, el: &ActiveEventLoop, ev: UserEv) {
        if let Some(t) = &mut self.timing {
            t.event_in();
        }
        match ev {
            UserEv::Menu(id) => self.emit(Some(el), "menu", vec![("id", Val::S(id))]),
            UserEv::Wake => self.emit(Some(el), "wake", vec![]),
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if !self.relative {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            // Queued and added up like the gestures: one per loop turn.
            let fields = vec![("dx", Val::F(delta.0)), ("dy", Val::F(delta.1))];
            if let Some(last) = self.queue.back_mut() {
                if last.0 == "mouse_delta" {
                    for ((_, old), (_, new)) in last.1.iter_mut().zip(fields.iter()) {
                        if let (Val::F(a), Val::F(b)) = (old, new) {
                            *a += *b;
                        }
                    }
                    return;
                }
            }
            self.queue.push_back(("mouse_delta", fields));
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        self.pump(Some(el)); // a mouse move that waited for this turn of the loop
        let mut frame_ms = 0.0;
        if self.want_frame {
            // The frame renders here, in the turn that handled the events; a
            // RedrawRequested would wait for a quiet loop (see apply).
            self.want_frame = false;
            let t0 = Instant::now();
            self.render();
            frame_ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        if let Some(t) = &mut self.timing {
            t.turn_end(frame_ms);
        }
        self.settle();
        if let Some(at) = self.tick_at {
            if Instant::now() >= at {
                self.tick_at = None;
                self.emit(Some(el), "tick", vec![]);
            }
        }
        let wake = match (self.tick_at, self.gesture_until) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        match wake {
            Some(at) => el.set_control_flow(ControlFlow::WaitUntil(at)),
            None => el.set_control_flow(ControlFlow::Wait),
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let sc = self.scale;
        if let Some(t) = &mut self.timing {
            t.event_in();
            if matches!(event, WindowEvent::CursorMoved { .. }) {
                t.moves_in += 1;
            }
        }
        match event {
            WindowEvent::RedrawRequested => {
                self.render();
                self.pump(Some(el));
            }
            WindowEvent::CloseRequested => self.emit(Some(el), "close", vec![]),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.configure();
                let (w, h) = self.logical();
                let scale = self.scale;
                self.emit(Some(el), "resize", vec![("width", Val::F(w)), ("height", Val::F(h)), ("scale", Val::F(scale))]);
            }
            WindowEvent::Focused(f) => self.emit(Some(el), "focus", vec![("focused", Val::B(f))]),
            WindowEvent::ModifiersChanged(m) => {
                let s = m.state();
                self.ctrl = s.control_key();
                self.emit(Some(el), "modifiers", vec![
                    ("shift", Val::B(s.shift_key())),
                    ("ctrl", Val::B(s.control_key())),
                    ("alt", Val::B(s.alt_key())),
                    ("meta", Val::B(s.super_key())),
                ]);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let kind = if event.state == ElementState::Pressed { "key_down" } else { "key_up" };
                let code = match event.physical_key {
                    PhysicalKey::Code(c) => format!("{c:?}"),
                    _ => "Unidentified".into(),
                };
                let text = event.text.as_ref().map(|t| t.to_string()).unwrap_or_default();
                self.emit(Some(el), kind, vec![
                    ("key", Val::S(key_name(&event.key_without_modifiers()))),
                    ("code", Val::S(code)),
                    ("text", Val::S(text)),
                    ("repeat", Val::B(event.repeat)),
                ]);
            }
            WindowEvent::CursorMoved { position, .. } => {
                // The mouse reports up to 120 moves a second and each one ran the
                // whole Python handler and redraw before a frame could show
                // (a stutter in strokes). Moves queue, the newest
                // replaces an older one still waiting, and about_to_wait delivers
                // one per turn of the loop. The stroke samples the way between
                // two positions itself, so no voxel is skipped. Other events
                // still deliver at once, in order behind a waiting move.
                let fields = vec![("x", Val::F(position.x / sc)), ("y", Val::F(position.y / sc))];
                if self.ctrl {
                    self.gesture(); // Ctrl+move orbits
                }
                if let Some(t) = &mut self.timing {
                    if t.input_at.is_none() {
                        t.input_at = Some(Instant::now()); // the oldest move still waiting for its frame
                    }
                }
                if let Some(last) = self.queue.back_mut() {
                    if last.0 == "mouse_move" {
                        last.1 = fields;
                        return;
                    }
                }
                self.queue.push_back(("mouse_move", fields));
            }
            WindowEvent::CursorLeft { .. } => self.emit(Some(el), "mouse_leave", vec![]),
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed {
                    self.buttons += 1;
                    self.set_interacting(true);
                } else {
                    self.buttons = self.buttons.saturating_sub(1);
                }
                let kind = if pressed { "mouse_down" } else { "mouse_up" };
                self.emit(Some(el), kind, vec![("button", Val::S(button_name(button)))]);
                if !pressed {
                    self.settle();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy, precise) = match delta {
                    MouseScrollDelta::PixelDelta(p) => (p.x / sc, p.y / sc, true),
                    MouseScrollDelta::LineDelta(x, y) => (x as f64 * 20.0, y as f64 * 20.0, false),
                };
                self.queue_gesture("scroll", vec![("dx", Val::F(dx)), ("dy", Val::F(dy)), ("precise", Val::B(precise))]);
            }
            WindowEvent::PinchGesture { delta, .. } => self.queue_gesture("pinch", vec![("delta", Val::F(delta))]),
            WindowEvent::RotationGesture { delta, .. } => {
                self.queue_gesture("rotate", vec![("delta", Val::F(delta as f64))])
            }
            _ => {}
        }
    }
}

/// Copy a rendered surface texture into RGBA bytes, top row first.
fn read_back(gpu: &Gpu, tex: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let row = (w * 4).div_ceil(256) * 256;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("screenshot") });
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let bgra = matches!(tex.format(), wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let data = buf.slice(..).get_mapped_range().expect("mapped");
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        let line = &data[y * row as usize..y * row as usize + w as usize * 4];
        for px in line.chunks_exact(4) {
            if bgra {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            } else {
                out.extend_from_slice(px);
            }
        }
    }
    out
}


/// macOS's own switches for two of the items it adds to the Edit menu.
#[cfg(target_os = "macos")]
fn hide_apple_edit_items() {
    use objc2_foundation::{NSString, NSUserDefaults};
    let defaults = NSUserDefaults::standardUserDefaults();
    for key in ["NSDisabledDictationMenuItem", "NSDisabledCharacterPaletteMenuItem"] {
        defaults.setBool_forKey(true, &NSString::from_str(key));
    }
}


/// One item of a menu: (id, label, accelerator, check), or a submenu (label, items).
/// The ids "@hide", "@hide_others" and "@show_all" are the system's own items,
/// with their own keys (Cmd+H, Alt+Cmd+H); macOS does their work.
#[derive(FromPyObject)]
enum MenuEntry {
    Item(String, String, String, i32),
    Sub(String, Vec<Option<MenuEntry>>),
}

#[cfg(feature = "gui")]
fn append_entries(sub: &muda::Submenu, items: Vec<Option<MenuEntry>>) {
    for item in items {
        match item {
            None => {
                let _ = sub.append(&muda::PredefinedMenuItem::separator());
            }
            Some(MenuEntry::Item(id, label, _, _)) if id.starts_with('@') => {
                let text = Some(label.as_str());
                let item = match id.as_str() {
                    "@hide" => muda::PredefinedMenuItem::hide(text),
                    "@hide_others" => muda::PredefinedMenuItem::hide_others(text),
                    "@show_all" => muda::PredefinedMenuItem::show_all(text),
                    _ => continue,
                };
                let _ = sub.append(&item);
            }
            Some(MenuEntry::Item(id, label, accel, check)) => {
                let acc = if accel.is_empty() { None } else { accel.parse::<muda::accelerator::Accelerator>().ok() };
                if check < 0 {
                    let _ = sub.append(&muda::MenuItem::with_id(id, &label, true, acc));
                } else {
                    let _ = sub.append(&muda::CheckMenuItem::with_id(id, &label, true, check > 0, acc));
                }
            }
            Some(MenuEntry::Sub(label, inner)) => {
                let child = muda::Submenu::new(&label, true);
                append_entries(&child, inner);
                let _ = sub.append(&child);
            }
        }
    }
}

/// Show the picture at `path` as the app's icon in the Dock.
#[cfg(target_os = "macos")]
fn set_icon(path: Option<&str>) {
    use objc2::{AnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSString;
    let (Some(path), Some(mtm)) = (path, MainThreadMarker::new()) else { return };
    let image = NSImage::initWithContentsOfFile(NSImage::alloc(), &NSString::from_str(path));
    if let Some(image) = image {
        let app = NSApplication::sharedApplication(mtm);
        unsafe { app.setApplicationIconImage(Some(&image)) };
    }
}

#[cfg(not(target_os = "macos"))]
fn set_icon(_path: Option<&str>) {}
