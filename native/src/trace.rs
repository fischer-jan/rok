//! The rendered view: a progressive voxel path tracer (trace.wgsl).
//!
//! Every visible model goes into one scene volume in world cells, with an 8^3
//! brick occupancy grid for skipping empty space and a list of emitting voxels
//! for direct light. Each frame traces one more sample per pixel and filters
//! the mean; the display pass shows it and writes depth, so the overlays still
//! test against the voxels. While the camera moves, the trace runs at half
//! resolution; once it stops, full resolution adds up to `max_samples` and the
//! GPU rests.

use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::scene::Model;
use bytemuck::{Pod, Zeroable};
use std::time::{Duration, Instant};

/// Most cells in the scene volume (256 MB of 3D texture).
const MAX_CELLS: usize = 256 * 1024 * 1024;
/// Most emitting voxels in the light list; more are thinned out evenly.
const MAX_LIGHTS: usize = 1 << 16;
/// Most pixels traced: a 5K window is traced a little smaller.
const MAX_PIXELS: u64 = 4_200_000;
/// The bloom's blur passes after the downsample: (step, horizontal), ping and
/// pong in turn at a quarter of the size; the glow ends in ping.
const BLOOM_PASSES: [(i32, bool); 4] = [(1, true), (1, false), (2, true), (2, false)];
/// The filter's passes, each twice as wide as the one before.
const DENOISE_STEPS: [i32; 5] = [1, 2, 4, 8, 16];
const DN_STRIDE: u64 = 256;
/// How long after the last camera change the trace stays at half resolution.
const SETTLE: Duration = Duration::from_millis(180);
/// More pixels than this: after the first sample, each frame traces one pixel
/// of every 2 x 2 block, so a frame costs a quarter and the UI stays quick.
const INTERLEAVE_PIXELS: u64 = 1_200_000;
/// After this many samples the filter runs once per full round, not every frame.
const FILTER_EVERY_UNTIL: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TraceU {
    inv_vp: [[f32; 4]; 4],
    vp: [[f32; 4]; 4],
    cam: [f32; 4],
    rect: [f32; 4],
    dims: [u32; 4],
    vol_lo: [i32; 4],
    vol_size: [i32; 4],
    bdim: [i32; 4],
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    ground: [f32; 4],
    backdrop: [f32; 4],
    params: [f32; 4],
    part: [u32; 4],
    look: [f32; 4],
    lamps: [[f32; 4]; 2 * MAX_LAMPS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenoiseU {
    step: i32,
    src: u32,
    dst: u32,
    last: u32,
}

/// Most lamps a picture has (a render preset's lights).
pub const MAX_LAMPS: usize = 4;

/// A lamp: kind 0 is a sun (`v` points toward it, `colour` is its irradiance,
/// `size` its width in degrees); kind 1 is a glowing sphere (`v` is its
/// offset from the scene's middle, `colour` its radiant intensity, `size` its
/// radius in cells). See rok/core/render_presets.py.
#[derive(Clone, Copy, PartialEq)]
pub struct Lamp {
    pub kind: u32,
    pub v: [f32; 3],
    pub colour: [f32; 3],
    pub size: f32,
}

/// What Python sets (Host.set_render, from a render preset).
#[derive(Clone, Copy, PartialEq)]
pub struct Settings {
    pub on: bool,
    pub max_samples: u32,
    pub bounces: u32,
    pub lamps: [Lamp; MAX_LAMPS],
    pub lamp_count: u32,
    /// The sky, linear RGB radiance: the zenith and the horizon (equal: a plain world colour).
    pub sky_top: [f32; 3],
    pub sky_horizon: [f32; 3],
    /// What the camera sees where nothing is, instead of the sky; None: the sky.
    pub backdrop: Option<[f32; 3]>,
    /// The ground's albedo; None: no ground.
    pub ground: Option<[f32; 3]>,
    /// In stops.
    pub exposure: f32,
    /// 0 PBR Neutral, 1 clipped, 2 film, 3 film punchy, 4 film high contrast.
    pub tone: u32,
    pub bloom: f32,
    pub outline: f32,
    pub voxel_grid: f32,
    pub emission: f32,
    pub transparent: bool,
}

impl Default for Settings {
    fn default() -> Settings {
        let off = Lamp { kind: 0, v: [0.0, 0.0, 1.0], colour: [0.0; 3], size: 1.0 };
        let mut lamps = [off; MAX_LAMPS];
        lamps[0] = Lamp { kind: 0, v: [-0.45, -0.6, 0.66], colour: [2.6, 2.42, 2.18], size: 2.5 };
        Settings {
            on: false,
            max_samples: 512,
            bounces: 4,
            lamps,
            lamp_count: 1,
            sky_top: [0.42, 0.55, 0.78],
            sky_horizon: [0.86, 0.88, 0.9],
            backdrop: None,
            ground: Some([0.42, 0.42, 0.42]),
            exposure: 0.0,
            tone: 0,
            bloom: 0.0,
            outline: 0.0,
            voxel_grid: 0.0,
            emission: 1.0,
            transparent: false,
        }
    }
}

struct SceneGpu {
    _vox: wgpu::Texture,
    vox_view: wgpu::TextureView,
    /// The region masks, then the block masks (from `block_at` on).
    occ: wgpu::Buffer,
    block_at: i32,
    lights: wgpu::Buffer,
    /// How many lights the buffer holds room for.
    light_cap: usize,
    lo: [i32; 3],
    size: [i32; 3],
    light_count: i32,
    light_weight: f32,
}

/// The occupancy masks of a volume (x, y, z in C order): a bit per cell in
/// each 4^3 block (bit = x + 4 y + 16 z, low word first), and a bit per
/// block in each 16^3 region, the same way.
pub fn occupancy(vol: &[u8], size: [usize; 3]) -> (Vec<[u32; 2]>, Vec<[u32; 2]>) {
    use rayon::prelude::*;
    let bd = [size[0].div_ceil(4), size[1].div_ceil(4), size[2].div_ceil(4)];
    let mut blocks = vec![[0u32; 2]; bd[0] * bd[1] * bd[2]];
    blocks.par_chunks_mut(bd[1] * bd[2]).enumerate().for_each(|(bx, slab)| {
        for lx in 0..4 {
            let x = bx * 4 + lx;
            if x >= size[0] {
                break;
            }
            for y in 0..size[1] {
                let row = &vol[(x * size[1] + y) * size[2]..(x * size[1] + y + 1) * size[2]];
                for (z, &v) in row.iter().enumerate() {
                    if v != 0 {
                        let bit = lx + 4 * (y & 3) + 16 * (z & 3);
                        slab[(y >> 2) * bd[2] + (z >> 2)][bit >> 5] |= 1 << (bit & 31);
                    }
                }
            }
        }
    });
    let cd = [size[0].div_ceil(16), size[1].div_ceil(16), size[2].div_ceil(16)];
    let mut coarse = vec![[0u32; 2]; cd[0] * cd[1] * cd[2]];
    for bx in 0..bd[0] {
        for by in 0..bd[1] {
            for bz in 0..bd[2] {
                let m = blocks[(bx * bd[1] + by) * bd[2] + bz];
                if m[0] | m[1] != 0 {
                    let bit = (bx & 3) + 4 * (by & 3) + 16 * (bz & 3);
                    coarse[((bx >> 2) * cd[1] + (by >> 2)) * cd[2] + (bz >> 2)][bit >> 5] |= 1 << (bit & 31);
                }
            }
        }
    }
    (blocks, coarse)
}

struct Pixels {
    accum: wgpu::Buffer,
    gbuf: wgpu::Buffer,
    ping: wgpu::Buffer,
    pong: wgpu::Buffer,
    out: wgpu::Buffer,
    cap: u64,
}

pub struct Tracer {
    trace_pipe: wgpu::ComputePipeline,
    denoise_pipe: wgpu::ComputePipeline,
    bloom_down_pipe: wgpu::ComputePipeline,
    bloom_h_pipe: wgpu::ComputePipeline,
    bloom_v_pipe: wgpu::ComputePipeline,
    show_pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    ubuf: wgpu::Buffer,
    dnbuf: wgpu::Buffer,
    mats: wgpu::Buffer,
    scene: Option<SceneGpu>,
    pixels: Option<Pixels>,
    bind: Option<wgpu::BindGroup>,
    pub settings: Settings,
    /// The scene changed since the volume was built (models added, moved, hidden...).
    pub dirty: bool,
    /// The palette or the materials changed.
    mats_dirty: bool,
    /// Changed boxes of the models' voxels, for the next frame (model, origin, dims).
    pending: Vec<(usize, [usize; 3], [usize; 3])>,
    /// The scene on the CPU, for updates by box.
    cpu: Option<CpuScene>,
    materials: Vec<f32>,
    sample: u32,
    key: Vec<u32>,
    moving_until: Instant,
    trace_size: (u32, u32),
    last: Option<(u32, u32)>,
    /// Which quarter of the pixels the next interleaved frame traces (0..4).
    quarter: u32,
}

/// One palette slot's material rows (MaterialTable.channels plus the type):
/// metallic, roughness, transmission, emission, ior, specular, type, 0.
pub const MAT_COLS: usize = 8;

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// The scene volume on the CPU, kept for updates by box.
struct CpuScene {
    vol: Vec<u8>,
    size: [usize; 3],
    blocks: Vec<[u32; 2]>,
    coarse: Vec<[u32; 2]>,
    lights: Vec<[i32; 4]>,
    /// More emitters than the list holds: a change needs a full build.
    thinned: bool,
    emits: [bool; 256],
    /// Where each model sits in the volume (offset, size); None while hidden.
    places: Vec<Option<([usize; 3], [usize; 3])>>,
}

impl CpuScene {
    fn at(&self, x: usize, y: usize, z: usize) -> u8 {
        self.vol[(x * self.size[1] + y) * self.size[2] + z]
    }

    fn block_dims(&self) -> [usize; 3] {
        [self.size[0].div_ceil(4), self.size[1].div_ceil(4), self.size[2].div_ceil(4)]
    }

    fn coarse_dims(&self) -> [usize; 3] {
        [self.size[0].div_ceil(16), self.size[1].div_ceil(16), self.size[2].div_ceil(16)]
    }

    /// Emitting voxels with an open face in the box lo..=hi: the ones that can light anything.
    fn find_lights(&self, lo: [usize; 3], hi: [usize; 3]) -> Vec<[i32; 4]> {
        let size = self.size;
        let mut out = Vec::new();
        if !self.emits.iter().any(|&e| e) {
            return out;
        }
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let v = self.at(x, y, z);
                    if v == 0 || !self.emits[v as usize] {
                        continue;
                    }
                    let open = x == 0 || x + 1 == size[0] || y == 0 || y + 1 == size[1] || z == 0 || z + 1 == size[2]
                        || self.at(x - 1, y, z) == 0 || self.at(x + 1, y, z) == 0 || self.at(x, y - 1, z) == 0
                        || self.at(x, y + 1, z) == 0 || self.at(x, y, z - 1) == 0 || self.at(x, y, z + 1) == 0;
                    if open {
                        out.push([x as i32, y as i32, z as i32, v as i32]);
                    }
                }
            }
        }
        out
    }

    /// Recompute the block masks lo..=hi (block coordinates) and the region
    /// masks above them.
    fn refresh_masks(&mut self, lo: [usize; 3], hi: [usize; 3]) {
        let size = self.size;
        let bd = self.block_dims();
        for bx in lo[0]..=hi[0] {
            for by in lo[1]..=hi[1] {
                for bz in lo[2]..=hi[2] {
                    let mut m = [0u32; 2];
                    for lx in 0..4 {
                        for ly in 0..4 {
                            for lz in 0..4 {
                                let (x, y, z) = (bx * 4 + lx, by * 4 + ly, bz * 4 + lz);
                                if x < size[0] && y < size[1] && z < size[2] && self.at(x, y, z) != 0 {
                                    let bit = lx + 4 * ly + 16 * lz;
                                    m[bit >> 5] |= 1 << (bit & 31);
                                }
                            }
                        }
                    }
                    self.blocks[(bx * bd[1] + by) * bd[2] + bz] = m;
                }
            }
        }
        let cd = self.coarse_dims();
        for cx in lo[0] / 4..=hi[0] / 4 {
            for cy in lo[1] / 4..=hi[1] / 4 {
                for cz in lo[2] / 4..=hi[2] / 4 {
                    let mut m = [0u32; 2];
                    for lx in 0..4 {
                        for ly in 0..4 {
                            for lz in 0..4 {
                                let (bx, by, bz) = (cx * 4 + lx, cy * 4 + ly, cz * 4 + lz);
                                if bx < bd[0] && by < bd[1] && bz < bd[2] {
                                    let b = self.blocks[(bx * bd[1] + by) * bd[2] + bz];
                                    if b[0] | b[1] != 0 {
                                        let bit = lx + 4 * ly + 16 * lz;
                                        m[bit >> 5] |= 1 << (bit & 31);
                                    }
                                }
                            }
                        }
                    }
                    self.coarse[(cx * cd[1] + cy) * cd[2] + cz] = m;
                }
            }
        }
    }
}

impl Tracer {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Tracer {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace"),
            source: wgpu::ShaderSource::Wgsl(include_str!("trace.wgsl").into()),
        });
        let vis = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let tex3 = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex3(1),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, false),
                storage(7, false),
                storage(8, false),
                storage(9, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: vis,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<DenoiseU>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trace"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let trace_pipe = compute("cs_trace");
        let denoise_pipe = compute("cs_denoise");
        let bloom_down_pipe = compute("cs_bloom_down");
        let bloom_h_pipe = compute("cs_bloom_h");
        let bloom_v_pipe = compute("cs_bloom_v");
        let show_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trace show"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_show"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_show"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trace u"),
            size: std::mem::size_of::<TraceU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dnbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoise u"),
            size: DN_STRIDE * (DENOISE_STEPS.len() + 1 + BLOOM_PASSES.len()) as u64,  // the filter, the bloom's downsample, its blurs
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trace materials"),
            size: 256 * 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Tracer {
            trace_pipe,
            denoise_pipe,
            bloom_down_pipe,
            bloom_h_pipe,
            bloom_v_pipe,
            show_pipe,
            bgl,
            ubuf,
            dnbuf,
            mats,
            scene: None,
            pixels: None,
            bind: None,
            settings: Settings::default(),
            dirty: true,
            mats_dirty: false,
            pending: Vec::new(),
            cpu: None,
            materials: Vec::new(),
            sample: 0,
            key: Vec::new(),
            moving_until: Instant::now(),
            trace_size: (0, 0),
            last: None,
            quarter: 0,
        }
    }

    /// The materials' rows (Host.set_materials); materials_changed applies them.
    pub fn set_materials(&mut self, rows: Vec<f32>) {
        self.materials = rows;
    }

    /// True while more samples would change the picture.
    pub fn busy(&self) -> bool {
        self.settings.on && (self.sample < self.settings.max_samples || Instant::now() < self.moving_until)
    }

    pub fn samples(&self) -> u32 {
        self.sample
    }

    /// Start the samples over (new settings).
    pub fn restart(&mut self) {
        self.sample = 0;
        self.quarter = 0;
    }

    /// No half-size frames: a screenshot traces at full size from the start.
    pub fn settle(&mut self) {
        self.last = None;
        self.moving_until = Instant::now();
    }

    fn material_rows(&self, palette: &[u8]) -> Vec<[f32; 4]> {
        let mut rows = vec![[0.0f32; 4]; 256 * 3];
        for i in 0..256 {
            let p = &palette[i * 4..i * 4 + 4];
            let m = if self.materials.len() >= (i + 1) * MAT_COLS {
                &self.materials[i * MAT_COLS..(i + 1) * MAT_COLS]
            } else {
                &[0.0, 0.9, 0.0, 0.0, 1.45, 0.5, 0.0, 0.0][..]
            };
            let emission = if m[6] as i32 == 3 { m[3].max(0.0) } else { 0.0 };
            rows[i * 3] = [srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2]), emission];
            rows[i * 3 + 1] = [m[0], m[1], m[2], m[4].max(1.0)];
            rows[i * 3 + 2] = [m[5], m[6], 0.0, 0.0];
        }
        rows
    }

    /// Build the scene volume from the visible models, the palette and the materials.
    fn build(&mut self, gpu: &Gpu, models: &[Model], palette: &[u8]) {
        self.dirty = false;
        self.mats_dirty = false;
        self.pending.clear();
        self.bind = None;
        self.scene = None;
        self.cpu = None;
        let rows = self.material_rows(palette);
        gpu.queue.write_buffer(&self.mats, 0, bytemuck::cast_slice(&rows));
        let corner = |m: &Model| {
            let c = m.center - m.half();
            [c.x.round() as i32, c.y.round() as i32, c.z.round() as i32]
        };
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for m in models.iter().filter(|m| !m.hidden) {
            let c = corner(m);
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k] + m.size[k] as i32);
            }
        }
        if lo[0] == i32::MAX {
            return; // nothing shows
        }
        let size = [(hi[0] - lo[0]) as usize, (hi[1] - lo[1]) as usize, (hi[2] - lo[2]) as usize];
        if size.iter().product::<usize>() > MAX_CELLS || size.iter().any(|&s| s > 2048) {
            eprintln!("rok: the scene is too big to render ({size:?} cells)");
            return;
        }
        let mut vol = vec![0u8; size[0] * size[1] * size[2]];
        let mut places = Vec::with_capacity(models.len());
        for m in models {
            if m.hidden {
                places.push(None);
                continue;
            }
            let c = corner(m);
            let o = [(c[0] - lo[0]) as usize, (c[1] - lo[1]) as usize, (c[2] - lo[2]) as usize];
            places.push(Some((o, m.size)));
            let s = m.size;
            for x in 0..s[0] {
                for y in 0..s[1] {
                    let src = (x * s[1] + y) * s[2];
                    let dst = ((o[0] + x) * size[1] + o[1] + y) * size[2] + o[2];
                    for z in 0..s[2] {
                        let v = m.voxels[src + z];
                        if v != 0 {
                            vol[dst + z] = v;
                        }
                    }
                }
            }
        }
        let (blocks, coarse) = occupancy(&vol, size);
        let mut emits = [false; 256];
        for (i, e) in emits.iter_mut().enumerate() {
            *e = rows[i * 3][3] > 0.0;
        }
        let mut cpu = CpuScene { vol, size, blocks, coarse, lights: Vec::new(), thinned: false, emits, places };
        cpu.lights = cpu.find_lights([0, 0, 0], [size[0] - 1, size[1] - 1, size[2] - 1]);
        let total = cpu.lights.len();
        if total > MAX_LIGHTS {
            let stride = total.div_ceil(MAX_LIGHTS);
            cpu.lights = cpu.lights.iter().copied().step_by(stride).collect();
            cpu.thinned = true;
        }
        // Each listed emitter stands for this many (more than one when thinned out).
        let light_weight = if cpu.lights.is_empty() { 0.0 } else { total as f32 / cpu.lights.len() as f32 };
        let lights = Self::light_buffer(gpu, &cpu.lights);
        let ext = wgpu::Extent3d { width: size[2] as u32, height: size[1] as u32, depth_or_array_layers: size[0] as u32 };
        let vt = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trace voxels"),
            size: ext,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            vt.as_image_copy(),
            &cpu.vol,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[2] as u32), rows_per_image: Some(size[1] as u32) },
            ext,
        );
        let block_at = cpu.coarse.len() as i32;
        let mut occ = cpu.coarse.clone();
        occ.extend_from_slice(&cpu.blocks);
        let occ = wgpu::util::DeviceExt::create_buffer_init(&gpu.device, &wgpu::util::BufferInitDescriptor {
            label: Some("trace occupancy"),
            contents: bytemuck::cast_slice(&occ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.scene = Some(SceneGpu {
            vox_view: vt.create_view(&Default::default()),
            _vox: vt,
            occ,
            block_at,
            light_cap: cpu.lights.len().max(1),
            lights,
            lo,
            size: [size[0] as i32, size[1] as i32, size[2] as i32],
            light_count: cpu.lights.len() as i32,
            light_weight,
        });
        self.cpu = Some(cpu);
    }

    fn light_buffer(gpu: &Gpu, lights: &[[i32; 4]]) -> wgpu::Buffer {
        let mut data = lights.to_vec();
        if data.is_empty() {
            data.push([0; 4]);
        }
        // Room to grow, so a few new emitters do not need a new buffer.
        let cap = (data.len() * 5 / 4 + 64).max(data.len());
        data.resize(cap, [0; 4]);
        wgpu::util::DeviceExt::create_buffer_init(&gpu.device, &wgpu::util::BufferInitDescriptor {
            label: Some("trace lights"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// A model's voxels changed in a box (Host.update_region): the scene takes
    /// the change at the next frame without a full build.
    pub fn region(&mut self, idx: usize, origin: [usize; 3], dims: [usize; 3]) {
        if !self.dirty {
            self.pending.push((idx, origin, dims));
        }
    }

    /// The palette or the materials changed: new material rows, and a full
    /// build only when other slots give light now.
    pub fn materials_changed(&mut self) {
        self.mats_dirty = true;
    }

    /// Bring the scene up to date: a full build, or the changed boxes only.
    fn update(&mut self, gpu: &Gpu, models: &[Model], palette: &[u8]) {
        if !self.dirty && self.mats_dirty {
            self.mats_dirty = false;
            let rows = self.material_rows(palette);
            gpu.queue.write_buffer(&self.mats, 0, bytemuck::cast_slice(&rows));
            let same = self.cpu.as_ref().is_some_and(|c| (0..256).all(|i| c.emits[i] == (rows[i * 3][3] > 0.0)));
            if !same {
                self.dirty = true;
            }
            self.restart();
        }
        if !self.dirty && !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            for (idx, origin, dims) in pending {
                if !self.apply_region(gpu, models, idx, origin, dims) {
                    self.dirty = true;
                    break;
                }
            }
            self.restart();
        }
        if self.dirty {
            self.build(gpu, models, palette);
            self.restart();
        }
    }

    /// Copy one changed box of model `idx` into the scene and upload what it
    /// touches. False when only a full build will do (a hidden or moved model,
    /// another model in the box, thinned-out lights).
    fn apply_region(&mut self, gpu: &Gpu, models: &[Model], idx: usize, origin: [usize; 3], dims: [usize; 3]) -> bool {
        let (Some(cpu), Some(scene)) = (self.cpu.as_mut(), self.scene.as_mut()) else { return false };
        let Some(m) = models.get(idx) else { return false };
        let Some(Some((off, msize))) = cpu.places.get(idx).copied() else { return false };
        if msize != m.size || m.hidden || cpu.thinned || dims.contains(&0) {
            return false;
        }
        let lo = [off[0] + origin[0], off[1] + origin[1], off[2] + origin[2]];
        let hi = [lo[0] + dims[0] - 1, lo[1] + dims[1] - 1, lo[2] + dims[2] - 1];
        // Another model in the box: which voxel shows is not this one's to say.
        for (j, p) in cpu.places.iter().enumerate() {
            if let Some((o, s)) = p {
                if j != idx && (0..3).all(|k| o[k] <= hi[k] && lo[k] < o[k] + s[k]) {
                    return false;
                }
            }
        }
        let size = cpu.size;
        let mut boxed = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let src = ((origin[0] + x) * msize[1] + origin[1] + y) * msize[2] + origin[2];
                let dst = ((lo[0] + x) * size[1] + lo[1] + y) * size[2] + lo[2];
                let row = &m.voxels[src..src + dims[2]];
                cpu.vol[dst..dst + dims[2]].copy_from_slice(row);
                boxed.extend_from_slice(row);
            }
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &scene._vox,
                mip_level: 0,
                origin: wgpu::Origin3d { x: lo[2] as u32, y: lo[1] as u32, z: lo[0] as u32 },
                aspect: wgpu::TextureAspect::All,
            },
            &boxed,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[2] as u32), rows_per_image: Some(dims[1] as u32) },
            wgpu::Extent3d { width: dims[2] as u32, height: dims[1] as u32, depth_or_array_layers: dims[0] as u32 },
        );
        // The masks of the blocks and regions the box touches.
        let (b_lo, b_hi) = ([lo[0] / 4, lo[1] / 4, lo[2] / 4], [hi[0] / 4, hi[1] / 4, hi[2] / 4]);
        cpu.refresh_masks(b_lo, b_hi);
        let bd = cpu.block_dims();
        let cd = cpu.coarse_dims();
        for bx in b_lo[0]..=b_hi[0] {
            for by in b_lo[1]..=b_hi[1] {
                let first = (bx * bd[1] + by) * bd[2] + b_lo[2];
                let run = &cpu.blocks[first..=first + b_hi[2] - b_lo[2]];
                let at = (scene.block_at as usize + first) as u64 * 8;
                gpu.queue.write_buffer(&scene.occ, at, bytemuck::cast_slice(run));
            }
        }
        let (c_lo, c_hi) = ([b_lo[0] / 4, b_lo[1] / 4, b_lo[2] / 4], [b_hi[0] / 4, b_hi[1] / 4, b_hi[2] / 4]);
        for cx in c_lo[0]..=c_hi[0] {
            for cy in c_lo[1]..=c_hi[1] {
                let first = (cx * cd[1] + cy) * cd[2] + c_lo[2];
                let run = &cpu.coarse[first..=first + c_hi[2] - c_lo[2]];
                gpu.queue.write_buffer(&scene.occ, first as u64 * 8, bytemuck::cast_slice(run));
            }
        }
        // The lights in and around the box: a voxel next to it may have lost
        // or gained an open face.
        if cpu.emits.iter().any(|&e| e) {
            let e_lo = [lo[0].saturating_sub(1), lo[1].saturating_sub(1), lo[2].saturating_sub(1)];
            let e_hi = [(hi[0] + 1).min(size[0] - 1), (hi[1] + 1).min(size[1] - 1), (hi[2] + 1).min(size[2] - 1)];
            let inside = |l: &[i32; 4]| (0..3).all(|k| l[k] as usize >= e_lo[k] && l[k] as usize <= e_hi[k]);
            cpu.lights.retain(|l| !inside(l));
            let found = cpu.find_lights(e_lo, e_hi);
            cpu.lights.extend(found);
            if cpu.lights.len() > MAX_LIGHTS {
                return false;
            }
            if cpu.lights.len() > scene.light_cap {
                scene.lights = Self::light_buffer(gpu, &cpu.lights);
                scene.light_cap = cpu.lights.len() * 5 / 4 + 64;
                self.bind = None;
            } else {
                gpu.queue.write_buffer(&scene.lights, 0, bytemuck::cast_slice(&cpu.lights));
            }
            scene.light_count = cpu.lights.len() as i32;
            scene.light_weight = if cpu.lights.is_empty() { 0.0 } else { 1.0 };
        }
        true
    }

    fn ensure_pixels(&mut self, gpu: &Gpu, n: u64) {
        if self.pixels.as_ref().is_some_and(|p| p.cap >= n) {
            return;
        }
        let make = |label, bytes: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes * n).max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        self.pixels = Some(Pixels {
            accum: make("trace accum", 16),
            gbuf: make("trace gbuf", 16),
            ping: make("trace ping", 8),
            pong: make("trace pong", 8),
            out: make("trace out", 8),
            cap: n,
        });
        self.bind = None;
    }

    fn bind_group(&mut self, gpu: &Gpu) {
        if self.bind.is_some() {
            return;
        }
        let (Some(s), Some(p)) = (&self.scene, &self.pixels) else { return };
        fn e(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        self.bind = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace"),
            layout: &self.bgl,
            entries: &[
                e(0, &self.ubuf),
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&s.vox_view) },
                e(2, &s.occ),
                e(3, &self.mats),
                e(4, &s.lights),
                e(5, &p.accum),
                e(6, &p.gbuf),
                e(7, &p.ping),
                e(8, &p.pong),
                e(9, &p.out),
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.dnbuf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<DenoiseU>() as u64),
                    }),
                },
            ],
        }));
    }

    /// Trace one more sample for the view `rect` (physical pixels) and filter
    /// it. Returns false when there is nothing to show (no scene).
    pub fn encode(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        models: &[Model],
        palette: &[u8],
        vp: glam::DMat4,
        eye: glam::DVec3,
        rect: [u32; 4],
    ) -> bool {
        self.update(gpu, models, palette);
        if self.scene.is_none() {
            return false;
        }
        let now = Instant::now();
        // The camera or the window changed: start over, at half size for a moment.
        let inv = vp.inverse();
        let mut key: Vec<u32> = inv.to_cols_array().iter().map(|v| (*v as f32).to_bits()).collect();
        key.extend_from_slice(&rect);
        if key != self.key {
            self.key = key;
            if self.last.is_some() {
                self.moving_until = now + SETTLE;
            }
            self.restart();
        }
        let (w, h) = (rect[2].max(1), rect[3].max(1));
        let full = {
            let px = w as u64 * h as u64;
            if px > MAX_PIXELS { (MAX_PIXELS as f64 / px as f64).sqrt() } else { 1.0 }
        };
        let scale = if now < self.moving_until { 0.5 * full } else { full };
        let size = (((w as f64 * scale).round() as u32).max(1), ((h as f64 * scale).round() as u32).max(1));
        if size != self.trace_size {
            self.trace_size = size;
            self.restart();
        }
        self.last = Some(size);
        self.ensure_pixels(gpu, size.0 as u64 * size.1 as u64);
        self.bind_group(gpu);
        let st = self.settings;
        let s = self.scene.as_ref().unwrap();
        let m = |m: glam::DMat4| m.as_mat4().to_cols_array_2d();
        let flags = st.ground.is_some() as u32 | ((st.transparent as u32) << 1) | ((st.backdrop.is_some() as u32) << 2);
        // Interleave on a big view once every pixel has its first sample.
        let interleave = self.sample > 0 && size.0 as u64 * size.1 as u64 > INTERLEAVE_PIXELS;
        let stride = if interleave { 2 } else { 1 };
        let q = if interleave { self.quarter } else { 0 };
        // The lamps in the shader's terms: a sun's direction and the cosine of
        // its half width; a sphere's world position (the volume's middle plus
        // its offset) and radius.
        let mid = [
            s.lo[0] as f32 + s.size[0] as f32 * 0.5,
            s.lo[1] as f32 + s.size[1] as f32 * 0.5,
            s.lo[2] as f32 + s.size[2] as f32 * 0.5,
        ];
        let mut lamps = [[0.0f32; 4]; 2 * MAX_LAMPS];
        for (k, l) in st.lamps.iter().take(st.lamp_count as usize).enumerate() {
            if l.kind == 0 {
                let d = glam::Vec3::from(l.v).normalize_or(glam::Vec3::Z);
                lamps[2 * k] = [d.x, d.y, d.z, 0.0];
                lamps[2 * k + 1] = [l.colour[0], l.colour[1], l.colour[2], (l.size * 0.5).to_radians().cos()];
            } else {
                lamps[2 * k] = [mid[0] + l.v[0], mid[1] + l.v[1], mid[2] + l.v[2], 1.0];
                lamps[2 * k + 1] = [l.colour[0], l.colour[1], l.colour[2], l.size.max(0.1)];
            }
        }
        let ground = st.ground.unwrap_or([0.0; 3]);
        let backdrop = st.backdrop.unwrap_or([0.0; 3]);
        let u = TraceU {
            inv_vp: m(inv),
            vp: m(vp),
            cam: [eye.x as f32, eye.y as f32, eye.z as f32, st.voxel_grid],
            rect: [rect[0] as f32, rect[1] as f32, w as f32, h as f32],
            dims: [size.0, size.1, self.sample.min(st.max_samples.saturating_sub(1)), flags],
            vol_lo: [s.lo[0], s.lo[1], s.lo[2], s.light_count],
            vol_size: [s.size[0], s.size[1], s.size[2], 1],
            bdim: [s.block_at, 0, 0, 0],
            sky_top: [st.sky_top[0], st.sky_top[1], st.sky_top[2], 1.0],
            sky_horizon: [st.sky_horizon[0], st.sky_horizon[1], st.sky_horizon[2], 1.0],
            ground: [ground[0], ground[1], ground[2], s.lo[2] as f32],
            backdrop: [backdrop[0], backdrop[1], backdrop[2], 0.0],
            params: [2f32.powf(st.exposure), st.bounces as f32, s.light_weight, st.emission],
            part: [stride, q % 2, q / 2, 0],
            look: [st.lamp_count as f32, st.tone as f32, st.bloom, st.outline],
            lamps,
        };
        gpu.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        let Some(bind) = &self.bind else { return false };
        let groups = (size.0.div_ceil(8), size.1.div_ceil(8), 1);
        if self.sample < st.max_samples {
            let tg = (size.0.div_ceil(8 * stride), size.1.div_ceil(8 * stride), 1);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("trace"), timestamp_writes: None });
            pass.set_pipeline(&self.trace_pipe);
            pass.set_bind_group(0, bind, &[0]);
            pass.dispatch_workgroups(tg.0, tg.1, tg.2);
            drop(pass);
            // A round is done when every pixel has one more sample.
            let round_done = !interleave || self.quarter == 3;
            self.quarter = if interleave { (self.quarter + 1) % 4 } else { 0 };
            // The filter: from the mean through ping and pong into out. Later it
            // runs once a round: a quarter of the pixels changes the picture little.
            let filter = self.sample < FILTER_EVERY_UNTIL || round_done;
            let n = DENOISE_STEPS.len();
            for (k, step) in DENOISE_STEPS.iter().enumerate() {
                let d = DenoiseU {
                    step: *step,
                    src: if k == 0 { 0 } else if k % 2 == 1 { 1 } else { 2 },
                    dst: if k + 1 == n { 3 } else if k % 2 == 0 { 1 } else { 2 },
                    last: (k + 1 == n) as u32,
                };
                gpu.queue.write_buffer(&self.dnbuf, DN_STRIDE * k as u64, bytemuck::bytes_of(&d));
            }
            let n = if !filter || std::env::var("ROK_TRACE_NO_FILTER").is_ok() { 0 } else { n };
            for k in 0..n {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("denoise"), timestamp_writes: None });
                pass.set_pipeline(&self.denoise_pipe);
                pass.set_bind_group(0, bind, &[(DN_STRIDE * k as u64) as u32]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            // The bloom: the bright parts of out, smaller and blurred, into
            // ping (free again once the filter is done with it).
            if filter && st.bloom > 0.0 {
                let bg = (size.0.div_ceil(4).div_ceil(8), size.1.div_ceil(4).div_ceil(8), 1);
                let base = DENOISE_STEPS.len();
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bloom"), timestamp_writes: None });
                pass.set_pipeline(&self.bloom_down_pipe);
                pass.set_bind_group(0, bind, &[(DN_STRIDE * base as u64) as u32]);
                pass.dispatch_workgroups(bg.0, bg.1, bg.2);
                for (k, (step, horizontal)) in BLOOM_PASSES.iter().enumerate() {
                    let d = DenoiseU { step: *step, src: if k % 2 == 0 { 1 } else { 2 }, dst: if k % 2 == 0 { 2 } else { 1 }, last: 0 };
                    let slot = base + 1 + k;
                    gpu.queue.write_buffer(&self.dnbuf, DN_STRIDE * slot as u64, bytemuck::bytes_of(&d));
                    pass.set_pipeline(if *horizontal { &self.bloom_h_pipe } else { &self.bloom_v_pipe });
                    pass.set_bind_group(0, bind, &[(DN_STRIDE * slot as u64) as u32]);
                    pass.dispatch_workgroups(bg.0, bg.1, bg.2);
                }
            }
            if round_done {
                self.sample += 1;
            }
        }
        true
    }

    /// Show the traced picture in the view (inside a render pass with depth).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(bind) = &self.bind {
            pass.set_pipeline(&self.show_pipe);
            pass.set_bind_group(0, bind, &[0]);
            pass.draw(0..3, 0..1);
        }
    }
}
