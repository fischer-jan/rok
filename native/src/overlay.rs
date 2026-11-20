//! Overlays and UI for the app host, in named layers.
//!
//! Python sends each layer as flat f32 lists (see the *_F constants) plus text
//! runs and icon runs. A layer stays until Python replaces or clears it, so a
//! big selection surface is uploaded once, not every frame. Layers draw in
//! order of (z, name): their 3D parts in the 3D pass (against the voxel depth,
//! then on top), their 2D parts in the UI pass (triangles, rects, lines, icons,
//! text; one layer after the other, so a higher layer covers a lower one's text).

use crate::gpu::{Gpu, DEPTH_FORMAT};
use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas,
    TextBounds, TextRenderer, Viewport,
};
use std::collections::HashMap;

/// Floats per item, as Python packs them.
pub const LINE3_F: usize = 11; // x0 y0 z0 x1 y1 z1 r g b a width
pub const LINE2_F: usize = 9; // x0 y0 x1 y1 r g b a width
pub const RECT_F: usize = 10; // x y w h r g b a radius stroke
pub const TRI3_F: usize = 7; // per vertex: x y z r g b a
pub const TRI2_F: usize = 6; // per vertex: x y r g b a
const ICON_F: usize = 13; // x y w h u0 v0 u1 v1 r g b a mode

const ATLAS: u32 = 4096; // icons, the welcome picture and every shader thumbnail (2048 ran out)
const U_STRIDE: u64 = 256; // uniform slots: 0 for the 2D UI, 1 + i for view i
/// Distinct shifts the layers may carry on the GPU (set_layer_offset). Each
/// distinct offset has its own uniform slot per view, after the views':
/// 1 + MAX_VIEWS * (1 + k) + view, holding the view-projection times the shift
/// and the eye moved back by it; layers with the same offset share a slot. A
/// shift then costs no upload: a big selection's moving copy, and every skin block and preview sent in grid space with the
/// model's origin as its offset (with four slots and dozens of
/// block layers, most blocks drew at the grid's zero corner).
const MAX_SHIFTED: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayU {
    view_proj: [[f32; 4]; 4],
    screen: [f32; 4],
    /// xyz: the view's eye. 3D lines and faces move toward it so they win against
    /// the voxel faces they lie on, by `pull`: x = a share of their distance
    /// (perspective), y = world units (orthographic).
    eye: [f32; 4],
    pull: [f32; 4],
}

/// One view for the overlay: view-projection, size in physical pixels, eye, pull.
pub type OverlayView = ([[f32; 4]; 4], u32, u32, [f32; 4], [f32; 4]);

/// One text run: logical pixels, top left of the text box.
#[derive(Clone, Debug)]
pub struct TextRun {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub color: [u8; 4],
    pub text: String,
}

/// One icon or image from the atlas: logical pixels, sRGB tint 0..1.
#[derive(Clone, Debug)]
pub struct IconRun {
    pub name: String,
    pub rect: [f32; 4],
    pub tint: [f32; 4],
    pub image: bool,
}

/// One layer's content.
#[derive(Default, Clone)]
pub struct DrawList {
    pub lines3_depth: Vec<f32>,
    pub lines3_top: Vec<f32>,
    pub tris3_depth: Vec<f32>,
    pub tris3_top: Vec<f32>,
    pub tris2: Vec<f32>,
    pub rects: Vec<f32>,
    pub lines2: Vec<f32>,
    pub icons: Vec<IconRun>,
    pub texts: Vec<TextRun>,
}

struct Batch {
    buf: Option<wgpu::Buffer>,
    cap: u64,
    count: u32,
    instanced: bool,
}

impl Batch {
    fn new(instanced: bool) -> Batch {
        Batch { buf: None, cap: 0, count: 0, instanced }
    }

    fn upload(&mut self, gpu: &Gpu, data: &[f32], floats: usize) {
        self.count = (data.len() / floats) as u32;
        if self.count == 0 {
            return;
        }
        let bytes = (self.count as usize * floats * 4) as u64;
        if self.buf.is_none() || self.cap < bytes {
            let cap = bytes.next_power_of_two().max(4096);
            self.buf = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay items"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cap = cap;
        }
        gpu.queue.write_buffer(self.buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&data[..self.count as usize * floats]));
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(pipe);
        pass.set_vertex_buffer(0, self.buf.as_ref().unwrap().slice(..));
        if self.instanced {
            pass.draw(0..6, 0..self.count);
        } else {
            pass.draw(0..self.count, 0..1);
        }
    }
}

struct Layer {
    z: i32,
    list: DrawList,
    dirty: bool,
    offset: [f32; 3],     // world units, applied on the GPU
    slot: Option<usize>,  // its shifted uniform slot this frame, when offset is not zero
    line3_depth: Batch,
    line3_top: Batch,
    tri3_depth: Batch,
    tri3_top: Batch,
    tri2: Batch,
    rect: Batch,
    line2: Batch,
    icon: Batch,
    text: TextRenderer,
}

/// A shelf-packed texture of icons and images, looked up by name.
struct Atlas {
    tex: wgpu::Texture,
    bg: wgpu::BindGroup,
    slots: HashMap<String, (u32, u32, u32, u32)>, // x, y, w, h in texels
    x: u32,
    y: u32,
    row: u32,
}

impl Atlas {
    fn uv(&self, name: &str) -> Option<[f32; 4]> {
        let (x, y, w, h) = *self.slots.get(name)?;
        let s = ATLAS as f32;
        Some([x as f32 / s, y as f32 / s, (x + w) as f32 / s, (y + h) as f32 / s])
    }

    fn add(&mut self, gpu: &Gpu, name: &str, w: u32, h: u32, rgba: &[u8]) -> bool {
        let place = match self.slots.get(name) {
            Some(&(x, y, sw, sh)) if sw == w && sh == h => (x, y),
            _ => {
                // One texel of air around each image, so linear filtering never bleeds.
                if self.x + w + 1 > ATLAS {
                    self.x = 0;
                    self.y += self.row + 1;
                    self.row = 0;
                }
                if w + 1 > ATLAS || self.y + h + 1 > ATLAS {
                    return false;
                }
                let p = (self.x, self.y);
                self.x += w + 1;
                self.row = self.row.max(h);
                p
            }
        };
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: place.0, y: place.1, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.slots.insert(name.to_string(), (place.0, place.1, w, h));
        true
    }
}

/// The name prefix of the layers only agent pictures draw.
pub const AGENT: &str = "agent:";

pub struct Overlay {
    ubuf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    line3_depth: wgpu::RenderPipeline,
    line3_top: wgpu::RenderPipeline,
    tri3_depth: wgpu::RenderPipeline,
    tri3_top: wgpu::RenderPipeline,
    tri2: wgpu::RenderPipeline,
    line2: wgpu::RenderPipeline,
    rect: wgpu::RenderPipeline,
    icon: wgpu::RenderPipeline,
    atlas: Atlas,
    layers: HashMap<String, Layer>,
    order: Vec<String>,
    // Agent pictures (Host.render_views) draw only the "agent:" layers; the
    // window never draws them.
    pub agent: bool,
    // Text
    swash: SwashCache,
    viewport: Viewport,
    text_atlas: TextAtlas,
    buffers: HashMap<(String, u32), Buffer>,
}

fn depth_state(cmp: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(cmp),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

impl Overlay {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Overlay {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<OverlayU>() as u64),
                },
                count: None,
            }],
        });
        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay u"),
            size: U_STRIDE * (1 + (crate::gpu::MAX_VIEWS * (1 + MAX_SHIFTED)) as u64),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ubuf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<OverlayU>() as u64),
                }),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let icon_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay icons"),
            bind_group_layouts: &[Some(&bgl), Some(&atlas_bgl)],
            immediate_size: 0,
        });
        let make = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: &str,
                    attrs: &[wgpu::VertexAttribute],
                    floats: usize,
                    step: wgpu::VertexStepMode,
                    cmp: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout { array_stride: (floats * 4) as u64, step_mode: step, attributes: attrs })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(depth_state(cmp)),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::CompareFunction::{Always, LessEqual};
        use wgpu::VertexStepMode::{Instance, Vertex};
        let l3 = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32];
        let l2 = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32];
        let lr = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x2];
        let t3 = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let t2 = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
        let ic = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32];
        let line3_depth = make("line3 depth", &layout, "vs_line3", "fs_flat", &l3, LINE3_F, Instance, LessEqual);
        let line3_top = make("line3 top", &layout, "vs_line3", "fs_flat", &l3, LINE3_F, Instance, Always);
        let tri3_depth = make("tris3 depth", &layout, "vs_tris3", "fs_flat", &t3, TRI3_F, Vertex, LessEqual);
        let tri3_top = make("tris3 top", &layout, "vs_tris3", "fs_flat", &t3, TRI3_F, Vertex, Always);
        let tri2 = make("tris2", &layout, "vs_tris2", "fs_flat", &t2, TRI2_F, Vertex, Always);
        let line2 = make("line2", &layout, "vs_line2", "fs_flat", &l2, LINE2_F, Instance, Always);
        let rect = make("rect", &layout, "vs_rect", "fs_rect", &lr, RECT_F, Instance, Always);
        let icon = make("icon", &icon_layout, "vs_icon", "fs_icon", &ic, ICON_F, Instance, Always);

        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("icon atlas"),
            size: wgpu::Extent3d { width: ATLAS, height: ATLAS, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("icon atlas"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let atlas_view = atlas_tex.create_view(&Default::default());
        let atlas_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atlas"),
            layout: &atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let text_atlas = TextAtlas::new(device, &gpu.queue, &cache, format);
        Overlay {
            ubuf,
            bg,
            line3_depth,
            line3_top,
            tri3_depth,
            tri3_top,
            tri2,
            line2,
            rect,
            icon,
            atlas: Atlas { tex: atlas_tex, bg: atlas_bg, slots: HashMap::new(), x: 0, y: 0, row: 0 },
            layers: HashMap::new(),
            agent: false,
            order: Vec::new(),
            swash: SwashCache::new(),
            viewport,
            text_atlas,
            buffers: HashMap::new(),
        }
    }

    /// Put an image into the atlas under `name` (RGBA, straight alpha, sRGB).
    pub fn add_image(&mut self, gpu: &Gpu, name: &str, w: u32, h: u32, rgba: &[u8]) {
        if !self.atlas.add(gpu, name, w, h, rgba) {
            eprintln!("rok: the icon atlas is full; {name} ({w} x {h}) is left out");
        }
        for layer in self.layers.values_mut() {
            if layer.list.icons.iter().any(|i| i.name == name) {
                layer.dirty = true;
            }
        }
    }

    pub fn set_layer(&mut self, gpu: &Gpu, name: &str, z: i32, list: DrawList) {
        match self.layers.get_mut(name) {
            Some(layer) => {
                layer.z = z;
                layer.list = list;
                layer.dirty = true;
            }
            None => {
                let text = TextRenderer::new(
                    &mut self.text_atlas,
                    &gpu.device,
                    wgpu::MultisampleState::default(),
                    Some(depth_state(wgpu::CompareFunction::Always)),
                );
                self.layers.insert(
                    name.to_string(),
                    Layer {
                        z,
                        list,
                        dirty: true,
                        offset: [0.0; 3],
                        slot: None,
                        line3_depth: Batch::new(true),
                        line3_top: Batch::new(true),
                        tri3_depth: Batch::new(false),
                        tri3_top: Batch::new(false),
                        tri2: Batch::new(false),
                        rect: Batch::new(true),
                        line2: Batch::new(true),
                        icon: Batch::new(true),
                        text,
                    },
                );
            }
        }
        self.sort();
    }

    /// Shift a layer's 3D parts by `offset` world units, without sending them again.
    pub fn set_layer_offset(&mut self, name: &str, offset: [f32; 3]) {
        if let Some(layer) = self.layers.get_mut(name) {
            layer.offset = offset;
        }
    }

    pub fn clear_layer(&mut self, name: &str) {
        self.layers.remove(name);
        self.sort();
    }

    /// The layers this picture draws, in order.
    fn shown(&self) -> impl Iterator<Item = &Layer> {
        let agent = self.agent;
        self.order.iter().filter(move |n| n.starts_with(AGENT) == agent).map(|n| &self.layers[n])
    }

    fn sort(&mut self) {
        let mut order: Vec<(i32, String)> = self.layers.iter().map(|(n, l)| (l.z, n.clone())).collect();
        order.sort();
        self.order = order.into_iter().map(|(_, n)| n).collect();
    }

    /// Upload changed layers and lay out the text. Call before the passes begin.
    /// `views` holds each view's view-projection and size in physical pixels.
    pub fn prepare(&mut self, gpu: &Gpu, fonts: &mut FontSystem, views: &[OverlayView], w: u32, h: u32, scale: f32) {
        let u = OverlayU { view_proj: [[0.0; 4]; 4], screen: [w as f32, h as f32, scale, 0.0], eye: [0.0; 4], pull: [0.0; 4] };
        gpu.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        for (i, (vp, vw, vh, eye, pull)) in views.iter().take(crate::gpu::MAX_VIEWS).enumerate() {
            let u = OverlayU { view_proj: *vp, screen: [*vw as f32, *vh as f32, scale, 0.0], eye: *eye, pull: *pull };
            gpu.queue.write_buffer(&self.ubuf, U_STRIDE * (1 + i as u64), bytemuck::bytes_of(&u));
        }
        // The shifted layers: their own slots, the view-projection times the shift
        // and the eye moved back by it, so the pull toward the eye stays right.
        let mut k = 0usize;
        let mut slots: Vec<([f32; 3], usize)> = Vec::new(); // one slot per distinct offset
        for name in self.order.clone() {
            let Some(layer) = self.layers.get_mut(&name) else { continue };
            layer.slot = None;
            if layer.offset == [0.0; 3] {
                continue;
            }
            if let Some((_, s)) = slots.iter().find(|(o, _)| *o == layer.offset) {
                layer.slot = Some(*s);
                continue;
            }
            if k >= MAX_SHIFTED {
                continue;
            }
            let shift = glam::Vec3::from(layer.offset);
            for (i, (vp, vw, vh, eye, pull)) in views.iter().take(crate::gpu::MAX_VIEWS).enumerate() {
                let m = glam::Mat4::from_cols_array_2d(vp) * glam::Mat4::from_translation(shift);
                let e = [eye[0] - shift.x, eye[1] - shift.y, eye[2] - shift.z, eye[3]];
                let u = OverlayU { view_proj: m.to_cols_array_2d(), screen: [*vw as f32, *vh as f32, scale, 0.0], eye: e, pull: *pull };
                let slot = 1 + crate::gpu::MAX_VIEWS * (1 + k) + i;
                gpu.queue.write_buffer(&self.ubuf, U_STRIDE * slot as u64, bytemuck::bytes_of(&u));
            }
            slots.push((layer.offset, k));
            layer.slot = Some(k);
            k += 1;
        }
        for layer in self.layers.values_mut() {
            if !layer.dirty {
                continue;
            }
            let l = &layer.list;
            layer.line3_depth.upload(gpu, &l.lines3_depth, LINE3_F);
            layer.line3_top.upload(gpu, &l.lines3_top, LINE3_F);
            layer.tri3_depth.upload(gpu, &l.tris3_depth, TRI3_F);
            layer.tri3_top.upload(gpu, &l.tris3_top, TRI3_F);
            layer.tri2.upload(gpu, &l.tris2, TRI2_F);
            layer.rect.upload(gpu, &l.rects, RECT_F);
            layer.line2.upload(gpu, &l.lines2, LINE2_F);
            let mut icons = Vec::with_capacity(l.icons.len() * ICON_F);
            for i in &l.icons {
                let Some(uv) = self.atlas.uv(&i.name) else { continue };
                icons.extend_from_slice(&i.rect);
                icons.extend_from_slice(&uv);
                icons.extend_from_slice(&i.tint);
                icons.push(if i.image { 1.0 } else { 0.0 });
            }
            layer.icon.upload(gpu, &icons, ICON_F);
            layer.dirty = false;
        }

        // Text: one shaped buffer per (string, size), shared by all layers; unused ones are dropped.
        let mut keep: HashMap<(String, u32), Buffer> = HashMap::new();
        for layer in self.layers.values() {
            for run in &layer.list.texts {
                let key = (run.text.clone(), (run.size * scale).to_bits());
                if keep.contains_key(&key) {
                    continue;
                }
                let buf = match self.buffers.remove(&key) {
                    Some(b) => b,
                    None => shaped(fonts, &run.text, run.size * scale),
                };
                keep.insert(key, buf);
            }
        }
        self.buffers = keep;
        self.viewport.update(&gpu.queue, Resolution { width: w, height: h });
        for layer in self.layers.values_mut() {
            let areas: Vec<TextArea> = layer
                .list
                .texts
                .iter()
                .map(|run| TextArea {
                    buffer: &self.buffers[&(run.text.clone(), (run.size * scale).to_bits())],
                    left: (run.x * scale).round(),
                    top: (run.y * scale).round(),
                    scale: 1.0,
                    bounds: TextBounds { left: 0, top: 0, right: w as i32, bottom: h as i32 },
                    default_color: Color::rgba(run.color[0], run.color[1], run.color[2], run.color[3]),
                    custom_glyphs: &[],
                })
                .collect();
            if let Err(e) = layer.text.prepare(&gpu.device, &gpu.queue, fonts, &mut self.text_atlas, &self.viewport, areas, &mut self.swash) {
                eprintln!("rok: text prepare failed: {e:?}");
            }
        }
    }

    /// The 3D parts of every layer: against the voxel depth first, then on top.
    pub fn draw_3d(&self, pass: &mut wgpu::RenderPass<'_>, view: usize) {
        let slot = |l: &Layer| -> u32 {
            let s = match l.slot {
                Some(k) => 1 + crate::gpu::MAX_VIEWS * (1 + k) + view,
                None => 1 + view,
            };
            (U_STRIDE * s as u64) as u32
        };
        for l in self.shown() {
            pass.set_bind_group(0, &self.bg, &[slot(l)]);
            l.tri3_depth.draw(pass, &self.tri3_depth);
            l.line3_depth.draw(pass, &self.line3_depth);
        }
        for l in self.shown() {
            pass.set_bind_group(0, &self.bg, &[slot(l)]);
            l.tri3_top.draw(pass, &self.tri3_top);
            l.line3_top.draw(pass, &self.line3_top);
        }
    }

    /// The 2D parts, one layer after the other.
    pub fn draw_2d(&self, pass: &mut wgpu::RenderPass<'_>) {
        for l in self.shown() {
            pass.set_bind_group(0, &self.bg, &[0]);
            l.tri2.draw(pass, &self.tri2);
            l.rect.draw(pass, &self.rect);
            l.line2.draw(pass, &self.line2);
            if l.icon.count > 0 {
                pass.set_bind_group(1, &self.atlas.bg, &[]);
                l.icon.draw(pass, &self.icon);
            }
            if let Err(e) = l.text.render(&self.text_atlas, &self.viewport, pass) {
                eprintln!("rok: text render failed: {e:?}");
            }
        }
    }

    pub fn trim(&mut self) {
        self.text_atlas.trim();
    }
}

fn shaped(fonts: &mut FontSystem, text: &str, px: f32) -> Buffer {
    let mut buf = Buffer::new(fonts, Metrics::new(px, (px * 1.25).ceil()));
    buf.set_size(None, None);
    buf.set_text(text, &Attrs::new().family(Family::SansSerif), Shaping::Advanced, None);
    buf.shape_until_scroll(fonts, false);
    buf
}

/// `text` at `px` pixels as an alpha mask: (width, height, one byte per pixel,
/// rows from the top). Python turns it and keeps it in the icon atlas, for text
/// the text runs cannot draw, such as text turned upright.
pub fn text_mask(fonts: &mut FontSystem, text: &str, px: f32) -> (u32, u32, Vec<u8>) {
    let mut buf = shaped(fonts, text, px);
    let (w, h) = measure(fonts, text, px);
    let (w, h) = (w.ceil().max(1.0) as u32, h.ceil().max(1.0) as u32);
    let mut data = vec![0u8; (w * h) as usize];
    let mut cache = SwashCache::new();
    buf.draw(fonts, &mut cache, Color::rgb(255, 255, 255), |x, y, cw, ch, c| {
        for yy in y.max(0)..(y + ch as i32).min(h as i32) {
            for xx in x.max(0)..(x + cw as i32).min(w as i32) {
                let i = (yy as u32 * w + xx as u32) as usize;
                data[i] = data[i].max(c.a());
            }
        }
    });
    (w, h, data)
}

/// Width and height of `text` at `size` logical pixels, in logical pixels.
pub fn measure(fonts: &mut FontSystem, text: &str, size: f32) -> (f32, f32) {
    let buf = shaped(fonts, text, size);
    let mut w: f32 = 0.0;
    let mut lines = 0;
    for run in buf.layout_runs() {
        w = w.max(run.line_w);
        lines += 1;
    }
    (w, lines.max(1) as f32 * (size * 1.25).ceil())
}
