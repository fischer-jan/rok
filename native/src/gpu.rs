//! wgpu side: device, pipelines, one 3D texture per model.

use crate::scene::Model;
use bytemuck::{Pod, Zeroable};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameU {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub key_dir: [f32; 4],
    pub fill_dir: [f32; 4],
    pub viewport: [f32; 4],
    /// x, y: the view's top-left corner in the target, physical pixels (quad view).
    pub origin: [f32; 4],
    /// The look (Host.set_look): see `Look` in host.rs and shader.wgsl. Zeros: the plain look.
    pub look: [[f32; 4]; 3],
}

/// Most views drawn at once (the quad view), and the uniform stride between them.
pub const MAX_VIEWS: usize = 8; // the quad view uses 4; an agent picture with its close-up strip 7
pub const VIEW_STRIDE: u64 = 256;

/// One view of a frame: its uniforms and its rectangle in the target, physical pixels.
pub struct ViewFrame {
    pub frame: FrameU,
    pub rect: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelU {
    pub model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub size: [u32; 4],
    pub info: [u32; 4],
}

pub struct GpuModel {
    pub tex: wgpu::Texture,
    pub bricks: wgpu::Texture,
    pub cad: wgpu::Texture,
    /// The seams field of a brick model; one empty cell for a voxel model.
    pub seams: wgpu::Texture,
    pub ubuf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

pub struct Gpu {
    #[allow(dead_code)] // kept with the other handles; nothing reads it
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    pub fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface>) -> Gpu {
        let ask = |fallback: bool| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: fallback,
                apply_limit_buckets: false,
            }))
        };
        // A machine with no GPU (a server, a container) only has a software
        // adapter, which wgpu offers as the fallback one.
        let adapter = ask(false).or_else(|_| ask(true)).expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rok"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("no GPU device");
        Gpu { instance, adapter, device, queue }
    }
}

pub struct Renderer {
    #[allow(dead_code)] // the format the pipelines were built for; nothing reads it
    pub format: wgpu::TextureFormat,
    bg_pipe: wgpu::RenderPipeline,
    model_pipe: wgpu::RenderPipeline,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    pal_tex: wgpu::Texture,
    model_bgl: wgpu::BindGroupLayout,
    depth: Option<(wgpu::TextureView, u32, u32)>,
    pub models: Vec<GpuModel>,
}

fn mat(m: glam::DMat4) -> [[f32; 4]; 4] {
    m.as_mat4().to_cols_array_2d()
}

pub fn model_uniform(m: &Model, use_bricks: bool, margin: usize) -> ModelU {
    let mm = m.matrix();
    ModelU {
        model: mat(mm),
        inv_model: mat(mm.inverse()),
        size: [m.size[0] as u32, m.size[1] as u32, m.size[2] as u32, use_bricks as u32],
        // w: bit 0 a CAD field is uploaded, bit 1 a brick model's seams field.
        info: [m.step_cap(margin), m.fade.to_bits(), m.hidden as u32, (m.cad as u32) | ((m.seams.is_some() as u32) << 1)],
    }
}

impl Renderer {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat, palette: &[u8]) -> Renderer {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("march"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<FrameU>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let uint3d = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                uint3d(1),
                uint3d(2),
                uint3d(3),
                uint3d(4),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&model_bgl)],
            immediate_size: 0,
        });
        let make = |label: &str, vs: &str, fs: &str, topo: wgpu::PrimitiveTopology, cull: Option<wgpu::Face>, write: bool, cmp: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { topology: topo, cull_mode: cull, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(cmp),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let bg_pipe = make("bg", "vs_bg", "fs_bg", wgpu::PrimitiveTopology::TriangleList, None, false, wgpu::CompareFunction::Always);
        let model_pipe = make("model", "vs_model", "fs_model", wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Front), true, wgpu::CompareFunction::Less);

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: VIEW_STRIDE * MAX_VIEWS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("palette"),
            size: wgpu::Extent3d { width: 256, height: 2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Row 0 holds the colours (sRGB bytes), row 1 each slot's material
        // (set_materials; plain until then).
        let mut rows = palette.to_vec();
        rows.resize(2048, 0);
        gpu.queue.write_texture(
            pal_tex.as_image_copy(),
            &rows,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1024), rows_per_image: Some(2) },
            wgpu::Extent3d { width: 256, height: 2, depth_or_array_layers: 1 },
        );
        let pal_view = pal_tex.create_view(&Default::default());
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &frame_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<FrameU>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&pal_view) },
            ],
        });
        Renderer {
            format,
            bg_pipe,
            model_pipe,
            frame_buf,
            frame_bg,
            pal_tex,
            model_bgl,
            depth: None,
            models: Vec::new(),
        }
    }

    fn texture3d(gpu: &Gpu, label: &str, data: &[u8], dims: [usize; 3]) -> wgpu::Texture {
        // The voxel array is (x, y, z) in C order: width = z, height = y, depth = x.
        Self::texture3d_fmt(gpu, label, data, dims, wgpu::TextureFormat::R8Uint, 1, false)
    }

    /// `align_rows`: pad each row to 256 bytes. The voxel textures upload
    /// unaligned and that works here; a RGBA volume is padded so a wide grid
    /// still copies.
    fn texture3d_fmt(gpu: &Gpu, label: &str, data: &[u8], dims: [usize; 3], format: wgpu::TextureFormat, bpp: usize, align_rows: bool) -> wgpu::Texture {
        let ext = wgpu::Extent3d { width: dims[2] as u32, height: dims[1] as u32, depth_or_array_layers: dims[0] as u32 };
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: ext,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row = dims[2] * bpp;
        let stride = if align_rows { row.div_ceil(256) * 256 } else { row };
        let upload: std::borrow::Cow<[u8]> = if stride == row {
            std::borrow::Cow::Borrowed(data)
        } else {
            let mut padded = vec![0u8; stride * dims[1] * dims[0]];
            for x in 0..dims[0] {
                for y in 0..dims[1] {
                    let src = (x * dims[1] + y) * row;
                    let dst = (x * dims[1] + y) * stride;
                    padded[dst..dst + row].copy_from_slice(&data[src..src + row]);
                }
            }
            std::borrow::Cow::Owned(padded)
        };
        gpu.queue.write_texture(
            tex.as_image_copy(),
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(dims[1] as u32),
            },
            ext,
        );
        tex
    }

    fn bind_model(gpu: &Gpu, layout: &wgpu::BindGroupLayout, ubuf: &wgpu::Buffer, tex: &wgpu::Texture, bricks: &wgpu::Texture, cad: &wgpu::Texture, seams: &wgpu::Texture) -> wgpu::BindGroup {
        let tv = tex.create_view(&Default::default());
        let bv = bricks.create_view(&Default::default());
        let cv = cad.create_view(&Default::default());
        let sv = seams.create_view(&Default::default());
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tv) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&bv) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cv) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&sv) },
            ],
        })
    }

    /// Upload (or re-upload) model `idx`: new textures replace the old ones.
    pub fn upload_model(&mut self, gpu: &Gpu, idx: usize, m: &Model) {
        let tex = Self::texture3d(gpu, "voxels", &m.voxels, m.size);
        let bricks = Self::texture3d(gpu, "bricks", &m.bricks, m.bdim);
        // No CAD field yet: a 1x1 zero the shader does not sample (model.cad is false).
        let cad = Self::texture3d_fmt(gpu, "cad", &[0, 0, 0, 0], [1, 1, 1], wgpu::TextureFormat::Rgba8Uint, 4, true);
        let seams = match &m.seams {
            Some(field) => Self::texture3d(gpu, "seams", field, m.size),
            None => Self::texture3d(gpu, "seams", &[0u8], [1, 1, 1]),
        };
        let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model"),
            size: std::mem::size_of::<ModelU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Self::bind_model(gpu, &self.model_bgl, &ubuf, &tex, &bricks, &cad, &seams);
        let gm = GpuModel { tex, bricks, cad, seams, ubuf, bind_group };
        if idx < self.models.len() {
            self.models[idx] = gm;
        } else {
            self.models.push(gm);
        }
    }

    /// Write one box of a 3D texture. `origin` and `dims` are (x, y, z) cells;
    /// `data` is that box in C order. The texture is stored as (z, y, x).
    fn write_box(gpu: &Gpu, tex: &wgpu::Texture, origin: [usize; 3], dims: [usize; 3], data: &[u8]) {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: origin[2] as u32, y: origin[1] as u32, z: origin[0] as u32 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[2] as u32), rows_per_image: Some(dims[1] as u32) },
            wgpu::Extent3d { width: dims[2] as u32, height: dims[1] as u32, depth_or_array_layers: dims[0] as u32 },
        );
    }

    /// Replace model `idx`'s CAD field and bind it. `data` is (x, y, z, 4) in C order.
    pub fn set_cad_field(&mut self, gpu: &Gpu, idx: usize, data: &[u8], size: [usize; 3]) {
        if idx >= self.models.len() {
            return;
        }
        let cad = Self::texture3d_fmt(gpu, "cad", data, size, wgpu::TextureFormat::Rgba8Uint, 4, true);
        let gm = &self.models[idx];
        let bind_group = Self::bind_model(gpu, &self.model_bgl, &gm.ubuf, &gm.tex, &gm.bricks, &cad, &gm.seams);
        let gm = &mut self.models[idx];
        gm.cad = cad;
        gm.bind_group = bind_group;
    }

    /// A new seams texture for model `idx` (the whole field, or None for a
    /// voxel model's one empty cell); the voxels and the CAD field stay.
    pub fn set_seams_field(&mut self, gpu: &Gpu, idx: usize, field: Option<&[u8]>, size: [usize; 3]) {
        if idx >= self.models.len() {
            return;
        }
        let seams = match field {
            Some(data) => Self::texture3d(gpu, "seams", data, size),
            None => Self::texture3d(gpu, "seams", &[0u8], [1, 1, 1]),
        };
        let gm = &self.models[idx];
        let bind_group = Self::bind_model(gpu, &self.model_bgl, &gm.ubuf, &gm.tex, &gm.bricks, &gm.cad, &seams);
        let gm = &mut self.models[idx];
        gm.seams = seams;
        gm.bind_group = bind_group;
    }

    /// Upload a changed box of model `idx`: the voxels, then the bricks around them.
    pub fn write_region(&self, gpu: &Gpu, idx: usize, origin: [usize; 3], dims: [usize; 3], data: &[u8], bricks: ([usize; 3], [usize; 3], Vec<u8>)) {
        let gm = &self.models[idx];
        Self::write_box(gpu, &gm.tex, origin, dims, data);
        let (bo, bd, bdata) = bricks;
        Self::write_box(gpu, &gm.bricks, bo, bd, &bdata);
    }

    /// Upload a changed box of model `idx`'s seams field (its texture has the model's size).
    pub fn write_seams(&self, gpu: &Gpu, idx: usize, origin: [usize; 3], dims: [usize; 3], data: &[u8]) {
        Self::write_box(gpu, &self.models[idx].seams, origin, dims, data);
    }

    pub fn remove_model(&mut self, idx: usize) {
        if idx < self.models.len() {
            self.models.remove(idx);
        }
    }

    /// Draw a frame for the app host. The first pass draws the background, the
    /// voxels and `draw3d` (overlays that test against the voxel depth). With
    /// `fx`, that pass goes to an off-screen texture and a second pass smooths
    /// it onto `target`; `draw2d` (the UI) always comes last, on `target`.
    pub fn render_with(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        w: u32,
        h: u32,
        views: &[ViewFrame],
        uniforms: &[ModelU],
        fx: Option<&mut crate::postfx::PostFx>,
        traced: Option<&crate::trace::Tracer>,
        draw3d: impl Fn(&mut wgpu::RenderPass<'_>, usize),
        draw2d: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) {
        for (i, v) in views.iter().take(MAX_VIEWS).enumerate() {
            gpu.queue.write_buffer(&self.frame_buf, VIEW_STRIDE * i as u64, bytemuck::bytes_of(&v.frame));
        }
        for (gm, u) in self.models.iter().zip(uniforms) {
            gpu.queue.write_buffer(&gm.ubuf, 0, bytemuck::bytes_of(u));
        }
        let depth = self.depth_view(gpu, w, h);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("host frame") });
        let smooth = fx.is_some();
        let mut fx = fx;
        let scene_target = match fx.as_mut() {
            Some(f) => f.scene_view(gpu, w, h),
            None => target.clone(),
        };
        let mut draw2d = Some(draw2d);
        // The rendered view replaces the look's voxels in the first view.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scene_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (i, v) in views.iter().take(MAX_VIEWS).enumerate() {
                let [x, y, vw, vh] = v.rect;
                let (vw, vh) = (vw.min(w.saturating_sub(x)).max(1), vh.min(h.saturating_sub(y)).max(1));
                pass.set_viewport(x as f32, y as f32, vw as f32, vh as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, vw, vh);
                if let (0, Some(t)) = (i, traced) {
                    t.draw(&mut pass);
                    draw3d(&mut pass, i);
                    continue;
                }
                pass.set_bind_group(0, &self.frame_bg, &[(VIEW_STRIDE * i as u64) as u32]);
                // The background shader declares the model group too: bind any model.
                if let Some(first) = self.models.first() {
                    pass.set_bind_group(1, &first.bind_group, &[]);
                    pass.set_pipeline(&self.bg_pipe);
                    pass.draw(0..3, 0..1);
                }
                pass.set_pipeline(&self.model_pipe);
                for (gm, u) in self.models.iter().zip(uniforms) {
                    if u.info[2] != 0 {
                        continue; // hidden (its layer is off)
                    }
                    pass.set_bind_group(1, &gm.bind_group, &[]);
                    pass.draw(0..36, 0..1);
                }
                draw3d(&mut pass, i);
            }
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, w, h);
            if !smooth {
                (draw2d.take().unwrap())(&mut pass);
            }
        }
        if smooth {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("smooth and ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            fx.as_ref().unwrap().draw(&mut pass);
            (draw2d.take().unwrap())(&mut pass);
        }
        gpu.queue.submit([enc.finish()]);
    }

    /// Replace the palette (256 RGBA colours).
    pub fn set_palette(&self, gpu: &Gpu, palette: &[u8]) {
        gpu.queue.write_texture(
            self.pal_tex.as_image_copy(),
            palette,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1024), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
    }

    /// Each slot's material for the looks, from the rendered view's rows
    /// (MAT_COLS per slot: metallic, roughness, transmission, emission, ...):
    /// palette row 1 holds glow (log10 of 1 + emission, over 2), metallic,
    /// roughness and the type code (0 plain, 1 metal, 2 glass, 3 light,
    /// 4 plastic, 5 other) times 40.
    pub fn set_materials(&self, gpu: &Gpu, rows: &[f32]) {
        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let mut texels = vec![0u8; 1024];
        for (slot, r) in rows.chunks_exact(crate::trace::MAT_COLS).take(256).enumerate() {
            texels[slot * 4] = byte((1.0 + r[3].max(0.0)).log10() / 2.0);
            texels[slot * 4 + 1] = byte(r[0]);
            texels[slot * 4 + 2] = byte(r[1]);
            texels[slot * 4 + 3] = (r[6].clamp(0.0, 5.0) as u8) * 40;
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.pal_tex, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 1, z: 0 }, aspect: wgpu::TextureAspect::All },
            &texels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(1024), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
    }

    fn depth_view(&mut self, gpu: &Gpu, w: u32, h: u32) -> wgpu::TextureView {
        if let Some((v, dw, dh)) = &self.depth {
            if *dw == w && *dh == h {
                return v.clone();
            }
        }
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let v = tex.create_view(&Default::default());
        self.depth = Some((v.clone(), w, h));
        v
    }
}
