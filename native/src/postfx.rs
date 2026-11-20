//! Smooth edges: the 3D view is drawn into an off-screen texture, then FXAA
//! copies it to the screen. The UI is drawn after that, so text stays sharp.

use crate::gpu::{Gpu, DEPTH_FORMAT};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct P {
    inv_size: [f32; 2],
    subpix: f32,
    edge_min: f32,
}

/// How strongly FXAA softens detail of one pixel (FXAA's own default is 0.75).
pub const SUBPIX: f32 = 1.0;
/// Below this luma contrast a pixel is left alone (FXAA's "high quality" 0.0312).
pub const EDGE_MIN: f32 = 0.0312;

pub struct PostFx {
    pipe: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ubuf: wgpu::Buffer,
    format: wgpu::TextureFormat,
    target: Option<(wgpu::TextureView, wgpu::BindGroup, u32, u32)>,
}

impl PostFx {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> PostFx {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fxaa.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fxaa"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_full"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fxaa"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            // The UI pass that follows keeps the depth buffer attached; this pipeline ignores it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fxaa"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fxaa u"),
            size: std::mem::size_of::<P>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        PostFx { pipe, bgl, sampler, ubuf, format, target: None }
    }

    /// The off-screen texture the 3D view is drawn into, at the screen's size.
    pub fn scene_view(&mut self, gpu: &Gpu, w: u32, h: u32) -> wgpu::TextureView {
        if let Some((v, _, tw, th)) = &self.target {
            if *tw == w && *th == h {
                return v.clone();
            }
        }
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let p = P { inv_size: [1.0 / w as f32, 1.0 / h as f32], subpix: SUBPIX, edge_min: EDGE_MIN };
        gpu.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&p));
        self.target = Some((view.clone(), bg, w, h));
        view
    }

    /// Draw the smoothed 3D view over the whole target.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some((_, bg, _, _)) = &self.target {
            pass.set_pipeline(&self.pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
