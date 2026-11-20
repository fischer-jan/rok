//! xs shaders on the GPU.
//!
//! The model is a 2D atlas of its z slices (rok.core.xs.atlas). For each z,
//! one full-screen fragment pass the size of the x-y slice calls the shader's
//! map() for its cell and writes the palette index; the slices are read back.
//! Several iterations feed each result into the next, with i_iter counting.
//! `rok.core.xs.fragment_source` gives the shader as GLSL; this module adds
//! the declarations (one uniform block, three textures) and its own main,
//! which fills the argument globals and calls the shader's main.
//!
//! The runner has its own headless GPU device, so Python can call it and wait
//! for the answer outside the window's event loop.

use numpy::{PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::time::Instant;

const MAX_ARGS: usize = 16;
const U_STRIDE: u64 = 256; // uniform offsets must be 256-aligned

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct XsU {
    volume: [f32; 3],
    color_index: f32,
    mirror: [f32; 3],
    num_sels: i32,
    axis: [f32; 3],
    iter: f32,
    z: f32,
    tiles: i32,
    loop_limit: i32,
    _pad: f32,
    args: [f32; MAX_ARGS],
    vars: [f32; MAX_ARGS],
}

fn prelude(arg_vars: &[String]) -> String {
    let mut s = String::from(
        "#version 450\n\
         layout(set = 0, binding = 0, std140) uniform XsU {\n\
           vec3 i_volume_size; float i_color_index;\n\
           vec3 i_mirror; int i_num_color_sels;\n\
           vec3 i_axis; float i_iter;\n\
           float u_z; int u_tiles; int u_loop_limit; float xs_pad0;\n\
           vec4 xs_args4[4];\n\
           vec4 xs_vars4[4];\n\
         };\n\
         layout(set = 0, binding = 1) uniform texture2D xs_vol;\n\
         layout(set = 0, binding = 2) uniform texture2D xs_sel;\n\
         layout(set = 0, binding = 3) uniform texture2D xs_pal;\n\
         layout(set = 0, binding = 4) uniform sampler xs_samp;\n\
         #define u_vol sampler2D(xs_vol, xs_samp)\n\
         #define u_sel sampler2D(xs_sel, xs_samp)\n\
         #define u_pal sampler2D(xs_pal, xs_samp)\n\
         layout(location = 0) out vec4 fragColor;\n\
         float i_args[16];\n",
    );
    for v in arg_vars {
        s.push_str(&format!("float {v};\n"));
    }
    s.push_str("#define main xs_shader_main\n");
    s
}

fn epilogue(arg_vars: &[String]) -> String {
    let mut s = String::from("#undef main\nvoid main() {\n");
    for k in 0..MAX_ARGS {
        s.push_str(&format!("  i_args[{k}] = xs_args4[{}][{}];\n", k / 4, k % 4));
    }
    for (k, v) in arg_vars.iter().enumerate().take(MAX_ARGS) {
        s.push_str(&format!("  {v} = xs_vars4[{}][{}];\n", k / 4, k % 4));
    }
    s.push_str("  xs_shader_main();\n}\n");
    s
}

const VS: &str = "#version 450\nvoid main() {\n  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));\n  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n}\n";

#[pyclass(unsendable)]
pub struct ShaderRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_texture: u32,
    adapter_name: String,
}

fn tex2d(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, format: wgpu::TextureFormat, bpp: u32, data: &[u8]) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("xs input"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpp), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    tex
}

#[pymethods]
impl ShaderRunner {
    #[new]
    fn new() -> PyResult<ShaderRunner> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let ask = |fallback: bool| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: fallback,
                apply_limit_buckets: false,
            }))
        };
        let adapter = ask(false)
            .or_else(|_| ask(true))
            .map_err(|e| PyRuntimeError::new_err(format!("no GPU adapter: {e}")))?;
        let supported = adapter.limits();
        let limits = wgpu::Limits { max_texture_dimension_2d: supported.max_texture_dimension_2d, ..wgpu::Limits::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rok xs"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| PyRuntimeError::new_err(format!("no GPU device: {e}")))?;
        Ok(ShaderRunner { device, queue, max_texture: supported.max_texture_dimension_2d, adapter_name: adapter.get_info().name })
    }

    #[getter]
    fn max_texture(&self) -> u32 {
        self.max_texture
    }

    #[getter]
    fn adapter(&self) -> String {
        self.adapter_name.clone()
    }

    /// Run a shader. `source` is rok.core.xs.fragment_source(shader);
    /// `arg_vars` the shader's argument names (xs.arg_names) and `arg_values`
    /// their values; `args` the xs format's i_args (16 floats). `atlas` is
    /// xs.atlas(voxels, cols) for the first iteration's voxels, `size` the
    /// model's (x, y, z). Returns (voxels (x, y, z) uint8, stopped cells).
    #[pyo3(signature = (source, arg_vars, arg_values, args, voxels, cols, palette, sels, colour, iterations=1, mirror=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0), loop_limit=131072, seconds=30.0))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        source: &str,
        arg_vars: Vec<String>,
        arg_values: Vec<f32>,
        args: Vec<f32>,
        voxels: PyReadonlyArray3<u8>,
        cols: u32,
        palette: PyReadonlyArray2<u8>,
        sels: Vec<u32>,
        colour: f32,
        iterations: u32,
        mirror: (f32, f32, f32),
        axis: (f32, f32, f32),
        loop_limit: i32,
        seconds: f64,
    ) -> PyResult<(Bound<'py, PyArray3<u8>>, u64)> {
        let shape = voxels.shape();
        let (sx, sy, sz) = (shape[0], shape[1], shape[2]);
        let mut current: Vec<u8> = voxels.as_array().iter().copied().collect();
        let pal: Vec<u8> = palette.as_array().iter().copied().collect();
        if pal.len() != 1024 {
            return Err(PyRuntimeError::new_err("the palette must be (256, 4)"));
        }
        let cols = cols.max(1) as usize;
        let rows = sz.div_ceil(cols);
        let (aw, ah) = ((cols * sx) as u32, (rows * sy) as u32);
        if aw > self.max_texture || ah > self.max_texture {
            return Err(PyRuntimeError::new_err(format!("the model is too large for this GPU ({aw} x {ah} > {})", self.max_texture)));
        }
        let device = &self.device;
        let queue = &self.queue;

        // Compile, catching errors instead of panicking.
        let full = format!("{}{}\n{}", prelude(&arg_vars), source, epilogue(&arg_vars));
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xs fragment"),
            source: wgpu::ShaderSource::Glsl { shader: full.into(), stage: wgpu::naga::ShaderStage::Fragment, defines: &[] },
        });
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xs vertex"),
            source: wgpu::ShaderSource::Glsl { shader: VS.into(), stage: wgpu::naga::ShaderStage::Vertex, defines: &[] },
        });
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xs"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<XsU>() as u64),
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xs"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let target_format = wgpu::TextureFormat::Rgba8Unorm;
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xs"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &vs, entry_point: Some("main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &fs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: target_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(PyRuntimeError::new_err(format!("the shader does not compile: {err}")));
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("xs"), ..Default::default() });
        let mut sel_row = vec![0u8; 256];
        for (i, s) in sels.iter().take(256).enumerate() {
            sel_row[i] = (*s).min(255) as u8;
        }
        let t_sel = tex2d(device, queue, 256, 1, wgpu::TextureFormat::R8Unorm, 1, &sel_row);
        let t_pal = tex2d(device, queue, 256, 1, wgpu::TextureFormat::Rgba8Unorm, 4, &pal);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xs target"),
            size: wgpu::Extent3d { width: sx as u32, height: sy as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let row_bytes = ((sx as u32 * 4).div_ceil(256) * 256) as u64;
        let slice_bytes = row_bytes * sy as u64;
        let mut values = [0f32; MAX_ARGS];
        for (k, v) in arg_values.iter().take(MAX_ARGS).enumerate() {
            values[k] = *v;
        }
        let mut iargs = [0f32; MAX_ARGS];
        for (k, v) in args.iter().take(MAX_ARGS).enumerate() {
            iargs[k] = *v;
        }
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xs uniforms"),
            size: U_STRIDE * sz as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let started = Instant::now();
        let mut stopped: u64 = 0;
        // Slices per submit: enough to keep the GPU busy, few enough that no
        // single command buffer runs long.
        let batch = ((1u64 << 20) / (sx as u64 * sy as u64).max(1)).clamp(1, 64) as usize;

        for it in 0..iterations.max(1) {
            // The atlas of the current voxels: xs.atlas's layout, tile (z % cols, z / cols).
            let mut atlas = vec![0u8; (aw * ah) as usize];
            for z in 0..sz {
                let (tx, ty) = (z % cols, z / cols);
                for y in 0..sy {
                    let row = (ty * sy + y) * aw as usize + tx * sx;
                    for x in 0..sx {
                        atlas[row + x] = current[(x * sy + y) * sz + z];
                    }
                }
            }
            let t_vol = tex2d(device, queue, aw, ah, wgpu::TextureFormat::R8Unorm, 1, &atlas);
            let view = |t: &wgpu::Texture| t.create_view(&Default::default());
            let (v_vol, v_sel, v_pal) = (view(&t_vol), view(&t_sel), view(&t_pal));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("xs"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &ubuf,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<XsU>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&v_vol) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&v_sel) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&v_pal) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            for z in 0..sz {
                let u = XsU {
                    volume: [sx as f32, sy as f32, sz as f32],
                    color_index: colour,
                    mirror: [mirror.0, mirror.1, mirror.2],
                    num_sels: sels.len().max(1) as i32,
                    axis: [axis.0, axis.1, axis.2],
                    iter: it as f32,
                    z: z as f32,
                    tiles: cols as i32,
                    loop_limit,
                    _pad: 0.0,
                    args: iargs,
                    vars: values,
                };
                queue.write_buffer(&ubuf, U_STRIDE * z as u64, bytemuck::bytes_of(&u));
            }
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xs readback"),
                size: slice_bytes * sz as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut z0 = 0;
            while z0 < sz {
                let z1 = (z0 + batch).min(sz);
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("xs") });
                for z in z0..z1 {
                    {
                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("xs slice"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &target_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_pipeline(&pipe);
                        pass.set_bind_group(0, &bg, &[(U_STRIDE * z as u64) as u32]);
                        pass.draw(0..3, 0..1);
                    }
                    enc.copy_texture_to_buffer(
                        target.as_image_copy(),
                        wgpu::TexelCopyBufferInfo {
                            buffer: &readback,
                            layout: wgpu::TexelCopyBufferLayout { offset: slice_bytes * z as u64, bytes_per_row: Some(row_bytes as u32), rows_per_image: Some(sy as u32) },
                        },
                        wgpu::Extent3d { width: sx as u32, height: sy as u32, depth_or_array_layers: 1 },
                    );
                }
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                if started.elapsed().as_secs_f64() > seconds {
                    return Err(PyRuntimeError::new_err(format!(
                        "it ran longer than {} seconds and was stopped; try it on a smaller model or a selection",
                        seconds as i64
                    )));
                }
                z0 = z1;
            }
            readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let data = readback.slice(..).get_mapped_range().map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
            let mut out = vec![0u8; sx * sy * sz];
            let mut dropped = false;
            for z in 0..sz {
                for y in 0..sy {
                    let row = slice_bytes as usize * z + row_bytes as usize * y;
                    for x in 0..sx {
                        let px = &data[row + x * 4..row + x * 4 + 4];
                        if px[3] == 0 {
                            dropped = true;
                        }
                        if px[1] != 0 {
                            stopped += 1;
                        }
                        out[(x * sy + y) * sz + z] = px[0];
                    }
                }
            }
            drop(data);
            readback.unmap();
            if dropped {
                return Err(PyRuntimeError::new_err(
                    "the GPU dropped it because one voxel's work ran too long; the model is unchanged",
                ));
            }
            current = out;
            let _ = py;
        }
        let arr = numpy::ndarray::Array3::from_shape_vec((sx, sy, sz), current).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((PyArray3::from_owned_array(py, arr), stopped))
    }
}
