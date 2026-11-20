// Overlays and UI: 3D lines, 2D lines and rounded rectangles.
// Every item is one instance; the vertex shader builds its quad.
// Colours arrive as sRGB with straight alpha; the target is an sRGB surface.

struct O {
    view_proj: mat4x4<f32>,
    // x, y = target size in physical pixels, z = scale factor (physical / logical)
    screen: vec4<f32>,
    eye: vec4<f32>,   // the view's eye (world)
    pull: vec4<f32>,  // x = share of the distance (perspective), y = world units (orthographic)
};

@group(0) @binding(0) var<uniform> o: O;

// A 3D overlay point moved toward the eye along its own sight line: the same
// pixel, a little nearer, so it wins against the voxel face it lies on.
fn pulled(p: vec3<f32>) -> vec3<f32> {
    let to_eye = o.eye.xyz - p;
    let d = length(to_eye);
    if (d < 1e-6) { return p; }
    let step = min(d * o.pull.x + o.pull.y, d * 0.5);
    return p + to_eye / d * step;
}

var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
);

fn to_linear(c: vec4<f32>) -> vec4<f32> {
    let lo = c.rgb / 12.92;
    let hi = pow((c.rgb + 0.055) / 1.055, vec3<f32>(2.4));
    return vec4<f32>(select(hi, lo, c.rgb <= vec3<f32>(0.04045)), c.a);
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    // rects: position inside the rect relative to its centre, in physical pixels
    @location(1) local: vec2<f32>,
    // rects: half size (xy), corner radius (z), stroke width (w), physical pixels
    @location(2) shape: vec4<f32>,
};

// Physical pixels (top left origin) to clip space.
fn px_to_clip(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(p.x / o.screen.x * 2.0 - 1.0, 1.0 - p.y / o.screen.y * 2.0, 0.0, 1.0);
}

// A segment in physical pixels, widened to `width` physical pixels, ends extended by half the width.
fn segment_px(a: vec2<f32>, b: vec2<f32>, width: f32, vi: u32) -> vec2<f32> {
    let q = QUAD[vi];
    var dir = b - a;
    let len = length(dir);
    if (len < 1e-4) { dir = vec2<f32>(1.0, 0.0); } else { dir = dir / len; }
    let n = vec2<f32>(-dir.y, dir.x);
    let h = width * 0.5;
    let along = mix(a - dir * h, b + dir * h, q.x);
    return along + n * q.y * h;
}

// --- 3D lines -------------------------------------------------------------------

struct Line3 {
    @location(0) p0: vec3<f32>,
    @location(1) p1: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
};

fn clip_near(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // Move `a` toward `b` until it is in front of the camera.
    let eps = 1e-3;
    if (a.w >= eps) { return a; }
    let t = (eps - a.w) / (b.w - a.w);
    return mix(a, b, t);
}

@vertex
fn vs_line3(@builtin(vertex_index) vi: u32, l: Line3) -> VOut {
    var c0 = o.view_proj * vec4<f32>(pulled(l.p0), 1.0);
    var c1 = o.view_proj * vec4<f32>(pulled(l.p1), 1.0);
    var out: VOut;
    out.color = to_linear(l.color);
    out.local = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    if (c0.w < 1e-3 && c1.w < 1e-3) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);  // behind the camera: off screen
        return out;
    }
    c0 = clip_near(c0, c1);
    c1 = clip_near(c1, c0);
    let half = o.screen.xy * 0.5;
    let s0 = vec2<f32>(c0.x / c0.w, -c0.y / c0.w) * half + half;
    let s1 = vec2<f32>(c1.x / c1.w, -c1.y / c1.w) * half + half;
    let p = segment_px(s0, s1, l.width * o.screen.z, vi);
    let t = QUAD[vi].x;
    let z = mix(c0.z / c0.w, c1.z / c1.w, t);
    var clip = px_to_clip(p);
    clip.z = clamp(z, 0.0, 1.0);
    out.pos = clip;
    return out;
}

// --- 2D lines -------------------------------------------------------------------

struct Line2 {
    @location(0) ab: vec4<f32>,  // x0, y0, x1, y1 in logical pixels
    @location(1) color: vec4<f32>,
    @location(2) width: f32,
};

@vertex
fn vs_line2(@builtin(vertex_index) vi: u32, l: Line2) -> VOut {
    let s = o.screen.z;
    let p = segment_px(l.ab.xy * s, l.ab.zw * s, l.width * s, vi);
    var out: VOut;
    out.pos = px_to_clip(p);
    out.color = to_linear(l.color);
    out.local = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    return out;
}

@fragment
fn fs_flat(v: VOut) -> @location(0) vec4<f32> {
    return v.color;
}

// --- rounded rectangles ---------------------------------------------------------

struct Rect {
    @location(0) xywh: vec4<f32>,  // logical pixels, top left origin
    @location(1) color: vec4<f32>,
    @location(2) rs: vec2<f32>,  // corner radius, stroke width (0: filled), logical pixels
};

var<private> CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
);

@vertex
fn vs_rect(@builtin(vertex_index) vi: u32, r: Rect) -> VOut {
    let s = o.screen.z;
    let pad = 1.0;  // one physical pixel around the rect for the soft edge
    let lo = r.xywh.xy * s - vec2<f32>(pad);
    let size = r.xywh.zw * s + vec2<f32>(2.0 * pad);
    let p = lo + CORNERS[vi] * size;
    let centre = (r.xywh.xy + r.xywh.zw * 0.5) * s;
    var out: VOut;
    out.pos = px_to_clip(p);
    out.color = to_linear(r.color);
    out.local = p - centre;
    out.shape = vec4<f32>(r.xywh.zw * 0.5 * s, r.rs.x * s, r.rs.y * s);
    return out;
}

@fragment
fn fs_rect(v: VOut) -> @location(0) vec4<f32> {
    let half = v.shape.xy;
    let rad = min(v.shape.z, min(half.x, half.y));
    let q = abs(v.local) - half + vec2<f32>(rad);
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rad;
    var cover = clamp(0.5 - d, 0.0, 1.0);
    if (v.shape.w > 0.0) {
        cover = cover * clamp(0.5 + d + v.shape.w, 0.0, 1.0);
    }
    return vec4<f32>(v.color.rgb, v.color.a * cover);
}

// --- filled triangles -----------------------------------------------------------

struct Tri3 {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_tris3(t: Tri3) -> VOut {
    var out: VOut;
    let clip = o.view_proj * vec4<f32>(pulled(t.pos), 1.0);  // faces lie on voxel faces: pulled toward the eye
    out.pos = clip;
    out.color = to_linear(t.color);
    out.local = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    return out;
}

struct Tri2 {
    @location(0) pos: vec2<f32>,  // logical pixels
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_tris2(t: Tri2) -> VOut {
    var out: VOut;
    out.pos = px_to_clip(t.pos * o.screen.z);
    out.color = to_linear(t.color);
    out.local = vec2<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    return out;
}

// --- icons and images from the atlas ----------------------------------------------

@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var atlas_samp: sampler;

struct Icon {
    @location(0) xywh: vec4<f32>,  // logical pixels
    @location(1) uv: vec4<f32>,    // u0 v0 u1 v1 in the atlas
    @location(2) tint: vec4<f32>,
    @location(3) mode: f32,        // 0: a mask, tinted; 1: an image, multiplied by the tint
};

struct IOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tint: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) mode: f32,
};

@vertex
fn vs_icon(@builtin(vertex_index) vi: u32, i: Icon) -> IOut {
    let c = CORNERS[vi];
    let p = (i.xywh.xy + c * i.xywh.zw) * o.screen.z;
    var out: IOut;
    out.pos = px_to_clip(p);
    out.tint = to_linear(i.tint);
    out.uv = mix(i.uv.xy, i.uv.zw, c);
    out.mode = i.mode;
    return out;
}

@fragment
fn fs_icon(v: IOut) -> @location(0) vec4<f32> {
    let t = textureSample(atlas, atlas_samp, v.uv);
    if (v.mode < 0.5) {
        return vec4<f32>(v.tint.rgb, v.tint.a * t.a);
    }
    return t * v.tint;
}
