// Rok's rendered view: a progressive voxel path tracer.
//
// One compute pass traces one sample per pixel through the scene volume (all
// visible models in one 3D texture) and adds it to the running mean.
// Occupancy masks (a bit per cell in 4^3 blocks, a bit per block in 16^3
// regions) let a ray skip empty space and test cells without a texture read. Then a few
// a-trous passes filter the light (the colour divided by the first hit's
// albedo, so voxel colours stay sharp), guided by the first hit's normal
// and distance, which a voxel scene gives exactly. A last render pass shows
// the result, tone-mapped, and writes depth for the overlays.

struct TraceU {
    inv_vp: mat4x4<f32>,
    vp: mat4x4<f32>,
    cam: vec4<f32>,        // xyz = eye; w = the voxel grid lines' strength
    rect: vec4<f32>,       // the view in the target: x, y, w, h (physical pixels)
    dims: vec4<u32>,       // x, y = trace size; z = sample index; w = flags (1 ground, 2 transparent, 4 backdrop)
    vol_lo: vec4<i32>,     // xyz = the volume's world corner; w = emitter count
    vol_size: vec4<i32>,   // xyz = the volume's size in cells; w = 1 when there is a volume
    bdim: vec4<i32>,       // x = where the block masks start in occ
    sky_top: vec4<f32>,    // rgb: the sky's radiance at the zenith
    sky_horizon: vec4<f32>,
    ground: vec4<f32>,     // rgb = albedo; w = the ground's height
    backdrop: vec4<f32>,   // rgb: what the camera sees where nothing is (flag 4)
    params: vec4<f32>,     // x = exposure (a factor), y = bounces, z = emitter weight, w = emitter scale
    part: vec4<u32>,       // x = stride (1, or 2: this pass traces one pixel of each 2 x 2), yz = offset
    look: vec4<f32>,       // x = lamp count, y = tone (0 neutral, 1 clip, 2-4 film), z = bloom, w = outline
    // Two per lamp: a sun is (direction, 0), (irradiance, cos of the half
    // width); a sphere is (world position, 1), (radiant intensity, radius).
    lamps: array<vec4<f32>, 8>,
};

struct DenoiseU {
    step: i32,
    src: u32,   // 0 the mean, 1 ping, 2 pong
    dst: u32,   // 1 ping, 2 pong, 3 out
    last: u32,
};

@group(0) @binding(0) var<uniform> u: TraceU;
@group(0) @binding(1) var vox: texture_3d<u32>;
// The occupancy masks: first one per 16^3 region (a bit per 4^3 block), then
// from bdim.x on one per 4^3 block (a bit per cell).
@group(0) @binding(2) var<storage, read> occ: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> mats: array<vec4<f32>>;     // 3 per palette slot
@group(0) @binding(4) var<storage, read> lights: array<vec4<i32>>;   // emitter cells (volume space), w = slot
@group(0) @binding(5) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gbuf: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> ping: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> pong: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> outbuf: array<vec2<u32>>;
@group(0) @binding(10) var<uniform> dn: DenoiseU;

const PI: f32 = 3.14159265;
const RIS_CANDIDATES: i32 = 8;
const INDIRECT_CLAMP: f32 = 2.0;
const BRICK: i32 = 8;
const MISS: u32 = 0u;
const VOXEL: u32 = 1u;
const GROUND: u32 = 2u;
// gbuf.z flags. The low 16 bits are the flags below; the high 16 bits are the outline ink: the
// running mean over the samples of "this sample's first hit is next to a step back", 0 to 65535.
const F_SKY: u32 = 16u;
const F_SPEC: u32 = 32u;
const F_EMIT: u32 = 64u;
const F_GROUND: u32 = 128u;

// Material types (MaterialTable.type_codes).
const T_METAL: f32 = 1.0;
const T_GLASS: f32 = 2.0;
const T_EMIT: f32 = 3.0;
const T_PLASTIC: f32 = 4.0;

// --- random numbers -----------------------------------------------------------------------
//
// A per-pixel hash rotates a rank-1 lattice (the R2 sequence) per dimension:
// low-discrepancy over the samples of one pixel, decorrelated between pixels.

var<private> seed: u32;
var<private> dim: u32;
var<private> sample_n: u32;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash01(v: u32) -> f32 {
    return f32(pcg(v) >> 8u) / 16777216.0;
}

fn rand2() -> vec2<f32> {
    let r = vec2<f32>(hash01(seed ^ (dim * 0x9e3779b9u)), hash01(seed + dim * 0x85ebca6bu + 17u));
    dim = dim + 1u;
    let n = f32(sample_n);
    return fract(r + n * vec2<f32>(0.7548776662, 0.5698402910));
}

fn rand1() -> f32 {
    return rand2().x;
}

// --- the volume -----------------------------------------------------------------------------

fn load_v(c: vec3<i32>) -> u32 {
    return textureLoad(vox, vec3<i32>(c.z, c.y, c.x), 0).r;
}

fn mat_a(i: u32) -> vec4<f32> { return mats[i * 3u]; }        // albedo (linear), emission strength
fn mat_b(i: u32) -> vec4<f32> { return mats[i * 3u + 1u]; }   // metallic, roughness, transmission, ior
fn mat_c(i: u32) -> vec4<f32> { return mats[i * 3u + 2u]; }   // specular, type

struct Hit {
    kind: u32,
    t: f32,
    n: vec3<f32>,     // faces the incoming ray
    cell: vec3<i32>,  // volume space
    v: u32,           // the palette slot hit; 0 for air left from inside a medium
};

fn hit_here(v: u32, medium: u32) -> bool {
    if (medium == 0u) { return v != 0u; }
    return v != medium;
}

// Where the ray enters the volume: t, the first cell, and the axis it came
// in across. t < 0: it misses the volume (or reaches it after tmax).
struct Entry {
    t: f32,
    cell: vec3<i32>,
    axis: i32,
};

fn enter(ol: vec3<f32>, d: vec3<f32>, inv: vec3<f32>, tmax: f32) -> Entry {
    var e: Entry;
    e.t = -1.0;
    let size = u.vol_size.xyz;
    let t1 = -ol * inv;
    let t2 = (vec3<f32>(size) - ol) * inv;
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    let tin = max(max(lo.x, lo.y), lo.z);
    let tout = min(min(hi.x, hi.y), hi.z);
    if (tout < max(tin, 0.0) || tin > tmax) { return e; }
    e.t = max(tin, 0.0);
    e.axis = 0;
    if (lo.y >= lo.x && lo.y >= lo.z) { e.axis = 1; }
    if (lo.z >= lo.x && lo.z >= lo.y) { e.axis = 2; }
    e.cell = clamp(vec3<i32>(floor(ol + d * (e.t + 1e-4))), vec3<i32>(0), size - vec3<i32>(1));
    return e;
}

fn bit64(m: vec2<u32>, i: u32) -> bool {
    if (i < 32u) { return ((m.x >> i) & 1u) != 0u; }
    return ((m.y >> (i - 32u)) & 1u) != 0u;
}

fn hit_voxel(t: f32, cell: vec3<i32>, v: u32, axis: i32, step: vec3<i32>) -> Hit {
    var h: Hit;
    h.kind = VOXEL;
    h.t = t;
    h.cell = cell;
    h.v = v;
    var n = vec3<f32>(0.0);
    n[axis] = -f32(step[axis]);
    h.n = n;
    return h;
}

// The first solid cell along the ray (or the ground), skipping empty space
// in three levels: 16^3 regions and 4^3 blocks whose occupancy masks say
// they are empty, then single cells by the block's 64-bit mask. The voxel's
// value is read only at the hit.
fn trace_air(o: vec3<f32>, d_in: vec3<f32>, tmax: f32) -> Hit {
    var h: Hit;
    h.kind = MISS;
    h.t = tmax;
    var d = d_in;
    d = select(d, vec3<f32>(1e-7), abs(d) < vec3<f32>(1e-7));
    let inv = 1.0 / d;
    if ((u.dims.w & 1u) != 0u && d.z < 0.0) {
        let tg = (u.ground.w - o.z) * inv.z;
        if (tg > 1e-4 && tg < h.t) {
            h.kind = GROUND;
            h.t = tg;
            h.n = vec3<f32>(0.0, 0.0, 1.0);
        }
    }
    if (u.vol_size.w == 0) { return h; }
    let size = u.vol_size.xyz;
    let ol = o - vec3<f32>(u.vol_lo.xyz);
    let e = enter(ol, d, inv, h.t);
    if (e.t < 0.0) { return h; }
    var t = e.t;
    var cell = e.cell;
    var axis = e.axis;
    let step = select(vec3<i32>(-1), vec3<i32>(1), d >= vec3<f32>(0.0));
    let fwd = select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
    let td = abs(inv);
    var tm = (vec3<f32>(cell) + fwd - ol) * inv;
    let bd = (size + vec3<i32>(3)) / 4;
    let cd = (size + vec3<i32>(15)) / 16;
    var last_c = -1;
    var cmask = vec2<u32>(0u);
    var last_b = -1;
    var bmask = vec2<u32>(0u);
    for (var i = 0; i < 4096; i = i + 1) {
        if (t > h.t) { return h; }
        let c = cell >> vec3<u32>(4u);
        let ci = (c.x * cd.y + c.y) * cd.z + c.z;
        if (ci != last_c) {
            cmask = occ[ci];
            last_c = ci;
        }
        var skip = 0;
        var box = vec3<i32>(0);
        let b = cell >> vec3<u32>(2u);
        if (cmask.x == 0u && cmask.y == 0u) {
            skip = 16;
            box = c * 16;
        } else {
            let lb = b & vec3<i32>(3);
            if (!bit64(cmask, u32(lb.x + 4 * lb.y + 16 * lb.z))) {
                skip = 4;
                box = b * 4;
            }
        }
        if (skip == 0) {
            let bi = (b.x * bd.y + b.y) * bd.z + b.z;
            if (bi != last_b) {
                bmask = occ[u.bdim.x + bi];
                last_b = bi;
            }
            let lc = cell & vec3<i32>(3);
            if (bit64(bmask, u32(lc.x + 4 * lc.y + 16 * lc.z))) {
                let v = load_v(cell);
                if (h.kind == GROUND && t > h.t) { return h; }
                return hit_voxel(t, cell, v, axis, step);
            }
            // The next cell.
            var ax = 0;
            if (tm.y < tm.x && tm.y <= tm.z) { ax = 1; }
            if (tm.z < tm.x && tm.z < tm.y) { ax = 2; }
            t = tm[ax];
            tm[ax] = tm[ax] + td[ax];
            cell[ax] = cell[ax] + step[ax];
            axis = ax;
            if (cell[ax] < 0 || cell[ax] >= size[ax]) { return h; }
            continue;
        }
        // An empty box: jump to where the ray leaves it.
        let bo = vec3<f32>(box);
        let sf = f32(skip);
        let tb = (bo + fwd * sf - ol) * inv;
        var ax = 0;
        if (tb.y < tb.x && tb.y <= tb.z) { ax = 1; }
        if (tb.z < tb.x && tb.z < tb.y) { ax = 2; }
        let te = tb[ax];
        var nc = vec3<i32>(floor(ol + d * te));
        nc[ax] = box[ax] + select(-1, skip, step[ax] > 0);
        for (var k = 0; k < 3; k = k + 1) {
            if (k != ax) { nc[k] = clamp(nc[k], box[k], box[k] + skip - 1); }
        }
        // The grid can end inside the box. A ray that left the grid on the way is gone: a
        // cell at the grid's edge is not on its path.
        if (any(nc < vec3<i32>(0)) || any(nc >= size)) { return h; }
        cell = nc;
        axis = ax;
        t = te;
        tm = (vec3<f32>(cell) + fwd - ol) * inv;
    }
    return h;
}

// Inside glass of slot `medium`: the first cell that is not that glass. Air
// counts (v = 0), and so does leaving the volume. Glass is thin: cell by cell.
fn trace_medium(o: vec3<f32>, d_in: vec3<f32>, medium: u32, tmax: f32) -> Hit {
    var h: Hit;
    h.kind = MISS;
    h.t = tmax;
    var d = d_in;
    d = select(d, vec3<f32>(1e-7), abs(d) < vec3<f32>(1e-7));
    let inv = 1.0 / d;
    let size = u.vol_size.xyz;
    let ol = o - vec3<f32>(u.vol_lo.xyz);
    let e = enter(ol, d, inv, h.t);
    if (e.t < 0.0) { return h; }
    var t = e.t;
    var cell = e.cell;
    var axis = e.axis;
    let step = select(vec3<i32>(-1), vec3<i32>(1), d >= vec3<f32>(0.0));
    let fwd = select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
    let td = abs(inv);
    var tm = (vec3<f32>(cell) + fwd - ol) * inv;
    for (var i = 0; i < 1024; i = i + 1) {
        if (t > h.t) { return h; }
        let v = load_v(cell);
        if (v != medium) { return hit_voxel(t, cell, v, axis, step); }
        var ax = 0;
        if (tm.y < tm.x && tm.y <= tm.z) { ax = 1; }
        if (tm.z < tm.x && tm.z < tm.y) { ax = 2; }
        t = tm[ax];
        tm[ax] = tm[ax] + td[ax];
        cell[ax] = cell[ax] + step[ax];
        axis = ax;
        if (cell[ax] < 0 || cell[ax] >= size[ax]) {
            return hit_voxel(t, cell, 0u, ax, step);  // out of the volume: into the air
        }
    }
    return h;
}

fn trace(o: vec3<f32>, d: vec3<f32>, medium: u32, tmax: f32) -> Hit {
    if (medium == 0u) { return trace_air(o, d, tmax); }
    return trace_medium(o, d, medium, tmax);
}

// How much light gets from o along d over a distance tmax: 0 behind an
// opaque voxel; glass on the way tints it (no bending: a cheap shadow).
fn transmit(o_in: vec3<f32>, d: vec3<f32>, tmax_in: f32, target_cell: vec3<i32>, has_target: bool) -> vec3<f32> {
    var tr = vec3<f32>(1.0);
    var o = o_in;
    var tmax = tmax_in;
    for (var k = 0; k < 6; k = k + 1) {
        let h = trace(o, d, 0u, tmax);
        if (h.kind == MISS) { return tr; }
        if (h.kind == GROUND) { return vec3<f32>(0.0); }
        if (has_target && all(h.cell == target_cell)) { return tr; }
        let b = mat_b(h.v);
        if (b.z <= 0.0) { return vec3<f32>(0.0); }
        tr = tr * mix(vec3<f32>(1.0), mat_a(h.v).rgb, b.z);
        // Through the glass to where it ends.
        let p = o + d * (h.t + 1e-3);
        let e = trace(p, d, h.v, tmax - h.t);
        if (e.kind != VOXEL || e.v != 0u) {
            if (e.kind == MISS) { return tr; }
            if (has_target && e.kind == VOXEL && all(e.cell == target_cell)) { return tr; }
            return vec3<f32>(0.0);
        }
        o = p + d * (e.t + 1e-3);
        tmax = tmax - h.t - e.t;
        if (tmax <= 0.0) { return tr; }
    }
    return tr;
}

// --- light and materials ----------------------------------------------------------------------

fn sky(d: vec3<f32>) -> vec3<f32> {
    let up = clamp(d.z, 0.0, 1.0);
    let c = mix(u.sky_horizon.rgb, u.sky_top.rgb, sqrt(up));
    let below = mix(u.sky_horizon.rgb * 0.6, u.sky_horizon.rgb, clamp(d.z + 1.0, 0.0, 1.0));
    return select(below, c, d.z >= 0.0);
}

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_dir(n: vec3<f32>) -> vec3<f32> {
    let r = rand2();
    let phi = 2.0 * PI * r.x;
    let s = sqrt(r.y);
    let l = vec3<f32>(cos(phi) * s, sin(phi) * s, sqrt(max(0.0, 1.0 - r.y)));
    return normalize(onb(n) * l);
}

// A direction in a cone of half angle acos(cos_max) around axis.
fn cone_dir(axis: vec3<f32>, cos_max: f32) -> vec3<f32> {
    let r = rand2();
    let c = 1.0 - r.x * (1.0 - cos_max);
    let s = sqrt(max(0.0, 1.0 - c * c));
    let phi = 2.0 * PI * r.y;
    return normalize(onb(axis) * vec3<f32>(cos(phi) * s, sin(phi) * s, c));
}

// Rough reflection: the mirror direction, turned by up to the roughness.
fn glossy(d: vec3<f32>, n: vec3<f32>, rough: f32) -> vec3<f32> {
    let m = reflect(d, n);
    let r = rough * rough;
    return normalize(mix(m, cosine_dir(m), r));
}

fn fresnel(cos_i: f32, ior: f32) -> f32 {
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_i, 0.0, 1.0), 5.0);
}

// Light that reaches point p with normal n straight from the sun and (with
// `emitters`) from one emitting voxel, for a matte surface of albedo 1.
fn direct(p: vec3<f32>, n: vec3<f32>, emitters: bool) -> vec3<f32> {
    var l = vec3<f32>(0.0);
    // One lamp, picked at random, for all of them.
    let lamp_count = i32(u.look.x);
    if (lamp_count > 0) {
        let k = min(i32(rand1() * f32(lamp_count)), lamp_count - 1);
        let a = u.lamps[2 * k];
        let b = u.lamps[2 * k + 1];
        if (a.w < 0.5) {
            // A sun: a direction in its cone; the colour is its irradiance.
            let sd = cone_dir(a.xyz, b.w);
            let cs = dot(n, sd);
            if (cs > 0.0) {
                l = l + b.rgb * cs / PI * f32(lamp_count) * transmit(p, sd, 1e30, vec3<i32>(0), false);
            }
        } else {
            // A glowing sphere: a point on it; the colour is its intensity.
            let r = rand2();
            let z = 1.0 - 2.0 * r.x;
            let phi = 2.0 * PI * r.y;
            let sr = sqrt(max(0.0, 1.0 - z * z));
            let q = a.xyz + b.w * vec3<f32>(cos(phi) * sr, sin(phi) * sr, z);
            let to = q - p;
            let dist2 = max(dot(to, to), 0.25);
            let dist = sqrt(dist2);
            let ld = to / dist;
            let cs = dot(n, ld);
            if (cs > 0.0) {
                l = l + b.rgb * cs / dist2 / PI * f32(lamp_count) * transmit(p, ld, dist, vec3<i32>(0), false);
            }
        }
    }
    let count = u.vol_lo.w;
    if (count > 0 && emitters) {
        // Resampled importance sampling (RIS): look at a few random emitters,
        // guess what each gives here (its power, distance and angle, no ray),
        // keep one in proportion to that, and trace one shadow ray to it.
        // Near and bright emitters win often, far and dim ones seldom, and the
        // weight W keeps the sum right.
        var chosen = -1;
        var chosen_guess = 0.0;
        var wsum = 0.0;
        for (var k = 0; k < RIS_CANDIDATES; k = k + 1) {
            let i = min(i32(rand1() * f32(count)), count - 1);
            let e = lights[i];
            let to = vec3<f32>(u.vol_lo.xyz) + vec3<f32>(e.xyz) + vec3<f32>(0.5) - p;
            let dist2 = max(dot(to, to), 0.25);
            let c = dot(n, to) * inverseSqrt(dist2);
            if (c <= 0.0) { continue; }
            let a = mat_a(u32(e.w));
            let guess = dot(a.rgb * a.w, vec3<f32>(0.2126, 0.7152, 0.0722)) * c / dist2;
            let w = guess * f32(count);  // the guess over the chance of drawing it (1 / count)
            wsum = wsum + w;
            if (w > 0.0 && rand1() * wsum < w) {
                chosen = i;
                chosen_guess = guess;
            }
        }
        if (chosen >= 0 && chosen_guess > 0.0) {
            let e = lights[chosen];
            let q = vec3<f32>(u.vol_lo.xyz) + vec3<f32>(e.xyz) + vec3<f32>(rand2(), rand1());
            let to = q - p;
            let dist2 = max(dot(to, to), 0.25);
            let dist = sqrt(dist2);
            let ld = to / dist;
            let c = dot(n, ld);
            if (c > 0.0) {
                let a = mat_a(u32(e.w));
                // A unit cube shows about 1.5 cells of area on average.
                let le = a.rgb * a.w * u.params.w;
                let ris_w = wsum / (f32(RIS_CANDIDATES) * chosen_guess);
                let g = c * 1.5 / dist2 * u.params.z * ris_w;
                l = l + le * g / PI * transmit(p, ld, dist + 1.0, e.xyz, true);
            }
        }
    }
    return l;
}

// --- the trace pass ---------------------------------------------------------------------------

fn pixel_ray(px: vec2<f32>) -> array<vec3<f32>, 2> {
    let tsize = vec2<f32>(u.dims.xy);
    let ndc = vec2<f32>(px.x / tsize.x * 2.0 - 1.0, 1.0 - px.y / tsize.y * 2.0);
    let near4 = u.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let far4 = u.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let o = near4.xyz / near4.w;
    let f = far4.xyz / far4.w;
    return array<vec3<f32>, 2>(o, normalize(f - o));
}

// The distance from the eye to the first surface in pixel px of the first sample's picture, 1e30 for the sky.
fn first_t(px: vec2<i32>) -> f32 {
    let g = gb(px);
    if ((g.z & F_SKY) != 0u) { return 1e30; }
    return length(world_at(vec2<f32>(px) + 0.5, bitcast<f32>(g.y)) - u.cam.xyz);
}

// The outline ink of one sample: 1 when its first hit (distance t0, through the pixel position `at`)
// lies next to a step back to something much farther, or to the floor behind it, within the
// line's reach in pixels; else 0.
// The mean over the samples is the line's coverage of the pixel, so its edges have no stairs.
// The first sample has no picture to look at yet and traces its probes always; later ones trace
// them only near a depth step of the first sample's picture.
fn ink_sample(pix: vec2<i32>, at: vec2<f32>, t0: f32, first: bool) -> f32 {
    // Each sample reaches a little more or less (a hash of its pixel and number, so the
    // path's own random numbers stay as they were): the line's inner edge fades over about
    // two pixels instead of one.
    let spread = f32(pcg(u32(pix.x) * 1973u + u32(pix.y) * 9277u + sample_n * 26699u + 7u) & 0xFFFFu) / 65536.0;
    let reach = (2.5 + max(u.look.w - 1.0, 0.0) * 2.0) * (0.7 + 0.6 * spread);
    var dirs = array<vec2<f32>, 8>(
        vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0),
        vec2<f32>(0.70710678, 0.70710678), vec2<f32>(-0.70710678, 0.70710678),
        vec2<f32>(0.70710678, -0.70710678), vec2<f32>(-0.70710678, -0.70710678));
    if (!first) {
        let steps = i32(ceil((2.5 + max(u.look.w - 1.0, 0.0) * 2.0) * 1.3 + 1.5));
        var tmin = first_t(pix);
        var tmax = tmin;
        var floor_near = false;  // the floor shows nearby: this pixel is on the model's outline
        for (var k = 0; k < 8; k = k + 1) {
            for (var d = 1; d <= 10; d = d + 1) {
                if (d > steps) { break; }
                let q = pix + vec2<i32>(round(dirs[k] * f32(d)));
                let tn = first_t(q);
                tmin = min(tmin, tn);
                tmax = max(tmax, tn);
                if ((gb(q).z & F_GROUND) != 0u) { floor_near = true; }
            }
        }
        if (tmax <= tmin * 1.02 + 0.3 && !floor_near) { return 0.0; }
    }
    for (var k = 0; k < 8; k = k + 1) {
        let ray = pixel_ray(at + dirs[k] * reach);
        let h = trace(ray[0], ray[1], 0u, 1e30);
        // The floor counts at any depth: beside a model's foot it lies as near as the model,
        // in front of it nearer, and the line must still go round the model there.
        if (h.kind == MISS || h.kind == GROUND || h.t > t0 * 1.04 + 0.6) { return 1.0; }
    }
    return 0.0;
}

fn normal_code(n: vec3<f32>) -> u32 {
    if (n.x > 0.5) { return 0u; }
    if (n.x < -0.5) { return 1u; }
    if (n.y > 0.5) { return 2u; }
    if (n.y < -0.5) { return 3u; }
    if (n.z > 0.5) { return 4u; }
    return 5u;
}

@compute @workgroup_size(8, 8)
fn cs_trace(@builtin(global_invocation_id) gid_in: vec3<u32>) {
    let tw = u.dims.x;
    let th = u.dims.y;
    // A big view traces one pixel of each 2 x 2 block per frame (part.x = 2).
    let gid = vec3<u32>(gid_in.xy * u.part.x + u.part.yz, 0u);
    if (gid.x >= tw || gid.y >= th) { return; }
    let idx = gid.y * tw + gid.x;
    sample_n = u.dims.z;
    seed = pcg(idx * 9781u + 6271u);
    dim = 0u;
    // Sample 0 looks through the pixel's middle: a clean guide for the filter.
    var jitter = vec2<f32>(0.5);
    if (sample_n > 0u) { jitter = rand2(); }
    let ray = pixel_ray(vec2<f32>(gid.xy) + jitter);
    var o = ray[0];
    var d = ray[1];
    var tp = vec3<f32>(1.0);   // throughput
    var lsum = vec3<f32>(0.0);
    var alpha = 1.0;
    var medium = 0u;
    // An emitter hit counts unless the last vertex already sampled the
    // emitters directly (camera rays and mirrors do not).
    var count_emit = true;
    var blurred = false;  // the path went through a rough or matte bounce
    var g_albedo = vec3<f32>(1.0);
    var g_depth = 1.0;
    var g_t = 1e30;
    var g_flags = F_SKY;
    var ink = 0.0;  // this sample's outline ink (see ink_sample)
    // The filter's guide: the first matte surface along the path. Behind glass
    // or in a mirror that is the wall seen through it, tinted on the way, so
    // glass and metal get filtered too; the depth stays the first surface's.
    var guided = false;
    var travelled = 0.0;
    let bounces = i32(u.params.y);
    var diffuse_bounces = 0;
    for (var bounce = 0; bounce < 24; bounce = bounce + 1) {
        let h = trace(o, d, medium, 1e30);
        if (h.kind == MISS) {
            if (medium == 0u) {
                // A clear picture keeps only the model. A miss adds no sky, so
                // the colour stays the surface and the alpha stays the cover.
                // Otherwise the camera sees the backdrop where there is one,
                // and the light comes from the sky.
                let clear = bounce == 0 && (u.dims.w & 2u) != 0u;
                if (!clear) {
                    var bg = sky(d);
                    if (bounce == 0 && (u.dims.w & 4u) != 0u) { bg = u.backdrop.rgb; }
                    lsum = lsum + tp * bg;
                } else {
                    alpha = 0.0;
                }
            }
            break;
        }
        let p = o + d * h.t;
        travelled = travelled + h.t;
        var n = h.n;
        if (h.kind == VOXEL && medium != 0u) {
            if (h.v == 0u) {
                // Out of the glass into the air: bend away, or reflect inside.
                let ior = mat_b(medium).w;
                let cos_i = -dot(d, n);
                let f = fresnel(cos_i, ior);
                let rd = refract(d, n, ior);
                if (dot(rd, rd) < 1e-6 || rand1() < f) {
                    d = reflect(d, n);
                    o = p + n * 1e-3;
                } else {
                    d = normalize(rd);
                    o = p - n * 1e-3;
                    medium = 0u;
                }
                continue;
            }
            medium = 0u;  // another voxel inside the glass: shade it as from the air
        }
        var albedo = u.ground.rgb;
        var ma = vec4<f32>(albedo, 0.0);
        var mb = vec4<f32>(0.0, 0.9, 0.0, 1.45);
        var mc = vec4<f32>(0.5, 0.0, 0.0, 0.0);
        if (h.kind == VOXEL) {
            ma = mat_a(h.v);
            mb = mat_b(h.v);
            mc = mat_c(h.v);
            albedo = ma.rgb;
        }
        let emitting = h.kind == VOXEL && ma.w > 0.0;
        if (bounce == 0) {
            let clip = u.vp * vec4<f32>(p, 1.0);
            g_depth = clamp(clip.z / clip.w, 0.0, 1.0);
            g_flags = F_SPEC;  // until a matte surface is found
            if (u.look.w > 0.0 && h.kind == VOXEL) {  // the ground has no outline
                ink = ink_sample(vec2<i32>(gid.xy), vec2<f32>(gid.xy) + jitter, h.t, sample_n == 0u);
            }
        }
        // Glass and polished metal show what lies beyond them; rough metal
        // is its own surface for the filter, like a matte one.
        let matte = !(mb.z > 0.0 || (mb.x > 0.5 && mb.y < 0.3));
        if (!guided && (matte || emitting)) {
            guided = true;
            g_albedo = clamp(tp * albedo, vec3<f32>(0.0), vec3<f32>(1.0));
            g_t = travelled;
            g_flags = normal_code(n);
            if (emitting) { g_flags = g_flags | F_EMIT; }
            if (h.kind == GROUND) { g_flags = g_flags | F_GROUND; }
        }
        if (emitting) {
            // Emitters end the path. Their light counts here unless the
            // vertex before already sampled them.
            if (count_emit) {
                var e = tp * albedo * ma.w * u.params.w;
                // After a blurry bounce, a small bright emitter found by
                // chance counts only so much (clamped):
                // else it leaves a bright dot.
                if (blurred) {
                    let lum = dot(e, vec3<f32>(0.2126, 0.7152, 0.0722));
                    if (lum > INDIRECT_CLAMP) { e = e * (INDIRECT_CLAMP / lum); }
                }
                lsum = lsum + e;
            }
            break;
        }
        let pick = rand1();
        if (mb.z > 0.0 && pick < mb.z) {
            // Glass: reflect or go in.
            let cos_i = -dot(d, n);
            let ior = mb.w;
            let f = fresnel(cos_i, ior);
            if (rand1() < f) {
                d = glossy(d, n, mb.y * 0.5);
                o = p + n * 1e-3;
            } else {
                let rd = refract(d, n, 1.0 / ior);
                d = normalize(mix(rd, cosine_dir(-n), mb.y * mb.y * 0.5));
                o = p - n * 1e-3;
                medium = h.v;
                tp = tp * albedo;
            }
            count_emit = true;
        } else if (mb.x > 0.0 && pick < mb.z + mb.x) {
            // Metal: a coloured, rough mirror.
            d = glossy(d, n, mb.y);
            if (dot(d, n) <= 0.0) { break; }
            o = p + n * 1e-3;
            tp = tp * albedo;
            count_emit = true;
            if (mb.y >= 0.3) { blurred = true; }
        } else if (mc.y == T_PLASTIC && rand1() < mc.x * fresnel(-dot(d, n), 1.5) * 2.0) {
            // Plastic's clear coat: a white, rough mirror.
            d = glossy(d, n, mb.y);
            if (dot(d, n) <= 0.0) { break; }
            o = p + n * 1e-3;
            count_emit = true;
            if (mb.y >= 0.3) { blurred = true; }
        } else {
            // Matte: the direct light, then a bounce.
            // The first two matte vertices sample the emitters directly; later
            // ones let the bounce find them (much the same light, fewer rays).
            let q = p + n * 1e-3;
            let sample_emitters = diffuse_bounces < 2;
            lsum = lsum + tp * albedo * direct(q, n, sample_emitters);
            diffuse_bounces = diffuse_bounces + 1;
            if (diffuse_bounces > bounces) { break; }
            d = cosine_dir(n);
            o = q;
            tp = tp * albedo;
            count_emit = !sample_emitters;
            blurred = true;
        }
        // Russian roulette after a few bounces.
        if (bounce >= 3) {
            let keep = clamp(max(tp.x, max(tp.y, tp.z)), 0.05, 1.0);
            if (rand1() > keep) { break; }
            tp = tp / keep;
        }
    }
    // A cap on a single sample keeps rare bright paths from leaving white dots.
    let lum = dot(lsum, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (lum > 16.0) { lsum = lsum * (16.0 / lum); }
    let s = vec4<f32>(lsum, alpha);
    if (sample_n == 0u) {
        accum[idx] = s;
        gbuf[idx] = vec4<u32>(pack4x8unorm(vec4<f32>(clamp(g_albedo, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0)),
                              bitcast<u32>(g_depth), g_flags | (u32(ink * 65535.0 + 0.5) << 16u), bitcast<u32>(g_t));
    } else {
        accum[idx] = mix(accum[idx], s, 1.0 / f32(sample_n + 1u));
        if (u.look.w > 0.0) {
            let z = gbuf[idx].z;
            let mean = mix(f32(z >> 16u) / 65535.0, ink, 1.0 / f32(sample_n + 1u));
            gbuf[idx].z = (z & 0xFFFFu) | (u32(mean * 65535.0 + 0.5) << 16u);
        }
    }
}

// --- the filter -------------------------------------------------------------------------------

fn pack_h(c: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(c.rg), pack2x16float(c.ba));
}

fn unpack_h(p: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(p.x), unpack2x16float(p.y));
}

// The albedo the filter divides by. Lifted a little: a dark plastic's white
// clear coat divided by its near-black albedo would turn into bright dots.
fn albedo_of(g: vec4<u32>) -> vec3<f32> {
    return unpack4x8unorm(g.x).rgb + vec3<f32>(0.15);
}

// The light at pixel i, without its albedo: the filter's input.
fn light_at(i: u32) -> vec4<f32> {
    if (dn.src == 0u) {
        let a = accum[i];
        let g = gbuf[i];
        if ((g.z & (F_SKY | F_EMIT)) != 0u) { return a; }
        return vec4<f32>(a.rgb / albedo_of(g), a.a);
    }
    if (dn.src == 1u) { return unpack_h(ping[i]); }
    return unpack_h(pong[i]);
}

fn put(i: u32, c: vec4<f32>) {
    if (dn.dst == 1u) {
        ping[i] = pack_h(c);
    } else if (dn.dst == 2u) {
        pong[i] = pack_h(c);
    } else {
        let g = gbuf[i];
        var r = c;
        if ((g.z & (F_SKY | F_EMIT)) == 0u) { r = vec4<f32>(c.rgb * albedo_of(g), c.a); }
        outbuf[i] = pack_h(r);
    }
}

@compute @workgroup_size(8, 8)
fn cs_denoise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = u.dims.x;
    let th = u.dims.y;
    if (gid.x >= tw || gid.y >= th) { return; }
    let i = gid.y * tw + gid.x;
    let c = light_at(i);
    let g = gbuf[i];
    if (dn.step == 0 || (g.z & (F_SKY | F_EMIT | F_SPEC)) != 0u) {
        put(i, c);
        return;
    }
    let tc = bitcast<f32>(g.w);
    let lc = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // The light's spread shrinks as samples add up: less blur later.
    let spp = f32(u.dims.z + 1u);
    let sigma_l = 4.0 / sqrt(spp) * (lc + 0.05);
    // 3 x 3 taps a pass, five passes of growing step: about the reach of 5 x 5
    // in four passes for 45 taps instead of 100.
    var kern = array<f32, 2>(0.5, 0.25);
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let q = vec2<i32>(gid.xy) + vec2<i32>(dx, dy) * dn.step;
            if (q.x < 0 || q.y < 0 || q.x >= i32(tw) || q.y >= i32(th)) { continue; }
            let j = u32(q.y) * tw + u32(q.x);
            let gq = gbuf[j];
            if ((gq.z & 15u) != (g.z & 15u) || (gq.z & (F_SKY | F_EMIT)) != 0u) { continue; }
            let tq = bitcast<f32>(gq.w);
            let wz = exp(-abs(tq - tc) / (0.02 * tc * f32(dn.step) + 1e-3));
            let cq = light_at(j);
            let lq = dot(cq.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            let wl = exp(-abs(lq - lc) / sigma_l);
            let w = kern[abs(dx)] * kern[abs(dy)] * wz * wl;
            sum = sum + cq * w;
            wsum = wsum + w;
        }
    }
    put(i, select(c, sum / wsum, wsum > 1e-6));
}

// --- the bloom --------------------------------------------------------------------------------
//
// The parts brighter than white, at a quarter of the size, blurred in four
// passes (ping and pong in turn) and added to the picture before the tone map.

fn bloom_size() -> vec2<u32> {
    return (u.dims.xy + vec2<u32>(3u)) / 4u;
}

fn bloom_get(i: u32) -> vec4<f32> {
    if (dn.src == 1u) { return unpack_h(ping[i]); }
    return unpack_h(pong[i]);
}

fn bloom_put(i: u32, c: vec4<f32>) {
    if (dn.dst == 1u) { ping[i] = pack_h(c); } else { pong[i] = pack_h(c); }
}

@compute @workgroup_size(8, 8)
fn cs_bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bs = bloom_size();
    if (gid.x >= bs.x || gid.y >= bs.y) { return; }
    var sum = vec3<f32>(0.0);
    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let px = min(gid.xy * 4u + vec2<u32>(x, y), u.dims.xy - vec2<u32>(1u));
            let c = unpack_h(outbuf[px.y * u.dims.x + px.x]).rgb * u.params.x;
            let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            sum = sum + c * (max(lum - 1.0, 0.0) / max(lum, 1e-4));
        }
    }
    ping[gid.y * bs.x + gid.x] = pack_h(vec4<f32>(sum / 16.0, 1.0));
}

fn bloom_blur(gid: vec2<u32>, dir: vec2<i32>) {
    let bs = bloom_size();
    if (gid.x >= bs.x || gid.y >= bs.y) { return; }
    var w = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    var sum = bloom_get(gid.y * bs.x + gid.x).rgb * w[0];
    for (var k = 1; k < 5; k = k + 1) {
        for (var sgn = -1; sgn <= 1; sgn = sgn + 2) {
            let q = clamp(vec2<i32>(gid) + dir * (k * sgn * dn.step), vec2<i32>(0), vec2<i32>(bs) - vec2<i32>(1));
            sum = sum + bloom_get(u32(q.y) * bs.x + u32(q.x)).rgb * w[k];
        }
    }
    bloom_put(gid.y * bs.x + gid.x, vec4<f32>(sum, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_bloom_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    bloom_blur(gid.xy, vec2<i32>(1, 0));
}

@compute @workgroup_size(8, 8)
fn cs_bloom_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    bloom_blur(gid.xy, vec2<i32>(0, 1));
}

// --- the display pass -------------------------------------------------------------------------

struct VOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_show(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    return o;
}

struct SOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Khronos PBR Neutral (Apache-2.0, see CREDITS.md): keeps colours close to
// the palette, rolls off highlights.
fn neutral(c_in: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;
    let desat = 0.15;
    var c = c_in;
    let x = min(c.r, min(c.g, c.b));
    let off = select(0.04, x - 6.25 * x * x, x < 0.08);
    c = c - off;
    let peak = max(c.r, max(c.g, c.b));
    if (peak < start) { return c; }
    let d = 1.0 - start;
    let np = 1.0 - d * d / (peak + d - start);
    c = c * (np / peak);
    let g = 1.0 - 1.0 / (desat * (peak - np) + 1.0);
    return mix(c, vec3<f32>(np), g);
}

// Film: a film-like roll-off that keeps bright colours from turning to flat
// white. Each channel goes through a logistic curve in log2: mid grey (0.18)
// shows at 0.5 and one stop moves the display by about an eighth; the curve
// is scaled so that 16 (four stops over white) reaches white. Near the top a
// colour fades towards white, as a bright lamp does on film. `look` 0 base,
// 1 punchy (deeper mid tones, more saturation), 2 high contrast. Returns
// linear colour for the sRGB target.
fn film(c: vec3<f32>, look: i32) -> vec3<f32> {
    var slope = 0.5;
    var gamma = 1.0;
    var sat = 1.0;
    if (look == 1) { slope = 0.56; gamma = 1.4; sat = 1.2; }
    if (look == 2) { slope = 0.6; gamma = 1.05; sat = 1.0; }
    let mid = log2(0.18);
    let lo = -12.5;
    let hi = 4.0;
    let x = clamp(log2(max(c, vec3<f32>(1e-10))), vec3<f32>(lo), vec3<f32>(hi));
    let s = 1.0 / (1.0 + exp(-slope * (x - mid)));
    let s_lo = 1.0 / (1.0 + exp(-slope * (lo - mid)));
    let s_hi = 1.0 / (1.0 + exp(-slope * (hi - mid)));
    var v = pow(clamp((s - s_lo) / (s_hi - s_lo), vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(gamma));
    let luma = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
    v = luma + sat * (v - luma);
    let peak = max(v.r, max(v.g, v.b));
    v = mix(v, vec3<f32>(peak), 0.6 * smoothstep(0.85, 1.0, peak));
    return pow(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
}

fn tone(c: vec3<f32>) -> vec3<f32> {
    let mode = i32(u.look.y + 0.5);
    if (mode == 1) { return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)); }  // clipped
    if (mode >= 2) { return film(c, mode - 2); }
    return neutral(c);
}

fn gb(px: vec2<i32>) -> vec4<u32> {
    let q = clamp(px, vec2<i32>(0), vec2<i32>(i32(u.dims.x) - 1, i32(u.dims.y) - 1));
    return gbuf[u32(q.y) * u.dims.x + u32(q.x)];
}

// The world point of trace pixel px at clip depth z.
fn world_at(px: vec2<f32>, z: f32) -> vec3<f32> {
    let tsize = vec2<f32>(u.dims.xy);
    let ndc = vec2<f32>(px.x / tsize.x * 2.0 - 1.0, 1.0 - px.y / tsize.y * 2.0);
    let w = u.inv_vp * vec4<f32>(ndc, z, 1.0);
    return w.xyz / w.w;
}

// The outline ink of trace pixel px: the share of its samples on a shape edge.
fn ink_at(px: vec2<i32>) -> f32 {
    return f32(gb(px).z >> 16u) / 65535.0;
}

fn fetch(px: vec2<i32>) -> vec4<f32> {
    let tw = i32(u.dims.x);
    let th = i32(u.dims.y);
    let q = clamp(px, vec2<i32>(0), vec2<i32>(tw - 1, th - 1));
    return unpack_h(outbuf[u32(q.y * tw + q.x)]);
}

@fragment
fn fs_show(@builtin(position) frag: vec4<f32>) -> SOut {
    var o: SOut;
    let local = (frag.xy - u.rect.xy) / u.rect.zw;
    if (any(local < vec2<f32>(0.0)) || any(local >= vec2<f32>(1.0))) { discard; }
    let tsize = vec2<f32>(u.dims.xy);
    let tp = local * tsize - 0.5;
    let base = vec2<i32>(floor(tp));
    let f = tp - floor(tp);
    // Bilinear while the view moves at a lower resolution; exact at full size.
    var c = fetch(base);
    if (u.dims.x < u32(u.rect.z + 0.5)) {
        c = mix(mix(fetch(base), fetch(base + vec2<i32>(1, 0)), f.x),
                mix(fetch(base + vec2<i32>(0, 1)), fetch(base + vec2<i32>(1, 1)), f.x), f.y);
    } else {
        c = fetch(vec2<i32>(floor(local * tsize)));
    }
    // A clear picture stores the surface colour, and the alpha is the cover.
    // The mean is the surface times the cover, so divide the cover back out
    // before the tone map. A missed pixel stays black.
    var covered = c.rgb;
    if ((u.dims.w & 2u) != 0u) {
        covered = select(vec3<f32>(0.0), c.rgb / max(c.a, 1e-4), c.a > (1.0 / 512.0));
    }
    var rad = covered * u.params.x;
    // The bloom, read from ping at a quarter of the size.
    if (u.look.z > 0.0) {
        let bs = bloom_size();
        let bp = local * vec2<f32>(bs) - 0.5;
        let b0 = vec2<i32>(floor(bp));
        let bf = bp - floor(bp);
        let hi = vec2<i32>(bs) - vec2<i32>(1);
        let p00 = unpack_h(ping[u32(clamp(b0.y, 0, hi.y)) * bs.x + u32(clamp(b0.x, 0, hi.x))]).rgb;
        let p10 = unpack_h(ping[u32(clamp(b0.y, 0, hi.y)) * bs.x + u32(clamp(b0.x + 1, 0, hi.x))]).rgb;
        let p01 = unpack_h(ping[u32(clamp(b0.y + 1, 0, hi.y)) * bs.x + u32(clamp(b0.x, 0, hi.x))]).rgb;
        let p11 = unpack_h(ping[u32(clamp(b0.y + 1, 0, hi.y)) * bs.x + u32(clamp(b0.x + 1, 0, hi.x))]).rgb;
        rad = rad + u.look.z * mix(mix(p00, p10, bf.x), mix(p01, p11, bf.x), bf.y);
    }
    var col = tone(rad);
    let gp = clamp(vec2<i32>(floor(local * tsize)), vec2<i32>(0), vec2<i32>(i32(u.dims.x) - 1, i32(u.dims.y) - 1));
    let g = gb(gp);
    let surface = (g.z & F_SKY) == 0u;
    // Ink on shape edges, from the samples (ink_sample). The strength darkens up to 1 (black);
    // past 1 the line grows thicker.
    if (u.look.w > 0.0) {
        // Read like the colour: from the four nearest pixels while the trace is smaller than
        // the view, so the line keeps its width where a pixel would otherwise show twice.
        var cover = ink_at(gp);
        if (u.dims.x < u32(u.rect.z + 0.5)) {
            cover = mix(mix(ink_at(base), ink_at(base + vec2<i32>(1, 0)), f.x),
                        mix(ink_at(base + vec2<i32>(0, 1)), ink_at(base + vec2<i32>(1, 1)), f.x), f.y);
        }
        col = mix(col, col * (1.0 - min(u.look.w, 1.0)), cover);
    }
    // Lines on the voxel faces' borders. The strength darkens up to 1; past 1 the line grows wider.
    // Each line's distance is measured in screen pixels on the face's plane, so every line has the
    // same width whichever way the face turns, and its coverage of the pixel is exact. A line fades
    // out as its voxels shrink under about three pixels, where it would only turn the face dark.
    if (u.cam.w > 0.0 && surface && (g.z & F_GROUND) == 0u && (g.z & 15u) < 6u) {
        let pc = local * tsize;
        let axis = (g.z & 15u) / 2u;
        let lo = vec3<f32>(u.vol_lo.xyz);
        let hit = world_at(pc, bitcast<f32>(g.y));
        let plane = lo[axis] + round(hit[axis] - lo[axis]);
        let r0 = pixel_ray(pc);
        let rx = pixel_ray(pc + vec2<f32>(1.0, 0.0));
        let ry = pixel_ray(pc + vec2<f32>(0.0, 1.0));
        if (abs(r0[1][axis]) > 1e-4 && abs(rx[1][axis]) > 1e-4 && abs(ry[1][axis]) > 1e-4) {
            let c0 = r0[0] + r0[1] * ((plane - r0[0][axis]) / r0[1][axis]) - lo;
            let cx = rx[0] + rx[1] * ((plane - rx[0][axis]) / rx[1][axis]) - lo;
            let cy = ry[0] + ry[1] * ((plane - ry[0][axis]) / ry[1][axis]) - lo;
            let half_width = 0.75 + max(u.cam.w - 1.0, 0.0) * 1.2;
            var cover = 0.0;
            for (var a = 0u; a < 3u; a = a + 1u) {
                if (a != axis) {
                    let grad = max(length(vec2<f32>(cx[a] - c0[a], cy[a] - c0[a])), 1e-6);  // cells per pixel
                    let f = fract(c0[a]);
                    let edge_px = min(f, 1.0 - f) / grad;  // pixels to the nearest line of this axis
                    let sharp = clamp((1.0 / grad - 2.5) / 1.5, 0.0, 1.0);
                    // A line's edge fades over about 1.4 pixels (a smooth step, as wide as the line is).
                    cover = max(cover, (1.0 - smoothstep(half_width - 0.7, half_width + 0.7, edge_px)) * sharp);
                }
            }
            col = mix(col, col * (1.0 - min(u.cam.w, 1.0)), cover);
        }
    }
    o.color = vec4<f32>(col, c.a);
    // The overlays' 3D lines test against this depth. Each pixel holds one depth, from its middle
    // sample, at the trace's own size, so along an edge a line on the outline would test against
    // a point up to a pixel off and break up into beads. The farthest depth of the 3 x 3 pixels
    // around lets a line on an outline or a crease show whole; a line hidden behind a model stays
    // hidden, but for a pixel next to its outline.
    var far = bitcast<f32>(g.y);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            far = max(far, bitcast<f32>(gb(gp + vec2<i32>(dx, dy)).y));
        }
    }
    o.depth = far;
    return o;
}
