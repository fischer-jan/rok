// Ray march of one voxel model: a proxy cube is drawn, every pixel of it
// marches its ray through the model's 3D texture (DDA, one cell per step).

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    key_dir: vec4<f32>,
    fill_dir: vec4<f32>,
    viewport: vec4<f32>,  // x, y = view size; z = 1 for the voxel grid; w = pixel size of an orthographic view
    origin: vec4<f32>,    // x, y = the view's top-left corner in the target (quad view); z, w = depth cue near and length (agent pictures; w = 0: none)
    // The look (Host.set_look):
    look0: vec4<f32>,     // x = light (0 lit, 1 flat, 2 toon, 3 iso, 4 soft, 5 night, 6 ceramic), y = occlusion, z = ink lines, w = rim
    look1: vec4<f32>,     // x = sun shadows, y = flags (1 clay, 2 light ink, 4 CAD view, 8 see-through), z = ink width (UI pixels), w = UI scale
    look2: vec4<f32>,     // rgb = background (linear), w = 1 when set
};

struct ModelU {
    model: mat4x4<f32>,
    inv_model: mat4x4<f32>,
    size: vec4<u32>,   // xyz = grid size, w = 1 when the brick grid skips empty space
    info: vec4<u32>,   // x = step cap, y = fade (f32 bits), z = hidden, w: bit 0 a CAD field is uploaded, bit 1 a brick model (a seams field)
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var palette_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> mu: ModelU;
@group(1) @binding(1) var vox: texture_3d<u32>;
@group(1) @binding(2) var bricks: texture_3d<u32>;
@group(1) @binding(3) var cad_tex: texture_3d<u32>;
// A brick model's seams byte per cell: bit 2a (+axis a) and 2a+1 (-axis a)
// say the neighbour on that side is the same brick.
@group(1) @binding(4) var seams: texture_3d<u32>;

var<private> CUBE_IDX: array<u32, 36> = array<u32, 36>(
    0u, 4u, 6u, 0u, 6u, 2u,
    1u, 3u, 7u, 1u, 7u, 5u,
    0u, 1u, 5u, 0u, 5u, 4u,
    2u, 6u, 7u, 2u, 7u, 3u,
    0u, 2u, 3u, 0u, 3u, 1u,
    4u, 5u, 7u, 4u, 7u, 6u,
);

fn corner(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u));
}

// The voxel array is (x, y, z) in C order, so the texture is (z, y, x).
fn load_vox(c: vec3<i32>) -> u32 {
    return textureLoad(vox, vec3<i32>(c.z, c.y, c.x), 0).r;
}

// rgb: face normal, 1..255 around 128. a: corner bits, +x -x +y -y +z -z.
fn load_cad(c: vec3<i32>) -> vec4<u32> {
    return textureLoad(cad_tex, vec3<i32>(c.z, c.y, c.x), 0);
}

fn load_seams(c: vec3<i32>) -> u32 {
    return textureLoad(seams, vec3<i32>(c.z, c.y, c.x), 0).r;
}

// The same-brick bit of the side of a cell that faces `dir` (one unit axis vector).
fn same_bit(dir: vec3<i32>) -> u32 {
    var a = 0u;
    if (dir.y != 0) { a = 1u; }
    if (dir.z != 0) { a = 2u; }
    let neg = select(0u, 1u, (dir.x + dir.y + dir.z) < 0);
    return 1u << (2u * a + neg);
}

// A filled cell; outside the grid counts as empty.
fn solid(c: vec3<i32>) -> f32 {
    let size_i = vec3<i32>(mu.size.xyz);
    if (any(c < vec3<i32>(0)) || any(c >= size_i)) { return 0.0; }
    return select(0.0, 1.0, load_vox(c) != 0u);
}

// Occlusion at one corner of a face: 0 (open) .. 3 (closed in), from the two
// side cells and the corner cell in front of the face.
fn corner_ao(front: vec3<i32>, du: vec3<i32>, dv: vec3<i32>) -> f32 {
    let s1 = solid(front + du);
    let s2 = solid(front + dv);
    if (s1 > 0.5 && s2 > 0.5) { return 3.0; }
    return s1 + s2 + solid(front + du + dv);
}

// A ray toward the sun through this model: 1 when a filled cell blocks it.
fn shadowed(start: vec3<f32>, dir: vec3<f32>) -> f32 {
    var dl = dir;
    dl = select(dl, vec3<f32>(1e-7), abs(dl) < vec3<f32>(1e-7));
    let size_i = vec3<i32>(mu.size.xyz);
    var cell = vec3<i32>(floor(start));
    let step = select(vec3<i32>(-1), vec3<i32>(1), dl >= vec3<f32>(0.0));
    let fwd = select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
    let inv = 1.0 / dl;
    let td = abs(inv);
    var tm = (vec3<f32>(cell) + fwd - start) * inv;
    for (var i = 0; i < 160; i = i + 1) {
        var ax = 0;
        if (tm.y < tm.x && tm.y <= tm.z) { ax = 1; }
        if (tm.z < tm.x && tm.z < tm.y) { ax = 2; }
        cell[ax] = cell[ax] + step[ax];
        tm[ax] = tm[ax] + td[ax];
        if (any(cell < vec3<i32>(0)) || any(cell >= size_i)) { return 0.0; }
        if (load_vox(cell) != 0u) { return 1.0; }
    }
    return 0.0;
}

// Ambient occlusion in the voxels (View > Ambient Occlusion). Short rays leave the hit into the half
// space over its face, through the cells themselves; a ray that meets a voxel
// within `reach` cells darkens the point, the more the nearer the voxel. The
// directions are cosine weighted (a Fibonacci set) and turned per pixel by an
// interleaved-gradient angle, so the picture carries a fine, even grain
// instead of bands across a face. Returns the occluded share, 0..1.
const AO_RAYS: i32 = 12;
fn occlusion(start: vec3<f32>, nl: vec3<f32>, u: vec3<f32>, v: vec3<f32>, reach: f32, px: vec2<f32>) -> f32 {
    let size_i = vec3<i32>(mu.size.xyz);
    let rot = fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y)) * 6.2831853;
    var sum = 0.0;
    for (var k = 0; k < AO_RAYS; k = k + 1) {
        let z = sqrt((f32(k) + 0.5) / f32(AO_RAYS));
        let r = sqrt(max(1.0 - z * z, 0.0));
        let phi = f32(k) * 2.399963 + rot;
        var dir = u * (r * cos(phi)) + v * (r * sin(phi)) + nl * z;
        dir = select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
        let inv = 1.0 / dir;
        let step = select(vec3<i32>(-1), vec3<i32>(1), dir >= vec3<f32>(0.0));
        let fwd = select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
        let td = abs(inv);
        var cell = vec3<i32>(floor(start));
        var tm = (vec3<f32>(cell) + fwd - start) * inv;
        for (var i = 0; i < 48; i = i + 1) {
            var ax = 0;
            if (tm.y < tm.x && tm.y <= tm.z) { ax = 1; }
            if (tm.z < tm.x && tm.z < tm.y) { ax = 2; }
            let t = tm[ax];
            if (t > reach) { break; }
            cell[ax] = cell[ax] + step[ax];
            tm[ax] = tm[ax] + td[ax];
            if (any(cell < vec3<i32>(0)) || any(cell >= size_i)) { break; }
            if (load_vox(cell) != 0u) {
                sum = sum + (1.0 - t / reach);
                break;
            }
        }
    }
    return sum / f32(AO_RAYS);
}

fn load_brick(c: vec3<i32>) -> u32 {
    return textureLoad(bricks, vec3<i32>(c.z, c.y, c.x), 0).r;
}

fn pixel_ray(frag: vec4<f32>) -> array<vec3<f32>, 2> {
    let p = frag.xy - frame.origin.xy;
    let ndc = vec2<f32>(p.x / frame.viewport.x * 2.0 - 1.0, 1.0 - p.y / frame.viewport.y * 2.0);
    // From the near point to the far point: right for a perspective and an orthographic view.
    let near4 = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far4 = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let o = near4.xyz / near4.w;
    let far_p = far4.xyz / far4.w;
    return array<vec3<f32>, 2>(o, normalize(far_p - o));
}

@vertex
fn vs_model(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let local = corner(CUBE_IDX[vi]) * vec3<f32>(mu.size.xyz);
    return frame.view_proj * (mu.model * vec4<f32>(local, 1.0));
}

struct FOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

struct Shaded {
    color: vec4<f32>,
    normal: vec3<f32>,
    depth: f32,
};

// The looks' lights (look0.x).
const FLAT: i32 = 1;
const TOON: i32 = 2;
const ISO: i32 = 3;
const SOFT: i32 = 4;
const NIGHT: i32 = 5;
const CERAMIC: i32 = 6;

// A world normal in view space: x right, y up, z towards the camera.
fn view_normal(n: vec3<f32>) -> vec3<f32> {
    let right = normalize(vec3<f32>(frame.view_proj[0][0], frame.view_proj[1][0], frame.view_proj[2][0]));
    let up = normalize(vec3<f32>(frame.view_proj[0][1], frame.view_proj[1][1], frame.view_proj[2][1]));
    return vec3<f32>(dot(n, right), dot(n, up), dot(n, cross(right, up)));
}

// Iso: the three faces an isometric view shows get three clear tones, as in
// pixel art. The light is fixed to the view, from the upper left: from the
// usual angle the top shows at full brightness, the left side at 75 % and
// the right side at 50 % (display values; linear 1.0, 0.52 and 0.21). The
// lit part is a little warm, the shade a little cool.
fn iso_light(nv: vec3<f32>) -> vec3<f32> {
    let k = max(dot(nv, vec3<f32>(-0.432, 0.698, 0.571)), 0.0);
    return vec3<f32>(0.19, 0.21, 0.24) + k * vec3<f32>(0.90, 0.875, 0.84);
}

// CAD: one neutral grey on the voxel face itself, fixed in the world, so a
// step keeps the tone of its own face. The six directions sit far apart:
// the top is light, the front is mid, the side is dark, and a face that
// points down stays visible on the white sheet. The values are linear.
fn cad_light(n: vec3<f32>) -> vec3<f32> {
    var g = 0.0637; // +X
    if (n.x < -0.5) {
        g = 0.3672; // -X
    } else if (n.y > 0.5) {
        g = 0.1065; // +Y
    } else if (n.y < -0.5) {
        g = 0.2330; // -Y
    } else if (n.z > 0.5) {
        g = 0.5382; // +Z
    } else if (n.z < -0.5) {
        g = 0.0470; // -Z
    }
    return vec3<f32>(g, g, g);
}

// Soft: a bright sky above and a dim warm bounce from the floor, blended by
// how far the face looks up, plus a weak key.
fn soft_light(n: vec3<f32>, key: f32) -> vec3<f32> {
    let up = n.z * 0.5 + 0.5;
    let sky = mix(vec3<f32>(0.42, 0.38, 0.34), vec3<f32>(0.92, 0.95, 1.0), up);
    return sky + 0.18 * key * vec3<f32>(1.0, 0.95, 0.88);
}

// Night: a dim blue ambient and a cool moon key. Light colours glow on top
// (the glow in shade_model).
fn night_light(n: vec3<f32>, key: f32) -> vec3<f32> {
    let up = n.z * 0.5 + 0.5;
    return mix(vec3<f32>(0.015, 0.018, 0.035), vec3<f32>(0.03, 0.036, 0.07), up) + key * vec3<f32>(0.11, 0.14, 0.23);
}

// Ceramic: a soft diffuse light under a glaze. The glaze is ceramic_gloss.
fn ceramic_light(n: vec3<f32>, key_dir: vec3<f32>, to_eye: vec3<f32>, sun: f32) -> vec3<f32> {
    let key = max(dot(n, key_dir), 0.0) * sun;
    let fill = max(dot(n, frame.fill_dir.xyz), 0.0);
    // The glaze reflects at grazing angles and takes that light from the body.
    let facing = clamp(dot(n, normalize(to_eye)), 0.0, 1.0);
    let body = 1.0 - 0.25 * pow(1.0 - facing, 3.0);
    return vec3<f32>(0.36 + 0.58 * key + 0.22 * fill) * body;
}

// The glaze: a white highlight towards the key and a pale sheen at grazing
// angles. A rough material gets a wider, weaker highlight.
fn ceramic_gloss(n: vec3<f32>, key_dir: vec3<f32>, to_eye: vec3<f32>, rough: f32) -> vec3<f32> {
    let e = normalize(to_eye);
    let h = normalize(key_dir + e);
    let r = clamp(rough, 0.05, 1.0);
    let spec = pow(max(dot(n, h), 0.0), mix(40.0, 6.0, r)) * mix(0.45, 0.10, r);
    let sheen = 0.10 * pow(1.0 - clamp(dot(n, e), 0.0, 1.0), 4.0);
    return vec3<f32>(spec + sheen);
}


fn shade_model(frag: vec4<f32>) -> Shaded {
    let ray = pixel_ray(frag);
    let o = ray[0];
    let d = ray[1];
    let ol = (mu.inv_model * vec4<f32>(o, 1.0)).xyz;
    var dl = (mu.inv_model * vec4<f32>(d, 0.0)).xyz;
    dl = select(dl, vec3<f32>(1e-7), abs(dl) < vec3<f32>(1e-7));
    let inv = 1.0 / dl;
    let size_f = vec3<f32>(mu.size.xyz);
    let size_i = vec3<i32>(mu.size.xyz);
    let t1 = (vec3<f32>(0.0) - ol) * inv;
    let t2 = (size_f - ol) * inv;
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    let tmin = max(max(lo.x, lo.y), lo.z);
    let tmax = min(min(hi.x, hi.y), hi.z);
    let t0 = max(tmin, 0.0);
    if (tmax < t0) {
        discard;
    }
    var last_axis = 0;
    if (lo.y >= lo.x && lo.y >= lo.z) { last_axis = 1; }
    if (lo.z >= lo.x && lo.z >= lo.y) { last_axis = 2; }

    let step = select(vec3<i32>(-1), vec3<i32>(1), dl >= vec3<f32>(0.0));
    let stepf = vec3<f32>(step);
    let fwd = select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
    let td = abs(inv);
    let p0 = ol + dl * (t0 + 1e-4);
    var cell = clamp(vec3<i32>(floor(p0)), vec3<i32>(0), size_i - vec3<i32>(1));
    var tm = (vec3<f32>(cell) + fwd - ol) * inv;
    var t_hit = t0;
    var value = 0u;
    let use_bricks = mu.size.w != 0u;
    let cap = mu.info.x;
    for (var i = 0u; i < cap; i = i + 1u) {
        if (use_bricks) {
            let b = cell / vec3<i32>(8);
            if (load_brick(b) == 0u) {
                // Jump to the far side of this empty brick.
                let be = (vec3<f32>(b) + fwd) * 8.0;
                let te = (be - ol) * inv;
                var ax = 0;
                if (te.y < te.x && te.y <= te.z) { ax = 1; }
                if (te.z < te.x && te.z < te.y) { ax = 2; }
                let t_e = te[ax];
                var nc = vec3<i32>(floor(ol + dl * t_e));
                let b8 = b * vec3<i32>(8);
                nc = clamp(nc, b8, b8 + vec3<i32>(7));
                if (step[ax] > 0) { nc[ax] = b8[ax] + 8; } else { nc[ax] = b8[ax] - 1; }
                cell = nc;
                t_hit = t_e;
                last_axis = ax;
                if (any(cell < vec3<i32>(0)) || any(cell >= size_i)) { break; }
                tm = (vec3<f32>(cell) + fwd - ol) * inv;
                continue;
            }
        }
        value = load_vox(cell);
        if (value != 0u) { break; }
        var ax = 0;
        if (tm.y < tm.x && tm.y <= tm.z) { ax = 1; }
        if (tm.z < tm.x && tm.z < tm.y) { ax = 2; }
        t_hit = tm[ax];
        cell[ax] = cell[ax] + step[ax];
        tm[ax] = tm[ax] + td[ax];
        last_axis = ax;
        if (any(cell < vec3<i32>(0)) || any(cell >= size_i)) { break; }
    }
    if (value == 0u) {
        discard;
    }
    var nl = vec3<f32>(0.0);
    nl[last_axis] = -stepf[last_axis];
    let n = normalize((mu.model * vec4<f32>(nl, 0.0)).xyz);
    // Palette row 0: the colour (sRGB bytes). Row 1: the material, x = glow
    // (a Light colour), y = metallic, z = roughness, w = type (0 plain;
    // Renderer::set_materials).
    var base = to_linear(textureLoad(palette_tex, vec2<i32>(i32(value), 0), 0).rgb);
    let material = textureLoad(palette_tex, vec2<i32>(i32(value), 1), 0);
    // The fade: the other models inside a model keep 35 % of their colour.
    let fade = bitcast<f32>(mu.info.y);
    base = mix(base, base * 0.35 + vec3<f32>(0.19, 0.18, 0.17) * 0.65, fade);
    // View > Grid (frame.viewport.z = 1): a dark border round every voxel face,
    // like a dark outer ring on each palette tile. The CAD view draws its own
    // corner lines, so this grid stays off there.
    let cad = (u32(frame.look1.y + 0.5) & 4u) != 0u;
    if (frame.viewport.z > 0.5 && !cad) {
        let p = ol + dl * t_hit;
        var uv = vec2<f32>(p.y, p.z);
        if (last_axis == 1) { uv = vec2<f32>(p.x, p.z); }
        if (last_axis == 2) { uv = vec2<f32>(p.x, p.y); }
        let f = fract(uv);
        var gx0 = f.x;
        var gx1 = 1.0 - f.x;
        var gy0 = f.y;
        var gy1 = 1.0 - f.y;
        if ((mu.info.w & 2u) != 0u) {
            // A brick model: no grid line between two cells of one brick.
            let sb = load_seams(cell);
            var ga = 1;
            var gb = 2;
            if (last_axis == 1) { ga = 0; gb = 2; }
            if (last_axis == 2) { ga = 0; gb = 1; }
            var da = vec3<i32>(0);
            da[ga] = 1;
            var db = vec3<i32>(0);
            db[gb] = 1;
            if ((sb & same_bit(-da)) != 0u) { gx0 = 1.0; }
            if ((sb & same_bit(da)) != 0u) { gx1 = 1.0; }
            if ((sb & same_bit(-db)) != 0u) { gy0 = 1.0; }
            if ((sb & same_bit(db)) != 0u) { gy1 = 1.0; }
        }
        let edge = min(min(gx0, gx1), min(gy0, gy1));
        // One pixel in cells at this distance (35 degree view, ~1800 px tall).
        // An orthographic view sends its pixel size in viewport.w.
        let persp_px = t_hit * 0.00038 * 1800.0 / max(frame.viewport.y, 1.0);
        let px = max(select(persp_px, frame.viewport.w, frame.viewport.w > 0.0), 1e-4);
        let line = 1.0 - smoothstep(0.035, 0.035 + px * 1.5, edge);
        // Voxels only a few pixels wide: their borders would alias into a
        // moire (a 200-cube). The grid fades out from 10 px a
        // voxel down to 4, as the ink lines of a look do.
        let grid_cell_px = 1.0 / px;
        base = base * (1.0 - 0.38 * line * smoothstep(4.0, 10.0, grid_cell_px));
    }
    // --- the look -------------------------------------------------------------
    let mode = i32(frame.look0.x + 0.5);
    let flags = u32(frame.look1.y + 0.5);
    if (cad) {
        base = vec3<f32>(1.0);  // the light below is the grey
    } else if ((flags & 1u) != 0u) {
        base = vec3<f32>(0.55, 0.54, 0.52);  // clay: one neutral material
    }
    // The face: its cell, the empty cell in front, the two in-face axes, where on it.
    let ph = ol + dl * t_hit;
    var front = cell;
    front[last_axis] = front[last_axis] + i32(nl[last_axis]);
    let ua = (last_axis + 1) % 3;
    let va = (last_axis + 2) % 3;
    var U = vec3<i32>(0);
    U[ua] = 1;
    var V = vec3<i32>(0);
    V[va] = 1;
    let fu = fract(ph[ua]);
    let fv = fract(ph[va]);
    // One pixel in cells at this distance, as the grid lines measure it.
    let persp_px2 = t_hit * 0.00038 * 1800.0 / max(frame.viewport.y, 1.0);
    let px_c = max(select(persp_px2, frame.viewport.w, frame.viewport.w > 0.0), 1e-4);

    // CAD corner bits. The light uses the voxel face, not the fitted plane.
    var corner_mask = 0u;
    if (cad && (mu.info.w & 1u) != 0u) {
        corner_mask = load_cad(cell).a;
    }

    // Voxel occlusion: corners and creases darken, blended across the face.
    var occ = 1.0;
    if (frame.look0.y > 0.0 && !cad) {
        let a00 = corner_ao(front, -U, -V);
        let a10 = corner_ao(front, U, -V);
        let a01 = corner_ao(front, -U, V);
        let a11 = corner_ao(front, U, V);
        let a = mix(mix(a00, a10, fu), mix(a01, a11, fu), fv) / 3.0;
        occ = 1.0 - frame.look0.y * 0.75 * a;
    }

    // Ambient occlusion over several cells (frame.fill_dir.w = strength, cam_pos.w = reach).
    var sky = 1.0;
    if (frame.fill_dir.w > 0.0 && !cad) {
        let reach = max(frame.cam_pos.w, 1.0);
        sky = 1.0 - frame.fill_dir.w * occlusion(ph + nl * 0.01, nl, vec3<f32>(U), vec3<f32>(V), reach, frag.xy);
    }

    // Edges of this face: convex (the surface turns away), concave (a crease),
    // or a colour change. Distances in cells to each of the four edges.
    var ink = 0.0;
    var rim = 0.0;
    var seam = 0.0;  // a brick model: the line between two bricks, in every look but CAD
    let brick_model = (mu.info.w & 2u) != 0u && !cad;
    var sb = 0u;
    if (brick_model) { sb = load_seams(cell); }
    // The line, measured from the edge on each face: half the look's width each side,
    // but never more than a quarter of the voxel, so small voxels get thinner lines.
    var ink_px = min(max(frame.look1.z, 0.5) * max(frame.look1.w, 1.0), 0.25 / px_c);
    if (cad) {
        // About 1.25 px, and never more than a quarter of the voxel.
        ink_px = min(1.25 * max(frame.look1.w, 1.0), 0.25 / px_c);
    }
    let ink_w = px_c * ink_px * 0.5;
    for (var k = 0; k < 4; k = k + 1) {
        var dir = U;
        var dist = 1.0 - fu;
        if (k == 1) { dir = -U; dist = fu; }
        if (k == 2) { dir = V; dist = 1.0 - fv; }
        if (k == 3) { dir = -V; dist = fv; }
        let side = cell + dir;
        let side_front = front + dir;
        let convex = solid(side) < 0.5 && solid(side_front) < 0.5;
        let concave = solid(side_front) > 0.5;
        var colour_edge = false;
        if (solid(side) > 0.5 && !concave) {
            colour_edge = load_vox(side) != value;
        }
        if (brick_model && solid(side) > 0.5 && !concave && (sb & same_bit(dir)) == 0u) {
            colour_edge = true;
            seam = max(seam, 1.0 - smoothstep(max(px_c * 0.6, 0.015), max(px_c * 0.6, 0.015) + px_c, dist));
        }
        var mark = convex || concave || colour_edge;
        if (cad) {
            // The bit names an in-face direction: +U, -U, +V, -V are k = 0..3.
            let axis = select(ua, va, k >= 2);
            let positive = k == 0 || k == 2;
            let bit = u32(axis) * 2u + select(1u, 0u, positive);
            mark = (corner_mask & (1u << bit)) != 0u;
        }
        if (mark) {
            let line = 1.0 - smoothstep(ink_w, ink_w + px_c, dist);
            ink = max(ink, select(line, line * 0.55, colour_edge && !cad));
        }
        if (convex && !cad) {
            rim = max(rim, 1.0 - smoothstep(0.0, max(0.12, px_c * 2.0), dist));
        }
    }

    // Lines on voxels only a few pixels wide would cover them: fade from 9 px down to 3.
    // CAD lines stay until the voxel is only a couple of pixels tall.
    let cell_px = 1.0 / px_c;
    if (cad) {
        ink = ink * smoothstep(1.2, 2.5, cell_px);
    } else {
        ink = ink * smoothstep(2.5, 6.0, cell_px);
        rim = rim * smoothstep(2.0, 6.0, cell_px);
    }
    seam = seam * smoothstep(3.0, 7.0, cell_px);

    // Light: lit (camera lights or a fixed sun with shadows), flat, or toon bands.
    var key_dir = frame.key_dir.xyz;
    if (frame.look1.x > 0.0) {
        key_dir = normalize(vec3<f32>(0.42, -0.30, 0.86));  // a sun over the scene
    }
    let key = max(dot(n, key_dir), 0.0);
    let fill = max(dot(n, frame.fill_dir.xyz), 0.0);
    var sun = 1.0;
    if (frame.look1.x > 0.0 && key > 0.0 && !cad) {
        let sdir = normalize((mu.inv_model * vec4<f32>(key_dir, 0.0)).xyz);
        let start = ph + nl * 0.01;
        sun = 1.0 - frame.look1.x * shadowed(start, sdir);
    }
    var light = vec3<f32>(0.30 + 0.62 * key * sun + 0.28 * fill);
    var glow = vec3<f32>(0.0);  // light the colour gives off itself, added after the shading
    if (cad) {
        let a00 = corner_ao(front, -U, -V);
        let a10 = corner_ao(front, U, -V);
        let a01 = corner_ao(front, -U, V);
        let a11 = corner_ao(front, U, V);
        let closed = mix(mix(a00, a10, fu), mix(a01, a11, fu), fv) / 3.0;
        // A closed corner goes darker toward the joint, so a hole and a
        // socket stay readable. An open face keeps one tone.
        light = cad_light(n) * (1.0 - 0.50 * closed);
    } else if (mode == FLAT) {
        light = vec3<f32>(1.0);
    } else if (mode == TOON) {
        light = vec3<f32>(select(select(0.55, 0.8, key * sun > 0.25), 1.05, key * sun > 0.7) + 0.08 * fill);
    } else if (mode == ISO) {
        light = iso_light(view_normal(n));
    } else if (mode == SOFT) {
        light = soft_light(n, key * sun);
    } else if (mode == NIGHT) {
        light = night_light(n, key * sun);
        glow = select(vec3<f32>(0.0), base * (1.4 + 4.0 * material.x), material.x > 0.004);
    } else if (mode == CERAMIC) {
        light = ceramic_light(n, key_dir, -d, sun);
    }
    if (!cad) {
        light = light * occ * sky * (1.0 + frame.look0.w * 0.6 * rim);
    }
    let hit_w = o + d * t_hit;
    var shade = light;
    // Agent pictures: farther surfaces darker, so a hole shows in a flat view.
    if (frame.origin.w > 0.0 && !cad) {
        let dist = dot(hit_w - frame.cam_pos.xyz, normalize(d));
        let f = clamp((dist - frame.origin.z) / frame.origin.w, 0.0, 1.0);
        shade = shade * (1.0 - 0.55 * f);
    }
    let clip = frame.view_proj * vec4<f32>(hit_w, 1.0);
    var out: Shaded;
    out.normal = n;
    var col = base * shade + glow;
    if (mode == CERAMIC && !cad) {
        // The glaze also catches light along every convex edge.
        // A plain colour is glazed; a material colour keeps its own roughness.
        let rough = select(material.z, 0.15, material.w < 0.08);
        col = col + (ceramic_gloss(n, key_dir, -d, rough) * sun + vec3<f32>(0.22 * rim)) * occ;
    }
    if (cad) {
        // See-through: the faces wash toward the white sheet. The corner lines stay dark.
        if ((flags & 8u) != 0u) {
            col = mix(vec3<f32>(1.0), col, 0.45);
        }
        col = mix(col, vec3<f32>(0.0), ink);
    } else if (frame.look0.z > 0.0) {
        let ink_col = select(vec3<f32>(0.012, 0.012, 0.016), vec3<f32>(0.85, 0.88, 0.95), (flags & 2u) != 0u);
        col = mix(col, ink_col, ink * frame.look0.z);
    }
    col = col * (1.0 - 0.55 * seam);
    out.color = vec4<f32>(col, 1.0);
    out.depth = clip.z / clip.w;
    return out;
}

@fragment
fn fs_model(@builtin(position) frag: vec4<f32>) -> FOut {
    let s = shade_model(frag);
    var out: FOut;
    out.color = s.color;
    out.depth = s.depth;
    return out;
}

// Background: a grey gradient and a floor grid at z = 0.
struct BgOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_bg(@builtin(vertex_index) vi: u32) -> BgOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: BgOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn shade_bg(frag: vec4<f32>) -> vec4<f32> {
    // Animation pictures (sprite sheets): -1 - g asks for a flat grey g (linear),
    // no floor grid, in every look. Two pictures on 0 and 1 give the alpha.
    if (frame.key_dir.w <= -1.0) {
        return vec4<f32>(vec3<f32>(clamp(-frame.key_dir.w - 1.0, 0.0, 1.0)), 1.0);
    }
    if ((u32(frame.look1.y + 0.5) & 4u) != 0u) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);  // CAD view: a flat white sheet, no floor grid
    }
    let ray = pixel_ray(frag);
    let o = ray[0];
    let d = ray[1];
    // Agent pictures of dark models: a light, flat background (frame.key_dir.w, linear).
    var light_bg = frame.key_dir.w;
    var sky = select(mix(vec3<f32>(0.016, 0.017, 0.020), vec3<f32>(0.040, 0.042, 0.048), clamp(d.z * 0.5 + 0.5, 0.0, 1.0)),
                     vec3<f32>(light_bg), light_bg > 0.0);
    if (frame.look2.w > 0.5) {
        // A look with its own background: flat, with a faint floor grid.
        sky = frame.look2.rgb;
        light_bg = max(max(sky.r, sky.g), sky.b) + 1e-3;
    }
    var col = sky;
    if (d.z < -1e-4 && o.z > 0.0) {
        let t = -o.z / d.z;
        let p = o + d * t;
        let uv = p.xy;
        let fw = max(fwidth(uv), vec2<f32>(1e-4));
        let g1 = abs(fract(uv - 0.5) - 0.5) / fw;
        let g16 = abs(fract(uv / 16.0 - 0.5) - 0.5) * 16.0 / fw;
        let l1 = 1.0 - clamp(min(g1.x, g1.y), 0.0, 1.0);
        let l16 = 1.0 - clamp(min(g16.x, g16.y), 0.0, 1.0);
        // Fade by the distance from the camera (an orthographic ray starts far behind it).
        let fade = clamp(1.0 - length(p - frame.cam_pos.xyz) / 1500.0, 0.0, 1.0);
        if (light_bg > 0.0) {
            let k = select(1.0, 0.35, light_bg < 0.2);  // a dark colour gets lighter lines
            col = sky - (vec3<f32>(0.03) * l1 + vec3<f32>(0.10) * l16) * fade * select(k, -k, light_bg < 0.2);
        } else {
            let floor_col = vec3<f32>(0.030, 0.031, 0.036);
            col = mix(sky, floor_col, fade);
            col = col + vec3<f32>(0.006) * l1 * fade + vec3<f32>(0.020) * l16 * fade;
        }
    }
    return vec4<f32>(col, 1.0);
}

@fragment
fn fs_bg(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return shade_bg(frag);
}

