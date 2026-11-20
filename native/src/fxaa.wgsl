// FXAA 3.11 (quality): a port of Simon Rodriguez's FXAA code, after Timothy
// Lottes' NVIDIA FXAA 3.11 (licences in CREDITS.md). Find the edge through
// each pixel from the luma of its neighbours, search along it for both ends,
// and sample the scene once, shifted across the edge by the right amount.
// The input is the 3D view (voxels and 3D lines); the UI is drawn after it.

struct P {
    inv_size: vec2<f32>,  // 1 / target size in pixels
    subpix: f32,  // 0: edges only, 1: soften single-pixel detail fully
    edge_min: f32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

const EDGE_MAX: f32 = 0.125;
const STEPS: u32 = 12u;

fn quality(i: u32) -> f32 {
    if (i < 5u) { return 1.0; }
    if (i == 5u) { return 1.5; }
    if (i < 10u) { return 2.0; }
    if (i == 10u) { return 4.0; }
    return 8.0;
}

// Perceptual luma: the texture holds linear colour (it is sRGB-encoded).
fn luma(c: vec3<f32>) -> f32 {
    return sqrt(dot(c, vec3<f32>(0.299, 0.587, 0.114)));
}

fn at(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(scene, samp, uv, 0.0).rgb;
}

fn l_at(uv: vec2<f32>, dx: f32, dy: f32) -> f32 {
    return luma(at(uv + vec2<f32>(dx, dy) * p.inv_size));
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_fxaa(v: VOut) -> @location(0) vec4<f32> {
    let uv = v.uv;
    let centre = at(uv);
    let lc = luma(centre);
    let ld = l_at(uv, 0.0, -1.0);
    let lu = l_at(uv, 0.0, 1.0);
    let ll = l_at(uv, -1.0, 0.0);
    let lr = l_at(uv, 1.0, 0.0);
    let lmin = min(lc, min(min(ld, lu), min(ll, lr)));
    let lmax = max(lc, max(max(ld, lu), max(ll, lr)));
    let range = lmax - lmin;
    if (range < max(p.edge_min, lmax * EDGE_MAX)) {
        return vec4<f32>(centre, 1.0);
    }
    let ldl = l_at(uv, -1.0, -1.0);
    let lur = l_at(uv, 1.0, 1.0);
    let lul = l_at(uv, -1.0, 1.0);
    let ldr = l_at(uv, 1.0, -1.0);
    let l_du = ld + lu;
    let l_lr = ll + lr;
    let c_left = ldl + lul;
    let c_down = ldl + ldr;
    let c_right = ldr + lur;
    let c_up = lur + lul;
    let edge_h = abs(-2.0 * ll + c_left) + abs(-2.0 * lc + l_du) * 2.0 + abs(-2.0 * lr + c_right);
    let edge_v = abs(-2.0 * lu + c_up) + abs(-2.0 * lc + l_lr) * 2.0 + abs(-2.0 * ld + c_down);
    let horizontal = edge_h >= edge_v;

    let l1 = select(ll, ld, horizontal);
    let l2 = select(lr, lu, horizontal);
    let g1 = l1 - lc;
    let g2 = l2 - lc;
    let steep1 = abs(g1) >= abs(g2);
    let g_scaled = 0.25 * max(abs(g1), abs(g2));
    var step = select(p.inv_size.x, p.inv_size.y, horizontal);
    var l_avg = 0.5 * (l2 + lc);
    if (steep1) {
        step = -step;
        l_avg = 0.5 * (l1 + lc);
    }
    var cur = uv;
    if (horizontal) { cur.y += step * 0.5; } else { cur.x += step * 0.5; }
    let off = select(vec2<f32>(0.0, p.inv_size.y), vec2<f32>(p.inv_size.x, 0.0), horizontal);
    var uv1 = cur - off;
    var uv2 = cur + off;
    var e1 = luma(at(uv1)) - l_avg;
    var e2 = luma(at(uv2)) - l_avg;
    var r1 = abs(e1) >= g_scaled;
    var r2 = abs(e2) >= g_scaled;
    if (!r1) { uv1 -= off; }
    if (!r2) { uv2 += off; }
    if (!(r1 && r2)) {
        for (var i = 2u; i < STEPS; i = i + 1u) {
            if (!r1) { e1 = luma(at(uv1)) - l_avg; }
            if (!r2) { e2 = luma(at(uv2)) - l_avg; }
            r1 = abs(e1) >= g_scaled;
            r2 = abs(e2) >= g_scaled;
            if (!r1) { uv1 -= off * quality(i); }
            if (!r2) { uv2 += off * quality(i); }
            if (r1 && r2) { break; }
        }
    }
    let d1 = select(uv.x - uv1.x, uv.y - uv1.y, !horizontal);
    let d2 = select(uv2.x - uv.x, uv2.y - uv.y, !horizontal);
    let dir1 = d1 < d2;
    let d = min(d1, d2);
    let thickness = d1 + d2;
    let px_off = -d / thickness + 0.5;
    let centre_smaller = lc < l_avg;
    let right_way = (select(e2, e1, dir1) < 0.0) != centre_smaller;
    var final_off = select(0.0, px_off, right_way);

    let l_mean = (1.0 / 12.0) * (2.0 * (l_du + l_lr) + c_left + c_right);
    let s1 = clamp(abs(l_mean - lc) / range, 0.0, 1.0);
    let s2 = (-2.0 * s1 + 3.0) * s1 * s1;
    final_off = max(final_off, s2 * s2 * p.subpix);

    var out_uv = uv;
    if (horizontal) { out_uv.y += final_off * step; } else { out_uv.x += final_off * step; }
    return vec4<f32>(at(out_uv), 1.0);
}
