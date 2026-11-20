









float shape(vec3 p) { return sqrt((((p.x) * (p.x)) + ((p.y) * (p.y))) + ((p.z) * (p.z))) - m_radius; }

float detail2(float d) { return clamp(d / (2.0 * 3.14159265), 0.0, 1.0); }

float map(vec3 v) {
  float d = shape(v - i_volume_size * 0.5);
  return d <= 0.0 ? max(i_color_index, detail2(d)) : 0.0;
}
