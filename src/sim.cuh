#pragma once
#include <cstdint>

struct SimParams {
  int Nx = 512;
  int Ny = 512;
  float Lx = 80.0f;
  float Ly = 80.0f;
  float dt = 0.01f;

  float hbar = 1.0f;
  float m = 1.0f;

  // lattice
  float lattice_x0 = 14.0f;
  float lattice_y0 = 0.0f;
  int lattice_nx = 6;
  int lattice_ny = 12;
  float lattice_ax = 4.0f;
  float lattice_ay = 4.0f;
  float lattice_sigma = 0.58f;
  float lattice_V0 = 38.0f; // barrier height

  // wavepacket
  float wp_x0 = -14.0f;
  float wp_y0 = 0.0f;
  float wp_sigma = 2.8f;
  float wp_kx = 3.8f;
  float wp_ky = 0.0f;

  // absorbing boundary
  float absorb_margin = 14.0f;
  float absorb_strength = 7.0f;

  // rendering
  float render_gamma = 0.55f;
  float render_clip_percentile = 99.7f; // used approx by scale estimate
  float dot_threshold_frac = 0.60f;     // overlay dots where V > frac*Vmax
};

struct SimState {
  // Device buffers
  void* d_psi = nullptr;        // cufftComplex
  void* d_tmp = nullptr;        // cufftComplex
  float* d_V = nullptr;         // potential
  float* d_absorb = nullptr;    // absorbing mask
  float* d_kin_half = nullptr;  // kinetic phase half-step (complex stored as float2)
  uint8_t* d_dotmask = nullptr; // 0/1 lattice dots mask (Ny*Nx)

  // Render output RGBA
  void* d_rgba = nullptr;       // uchar4 (Ny*Nx)
};

bool sim_init(const SimParams& p, SimState& s);
void sim_step(const SimParams& p, SimState& s, int steps);
void sim_render_rgba(const SimParams& p, SimState& s, float* out_scale_host /* optional */);
void sim_shutdown(SimState& s);
