#include "sim.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    return false; \
  } \
} while(0)

#define CUDA_CHECK_VOID(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    return; \
  } \
} while(0)

#define CUFFT_CHECK(x) do { \
  cufftResult r = (x); \
  if (r != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %s:%d: code=%d\n", __FILE__, __LINE__, (int)r); \
    return false; \
  } \
} while(0)

#define CUFFT_CHECK_VOID(x) do { \
  cufftResult r = (x); \
  if (r != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %s:%d: code=%d\n", __FILE__, __LINE__, (int)r); \
    return; \
  } \
} while(0)

// -----------------------------
// Helpers
// -----------------------------
static inline __device__ __host__ float2 cexp_i(float a) {
  // exp(i a) = cos(a) + i sin(a)
  return make_float2(cosf(a), sinf(a));
}

static inline __device__ float2 cmul(float2 a, float2 b) {
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

static inline __device__ float2 cscale(float2 a, float s) {
  return make_float2(a.x*s, a.y*s);
}

// -----------------------------
// Kernels
// -----------------------------
__global__ void k_build_absorb(float* absorb, int Nx, int Ny, float Lx, float Ly, float margin, float strength) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= Nx || iy >= Ny) return;

  float dx = Lx / Nx;
  float dy = Ly / Ny;

  float x = (ix - Nx/2) * dx;
  float y = (iy - Ny/2) * dy;

  float halfLx = Lx * 0.5f;
  float halfLy = Ly * 0.5f;

  float dist_x = fminf(x + halfLx, halfLx - x);
  float dist_y = fminf(y + halfLy, halfLy - y);
  float dist = fminf(dist_x, dist_y);

  float t = dist / margin;
  t = fminf(fmaxf(t, 0.0f), 1.0f);
  float val = expf(-strength * powf(1.0f - t, 4.0f));
  absorb[iy * Nx + ix] = val;
}

__global__ void k_build_lattice_potential(float* V, int Nx, int Ny, float Lx, float Ly,
                                         float x0, float y0, int nx, int ny, float ax, float ay,
                                         float sigma, float V0) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= Nx || iy >= Ny) return;

  float dx = Lx / Nx;
  float dy = Ly / Ny;

  float x = (ix - Nx/2) * dx;
  float y = (iy - Ny/2) * dy;

  float s2 = 2.0f * sigma * sigma;
  float sum = 0.0f;

  // finite rectangular lattice centered at (x0,y0)
  for (int i = 0; i < nx; ++i) {
    float xi = x0 + (i - (nx - 1) * 0.5f) * ax;
    for (int j = 0; j < ny; ++j) {
      float yj = y0 + (j - (ny - 1) * 0.5f) * ay;
      float dxp = x - xi;
      float dyp = y - yj;
      sum += expf(-(dxp*dxp + dyp*dyp) / s2);
    }
  }

  V[iy * Nx + ix] = V0 * sum; // barrier bumps
}

__global__ void k_build_dotmask(uint8_t* dotmask, const float* V, int N, float thr) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  dotmask[i] = (fabsf(V[i]) > thr) ? 1 : 0;
}

__global__ void k_init_wavepacket(cufftComplex* psi, int Nx, int Ny, float Lx, float Ly,
                                  float x0, float y0, float sigma, float kx, float ky) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= Nx || iy >= Ny) return;

  float dx = Lx / Nx;
  float dy = Ly / Ny;

  float x = (ix - Nx/2) * dx;
  float y = (iy - Ny/2) * dy;

  // envelope exp(-((x-x0)^2+(y-y0)^2)/(4 sigma^2))
  float dx0 = x - x0;
  float dy0 = y - y0;
  float env = expf(-(dx0*dx0 + dy0*dy0) / (4.0f * sigma * sigma));

  float phase = kx * x + ky * y;
  float2 ph = cexp_i(phase);

  cufftComplex v;
  v.x = env * ph.x;
  v.y = env * ph.y;

  psi[iy * Nx + ix] = v;
}

__global__ void k_apply_potential(cufftComplex* psi, const float* V, int N, float dt_over_hbar) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  float a = -dt_over_hbar * V[i];
  float2 e = cexp_i(a);
  float2 z = make_float2(psi[i].x, psi[i].y);
  z = cmul(z, e);
  psi[i].x = z.x;
  psi[i].y = z.y;
}

__global__ void k_apply_absorb(cufftComplex* psi, const float* absorb, int N) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  float s = absorb[i];
  psi[i].x *= s;
  psi[i].y *= s;
}

// kinetic half phase stored as float2 array (same size as psi) in k-space layout
__global__ void k_apply_kinetic_half(cufftComplex* psi_k, const float2* kin_half, int N) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  float2 z = make_float2(psi_k[i].x, psi_k[i].y);
  float2 e = kin_half[i];
  z = cmul(z, e);
  psi_k[i].x = z.x;
  psi_k[i].y = z.y;
}

__global__ void k_scale(cufftComplex* psi, int N, float s) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  psi[i].x *= s;
  psi[i].y *= s;
}

__global__ void k_build_kinetic_half(float2* kin_half, int Nx, int Ny, float Lx, float Ly,
                                    float dt, float hbar, float m) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= Nx || iy >= Ny) return;

  // FFT frequencies (radians): k = 2*pi*fftfreq
  // Here we replicate numpy fftfreq layout:
  int kxi = (ix <= Nx/2) ? ix : (ix - Nx);
  int kyi = (iy <= Ny/2) ? iy : (iy - Ny);

  float dkx = 2.0f * (float)M_PI / Lx;
  float dky = 2.0f * (float)M_PI / Ly;

  float kx = kxi * dkx;
  float ky = kyi * dky;
  float k2 = kx*kx + ky*ky;

  // kinetic operator: exp(-i (dt/2) * (hbar k^2 / (2m)))
  float a = -(dt * 0.5f) * (hbar * k2 / (2.0f * m));
  float2 e = cexp_i(a);
  kin_half[iy * Nx + ix] = e;
}

// Estimate max(|psi|^2) cheaply by sampling (not exact percentile).
__global__ void k_reduce_max_prob(const cufftComplex* psi, float* out_max, int N) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  float v = 0.0f;
  if (i < N) {
    float re = psi[i].x;
    float im = psi[i].y;
    v = re*re + im*im;
  }
  sdata[tid] = v;
  __syncthreads();

  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out_max[blockIdx.x] = sdata[0];
}

__device__ float3 hsv2rgb(float h, float s, float v) {
  float h6 = fmodf(h * 6.0f, 6.0f);
  int i = (int)floorf(h6);
  float f = h6 - i;
  float p = v * (1.0f - s);
  float q = v * (1.0f - s * f);
  float t = v * (1.0f - s * (1.0f - f));

  float r, g, b;
  switch (i) {
    case 0: r=v; g=t; b=p; break;
    case 1: r=q; g=v; b=p; break;
    case 2: r=p; g=v; b=t; break;
    case 3: r=p; g=q; b=v; break;
    case 4: r=t; g=p; b=v; break;
    default: r=v; g=p; b=q; break;
  }
  return make_float3(r,g,b);
}

__global__ void k_render_rgba(uchar4* rgba, const cufftComplex* psi, const uint8_t* dotmask,
                             int Nx, int Ny, float gamma, float inv_vmax, float dot_alpha) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= Nx || iy >= Ny) return;
  int i = iy * Nx + ix;

  float re = psi[i].x;
  float im = psi[i].y;

  float phase = atan2f(im, re);           // [-pi, pi]
  float h = (phase + (float)M_PI) * (0.5f / (float)M_PI); // [0,1]

  float prob = re*re + im*im;
  float v = fminf(prob * inv_vmax, 1.0f);
  v = powf(v, gamma);

  float3 rgb = hsv2rgb(h, 1.0f, v);

  // overlay lattice dots as white
  if (dotmask && dotmask[i]) {
    rgb.x = rgb.x*(1.0f - dot_alpha) + dot_alpha*1.0f;
    rgb.y = rgb.y*(1.0f - dot_alpha) + dot_alpha*1.0f;
    rgb.z = rgb.z*(1.0f - dot_alpha) + dot_alpha*1.0f;
  }

  rgba[i] = make_uchar4(
    (unsigned char)(255.0f * fminf(fmaxf(rgb.x, 0.0f), 1.0f)),
    (unsigned char)(255.0f * fminf(fmaxf(rgb.y, 0.0f), 1.0f)),
    (unsigned char)(255.0f * fminf(fmaxf(rgb.z, 0.0f), 1.0f)),
    255
  );
}

// -----------------------------
// Global cuFFT plan (kept internal to this TU)
// -----------------------------
static cufftHandle g_plan = 0;

// -----------------------------
// Public API
// -----------------------------
bool sim_init(const SimParams& p, SimState& s) {
  const int Nx = p.Nx, Ny = p.Ny;
  const int N = Nx * Ny;

  // Allocate buffers
  CUDA_CHECK(cudaMalloc(&s.d_psi,  sizeof(cufftComplex) * N));
  CUDA_CHECK(cudaMalloc(&s.d_tmp,  sizeof(cufftComplex) * N));
  CUDA_CHECK(cudaMalloc(&s.d_V,    sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&s.d_absorb, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&s.d_kin_half, sizeof(float2) * N));
  CUDA_CHECK(cudaMalloc(&s.d_dotmask, sizeof(uint8_t) * N));
  CUDA_CHECK(cudaMalloc(&s.d_rgba, sizeof(uchar4) * N));

  // Build lattice potential
  dim3 bs2(16,16);
  dim3 gs2((Nx + bs2.x - 1)/bs2.x, (Ny + bs2.y - 1)/bs2.y);
  k_build_lattice_potential<<<gs2, bs2>>>(
    s.d_V, Nx, Ny, p.Lx, p.Ly,
    p.lattice_x0, p.lattice_y0,
    p.lattice_nx, p.lattice_ny,
    p.lattice_ax, p.lattice_ay,
    p.lattice_sigma, p.lattice_V0
  );
  CUDA_CHECK(cudaGetLastError());

  // Absorb mask
  k_build_absorb<<<gs2, bs2>>>(s.d_absorb, Nx, Ny, p.Lx, p.Ly, p.absorb_margin, p.absorb_strength);
  CUDA_CHECK(cudaGetLastError());

  // Kinetic half-step phase
  k_build_kinetic_half<<<gs2, bs2>>>((float2*)s.d_kin_half, Nx, Ny, p.Lx, p.Ly, p.dt, p.hbar, p.m);
  CUDA_CHECK(cudaGetLastError());

  // Init wavepacket
  k_init_wavepacket<<<gs2, bs2>>>((cufftComplex*)s.d_psi, Nx, Ny, p.Lx, p.Ly, p.wp_x0, p.wp_y0, p.wp_sigma, p.wp_kx, p.wp_ky);
  CUDA_CHECK(cudaGetLastError());

  // Normalize roughly by computing sum(|psi|^2) on CPU (simple one-time copy)
  std::vector<cufftComplex> hpsi(N);
  CUDA_CHECK(cudaMemcpy(hpsi.data(), s.d_psi, sizeof(cufftComplex)*N, cudaMemcpyDeviceToHost));
  double sum = 0.0;
  double dx = (double)p.Lx / (double)p.Nx;
  double dy = (double)p.Ly / (double)p.Ny;
  for (int i = 0; i < N; ++i) {
    double re = hpsi[i].x, im = hpsi[i].y;
    sum += (re*re + im*im);
  }
  double norm = sqrt(sum * dx * dy);
  float invnorm = (norm > 0.0) ? (float)(1.0 / norm) : 1.0f;

  int threads = 256;
  int blocks = (N + threads - 1)/threads;
  k_scale<<<blocks, threads>>>((cufftComplex*)s.d_psi, N, invnorm);
  CUDA_CHECK(cudaGetLastError());

  // Build dotmask threshold based on Vmax (host reduction for simplicity)
  std::vector<float> hV(N);
  CUDA_CHECK(cudaMemcpy(hV.data(), s.d_V, sizeof(float)*N, cudaMemcpyDeviceToHost));
  float vmax = 0.0f;
  for (int i = 0; i < N; ++i) vmax = std::max(vmax, std::abs(hV[i]));
  float thr = p.dot_threshold_frac * vmax;

  k_build_dotmask<<<blocks, threads>>>(s.d_dotmask, s.d_V, N, thr);
  CUDA_CHECK(cudaGetLastError());

  // cuFFT plan
  CUFFT_CHECK(cufftPlan2d(&g_plan, Ny, Nx, CUFFT_C2C)); // NOTE: cuFFT uses row-major with (ny,nx)
  CUFFT_CHECK(cufftSetStream(g_plan, 0));

  CUDA_CHECK(cudaDeviceSynchronize());
  return true;
}

void sim_step(const SimParams& p, SimState& s, int steps) {
  const int N = p.Nx * p.Ny;
  int threads = 256;
  int blocks = (N + threads - 1)/threads;

  for (int it = 0; it < steps; ++it) {
    // half kinetic: FFT -> multiply -> iFFT
    CUFFT_CHECK_VOID(cufftExecC2C(g_plan, (cufftComplex*)s.d_psi, (cufftComplex*)s.d_tmp, CUFFT_FORWARD));
    k_apply_kinetic_half<<<blocks, threads>>>((cufftComplex*)s.d_tmp, (float2*)s.d_kin_half, N);
    CUFFT_CHECK_VOID(cufftExecC2C(g_plan, (cufftComplex*)s.d_tmp, (cufftComplex*)s.d_psi, CUFFT_INVERSE));

    // cuFFT inverse is unnormalized => scale by 1/N
    k_scale<<<blocks, threads>>>((cufftComplex*)s.d_psi, N, 1.0f / (float)N);

    // potential full
    k_apply_potential<<<blocks, threads>>>((cufftComplex*)s.d_psi, s.d_V, N, p.dt / p.hbar);

    // half kinetic again
    CUFFT_CHECK_VOID(cufftExecC2C(g_plan, (cufftComplex*)s.d_psi, (cufftComplex*)s.d_tmp, CUFFT_FORWARD));
    k_apply_kinetic_half<<<blocks, threads>>>((cufftComplex*)s.d_tmp, (float2*)s.d_kin_half, N);
    CUFFT_CHECK_VOID(cufftExecC2C(g_plan, (cufftComplex*)s.d_tmp, (cufftComplex*)s.d_psi, CUFFT_INVERSE));
    k_scale<<<blocks, threads>>>((cufftComplex*)s.d_psi, N, 1.0f / (float)N);

    // absorb boundary
    k_apply_absorb<<<blocks, threads>>>((cufftComplex*)s.d_psi, s.d_absorb, N);
  }
}

void sim_render_rgba(const SimParams& p, SimState& s, float* out_scale_host) {
  const int Nx = p.Nx, Ny = p.Ny;
  const int N = Nx * Ny;

  // Estimate vmax = max(|psi|^2) using a two-stage reduction
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  float* d_part = nullptr;
  CUDA_CHECK_VOID(cudaMalloc(&d_part, sizeof(float) * blocks));
  k_reduce_max_prob<<<blocks, threads>>>((cufftComplex*)s.d_psi, d_part, N);
  CUDA_CHECK_VOID(cudaGetLastError());

  // reduce on CPU (blocks is small)
  std::vector<float> hpart(blocks);
  CUDA_CHECK_VOID(cudaMemcpy(hpart.data(), d_part, sizeof(float)*blocks, cudaMemcpyDeviceToHost));
  CUDA_CHECK_VOID(cudaFree(d_part));

  float vmax = 0.0f;
  for (float v : hpart) vmax = std::max(vmax, v);
  vmax = std::max(vmax, 1e-20f);

  // We want something like a percentile clip; approximate by scaling down vmax a bit
  // to bring out fringes (you can tune this factor).
  float clip = vmax * 0.35f;
  clip = std::max(clip, 1e-20f);
  float inv_vmax = 1.0f / clip;

  dim3 bs2(16,16);
  dim3 gs2((Nx + bs2.x - 1)/bs2.x, (Ny + bs2.y - 1)/bs2.y);

  k_render_rgba<<<gs2, bs2>>>(
    (uchar4*)s.d_rgba,
    (cufftComplex*)s.d_psi,
    s.d_dotmask,
    Nx, Ny,
    p.render_gamma,
    inv_vmax,
    0.65f  // dot alpha
  );
  CUDA_CHECK_VOID(cudaGetLastError());

  if (out_scale_host) *out_scale_host = clip;
}

void sim_shutdown(SimState& s) {
  if (g_plan) {
    cufftDestroy(g_plan);
    g_plan = 0;
  }
  if (s.d_rgba) cudaFree(s.d_rgba);
  if (s.d_dotmask) cudaFree(s.d_dotmask);
  if (s.d_kin_half) cudaFree(s.d_kin_half);
  if (s.d_absorb) cudaFree(s.d_absorb);
  if (s.d_V) cudaFree(s.d_V);
  if (s.d_tmp) cudaFree(s.d_tmp);
  if (s.d_psi) cudaFree(s.d_psi);

  s = SimState{};
}
