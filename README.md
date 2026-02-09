# TDSE Lattice Scattering (CUDA + cuFFT + OpenGL)

This project simulates a **2D time-dependent Schrödinger equation (TDSE)** wavepacket
scattering through a **finite periodic lattice** using a **split-step Fourier method**
accelerated on an **NVIDIA GPU (cuFFT)**, with **real-time OpenGL visualization**.

The visualization encodes:
- **Hue** → phase of the wavefunction  
- **Brightness** → probability density  
- **White dots** → lattice “atoms” (Gaussian scatterers)

The code supports **CUDA–OpenGL interop** for fast GPU→GPU rendering.
On hybrid Intel+NVIDIA systems, the program **must be run with NVIDIA PRIME offload**
so the OpenGL context is created on the NVIDIA driver.

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support

### OS
- Ubuntu 22.04 (or similar Linux distro)

### NVIDIA Driver
Verify the proprietary driver is installed:

```bash
nvidia-smi
```

### CUDA Toolkit
Verify CUDA is installed:

```bash
which nvcc
nvcc --version
```

If missing:

```bash
sudo apt install nvidia-cuda-toolkit
```

### Compiler Compatibility (IMPORTANT)
CUDA 11.5 **does not support GCC 11**.

Install GCC 10:

```bash
sudo apt install gcc-10 g++-10
```

---

## System Dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libglfw3-dev \
  libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev \
  mesa-utils
```

---

## Build

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-10 \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-10
cmake --build build -j
```

Executable:
```bash
./build/tdse
```

---

## Running with NVIDIA PRIME (Hybrid GPUs)

Force NVIDIA OpenGL:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
./build/tdse
```

Verify OpenGL vendor:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
glxinfo -B | egrep "OpenGL vendor|OpenGL renderer"
```

Expected:
```
OpenGL vendor string: NVIDIA Corporation
```

---

## Wayland Warning

CUDA–OpenGL interop is unreliable on Wayland.

Check:
```bash
echo $XDG_SESSION_TYPE
```

If `wayland`, log out and select **Ubuntu on Xorg** at login.

---

## Common Errors

### cudaGraphicsGLRegisterImage failed
Cause: OpenGL running on Intel/Mesa.

Fix: Run with PRIME offload and ensure NVIDIA OpenGL.

### CUDA compile errors with std headers
Cause: GCC 11 + CUDA 11.5 mismatch.

Fix: Use GCC/G++ 10.

---

## Tuning Parameters

Edit `src/sim.cuh`:

- Increase `wp_kx` → more fringes
- Increase `lattice_V0` → stronger scattering
- Reduce `lattice_sigma` → smaller lattice atoms
- Reduce `render_gamma` → brighter faint fringes

---

## License

Free for research and educational use.
