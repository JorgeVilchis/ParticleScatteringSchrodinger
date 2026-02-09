#include "sim.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef __APPLE__
  #error "This OpenGL/CUDA interop sample is intended for Linux/Windows with NVIDIA drivers."
#endif

// Minimal GL loader: we use glfwGetProcAddress via glad-like calls is typical,
// but to keep this self-contained, we rely on OpenGL 2.1-ish calls that exist.
#if defined(_WIN32)
  #include <windows.h>
#endif
#include <GL/gl.h>

static void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  std::exit(1);
}

static GLuint create_texture_rgba8(int w, int h) {
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

static void draw_fullscreen_quad(GLuint tex) {
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex);

  glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(-1,-1);
    glTexCoord2f(1,0); glVertex2f( 1,-1);
    glTexCoord2f(1,1); glVertex2f( 1, 1);
    glTexCoord2f(0,1); glVertex2f(-1, 1);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}

int main() {
  SimParams p;
  p.Nx = 512;
  p.Ny = 512;

  if (!glfwInit()) die("Failed to init GLFW");

  // Create OpenGL window
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  GLFWwindow* win = glfwCreateWindow(900, 900, "TDSE lattice (CUDA + cuFFT + OpenGL)", nullptr, nullptr);
  if (!win) die("Failed to create window");

  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  // Create texture
  GLuint tex = create_texture_rgba8(p.Nx, p.Ny);

  // Register texture with CUDA
  cudaGraphicsResource* cuda_res = nullptr;
  cudaError_t cerr = cudaGraphicsGLRegisterImage(&cuda_res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
  if (cerr != cudaSuccess) {
    die("cudaGraphicsGLRegisterImage failed. (Driver/GL context mismatch?)");
  }

  // Init simulation
  SimState s;
  if (!sim_init(p, s)) die("sim_init failed");

  auto t0 = std::chrono::high_resolution_clock::now();
  int frames = 0;

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();

    // Controls
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      glfwSetWindowShouldClose(win, 1);
    }

    // Step simulation
    sim_step(p, s, 6);

    // Render RGBA into s.d_rgba
    float clip = 0.0f;
    sim_render_rgba(p, s, &clip);

    // Map OpenGL texture for CUDA write
    cudaGraphicsMapResources(1, &cuda_res, 0);
    cudaArray_t arr = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&arr, cuda_res, 0, 0);

    // Copy device RGBA buffer -> CUDA array (texture storage)
    cudaMemcpy2DToArray(
      arr, 0, 0,
      s.d_rgba,
      p.Nx * 4,              // pitch in bytes (uchar4)
      p.Nx * 4,
      p.Ny,
      cudaMemcpyDeviceToDevice
    );

    cudaGraphicsUnmapResources(1, &cuda_res, 0);

    // Draw quad
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);

    draw_fullscreen_quad(tex);

    // FPS title
    frames++;
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (dt >= 1.0) {
      double fps = frames / dt;
      frames = 0;
      t0 = t1;
      char title[256];
      std::snprintf(title, sizeof(title), "TDSE lattice (CUDA/cuFFT/GL)  |  FPS %.1f  | clip %.3e", fps, clip);
      glfwSetWindowTitle(win, title);
    }

    glfwSwapBuffers(win);
  }

  // Cleanup
  sim_shutdown(s);
  cudaGraphicsUnregisterResource(cuda_res);
  glDeleteTextures(1, &tex);

  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}

