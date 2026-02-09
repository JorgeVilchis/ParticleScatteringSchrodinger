cmake_minimum_required(VERSION 3.18)
project(tdse_lattice_cuda_gl LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- USER: set your CUDA path if needed ---
find_package(CUDAToolkit REQUIRED)

# GLFW
find_package(glfw3 REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

add_executable(tdse
  src/main.cpp
  src/sim.cu
)

set_target_properties(tdse PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(tdse PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(tdse PRIVATE
  glfw
  OpenGL::GL
  CUDA::cudart
  CUDA::cufft
)

# On Linux you may need dl, pthread, X11; glfw3 config usually handles this.
