cmake_minimum_required(VERSION 3.18)
project(qwentin LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_C_STANDARD 11)

# Without this FindCUDA (and CUDA::cudart_static after it) may pick another toolkit on the box, e.g. a
# system CUDA under /usr: its runtime then fills cudaDeviceProp in its own layout (multiProcessorCount 1).
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    get_filename_component(_tq_cuda_root "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_tq_cuda_root "${_tq_cuda_root}" DIRECTORY)
    set(CUDA_TOOLKIT_ROOT_DIR "${_tq_cuda_root}" CACHE PATH "CUDA toolkit of CMAKE_CUDA_COMPILER")
endif()
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The Qwen3.6 inference engine: one CUDA translation unit (kernels + C ABI),
# loaded via ctypes by the Python tools (serve_batched.py, ...).
add_library(qwentin-forward-qwen SHARED
    src/forward_qwen.cu
)
set_target_properties(qwentin-forward-qwen PROPERTIES
    OUTPUT_NAME forward_qwen
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES OFF
)
target_link_libraries(qwentin-forward-qwen PRIVATE
    CUDA::cudart_static
)
target_compile_options(qwentin-forward-qwen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120f,code=sm_120f>
)

install(TARGETS qwentin-forward-qwen DESTINATION lib)
