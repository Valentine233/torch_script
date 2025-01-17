cmake_minimum_required(VERSION 3.26)
project(FlashAttentionTest)

# # Set CMake prefix path
set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
set(CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")

# PATH
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PARENT_DIR)
cmake_path(APPEND FLASH_ATTENTION_DIR "${PARENT_DIR}" "flash-attention")
find_package(Torch REQUIRED)
# set(CUDA_HOME "/usr/local/cuda")
# set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
# set(CUDACXX "/usr/local/cuda/bin/nvcc")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA)
set(CUDA_ARCH "-gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CUDA_ARCH} -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
set(CUDA_NVCC_FLAGS ${CMAKE_CUDA_FLAGS})

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -std=c++17")    

add_library(flash_attention_libs SHARED
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/flash_api.cpp
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim160_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim160_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim160_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim160_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu)


set_target_properties(flash_attention_libs PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".so"
)

add_executable(flash_attention_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/run.cpp
)

message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

set(FLASH_ATTENTION_INCLUDE_DIRS
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn
    ${FLASH_ATTENTION_DIR}/csrc/flash_attn/src
    ${FLASH_ATTENTION_DIR}/csrc/cutlass/include
)

target_link_libraries(
    flash_attention_libs PRIVATE
    "${TORCH_LIBRARIES}"
)

target_include_directories(
    flash_attention_libs PRIVATE
    ${FLASH_ATTENTION_INCLUDE_DIRS}
)

target_link_libraries(
    flash_attention_cuda PRIVATE
    flash_attention_libs
    "${TORCH_LIBRARIES}"
)

target_include_directories(
    flash_attention_cuda PRIVATE
    ${FLASH_ATTENTION_INCLUDE_DIRS}
)
