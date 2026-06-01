cmake_minimum_required(VERSION 3.18)

project(gpt_cu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

add_executable(simple_transformer_cuda src/simple_transformer.cu)

target_compile_options(simple_transformer_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

enable_testing()
add_test(NAME simple_transformer_cuda COMMAND simple_transformer_cuda)
