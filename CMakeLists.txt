cmake_minimum_required(VERSION 3.18)

project(gpt_cu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CTest)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

add_library(simple_transformer_reference STATIC
  src/simple_transformer_reference.cpp
)

target_include_directories(simple_transformer_reference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(simple_transformer_cuda src/simple_transformer.cu)

target_link_libraries(simple_transformer_cuda PRIVATE simple_transformer_reference)

target_compile_options(simple_transformer_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(BUILD_TESTING)
  add_executable(simple_transformer_reference_test
    tests/simple_transformer_reference_test.cpp
  )
  target_link_libraries(simple_transformer_reference_test PRIVATE
    simple_transformer_reference
  )

  add_test(NAME simple_transformer_reference_test
           COMMAND simple_transformer_reference_test)
  add_test(NAME simple_transformer_cuda COMMAND simple_transformer_cuda)
endif()
