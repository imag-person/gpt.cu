cmake_minimum_required(VERSION 3.16)

project(gptcu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GPTCU_ENABLE_CUDA "Build optional CUDA tokenizer accelerators when nvcc is available" ON)

add_library(gptcu_tokenizer_cpu
    src/tokenizer_bpe_cpu.cpp
)
target_include_directories(gptcu_tokenizer_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(tokenize_bpe_cpu tools/tokenize_bpe.cpp)
target_link_libraries(tokenize_bpe_cpu PRIVATE gptcu_tokenizer_cpu)

if(GPTCU_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_library(gptcu_tokenizer_cuda
            src/tokenizer_bpe_cuda.cu
        )
        target_include_directories(gptcu_tokenizer_cuda PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(gptcu_tokenizer_cuda PUBLIC gptcu_tokenizer_cpu)
        set_target_properties(gptcu_tokenizer_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )

        add_executable(tokenize_bytes_cuda tools/tokenize_bytes_cuda.cpp)
        target_link_libraries(tokenize_bytes_cuda PRIVATE gptcu_tokenizer_cuda)
    else()
        message(STATUS "CUDA compiler not found; skipping optional CUDA tokenizer targets")
    endif()
endif()

enable_testing()
add_executable(tokenizer_bpe_cpu_test tests/tokenizer_bpe_cpu_test.cpp)
target_link_libraries(tokenizer_bpe_cpu_test PRIVATE gptcu_tokenizer_cpu)
add_test(NAME tokenizer_bpe_cpu_test COMMAND tokenizer_bpe_cpu_test)

if(TARGET gptcu_tokenizer_cuda)
    add_executable(tokenizer_bpe_cuda_test tests/tokenizer_bpe_cuda_test.cpp)
    target_link_libraries(tokenizer_bpe_cuda_test PRIVATE gptcu_tokenizer_cuda)
    add_test(NAME tokenizer_bpe_cuda_test COMMAND tokenizer_bpe_cuda_test)
endif()