cmake_minimum_required(VERSION 3.16)

project(gptcu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gptcu_tokenizer_cpu
    src/tokenizer_bpe_cpu.cpp
)
target_include_directories(gptcu_tokenizer_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(tokenize_bpe_cpu tools/tokenize_bpe.cpp)
target_link_libraries(tokenize_bpe_cpu PRIVATE gptcu_tokenizer_cpu)

enable_testing()
add_executable(tokenizer_bpe_cpu_test tests/tokenizer_bpe_cpu_test.cpp)
target_link_libraries(tokenizer_bpe_cpu_test PRIVATE gptcu_tokenizer_cpu)
add_test(NAME tokenizer_bpe_cpu_test COMMAND tokenizer_bpe_cpu_test)