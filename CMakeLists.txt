cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(xact-build LANGUAGES CXX CUDA)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--use_fast_math>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-g>")

TRY_RUN(
    FIND_GPU_RESULT
    FIND_GPU_COMPILE_RESULT
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${CMAKE_CURRENT_BINARY_DIR}/../src/findgpu.cu
    RUN_OUTPUT_VARIABLE GPU_ARCH_OUTPUT
)
if("${FIND_GPU_COMPILE_RESULT}" AND ("${FIND_GPU_RESULT}" EQUAL 0))
    message(STATUS "CUDA-capable GPU found. GPU architecture: ${GPU_ARCH_OUTPUT}")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_OUTPUT})
else()
    message(STATUS "No CUDA-capable GPU found. Generating binary to support multiple GPU generations.")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TBB REQUIRED CONFIG)
if(TBB_FOUND)
    message(STATUS "Found TBB: ${TBB_DIR} (found version \"${TBB_VERSION}\")")
endif()

add_executable (aligner
    src/main.cpp
    src/alignment.cu
    src/gpuProperty.cu
)

add_executable (compare_alignment
    src/compare_alignment.cpp
)

add_executable (check_alignment
    src/check_alignment.cpp
)

target_link_libraries(aligner PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)
target_link_libraries(compare_alignment PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)
target_link_libraries(check_alignment PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)

