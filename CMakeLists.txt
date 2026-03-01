cmake_minimum_required(VERSION 3.10)

project(kmerTest-build)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA 10.0 REQUIRED)

project(xact-build LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -std=c++17 -Wall -g")

TRY_RUN(
	FIND_GPU_RESULT
	FIND_GPU_COMPILE_RESULT
	${CMAKE_CURRENT_BINARY_DIR}/
	${CMAKE_CURRENT_BINARY_DIR}/../src/findgpu.cu
	RUN_OUTPUT_VARIABLE GPU_ARCH_OUTPUT
)
if("${FIND_GPU_COMPILE_RESULT}" AND ("${FIND_GPU_RESULT}" EQUAL 0))
	message(STATUS "CUDA-capable GPU found. GPU architecture: ${GPU_ARCH_OUTPUT}")
	if(${CMAKE_VERSION} VERSION_LESS "3.18")
	    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${GPU_ARCH_OUTPUT}")
	else()
	    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_OUTPUT})
	endif()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
else()
	message(STATUS "No CUDA-capable GPU found. Generating binary to support multiple GPU generations.")
	set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

find_package(Boost COMPONENTS program_options REQUIRED)

find_package(ZLIB REQUIRED)
find_path(ZLIB_INCLUDE_DIR zlib.h)
include_directories(${ZLIB_INCLUDE_DIR})

find_package(TBB REQUIRED CONFIG)
if(TBB_FOUND)
    message(STATUS "Found TBB: ${TBB_DIR} (found version \"${TBB_VERSION}\")")
endif()

add_executable (kmerTest
    src/murmur3Hash.cu
	src/memTransfer.cu
    src/gpuProperty.cu
    src/twoBitCompressor.cpp
    src/kmer.cu
	src/mash.cu
    src/main.cpp
	src/newickHelper.cpp)

target_link_libraries(kmerTest PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)