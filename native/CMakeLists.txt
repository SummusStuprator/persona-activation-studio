cmake_minimum_required(VERSION 3.24)
project(persona_activation_studio_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

set(LLAMA_CPP_DIR "" CACHE PATH "Path to a llama.cpp checkout")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp-build EXCLUDE_FROM_ALL)

add_library(activation_bridge SHARED bridge.cpp)
target_include_directories(activation_bridge PRIVATE
  "${LLAMA_CPP_DIR}/include"
  "${LLAMA_CPP_DIR}/ggml/include"
)
target_link_libraries(activation_bridge PRIVATE llama ggml)
set_target_properties(activation_bridge PROPERTIES PREFIX "")
