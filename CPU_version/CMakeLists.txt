cmake_minimum_required(VERSION 3.3)
project(hw3 C)

set(CMAKE_C_STANDARD 99)

if (CMAKE_C_COMPILER_ID STREQUAL GNU)
  add_compile_options(-lm -mavx -fopt-info-vec)
elseif (CMAKE_C_COMPILER_ID STREQUAL Clang)
  add_compile_options(-Rpass=.*loop.*)
elseif (CMAKE_C_COMPILER_ID STREQUAL AppleClang)
  add_compile_options(-Rpass=.*loop.*)
elseif (CMAKE_C_COMPILER_ID STREQUAL Intel)
  add_compile_options(-xAVX -qopt-report-stdout)
endif()

set(LIB_SOURCES
  src/tick.c
  src/mat_types.c
  src/vec.c
  )

add_library(hw3_lib ${LIB_SOURCES})
add_executable(ray_ser src/ray_tracing.c)
if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  target_link_libraries(ray_ser PUBLIC m)
endif ()
target_link_libraries(ray_ser PRIVATE hw3_lib)

find_package(OpenMP)

if(OpenMP_FOUND)
  add_executable(ray_par src/ray_tracing.c)

  if(CMAKE_C_COMPILER_ID STREQUAL GNU)
    target_link_libraries(ray_par PUBLIC m)
  endif ()
  target_compile_options(ray_par PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(ray_par PRIVATE hw3_lib ${OpenMP_C_FLAGS})
endif()
