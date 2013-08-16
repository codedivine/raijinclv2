cmake_minimum_required(VERSION 2.6)
PROJECT(RAIJIN)
set(OPENCL_INCLUDE_DIR $ENV{OPENCLINCLUDEDIR})
message("OpenCL path is ${OPENCL_INCLUDE_DIR}")
IF (UNIX)
	FIND_LIBRARY(OPENCL_LIBRARY NAMES OpenCL PATHS ENV LD_LIBRARY_PATH)
	FIND_LIBRARY(RT_LIBRARY NAMES rt PATHS ENV LD_LIBRARY_PATH REQUIRED)
	message("rt library path is ${RT_LIBRARY}")
	message("OpenCL library path is ${OPENCL_LIBRARY}")
ENDIF (UNIX)

IF (WIN32)
	FIND_LIBRARY(OPENCL_LIBRARY OpenCL PATHS ENV OPENCLLIBDIR REQUIRED)
	message("OpenCL library path is ${OPENCL_LIBRARY}")
ENDIF (WIN32)

include_directories( $ENV{OPENCLINCLUDEDIR} )
include_directories("${PROJECT_SOURCE_DIR}/include")
file(GLOB RAIJIN_HEADERS "include/*hpp")
add_subdirectory(src)
add_subdirectory(test)


