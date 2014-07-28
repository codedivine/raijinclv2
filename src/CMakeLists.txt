add_library(Raijin gemm zgemm trans jsoncpp ${RAIJIN_HEADERS})
target_link_libraries(Raijin ${OPENCL_LIBRARY} ${RT_LIBRARY})
#file(GLOB OPENCL_SOURCE_FILES "*.cl")
#file(COPY ${OPENCL_SOURCE_FILES} DESTINATION $ENV{RAIJIN_TUNE_PATH})
add_executable(raijintune raijintune rgemmtune transtune zgemmtune ${RAIJIN_HEADERS})
target_link_libraries(raijintune Raijin ${OPENCL_LIBRARY})

