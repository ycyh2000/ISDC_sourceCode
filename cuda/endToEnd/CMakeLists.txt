cmake_minimum_required(VERSION 4.0)

project(MySDKProject LANGUAGES CXX CUDA)

add_library(decompress SHARED decompress.cu)
add_library(decompress_768_3072 SHARED decompress_768_3072.cu)
add_library(decompress_3072_768 SHARED decompress_3072_768.cu)
add_library(decompress_768_768 SHARED decompress_768_768.cu)
add_library(decompress_1024_1024 SHARED decompress_1024_1024.cu)
add_library(decompress_1024_4096 SHARED decompress_1024_4096.cu)
add_library(decompress_4096_1024 SHARED decompress_4096_1024.cu)
add_library(decompress_2048_2048 SHARED decompress_2048_2048.cu)
add_library(decompress_2048_8192 SHARED decompress_2048_8192.cu)
add_library(decompress_8192_2048 SHARED decompress_8192_2048.cu)


target_link_libraries(decompress PRIVATE cudart)
target_link_libraries(decompress_768_3072 PRIVATE cudart)
target_link_libraries(decompress_3072_768 PRIVATE cudart)
target_link_libraries(decompress_768_768 PRIVATE cudart)
target_link_libraries(decompress_1024_1024 PRIVATE cudart)
target_link_libraries(decompress_1024_4096 PRIVATE cudart)
target_link_libraries(decompress_4096_1024 PRIVATE cudart)
target_link_libraries(decompress_2048_2048 PRIVATE cudart)
target_link_libraries(decompress_2048_8192 PRIVATE cudart)
target_link_libraries(decompress_8192_2048 PRIVATE cudart)

target_compile_options(decompress PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86  
    >
)

target_compile_options(decompress_768_3072 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

target_compile_options(decompress_3072_768 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

target_compile_options(decompress_768_768 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

target_compile_options(decompress_1024_1024 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

target_compile_options(decompress_1024_4096 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

target_compile_options(decompress_4096_1024 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)


target_compile_options(decompress_2048_2048 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)


target_compile_options(decompress_2048_8192 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)


target_compile_options(decompress_8192_2048 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -lineinfo
        -Xcompiler=-fPIC
        -arch=sm_86
    >
)

set_target_properties(decompress PROPERTIES
    OUTPUT_NAME "decompress"
    CUDA_SEPARABLE_COMPILATION ON
)



