import numpy as np
from ctypes import *
import ctypes

libdecompress_768_768 = ctypes.CDLL("../cuda/endToEnd/libdecompress_768_768.so")
libdecompress_768_3072 = ctypes.CDLL("../cuda/endToEnd/libdecompress_768_3072.so")
libdecompress_3072_768 = ctypes.CDLL("../cuda/endToEnd/libdecompress_3072_768.so")

libdecompress_1024_1024 = ctypes.CDLL("../cuda/endToEnd/libdecompress_1024_1024.so")
libdecompress_1024_4096 = ctypes.CDLL("../cuda/endToEnd/libdecompress_1024_4096.so")
libdecompress_4096_1024 = ctypes.CDLL("../cuda/endToEnd/libdecompress_4096_1024.so")

libdecompress_2048_2048 = ctypes.CDLL("../cuda/endToEnd/libdecompress_2048_2048.so")
libdecompress_2048_8192 = ctypes.CDLL("../cuda/endToEnd/libdecompress_2048_8192.so")
libdecompress_8192_2048 = ctypes.CDLL("../cuda/endToEnd/libdecompress_8192_2048.so")

# libdecompress_768_768 = ctypes.CDLL("./kernel/libdecompress_768_768.
#
#
# `
#
#
# so")
# libdecompress_768_3072 = ctypes.CDLL("./kernel/libdecompress_768_3072.so")
# libdecompress_3072_768 = ctypes.CDLL("./kernel/libdecompress_3072_768.so")
#
# libdecompress_1024_1024 = ctypes.CDLL("./kernel/libdecompress_1024_1024.so")
# libdecompress_1024_4096 = ctypes.CDLL("./kernel/libdecompress_1024_4096.so")
# libdecompress_4096_1024 = ctypes.CDLL("./kernel/libdecompress_4096_1024.so")
#
# libdecompress_2048_2048 = ctypes.CDLL("./kernel/libdecompress_2048_2048.so")
# libdecompress_2048_8192 = ctypes.CDLL("./kernel/libdecompress_2048_8192.so")
# libdecompress_8192_2048 = ctypes.CDLL("./kernel/libdecompress_8192_2048.so")

def callDecompressedWeightCu_768_768(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,

        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):

    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")

    libdecompress_768_768.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))


    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_768_768.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value


def callDecompressedWeightCu_768_3072(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,

        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    # print(compressed_data_width)

    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")

    libdecompress_768_3072.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)

    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))


    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_768_3072.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                        decompressed_weight_ptr
                                         )
    return decompressed_weight, runtime_.value



def callDecompressedWeightCu_3072_768(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_3072_768.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_3072_768.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value


def callDecompressedWeightCu_1024_1024(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_1024_1024.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_1024_1024.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value


def callDecompressedWeightCu_1024_4096(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_1024_4096.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_1024_4096.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value

def callDecompressedWeightCu_4096_1024(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_4096_1024.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_4096_1024.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value



def callDecompressedWeightCu_2048_2048(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_2048_2048.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_2048_2048.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value



def callDecompressedWeightCu_2048_8192(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_2048_8192.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_2048_8192.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value



def callDecompressedWeightCu_8192_2048(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,


        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width
):
    decompressed_weight = np.random.randn(ocSize, icSize).astype(np.uint16)
    # print(f"shape = {decompressed_weight.shape}")
    libdecompress_8192_2048.call_decompress_kernel.argtypes = [
        POINTER(c_uint32), c_int,
        POINTER(c_uint32), c_int,
        POINTER(c_uint8), c_int,

        POINTER(c_uint8), c_int,
        POINTER(c_bool), c_int,

        POINTER(c_uint32),
        c_int,
        c_int,
        c_int,
        c_int,
        POINTER(c_float),

        POINTER(c_uint16)
    ]

    # compressed_data_width = compressed_data_width.cpu().numpy().astype(np.uint8)
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(POINTER(c_uint32))

    # compressed_weight_data = compressed_weight_data.cpu().numpy().astype(np.uint8)
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(POINTER(c_uint32))

    # compressed_index_data = compressed_index_data.cpu().numpy().astype(np.uint8)
    index_data_ptr = index_data.ctypes.data_as(POINTER(c_uint8))

    sign_data_ptr = sign_data.ctypes.data_as(POINTER(c_bool))
    exp_data_ptr = exp_data.ctypes.data_as(POINTER(c_uint8))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(POINTER(c_uint32))

    runtime_ = c_float(0.0)

    decompressed_weight_ptr = decompressed_weight.ctypes.data_as(POINTER(c_uint16))

    libdecompress_8192_2048.call_decompress_kernel(compressed_data_width_ptr, compressed_data_width_size,
                                         compressed_weight_data_ptr, compressed_weight_data_size,
                                         index_data_ptr, index_data_size,

                                         exp_data_ptr, exp_data_size,
                                         sign_data_ptr, sign_data_size,

                                         startAddr_ptr,
                                         ocSize, icSize,
                                         data_width_len,
                                         base_width,
                                         byref(runtime_),

                                         decompressed_weight_ptr
                                         )

    return decompressed_weight, runtime_.value


def callDecompressedWeightCu_768_768_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_768_768.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_768_768.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value


def callDecompressedWeightCu_768_3072_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_768_3072.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_768_3072.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value


def callDecompressedWeightCu_3072_768_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_3072_768.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_3072_768.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value

def callDecompressedWeightCu_1024_1024_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_1024_1024.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_1024_1024.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value


def callDecompressedWeightCu_1024_4096_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_1024_4096.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_1024_4096.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value


def callDecompressedWeightCu_4096_1024_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_4096_1024.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_4096_1024.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_2048_2048_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_2048_2048.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_2048_2048.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_2048_8192_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_2048_8192.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_2048_8192.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_8192_2048_toGPU(
        compressed_data_width, compressed_data_width_size,
        compressed_weight_data, compressed_weight_data_size,
        index_data, index_data_size,
        sign_data, sign_data_size,
        exp_data, exp_data_size,
        startAddr,
        ocSize, icSize,
        data_width_len,
        base_width,

        decompressed_weight_gpu
):
    # 1. Create output tensor on GPU (must be on the same device as the C function, e.g., cuda:3)
    # device = torch.device('cuda:3')
    # decompressed_weight_gpu = torch.empty((ocSize, icSize), dtype=torch.uint16, device=device)

    # 2. Set argument types for the C function (use the new function name call_decompress_kernel_toGPU)
    libdecompress_8192_2048.call_decompress_kernel_toGPU.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p   # GPU pointer, received as void*
    ]

    # 3. Convert input data to pointers (assuming inputs are already CPU numpy arrays or PyTorch CPU tensors)
    #    Note: if inputs are on GPU, additional handling is needed; here we assume CPU inputs
    compressed_data_width_ptr = compressed_data_width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    compressed_weight_data_ptr = compressed_weight_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    index_data_ptr = index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    exp_data_ptr = exp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    sign_data_ptr = sign_data.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    startAddr_np = np.array(startAddr, dtype=np.uint32)
    startAddr_ptr = startAddr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

    runtime_ = ctypes.c_float(0.0)

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    # 4. Call the C function, passing the GPU pointer
    libdecompress_8192_2048.call_decompress_kernel_toGPU(
        compressed_data_width_ptr, compressed_data_width_size,
        compressed_weight_data_ptr, compressed_weight_data_size,
        index_data_ptr, index_data_size,
        exp_data_ptr, exp_data_size,
        sign_data_ptr, sign_data_size,
        startAddr_ptr,
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)  # Key: pass the GPU pointer
    )

    # 5. Return the GPU tensor (no copy back to CPU)
    return decompressed_weight_gpu, runtime_.value


def callDecompressedWeightCu_768_768_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_768_768.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]


    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_768_768.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_768_3072_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_768_3072.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_768_3072.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_3072_768_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):


    libdecompress_3072_768.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_3072_768.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_1024_1024_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_1024_1024.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_1024_1024.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_1024_4096_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_1024_4096.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_1024_4096.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value



def callDecompressedWeightCu_4096_1024_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_4096_1024.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_4096_1024.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value




def callDecompressedWeightCu_2048_2048_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_2048_2048.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_2048_2048.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value




def callDecompressedWeightCu_2048_8192_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_2048_8192.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_2048_8192.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value





def callDecompressedWeightCu_8192_2048_DEVICE(
        d_compressed_data_width,          # GPU tensor (uint32) or pointer
        compressed_data_width_size,
        d_compressed_weight_data,         # GPU tensor (uint32)
        compressed_weight_data_size,
        d_index_data,                     # GPU tensor (uint8)
        index_data_size,
        d_sign_data,                      # GPU tensor (bool)
        sign_data_size,
        d_exp_data,                       # GPU tensor (uint8)
        exp_data_size,
        d_startAddr,                      # GPU tensor (uint32)
        ocSize, icSize,
        data_width_len,
        base_width,
        decompressed_weight_gpu           # output GPU tensor (uint16)
):

    libdecompress_8192_2048.call_decompress_kernel_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_data_width, size
        ctypes.c_void_p, ctypes.c_int,  # d_compressed_weight_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_index_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_exp_data, size
        ctypes.c_void_p, ctypes.c_int,  # d_sign_data, size
        ctypes.c_void_p,  # d_startAddr
        ctypes.c_int, ctypes.c_int,  # oc_size, ic_size
        ctypes.c_int, ctypes.c_int,  # data_width_len, base_width
        ctypes.POINTER(ctypes.c_float),  # runtime_
        ctypes.c_void_p  # d_decompressed_weight
    ]

    def get_ptr(t):
        if hasattr(t, 'data_ptr'):
            return ctypes.c_void_p(t.data_ptr())
        elif isinstance(t, int):
            return ctypes.c_void_p(t)
        else:
            return t

    runtime_ = ctypes.c_float(0.0)

    libdecompress_8192_2048.call_decompress_kernel_device(
        get_ptr(d_compressed_data_width), compressed_data_width_size,
        get_ptr(d_compressed_weight_data), compressed_weight_data_size,
        get_ptr(d_index_data), index_data_size,
        get_ptr(d_exp_data), exp_data_size,
        get_ptr(d_sign_data), sign_data_size,
        get_ptr(d_startAddr),
        ocSize, icSize,
        data_width_len, base_width,
        ctypes.byref(runtime_),
        get_ptr(decompressed_weight_gpu)   # output pointer
    )

    return decompressed_weight_gpu, runtime_.value