import os
import json
import torch
from ISDC import *
from nvCOMP import *
import math
import numpy as np
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def get_mantissa_bitwidth(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 23
    elif dtype == torch.float16:
        return 10
    elif dtype == torch.bfloat16:
        return 7
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def weight_int_to_binary(value_matrix, width_matrix, base_width):
    """
    weight_int_to_binary:
        convert weight data into binary storage
    Args:
        value_matrix
        width_matrix
    Returns:
        data_width_str, weight_data_str
    """
    value_matrix = value_matrix.int()
    width_matrix = width_matrix.int()

    data_width_str = ""
    weight_data_str = ""

    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[1]):
            width = width_matrix[i, j].item()
            if j == 1:
                data_width_str = data_width_str + bin(value_matrix[i, j].item())[2:].zfill(width)
            else:
                if width == 0:
                    weight_data_str = weight_data_str
                weight_data_str = weight_data_str + bin(value_matrix[i, j].item())[2:].zfill(width)

    return data_width_str, weight_data_str


def weight_int_to_binary_length(value_matrix, width_matrix, base_width):
    value_matrix = value_matrix.int()
    width_matrix = width_matrix.int()

    data_width_len = 0
    weight_data_len = 0

    rows, cols = value_matrix.shape

    for i in range(rows):
        for j in range(cols):
            width = width_matrix[i, j].item()

            if j == 1:
                data_width_len += width
            else:
                weight_data_len += width

    return data_width_len, weight_data_len

def index_int_to_binary(int_tensor, bit_size=None):
    """
    index_int_to_binary:
        convert index data into binary storage
    Args:
        int_tensor
        bit_size
    Returns:
        binary_string
    """
    if bit_size is None:
        max_val = int_tensor.max().item()
        bit_size = max(1, (max_val.bit_length()))  # 至少1位

    masks = 2 ** torch.arange(bit_size - 1, -1, -1, device=int_tensor.device)

    bits = (int_tensor.unsqueeze(-1) & masks) > 0

    binary_chars = bits.to(torch.int8).flatten().tolist()
    binary_string = ''.join(map(str, binary_chars))

    return binary_string


def weightMatrixWidthGet(weight, N, M, bit_width):
    """
    weightMatrixWidthGet:Translate the weight matrix into a matrix composed of 'HEAD DATA',
    'DATA WIDTH', and 'DIFF DATA', and calculate the bit width corresponding to each element
    after converting it into binary.
    Args:
        weight
        N
        M
        bit_width
    """

    sub_weight = abs(weight[:, 1:])
    max_values = torch.max(sub_weight, dim=1).values
    bitwidths = torch.ceil(torch.log2(max_values + 1)).to(torch.int32)

    unique_values = torch.unique(bitwidths)
    count_dict = {}
    for value in unique_values:
        count = (bitwidths == value).sum().item()
        count_dict[value.item()] = count
    for i in range(bit_width + 1):
        if i not in count_dict:
            count_dict[i] = 0

    optimized_len = -math.inf
    for i in range(bit_width + 1):
        adder_len = 0
        minus_len = 0
        for j in range(bit_width + 1):
            minus_len = minus_len + count_dict[j] * (
                    math.ceil(math.log2(bit_width + 1)) - math.ceil(math.log2(bit_width - i + 1)))
            if j <= i:
                adder_len = adder_len + count_dict[j] * (N - 1) * (i - j)

        middle_len = minus_len - adder_len
        if middle_len > optimized_len:
            optimized_len = middle_len
            base_width = i

    new_bitwidths = bitwidths - base_width
    device = weight.device
    zeros = torch.zeros_like(new_bitwidths, device=device)
    new_bitwidths = new_bitwidths.to(device)
    new_bitwidths = torch.max(new_bitwidths, zeros)

    new_bitwidths = new_bitwidths.unsqueeze(1)
    first_col = weight[:, :1].to(device)
    rest_cols = weight[:, 1:].to(device)
    new_weight = torch.cat((first_col, new_bitwidths, rest_cols), dim=1)

    head_data_len_vector = torch.full((bitwidths.numel(),), bit_width, dtype=torch.int32,
                                      device=device).unsqueeze(1)
    bit_width_len_vector = torch.full((bitwidths.numel(),), math.ceil(math.log2(bit_width - base_width + 1)),
                                      dtype=torch.int32,
                                      device=device).unsqueeze(1)
    diff_data_len_vector = (new_bitwidths + base_width).flatten()
    diff_data_len_matrix = diff_data_len_vector.repeat(N - 1, 1).T
    width_matrix = torch.cat((head_data_len_vector, bit_width_len_vector, diff_data_len_matrix), dim=1)

    return new_weight, width_matrix, base_width


def iter_linear_weights_from_json(json_dir):
    """
    Generator to load linear layer weights from JSON files one by one.

    Args:
        json_dir (str): Directory containing JSON weight files

    Yields:
        dict:
            {
                "layer_name": str,
                "weight": torch.Tensor,
                "shape": tuple
            }
    """
    # Sort files to keep layer order consistent
    json_files = sorted(os.listdir(json_dir))

    for file_name in json_files:
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        layer_name = file_name.replace(".json", "")
        weight = torch.tensor(data["weight"])
        shape = tuple(data["weight_shape"])
        weight = weight.reshape(shape)

        yield {
            "layer_name": layer_name,
            "weight": weight,
            "shape": shape
        }


if __name__ == "__main__":
    json_dir = "./opt125m_48_bfloat16_weight"
    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    out = {}
    for layer_data in iter_linear_weights_from_json(json_dir):
        layer_name = layer_data["layer_name"]
        exp_out = {}
        N_value = 4
        M_value = 8
        while N_value <= 4 and M_value <= 4:
            temp_out = {}
            weight = layer_data["weight"]
            weight = torch.tensor(weight, dtype=dtype_)
            weight_size = layer_data["weight"].shape
            # print("weight_size = ", weight_size)
            sorted_matrix, sorted_index = addIndex(weight, N_value, M_value)
            sign, exp, mant = extract_float_parts(sorted_matrix, sorted_matrix.dtype)

            exp_uint8 = exp.to(torch.uint8)
            print("exp_uint8 = ", exp_uint8)

            sign_bool = sign.to(torch.bool)
            print("sign_bool = ", sign_bool)

            with open(f"isdc_compressed_weight_{N_value}_{M_value}_/{layer_name}_exp.json", 'w') as f:
                json.dump(exp_uint8.tolist(), f, ensure_ascii=False, indent=4)


            with open(f"isdc_compressed_weight_{N_value}_{M_value}_/{layer_name}_sign.json", 'w') as f:
                json.dump(sign_bool.tolist(), f, ensure_ascii=False, indent=4)

            with open(f"isdc_compressed_weight_{N_value}_{M_value}_/{layer_name}_float.json", 'w') as f:
                json.dump(weight.tolist(), f, ensure_ascii=False, indent=4)

            # mant, sign, exp, index = sort_matrices(
            #     mant,
            #     sign=sign,
            #     exp=exp,
            #     index=sorted_index,
            #     use_torch=True,
            #     use_gpu=(device == 'cuda')
            # )
            #
            # print("mant = ", mant)
            # mant_diff_result, mant_bit_result = compute_difference_and_bit_requirements(
            #     mant,
            #     use_torch=True,
            #     device=device
            # )
            #
            # print("mant_diff_result = ", mant_diff_result)
            # print("mant_bit_result = ", mant_bit_result)
            #
            # value_matrix, width_matrix, base_width = weightMatrixWidthGet(mant_diff_result, N_value, M_value,
            #                                                               mantissa_bitwidth)
            #
            # print("value_matrix=",value_matrix)
            # print("width_matrix=",width_matrix)
            #
            # data_width_str, weight_data_str = weight_int_to_binary(value_matrix, width_matrix, base_width)
            #
            # with open(f"{layer_name}_original_weight.json", 'w') as f:
            #     json.dump(mant.tolist(), f, ensure_ascii=False, indent=4)
            #
            # with open(f"{layer_name}_quantized_weight.json", 'w') as f:
            #     json.dump(mant.tolist(), f, ensure_ascii=False, indent=4)
            #
            # with open(f"{layer_name}_data_width_str.txt", "w", encoding="utf-8") as file:
            #     file.write(data_width_str)
            #
            # with open(f"{layer_name}_weight_data_str.txt", "w", encoding="utf-8") as file:
            #     file.write(weight_data_str)
            #
            # with open(f"{layer_name}_index_data_str.json", "w", encoding="utf-8") as f:
            #     json.dump(sorted_index.tolist(), f, ensure_ascii=False, indent=4)
            #
            # rowNum = width_matrix.shape[0] / weight_size[1]
            # startAddr = cal_addr(width_matrix, rowNum)
            #
            # # print(f"width_matrix.shape[0] = {width_matrix.shape[0]}, weight_size[0] = {weight_size[1]}")
            # # print(f"rowNum = {rowNum}")
            # # print(f"width_matrix = {width_matrix}")
            # # print(f"startAddr = {startAddr}")
            #
            # with open(f"{layer_name}_startAddr.json", 'w', encoding="utf-8") as f:
            #     json.dump(startAddr, f)
            #
            # weight_bit_len_after_compressed = len(data_width_str) + len(weight_data_str) + len(startAddr) * 32
            #
            # all_len_after_compressed = weight_bit_len_after_compressed
            #
            # data_width_len, weight_data_len = weight_int_to_binary_length(value_matrix, width_matrix, base_width)
            # if (all_len_after_compressed)  != data_width_len + weight_data_len + (weight_size[1] + 1) * 32:
            #     print("something is wrong\n")
            #     print("(all_len_after_compressed - len(startAddr) * 32) = ", (all_len_after_compressed - len(startAddr) * 32))
            #     print("data_width_len + weight_data_len = ", data_width_len + weight_data_len)
            #     print(f"len(data_width_str) = {len(data_width_str)}, weight_size[0] * 32 = {(weight_size[0] + 1) * 32}")
            # else:
            #     print("everything is ok")
            #
            #
            # dict = {
            #     "all_len_after_compressed": all_len_after_compressed,
            #     "base_width": base_width,
            #     "ocSize": weight_size[1],
            #     "icSize": weight_size[0],
            #     "ISDC": math.ceil(all_len_after_compressed / 8)
            # }
            #
            # with open(layer_name + ".json", "w") as json_file:
            #     json.dump(dict, json_file, indent=4)
            #
            N_value = N_value * 2
            M_value = M_value * 2
            # break

        # out[f"f{layer_name}"] = dict

    # with open("bfloat16_result.json", "w") as f:
    #     json.dump(out, f)




