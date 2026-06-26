import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from callDecompressKernel import *
import json
import ml_dtypes
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ISDCLinearDevice(nn.Module):
    def __init__(self, in_features, out_features, compressed_representation, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Save raw compressed data (may be on CPU)
        self.compressed_data_width = compressed_representation["compressed_data_width"]
        self.compressed_data_width_size = compressed_representation["compressed_data_width_size"]
        self.compressed_weight_data = compressed_representation["compressed_weight_data"]
        self.compressed_weight_data_size = compressed_representation["compressed_weight_data_size"]
        self.index_data = compressed_representation["index_data"]
        self.index_data_size = compressed_representation["index_data_size"]
        self.sign_data = compressed_representation["sign_data"]
        self.sign_data_size = compressed_representation["sign_data_size"]
        self.exp_data = compressed_representation["exp_data"]
        self.exp_data_size = compressed_representation["exp_data_size"]
        self.startAddr = compressed_representation["startAddr"]
        self.data_width_len = compressed_representation["data_width_len"]
        self.base_width = compressed_representation["base_width"]

        # if self.out_features == 3072:
        #     print(self.startAddr)
        #     print(self.out_features,len(self.startAddr))


        # Move compressed data to GPU (cuda:3, consistent with cudaSetDevice(3) in the C function)
        device = torch.device('cuda:3')
        self.d_compressed_data_width = torch.tensor(self.compressed_data_width, dtype=torch.uint32, device=device)
        self.d_compressed_weight_data = torch.tensor(self.compressed_weight_data, dtype=torch.uint32, device=device)
        self.d_index_data = torch.tensor(self.index_data, dtype=torch.uint8, device=device)
        self.d_sign_data = torch.tensor(self.sign_data, dtype=torch.bool, device=device)
        self.d_exp_data = torch.tensor(self.exp_data, dtype=torch.uint8, device=device)
        self.d_startAddr = torch.tensor(self.startAddr, dtype=torch.uint32, device=device)



        8
        # Output weight buffer (decompressed result will be written directly here)
        # self.weight = torch.empty((self.out_features, self.in_features), dtype=torch.uint16, device=device)

        # Bias handling (optional)
        if bias is not None:
            self.register_buffer('bias', bias.to(device))
        else:
            self.bias = None

    def _decompress(self, weight):
        """
        Decompress the compressed representation directly into self.weight.
        Returns the weight tensor and runtime.
        """
        if self.in_features == 768 and self.out_features == 768:
            decompress_weight, runtime_ = callDecompressedWeightCu_768_768_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight  # Pass the output buffer
            )
        elif self.in_features == 3072 and self.out_features == 768:
            decompress_weight, runtime_ = callDecompressedWeightCu_3072_768_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 768 and self.out_features == 3072:
            decompress_weight, runtime_ = callDecompressedWeightCu_768_3072_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 1024 and self.out_features == 4096:
            decompress_weight, runtime_ = callDecompressedWeightCu_1024_4096_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 4096 and self.out_features == 1024:
            decompress_weight, runtime_ = callDecompressedWeightCu_4096_1024_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 1024 and self.out_features == 1024:
            decompress_weight, runtime_ = callDecompressedWeightCu_1024_1024_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 2048 and self.out_features == 8192:
            decompress_weight, runtime_ = callDecompressedWeightCu_2048_8192_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )

        elif self.in_features == 8192 and self.out_features == 2048:
            decompress_weight, runtime_ = callDecompressedWeightCu_8192_2048_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )

        elif self.in_features == 2048 and self.out_features == 2048:
            decompress_weight, runtime_ = callDecompressedWeightCu_2048_2048_DEVICE(
                self.d_compressed_data_width, self.compressed_data_width_size,
                self.d_compressed_weight_data, self.compressed_weight_data_size,
                self.d_index_data, self.index_data_size,
                self.d_sign_data, self.sign_data_size,
                self.d_exp_data, self.exp_data_size,
                self.d_startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        else:
            raise NotImplementedError(f"Unsupported dimensions: out={self.out_features}, in={self.in_features}")

        # self.weight has been updated in place, decompress_weight and self.weight refer to the same object
        return weight, runtime_

    def forward(self, x):
        # assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}"
        # weight, runtime_ = self._decompress()
        # weight = torch.from_numpy(weight).view(torch.bfloat16)
        #
        # # Convert to match input's device and dtype
        # weight = weight.to(device=x.device)


        weight = torch.zeros((self.out_features, self.in_features), dtype=x.dtype, device=x.device)
        weight, runtime_ = self._decompress(weight)
        # weight = weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype, device=x.device)
        # print(f"x = {x}, weight = {weight}, bias = {bias}")
        return F.linear(x, weight, bias)




class ISDCLinear(nn.Module):
    def __init__(self, in_features, out_features, compressed_representation, bias=None):
        """
        Custom linear layer that does not store the original weight,
        but stores a compressed representation instead.
        Suitable for inference (no gradient needed).

        Args:
            in_features, out_features: input / output dimensions
            compressed_representation: compressed weight data (format is user-defined)
            bias: bias tensor (if any)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.compressed_data_width = compressed_representation["compressed_data_width"]
        self.compressed_data_width_size = compressed_representation["compressed_data_width_size"]

        self.compressed_weight_data = compressed_representation["compressed_weight_data"]
        self.compressed_weight_data_size = compressed_representation["compressed_weight_data_size"]

        self.index_data = compressed_representation["index_data"]
        self.index_data_size = compressed_representation["index_data_size"]

        self.sign_data = compressed_representation["sign_data"]
        self.sign_data_size = compressed_representation["sign_data_size"]

        self.exp_data = compressed_representation["exp_data"]
        self.exp_data_size = compressed_representation["exp_data_size"]

        self.startAddr = compressed_representation["startAddr"]

        self.data_width_len = compressed_representation["data_width_len"]
        self.base_width = compressed_representation["base_width"]

        # self.weight = torch.randn(self.out_features, self.in_features, device='cuda')

        # Bias handling (optional)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None


    def _decompress(self, weight):
        """
        Decompress the compressed representation into the full weight matrix (out_features, in_features).
        The user should implement this according to the actual compression algorithm.
        """
        # Example: assume compressed_weight is a concatenation of two matrices from low-rank decomposition
        # This is just an illustration; follow your own algorithm.
        # For instance, compressed_weight could be of shape (rank, in_features+out_features) or other forms.
        # Return a tensor of shape (out_features, in_features) after decompression.


        if self.in_features == 768 and self.out_features == 768:
            decompress_weight, runtime_ = callDecompressedWeightCu_768_768_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 3072 and self.out_features == 768:
            decompress_weight, runtime_ = callDecompressedWeightCu_3072_768_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 768 and self.out_features == 3072:
            decompress_weight, runtime_ = callDecompressedWeightCu_768_3072_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 1024 and self.out_features == 4096:
            decompress_weight, runtime_ = callDecompressedWeightCu_1024_4096_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 4096 and self.out_features == 1024:
            decompress_weight, runtime_ = callDecompressedWeightCu_4096_1024_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 1024 and self.out_features == 1024:
            decompress_weight, runtime_ = callDecompressedWeightCu_1024_1024_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 2048 and self.out_features == 8192:
            decompress_weight, runtime_ = callDecompressedWeightCu_2048_8192_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 8192 and self.out_features == 2048:
            decompress_weight, runtime_ = callDecompressedWeightCu_8192_2048_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        elif self.in_features == 2048 and self.out_features == 2048:
            decompress_weight, runtime_ = callDecompressedWeightCu_2048_2048_toGPU(
                self.compressed_data_width, self.compressed_data_width_size,
                self.compressed_weight_data, self.compressed_weight_data_size,
                self.index_data, self.index_data_size,
                self.sign_data, self.sign_data_size,
                self.exp_data, self.exp_data_size,
                self.startAddr,
                self.in_features, self.out_features,
                self.data_width_len,
                self.base_width,
                weight
            )
        else:
            raise NotImplementedError(f"Unsupported dimensions: out={self.out_features}, in={self.in_features}")

        return decompress_weight, runtime_

    def forward(self, x):
        # assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}"
        # weight, runtime_ = self._decompress()
        # weight = torch.from_numpy(weight).view(torch.bfloat16)
        #
        # # Convert to match input's device and dtype
        # weight = weight.to(device=x.device)
        weight = torch.zeros((self.out_features, self.in_features), dtype=x.dtype, device=x.device)
        weight, runtime_ = self._decompress(weight)
        # weight = weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias
        # if bias is not None:
        #     bias = bias.to(device=x.device)
        # print(f"x = {x.device}, weight = {weight.device}, bias = {bias.device}")

        return F.linear(x, weight, bias)



def replace_linear_layer(layer, compressed_representation):
    """
    Replace a single nn.Linear layer with ISDCLinear using a compressed representation.
    If the input is not an nn.Linear, return it unchanged.

    Args:
        layer: A PyTorch module (e.g., nn.Linear or any other)
        compressed_representation: The compressed weight data (e.g., quantized tensor, low-rank factors, etc.)

    Returns:
        If layer is nn.Linear -> ISDCLinear with compressed parameters.
        Otherwise -> the original layer.
    """
    if isinstance(layer, nn.Linear):
        # Extract bias tensor if it exists
        bias_tensor = layer.bias.data if layer.bias is not None else None

        # Create new custom layer with compressed representation and original bias
        new_layer = ISDCLinear(
            layer.in_features,
            layer.out_features,
            compressed_representation,  # Pass the compressed data here
            bias=bias_tensor
        )
        return new_layer
    else:
        return layer




def replace_linear_layer_DEVICE(layer, compressed_representation):
    """
    Replace a single nn.Linear layer with ISDCLinear using a compressed representation.
    If the input is not an nn.Linear, return it unchanged.

    Args:
        layer: A PyTorch module (e.g., nn.Linear or any other)
        compressed_representation: The compressed weight data (e.g., quantized tensor, low-rank factors, etc.)

    Returns:
        If layer is nn.Linear -> ISDCLinear with compressed parameters.
        Otherwise -> the original layer.
    """
    if isinstance(layer, nn.Linear):
        # Extract bias tensor if it exists
        bias_tensor = layer.bias.data if layer.bias is not None else None

        # Create new custom layer with compressed representation and original bias
        new_layer = ISDCLinearDevice(
            layer.in_features,
            layer.out_features,
            compressed_representation,  # Pass the compressed data here
            bias=bias_tensor
        )

        # print(f"layer = {layer}, new_layer = {new_layer}, len(startAddr) = {len(compressed_representation['startAddr'])}")
    
        return new_layer
    else:
        return layer

def get_mantissa_bitwidth(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 23
    elif dtype == torch.float16:
        return 10
    elif dtype == torch.bfloat16:
        return 7
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def read_txt_to_string(file_path):
    """
    read_txt_to_string : Read the content of a txt file into a string
    Args:
        file_path (str): Path of the file to read
    Returns:
        str: File content string (on success)
        None: None (on failure)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all content using read() method and return
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found (path: {file_path})")
    except PermissionError:
        print(f"Error: No permission to access file (path: {file_path})")
    except UnicodeDecodeError:
        print(f"Error: File encoding may not be UTF-8 (path: {file_path})")
    except Exception as e:
        print(f"Error: Unexpected exception occurred - {str(e)}")
    return None



def string_to_uint32_array(s):
    """Converts a binary string (composed of '0's and '1's) to a uint32 array.
    Args:
        s (str): Input binary string containing only '0's and '1's
    Returns:
        np.ndarray: uint32 array where each element is the decimal value of 32-bit binary chunks
    """
    padding = (32 - (len(s) % 32)) % 32
    s_padded = s.ljust(len(s) + padding, '0')
    chunks = [s_padded[i:i + 32] for i in range(0, len(s_padded), 32)]
    uint32_values = [int(chunk, 2) for chunk in chunks]
    return np.array(uint32_values, dtype=np.uint32)



if __name__ == "__main__":
    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt125m_isdc_compressed_weight_2_4"
    name = "model_decoder_layers_0_fc2"
    layer_name = f"{folder_path}/{name}"
    with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
        message = json.load(f)

    ocSize = message["ocSize"]
    icSize = message["icSize"]
    minibatch = 1
    data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
    base_width = message["base_width"]

    print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

    data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
    weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
    with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    index_data_size = len(index_data) * len(index_data[0])
    index_data = np.array(index_data, dtype=np.uint8)

    with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
        startAddr = json.load(f)
    with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
        sign_data = json.load(f)
    with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
        exp_data = json.load(f)

    exp_data_size = len(exp_data) * len(exp_data[0])
    exp_data = np.array(exp_data, dtype=np.uint8)

    sign_data_size = len(sign_data) * len(sign_data[0])
    sign_data = np.array(sign_data, dtype=np.bool)

    compressed_data_width = string_to_uint32_array(data_width_str)
    compressed_data_width_size = compressed_data_width.size
    compressed_weight_data = string_to_uint32_array(weight_data_str)

    compressed_weight_data_size = compressed_weight_data.size
    # compressed_index_data = string_to_uint32_array(index_data_str)
    # compressed_index_data_size = compressed_index_data.size

    compressed = {}
    compressed['compressed_data_width'] = compressed_data_width
    compressed['compressed_data_width_size'] = compressed_data_width_size
    compressed['compressed_weight_data'] = compressed_weight_data
    compressed['compressed_weight_data_size'] = compressed_weight_data_size
    compressed['index_data'] = index_data
    compressed['index_data_size'] = index_data_size
    compressed['sign_data'] = sign_data
    compressed['sign_data_size'] = sign_data_size
    compressed['exp_data'] = exp_data
    compressed['exp_data_size'] = exp_data_size
    compressed['startAddr'] = startAddr
    compressed['data_width_len'] = data_width_len
    compressed['base_width'] = base_width


    batch_size = 1
    x = torch.randn(batch_size, icSize)
    # Original linear layer
    linear = nn.Linear(icSize, ocSize)
    with torch.no_grad():
        output = linear(x)

    # print(f"x = {x}, output = {output}")


    new_layer = replace_linear_layer(linear, compressed)
    # print(new_layer.state_dict().keys())  # odict_keys(['compressed_weight', 'bias'])

    pass