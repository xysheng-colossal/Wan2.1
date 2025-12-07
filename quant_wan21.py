import os
import json
import argparse
import re
from types import SimpleNamespace

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch import distributed as dist

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier

from wan.modules.model import WanModel
from wan.configs import WAN_CONFIGS
from wan.configs.shared_config import wan_shared_cfg

# 如果使用npu进行量化需开启二进制编译，避免在线编译算子
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False


def parse_quant_args():
    """Parse command-line arguments for Wan model quantization.

    Returns:
        argparse.Namespace: Parsed and validated arguments
    """
    parser = argparse.ArgumentParser(description="Wan model quantization script (no calibration data required)")

    # model arguments
    core_group = parser.add_argument_group(title="Core Model Args")
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")

    core_group.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing original model checkpoints"
    )
    core_group.add_argument(
        '--device_type', 
        type=str, 
        choices=["cpu", "npu"], 
        default="cpu"
    )
    core_group.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device ID to use (default: 0)"
    )

    # Quantization-specific arguments
    quant_group = parser.add_argument_group(title="Quantization Args")
    quant_group.add_argument(
        "--quant_dit_path",
        type=str,
        default="./quant_dit_w8a8_dynamic ",
        help="Directory to save quantized weights (default: ./quant_weights)"
    )
    quant_group.add_argument(
        "--quant_type",
        type=str,
        default="W8A8",
        choices=["W8A8"],
        help="Quantization mode: w8a8 (8bit weight + 8bit activation) or w8a16 (8bit weight + 16bit activation)"
    )
    quant_group.add_argument(
        "--is_dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic quantization (activation parameters generated dynamically)"
    )

    args = parser.parse_args()
    return args


def quant_dit_model(model_path=None, subfolder=None, dtype=torch.bfloat16, save_path=None, disable_name_list=None):
    model = WanModel.from_pretrained(model_path, subfolder=subfolder, 
                                     torch_dtype=dtype, device_map=args.device_type)

    # 由于llm ptq接口限制，模型补充dtype属性
    if not hasattr(model, 'config'):
        model.config = SimpleNamespace()  # 使用轻量级命名空间
    if not hasattr(model.config, 'torch_dtype'):
        model.config.torch_dtype = dtype
    if not hasattr(model, "dtype"):
        model.dtype = dtype

    match = re.match(r'W(\d+)A(\d+)', args.quant_type)
    if not match:
        raise ValueError(f"Invalid quant_type format: {args.quant_type}, expected W<num>A<num>")

    quant_config = QuantConfig(
        w_bit=int(match.group(1)),
        a_bit=int(match.group(2)),
        disable_names=disable_name_list,
        dev_type=args.device_type,
        dev_id=args.device_id,
        act_method=3,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=args.is_dynamic,
    )

    # 2.执行校准，不需要校准数据的场景不需要传calib_data
    calibrator = Calibrator(model, quant_config, disable_level='L0')  # disable_level: 自动回退n个linear
    calibrator.run()  # 执行PTQ量化校准

    # 3.多卡场景需要保存多个权重
    save_path = os.path.join(save_path, subfolder)
    os.makedirs(save_path, exist_ok=True)
    calibrator.save(save_path, save_type=["safe_tensor"])

def quant(args):
    config = WAN_CONFIGS[args.task]
    torch.npu.set_device(args.device_id)

    #回退层layer名
    disable_name_list = []

    quant_dit_model(
        model_path=args.ckpt_dir, 
        subfolder="", 
        dtype=config.param_dtype,
        save_path=args.quant_dit_path,
        disable_name_list=disable_name_list
    )


if __name__ == "__main__":
    args = parse_quant_args()
    quant(args)