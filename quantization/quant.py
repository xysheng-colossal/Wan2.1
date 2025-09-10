import argparse
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from wan import WanT2V, WanI2V
from wan.configs import WAN_CONFIGS
from quantization.config import get_wan_quant_config
from quantization.quantizer import WanQuantizer


def init_npu_env():
    """Initialize NPU environment (prevent online operator compilation).
    
    Disables JIT compilation and internal format optimization to ensure
    compatibility with quantized operations.
    """
    try:
        # Disable JIT compilation for consistent behavior
        torch_npu.npu.set_compile_mode(jit_compile=False)
        # Disable internal format to maintain quantized weight compatibility
        torch.npu.config.allow_internal_format = False
        print("NPU environment initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize NPU environment: {str(e)}")


def parse_quant_args():
    """Parse command-line arguments for Wan model quantization.
    
    Returns:
        argparse.Namespace: Parsed and validated arguments
    """
    parser = argparse.ArgumentParser(description="Wan model quantization script (no calibration data required)")
    
    # Core model arguments
    core_group = parser.add_argument_group(title="Core Model Args")
    core_group.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(WAN_CONFIGS.keys()),
        help=f"Task type, choices: {list(WAN_CONFIGS.keys())}"
    )
    core_group.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing original model checkpoints"
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
        "--quant_save_dir",
        type=str,
        default="./quant_weights",
        help="Directory to save quantized weights (default: ./quant_weights)"
    )
    quant_group.add_argument(
        "--quant_mode",
        type=str,
        default="w8a8",
        choices=["w8a8", "w8a16"],
        help="Quantization mode: w8a8 (8bit weight + 8bit activation) or w8a16 (8bit weight + 16bit activation)"
    )
    quant_group.add_argument(
        "--is_dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic quantization (activation parameters generated dynamically)"
    )
    quant_group.add_argument(
        "--w_sym",
        action="store_true",
        default=False,
        help="Use symmetric quantization for weights "
    )
    quant_group.add_argument(
        "--act_method",
        type=int,
        default=3,
        help="Activation quantization method (default: 3, compatible with library)"
    )
    quant_group.add_argument(
        "--disable_quant_layers",
        type=str,
        nargs="*",
        default=[],
        help="List of layer names to exclude from quantization"
    )

    # Parse and validate arguments
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.quant_save_dir, exist_ok=True)
    
    return args


def load_wan_model(args):
    """Load the core transformer model from WanT2V or WanI2V.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        torch.nn.Module: Core transformer model moved to NPU
    """
    try:
        print(f"Loading Wan model for task {args.task} from {args.ckpt_dir}...")
        
        # Initialize appropriate pipeline based on task type
        if "t2v" in args.task:
            pipe = WanT2V(
                config=WAN_CONFIGS[args.task],
                checkpoint_dir=args.ckpt_dir,
                device_id=args.device_id
            )
        elif "i2v" in args.task:
            pipe = WanI2V(
                config=WAN_CONFIGS[args.task],
                checkpoint_dir=args.ckpt_dir,
                device_id=args.device_id
            )
        else:
            raise ValueError(f"Unsupported task type: {args.task}")
        
        # Extract and move core model to NPU
        core_model = pipe.model
        device = torch.device(f"npu:{args.device_id}")
        core_model = core_model.to(device) 
        
        print(f"Model loaded successfully. Core transformer moved to NPU:{args.device_id}")
        return core_model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Wan model: {str(e)}")


def main():
    try:
        # 1. Initialize NPU environment
        init_npu_env()
        
        # 2. Parse and validate arguments
        args = parse_quant_args()
        
        # 3. Load core model to quantize
        core_model = load_wan_model(args)
        
        # 4. Generate quantization configuration
        quant_config = get_wan_quant_config(
            quant_mode=args.quant_mode,
            is_dynamic=args.is_dynamic,
            w_sym=args.w_sym,
            act_method=args.act_method,
            disable_names=args.disable_quant_layers,
            dev_id=args.device_id
        )
        print(f"Quantization config generated: {args.quant_mode} mode")
        
        # 5. Perform quantization
        quantizer = WanQuantizer(model=core_model, quant_config=quant_config)
        print("Starting quantization (no calibration data required)...")
        calibrator = quantizer.quantize()
        
        # 6. Save quantized weights
        quantizer.save_quantized_weights(calibrator, save_dir=args.quant_save_dir)
        
        # 7. Final status message
        print(f"\nQuantization completed successfully!")
        print(f"Quantized weights saved to: {args.quant_save_dir}")
        
    except Exception as e:
        print(f"Quantization failed: {str(e)}", flush=True)
        exit(1)


if __name__ == "__main__":
    main()