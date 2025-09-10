from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig


def get_wan_quant_config(
    quant_mode: str,
    is_dynamic: bool,
    w_sym: bool,
    act_method: int,
    disable_names: list = None,
    dev_type: str = "npu",
    dev_id: int = None,** kwargs
) -> QuantConfig:
    """Generate quantization configuration based on command-line arguments.

    Args:
        quant_mode: Quantization mode, choices are "w8a8" (8-bit weight + 8-bit activation)
                    or "w8a16" (8-bit weight + 16-bit activation).
        is_dynamic: Whether to enable dynamic quantization (activation params computed on-the-fly).
        w_sym: Whether to use symmetric quantization for weights.
        act_method: Activation quantization method (Label-Free scenarios):
                    1 = min-max quantization,
                    2 = histogram quantization,
                    3 = auto-mixed quantization (recommended for LLM large models).
                    Note: 3 is not supported when low-bit sparse quantization is enabled.
        disable_names: List of layer names to exclude from quantization (optional).
        dev_type: Device type (default: "npu" for Ascend devices).
        dev_id: Device ID to use (e.g., NPU device number). 
        **kwargs: Additional keyword arguments for QuantConfig initialization.

    Returns:
        QuantConfig: Initialized quantization configuration object.

    Raises:
        ValueError: If quant_mode is not in ["w8a8", "w8a16"].
        ValueError: If act_method is not in [1, 2, 3].
    """
    # Validate activation quantization method
    if act_method not in [1, 2, 3]:
        raise ValueError(f"Unsupported act_method {act_method}. Supported: 1 (min-max), 2 (histogram), 3 (auto-mixed)")

    # Parse quant_mode to determine bit widths
    if quant_mode == "w8a8":
        w_bit = 8
        a_bit = 8
    elif quant_mode == "w8a16":
        w_bit = 8
        a_bit = 16  # 16-bit activation = no quantization (original precision)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}. Choose 'w8a8' or 'w8a16'")

    # Initialize quantization configuration
    quant_config = QuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        disable_names=disable_names,
        dev_type=dev_type,
        dev_id=dev_id,
        act_method=act_method,
        pr=1.0,
        w_sym=w_sym,
        mm_tensor=False,
        is_dynamic=is_dynamic,** kwargs
    )
    
    return quant_config