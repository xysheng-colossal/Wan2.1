import os
import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
from types import SimpleNamespace


class WanQuantizer:
    """Quantizer for Wan2.1 model (no calibration data required).

    This quantizer leverages weight distribution characteristics of the model
    to compute quantization parameters statically, eliminating the need for
    additional calibration data. Suitable for scenarios with stable weight distribution.
    """
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config
        self._prepare_model_for_quant()  # Prepare model for quantization compatibility

    def _prepare_model_for_quant(self):
        """Supplement model attributes required by msmodelslim quantization interface.

        Adds missing attributes (config, dtype, etc.) to ensure compatibility with
        the quantization library's expectations.
        """
        # Add config namespace if missing
        if not hasattr(self.model, "config"):
            self.model.config = SimpleNamespace()
        
        # Supplement metadata required for quantization
        if not hasattr(self.model.config, "torch_dtype"):
            self.model.config.torch_dtype = torch.bfloat16
        if not hasattr(self.model, "dtype"):
            self.model.dtype = torch.bfloat16
        if not hasattr(self.model.config, "model_type"):
            self.model.config.model_type = "wan21"  # Identify model type for the library

    def quantize(self) -> Calibrator:
        """Perform quantization using msmodelslim's Calibrator.

        Disables automatic layer fallback (disable_level='L0') to ensure all specified
        layers are quantized (except those in disable_names).

        Returns:
            Calibrator: Calibrator object containing quantized parameters.
        """
        # Initialize calibrator with strict quantization (no layer fallback)
        calibrator = Calibrator(
            model=self.model,
            cfg=self.quant_config,
            disable_level="L0"  # Enforce quantization for all layers (except disabled ones)
        )
        
        # Run quantization (no calibration data required for this model)
        calibrator.run()
        return calibrator

    def save_quantized_weights(self, calibrator: Calibrator, save_dir: str):
        """Save quantized weights and configuration files.

        Args:
            calibrator: Calibrator object containing quantized parameters.
            save_dir: Directory to save output files.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            # Save as safe_tensor (compatible with Hugging Face ecosystem, avoids pickle risks)
            calibrator.save(save_dir, save_type=["safe_tensor"])
            print(f"Quantized weights saved to: {save_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to save quantized weights: {str(e)}")