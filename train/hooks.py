# train/hooks.py
import os
import accelerate
from diffusers import UNet2DConditionModel
from diffusers.training_utils import EMAModel
from accelerate.logging import get_logger
from packaging import version

logger = get_logger(__name__, log_level="INFO")

def register_accelerate_hooks(accelerator, args, ema_unet):
    """
    Registers custom save/load hooks for accelerator.
    These are taken verbatim from the original training code.
    """

    # Hooks are supported only on accelerate >= 0.16.0
    if version.parse(accelerate.__version__) < version.parse("0.16.0"):
        return

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=args.foreach_ema
            )
            ema_unet.load_state_dict(load_model.state_dict())
            if args.offload_ema:
                ema_unet.pin_memory()
            else:
                ema_unet.to(accelerator.device)
            del load_model

        for _ in range(len(models)):
            model = models.pop()
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
