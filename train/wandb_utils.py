from diffusers.utils import is_wandb_available

if is_wandb_available():
    import wandb
else:
    wandb = None


def log_wandb_validation_images(tracker, images, prompts):
    if wandb is None:
        return
    tracker.log(
        {
            "validation": [
                wandb.Image(image, caption=f"{i}: {prompts[i]}")
                for i, image in enumerate(images)
            ]
        }
    )
