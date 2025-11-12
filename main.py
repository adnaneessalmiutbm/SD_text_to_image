import os
from core.cli import parse_args
from core.engine import setup_environment
from train.trainer import train
from huggingface_hub import create_repo
from inference.runner import run_inference

def main():
    args = parse_args()
    if args.inference:
        run_inference(args)
        return
    accelerator = setup_environment(args)

    # 3. Create or reuse repository for push_to_hub (faithful to original)
    repo_id = None
    if accelerator.is_main_process and args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or os.path.basename(args.output_dir),
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    # 4. Launch training
    train(args, accelerator, repo_id)


if __name__ == "__main__":
    main()
