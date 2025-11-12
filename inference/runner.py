# inference/runner.py
import os
import csv
import torch
from diffusers import StableDiffusionPipeline


def run_inference(args):
    ckpt_path = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            ckpt_path = os.path.join(args.output_dir, checkpoints[-1])
            print(f"Using latest checkpoint: {ckpt_path}")
        else:
            print("No checkpoints found, using pretrained model instead.")
    else:
        print("Output directory not found, using pretrained model instead.")

    model_path = ckpt_path if ckpt_path else args.pretrained_model_name_or_path
    """
    prompts = {
        1: "a Naruto-style portrait of a young ninja",
        2: "a samurai in the Hidden Leaf Village, anime style",
        3: "a female ninja wearing a red scarf",
    }
    """
    prompts = {
        1: "A sedan driving ahead on an urban street, front-camera view from another car",
        2: "Two cars side by side in adjacent lanes, clear daylight, front-facing dashcam view",
        3: "A bus pulling over at a bus stop, seen from dashboard camera, urban street",
        4: "Motorcycle overtaking on the left lane, captured from rear-camera view",
        5: "Truck turning right at an intersection, mid-day, dry road, ground-level view",
        6: "A pedestrian crossing at a zebra crossing in front of the car, clear sky",
        7: "Group of people walking on the sidewalk near the lane, morning light, dashcam",
        8: "Cyclist moving in the same direction as traffic, self-driving car perspective",
        9: "Child running toward the crosswalk while cars are stopped, daytime",
        10: "Pedestrian standing near the curb waiting to cross, downtown scene",
        11: "Traffic light at an intersection ahead, green light, bright daylight, dashcam",
        12: "Red traffic light with several cars waiting, front-facing vehicle camera",
        13: "Speed limit 50 sign on the right side of the road, clear weather, ground view",
        14: "Stop sign partly occluded by tree leaves in a suburban street, car perspective",
        15: "Yield sign at a T-junction, captured from a moving car’s camera",
        16: "Multi-lane road with clear white markings, straight highway segment, dashcam",
        17: "Curved urban road section with dashed lane markings, dry surface",
        18: "City street with faded road paint and potholes, daylight, forward camera",
        19: "Construction zone with cones and temporary lane deviation, front view",
        20: "Rural two-lane road merging into a main road, sunny weather, car perspective",
        21: "Urban road in heavy rain, windshield water streaks, car lights on, forward view",
        22: "Dense fog reducing visibility ahead, headlights illuminating fog, dashcam",
        23: "Snow-covered city street, vehicles moving slowly, cloudy sky, vehicle camera",
        24: "Wet asphalt reflecting traffic lights after rainfall, night scene, car perspective",
        25: "Strong sunlight causing windshield glare on a busy boulevard, morning",
        26: "Urban street at night illuminated by street lamps, few cars visible, dashcam",
        27: "Evening rush hour under orange sunset, dense traffic ahead, ground level",
        28: "Early morning low-light scene with headlights on, forward camera",
        29: "Tunnel interior with artificial lighting, car ahead visible, dashcam",
        30: "Bright midday lighting with high contrast and sharp shadows, city street",
        31: "Downtown street with tall buildings and parked cars on both sides, forward view",
        32: "Residential neighborhood with driveways and parked vehicles, car perspective",
        33: "Industrial area road near warehouses, few pedestrians, forward camera",
        34: "Suburban avenue lined with trees, medium traffic density, dashcam",
        35: "Highway through an urban interchange with overpass structures, forward view",
        36: "Car braking suddenly as a pedestrian steps onto the road, city intersection",
        37: "Vehicle changing lanes on a highway with indicator blinking, forward camera",
        38: "Bicycle crossing the street diagonally in front of the car, urban scene",
        39: "Emergency vehicle with flashing lights passing through an intersection, dashcam",
        40: "Road partially blocked by construction barrier with detour sign, ground level",
        41: "Night scene with bright billboard lights and reflections on wet road, forward view",
        42: "Vehicle partially occluded by a parked truck at a junction, sunny day",
        43: "Dog crossing the street in a residential area, car slowing down, dashcam",
        44: "Rear-camera view of vehicles approaching from behind on a city street",
        45: "Side-mirror camera capturing an overtaking car in the adjacent lane",
        46: "Wide-angle fisheye front view of a multi-lane road with moderate traffic",
        47: "Close-range front camera view of bumper-to-bumper traffic in downtown",
        48: "Pedestrian carrying shopping bags crossing at a signalized crosswalk, daytime",
        49: "Scooter rider weaving through slow traffic, city street, forward camera",
        50: "Parked delivery truck blocking part of a lane with hazard lights on, urban street"
    }
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=None,
            revision=getattr(args, "revision", None),
            variant=getattr(args, "variant", None),
        )
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Could not load '{model_path}' as a pipeline ({e}). Falling back to pretrained base.")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            safety_checker=None,
            revision=getattr(args, "revision", None),
            variant=getattr(args, "variant", None),
        )
        print(f"Loaded base model from: {args.pretrained_model_name_or_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    if getattr(args, "mixed_precision", None) == "fp16":
        pipe.torch_dtype = torch.float16
    elif getattr(args, "mixed_precision", None) == "bf16":
        pipe.torch_dtype = torch.bfloat16

    generator = None
    if getattr(args, "seed", None) is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    outdir = os.path.join(args.output_dir, "inference_results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "prompts.csv")

    steps = 20
    pipe.set_progress_bar_config(disable=True)

    print("\nGenerating images...")
    with torch.inference_mode(), open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "prompt"])  # header

        for key, prompt in prompts.items():
            img = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]
            filename = os.path.join(outdir, f"{key}.png")
            img.save(filename)
            writer.writerow([key, prompt])
            print(f"Saved: {filename}")

    print(f"\nAll images saved in: {outdir}")
    print(f"Prompts CSV saved as: {csv_path}")
