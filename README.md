# Fine-tuning Stable Diffusion 

Ce projet propose une implémentation entièrement modulaire du fine-tuning texte-vers-image pour Stable Diffusion, en utilisant les bibliothèques Hugging Face Diffusers et Accelerate.  


---

## Structure du projet

```bash
project/
│
├── core/
│ ├── cli.py # Analyse des arguments (entraînement et inférence)
│ └── engine.py # Configuration de l’environnement (Accelerate, logs, seed)
│
├── data/
│ └── dataset.py # Chargement, prétraitement et DataLoader du dataset
│
├── models/
│ └── models.py # Chargement des modèles : VAE, UNet, Text Encoder, Scheduler
│
├── train/
│ ├── trainer.py # Boucle principale d’entraînement (pertes, checkpoints, EMA)
│ ├── validation.py # Validation via StableDiffusionPipeline
│ ├── hooks.py # Hooks personnalisés pour Accelerate (sauvegarde/chargement)
│ └── model_saving.py # Génération du model card 
│
├── inference/
│ └── runner.py # Inférence autonome à partir des checkpoints ou du modèle de base
│
├── outputs/ # Checkpoints, journaux et images générées
│
└── main.py # Point d’entrée pour l’entraînement ou l’inférence

```


---

## Utilisation

### Entraînement

```bash
python main.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --dataset_name "lambdalabs/naruto-blip-captions" --output_dir "outputs" --train_batch_size 8 --num_train_epochs 5 --mixed_precision fp16 --gradient_checkpointing --allow_tf32 --seed 42 --validation_prompts "Two people standing"
```

### Inférence 

Vous pouvez générer des images à partir du dernier checkpoint disponible ou directement depuis un modèle pré-entraîné :

```bash
python main.py --inference --output_dir "./outputs/sd-model-finetuned"
```


## Prérequis

Installation des dépendances principales :

```bash
pip install torch torchvision diffusers transformers accelerate datasets huggingface_hub tqdm
```

## Dépendances optionnelles :


```bash
pip install wandb bitsandbytes
```

