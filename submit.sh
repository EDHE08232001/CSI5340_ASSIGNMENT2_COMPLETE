#!/bin/bash
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/out/job-%j.out
#SBATCH --error=/home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/out/job-%j.err

mkdir -p /home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/out
cd /home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main
source /home/uottawa.o.univ/ehe058/CSI5340_Assignment2-main/myenv/local/bin/activate
python main.py

# Cleanup (optional)
find . -type f -name '._*' -delete 2>/dev/null                        # Remove macOS metadata files
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null      # Remove Python cache directories
find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null  # Remove Jupyter checkpoint directories