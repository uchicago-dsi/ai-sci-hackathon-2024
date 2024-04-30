#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00


module load python/miniforge-24.1.2 # python 3.10

echo "output of the visible GPU environment"
nvidia-smi

# Use hackathon enviroment
source /project/dfreedman/hackathon/hackathon-env/bin/activate

echo PyTorch
python example_torch.py
echo Tensorflow
python example_tf.py

# Use a different environment for JAX
source /project/dfreedman/hackathon/hackathon-env-jax/bin/activate
echo JAX
python example_jax.py
