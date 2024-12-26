CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py
CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py --split non-overlap
CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py --contrastive 0
CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py --margin 0.0
CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py --margin 0.2
CUDA_VISIBLE_DEVICES="1,2" mpirun -np 2 python -m train.py --margin 0.6


# CUDA_VISIBLE_DEVICES=2 python train.py

# CUDA_VISIBLE_DEVICES=1 python train.py --split non-overlap

# CUDA_VISIBLE_DEVICES=2 python train.py --contrastive 0

# CUDA_VISIBLE_DEVICES=1 python train.py --margin 0.0
# CUDA_VISIBLE_DEVICES=2 python train.py --margin 0.2
# CUDA_VISIBLE_DEVICES=1 python train.py --margin 0.6
