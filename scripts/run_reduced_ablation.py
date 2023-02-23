"""
This script starts multiple slurm jobs
The task is setting skip_every_for_train_split to 2,4,8,...,64
"""

import subprocess

for i in range(7):
    skip = 2 ** i
    output = subprocess.check_output([
        "sbatch",
        "--partition",
        "gpu",
        "--mem",
        "100GB",
        "--gres=gpu:1",
        "ns-train",
        "neus-facto-bigmlp",
        "--pipeline.model.sdf-field.use-grid-feature",
        "False",
        "--pipeline.model.sdf-field.use-appearance-embedding",
        "True",
        "--pipeline.model.sdf-field.geometric-init",
        "True",
        "--pipeline.model.sdf-field.inside-outside",
        "False",
        "--pipeline.model.sdf-field.bias",
        "0.3",
        "--pipeline.model.sdf-field.beta-init",
        "0.3",
        "--pipeline.model.eikonal-loss-mult",
        "0.0001",
        "--pipeline.model.num-samples-outside",
        "4",
        "--pipeline.model.background-model",
        "grid",
        "--steps-per-eval-image",
        "5000",
        "--vis",
        "wandb",
        "--experiment-name",
        f"neus-gate-skip-{i}",
        "--machine.num-gpus",
        "1",
        "heritage-data",
        "--data",
        "data/heritage/brandenburg_gate",
        "--skip_every_for_train_split",
        i])

    print(output)
