# ns-extract-mesh --load-config outputs/neus-facto-dtu65/neus-facto/XXX/config.yml --output-path meshes/neus-facto-dtu65.ply
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml
from tqdm import tqdm

from scripts.eval import ComputePSNR
from scripts.extract_mesh import ExtractMesh

save_path = Path("/mnt/hdd/3d_recon/sdfstudio/outputs")

for ckpt_file in tqdm(list(save_path.rglob("neus-gate-skip-*/**/step-000100000.ckpt"))):
    config_file = ckpt_file.parent.with_name("config.yml")
    name = config_file.parent.parent.parent.name
    out_path = save_path / f"{name}.json"

    ComputePSNR(config_file, out_path).main()
