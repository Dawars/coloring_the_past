from math import ceil

import pandas as pd

filelist = pd.read_csv("/mnt/hdd/3d_recon/neural_recon_w/heritage-recon/brandenburg_gate/brandenburg_gate.tsv",
                       sep="\t")

test = filelist[filelist["split"] == "test"]
train = filelist[filelist["split"] == "train"]
num_train = len(train)

for i in reversed(range(10)):
    n = ceil(num_train / 2 ** i)
    print(n)

    split = train.iloc[:n]
    df = pd.concat([test, split])
    filelist = df.to_csv(f"/mnt/hdd/3d_recon/neural_recon_w/heritage-recon/brandenburg_gate/brandenburg_gate_{i}.tsv",
                         sep="\t")
