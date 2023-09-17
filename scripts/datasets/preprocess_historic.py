"""Script to preprocess historic images before feature matching.
 Downscales and converts to grayscale
 """
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageOps
from PIL.Image import Resampling

color_list = [
    "C_ROB19610907002.jpg",
    "C_ROB19630501025.jpg",
    "C_ROB19630501026.jpg",
    "C_SZI19610505015.jpg",
    "C__VA19590709002.jpg",
    "C__ZE19590822006.jpg",
    "fortepan_112161.jpg",
    "fortepan_115002.jpg",
    "fortepan_153315.jpg",
    "fortepan_183722.jpg",
    "fortepan_187023.jpg",
    "fortepan_210900.jpg",
    "fortepan_250610.jpg",
    "fortepan_251232.jpg",
    "fortepan_251236.jpg",
    # hotel international
    "2456.jpeg",
    # observatory
    "109.jpeg",
    "1226.jpeg",
    "1561.jpeg",
    "2984.jpeg",
    "3000.jpeg",
    "3008.jpeg",
    "3015.jpeg",
    # st michael (jena)
    "A6P_1169.jpg",
    "A6P_1291.jpg" "A6P_1307.jpg" "A6P_1307.jpg" "fortepan_171225.jpg",
    # dresden
    "df_bs_0012202_postkarte.jpg",
    "df_bs_0030587_postkarte.jpg",
    "df_bs_0030619_postkarte.jpg",
    "df_ge_0000045.jpg",
    "df_ge_0025328.jpg",
    "df_ge_0025329.jpg",
    "df_ge_0025332.jpg",
    "df_hauptkatalog_0087560.jpg",
    "df_hauptkatalog_0664613.jpg",
    "df_pos-2020-a_0000216.jpg",
    "tu_kg_0248384.jpg",
    "tu_kg_0248385.jpg",
    "tu_kg_0260947.jpg",
    "tu_kg_0260948.jpg",
    "tu_kg_0260949.jpg",
    "tu_kg_0260950.jpg",
    "tu_kg_0260951.jpg",
    "tu_kg_0260953.jpg",
    "tu_kg_0260954.jpg",
    "tu_kg_0260955.jpg",
    "tu_kg_0260956.jpg",
    "tu_kg_0260967.jpg",
    "tu_kg_0260968.jpg",
    "tu_kg_0260969.jpg",
    "tu_kg_0260974.jpg",
    "tu_kg_0260975.jpg",
    "tu_kg_0260976.jpg",
    "tu_kg_0266461.jpg",
    "tu_kg_0266462.jpg",
    # stockholm
    "ARKM.1990-106-006.jpg",
    "ARKM.1990-106-015.jpg",
    "ARKM.1990-106-017.jpg",
    "ARKM.1990-106-018.jpg",
    "ARKM.1990-106-019.jpg",
    "ARKM.1990-106-020.jpg",
    "ARKM.1990-106-022.jpg",
    "ARKM.1990-106-023.jpg",
    "ARKM.1990-106-050.jpg",
    "ARKM.1990-106-111.jpg",
    "ARKM.1990-106-163.jpg",
]

parser = ArgumentParser()
parser.add_argument("--img-dir", type=Path, required=True, help="source image folder")
parser.add_argument("--out-dir", type=Path, required=True, help="output folder for processed images")
args = parser.parse_args()


def get_files(root: Path, extensions):
    all_files = []
    for ext in extensions:
        all_files.extend(Path(root).glob(ext))
    return all_files


image_dir = args.img_dir
out_dir = args.out_dir
out_dir.mkdir(exist_ok=True)
for url in get_files(Path(image_dir), ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")):
    image = Image.open(url)
    image = ImageOps.exif_transpose(image)

    if url.name not in color_list:
        image = image.convert("L")

    image.thumbnail((1600, 1600), resample=Resampling.LANCZOS)

    image.save(out_dir / url.with_suffix(".jpg").name, quality=100)
