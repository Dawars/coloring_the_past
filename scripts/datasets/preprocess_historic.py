"""Script to preprocess historic images before feature matching.
 Downscales and converts to grayscale
 """
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageOps

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
]

parser = ArgumentParser()
parser.add_argument("--img-dir", type=Path, required=True, help="source image folder")
parser.add_argument("--out-dir", type=Path, required=True, help="output folder for processed images")
args = parser.parse_args()

image_dir = args.img_dir
out_dir = args.out_dir
out_dir.mkdir(exist_ok=True)
for url in Path(image_dir).glob("*.jpg"):
    image = Image.open(url)
    image = ImageOps.exif_transpose(image)

    if url.name not in color_list:
        image = image.convert("L")

    image.thumbnail((2000, 2000))

    image.save(out_dir / url.name, quality=100)
