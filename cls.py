import argparse
import os
from duckduckgo_search import DDGS
from fastcore.all import *
from time import sleep
from fastdownload import download_url
from fastai.vision.all import *

parser = argparse.ArgumentParser()
parser.add_argument("--first", help="First image category to search",
                    default="jetfighter")
parser.add_argument("--second", help="Second image category to search",
                    default="rocket")
parser.add_argument("--file", help="picture to classify",
                    default="test/sr71-burners-below.jpg")

args = vars(parser.parse_args())

path = Path("data/" + args['first'] + "_or_" + args['second'])

dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
        ).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

classification,_,probs = learn.predict(PILImage.create(args['file']))
print(f"This is a: {classification}.")
print(f"Probability it's a {classification}: {probs[0]:.4f}")
