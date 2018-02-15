from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform, exposure, img_as_uint, img_as_float
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("DataPath", help="Please provide the path of csv file.")
parser.add_argument('-o',"--OutputPath", help="Please provide the path of the output directory.")
args = parser.parse_args()

output_dir = ""

filepath = os.path.join(args.DataPath,)
if args.OutputPath:
	output_dir = args.OutputPath

labels = pd.read_csv(filepath,header=None)
labels.columns = ["building","barren_land","trees","grassland","road","water"]

labels_rows, labels_cols = labels.shape

output_path = os.path.join(output_dir,"annotations.csv")

labels.to_csv(output_path)