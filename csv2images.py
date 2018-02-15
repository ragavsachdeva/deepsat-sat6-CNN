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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv(filepath,header=None)
data_rows, data_cols = data.shape

for i in range(data_rows):
	
	row_i = data.iloc[i, 0:].as_matrix()
	im = np.reshape(row_i, (28,28,4))

	image_name = str(i)+'.png'
	output_path = os.path.join(output_dir,image_name)
	io.use_plugin('freeimage')
	io.imsave(output_path, im)
	
	if i%100==0:
		print(i)
