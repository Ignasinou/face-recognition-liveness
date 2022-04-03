import argparse
import csv
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

from modules import FaceDetection, IdentityVerification

# Create the parser
parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
parser.add_argument('images',
                    metavar='path',
                    type=str,
                    help='the path to the images folder')
parser.add_argument('checkpoint',
                    metavar='path',
                    type=str,
                    help='the path to the resnet vggface2 onnx checkpoint')
parser.add_argument('csv',
                    metavar='path',
                    type=str,
                    help='the path to the output csv file')
args = parser.parse_args()

input_path = args.images
checkpoint_path = args.checkpoint
csv_path = args.csv

if not os.path.isdir(input_path):
    print('The path specified is not a directory')
    sys.exit()

images_list = glob(os.path.join(input_path, '*.jpg')) \
    + glob(os.path.join(input_path, '*.png')) \
    + glob(os.path.join(input_path, '*.jpeg'))

if not len(images_list):
    print('There is not any images in the specified path')
    sys.exit()

faceDetector = FaceDetection()
resnet = onnxruntime.InferenceSession(
    checkpoint_path, providers=['CPUExecutionProvider'])

f = open(csv_path, 'w')
writer = csv.writer(f)
for image_path in tqdm(images_list):
    image = cv2.imread(image_path)
    faces, boxes = faceDetector(image)
    if not len(faces):
        continue

    face_arr = faces[0]
    face_arr = np.moveaxis(face_arr, -1, 0)
    input_arr = np.expand_dims((face_arr - 127.5) / 128.0, 0)
    embeddings = resnet.run(["output"],
                            {"input": input_arr.astype(np.float32)})[0]
    writer.writerow(embeddings.flatten().tolist())
f.close()
