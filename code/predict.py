#python  predict.py  --save_weights_path=weights/ex1  --epoch_number=0  --test_images="data/dataset1/images_prepped_test/"  --output_path="data/predictions/"  --n_classes=10  --input_height=320  --input_width=480 --model_name="segnet"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import LoadBatches
from Models import Segnet, VGGSegnet, VGGUnet, FCN8, FCN32
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

#modelFns = { 'segnet':Segnet.Segnet, 'vgg_segnet':VGGSegnet.VGGSegnet , 'vgg_unet':VGGUnet.VGGUnet , 'vgg_unet2':VGGUnet.VGGUnet2 , 'fcn8':FCN8.FCN8 , 'fcn32':FCN32.FCN32 }
modelFns = { 'segnet':Segnet.Segnet }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

for imgName in images[0:1]:
	outName = imgName.replace( images_path ,  args.output_path )
	X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
	pr = m.predict( np.array([X]) )[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	cv2.imwrite(  outName , seg_img )

