#import datetime for handeling date and time data
from datetime import datetime
#import os modules for handeling file retreival and writing tasks
import os
from os import listdir
from os.path import isfile, join
#Import json for handeling data read/write tasks
import json
#Import regex for handeling text parsing
import regex as re
#Import numpy for calculating eye movement and sensor metrics
import numpy as np 
#Import CV2 for handleing image analysis tasks
import cv2
#Import PIL for handeling image formating and read/write tasks
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener
#Import Ultralytics and Supervision for object detection (with YOLO)
from ultralytics import YOLO
import supervision as sv
######################################################################################
######################################################################################
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.keras.preprocessing import image as tf_image

extractMetaData = True
extractFeatures = False
workspacePaths_ = {
	"00_resources":r"resource\\",
	"testImages":r"PATH TO IMAGES",
	"output":r"output\\",
	"output_test":r"output\test\\",
	"output_test_resized":r"output\test\resized\\",
	"output_features":r"output\features\\"
}
######################################################################################
classificationsJson_ = json.load(open("%s%s" % (workspacePaths_["00_resources"],"yolov8_classifications.json")))
classRef_ = {}
for i in range(len(classificationsJson_["classifications"])):
	obj_ = classificationsJson_["classifications"][i]
	classRef_[obj_["label"]]={"color":obj_["color"]}
	
######################################################################################
models_ = ["yolov8l.pt","yolov8x.pt","yolov8n.pt","yolov8l-cls.pt"]
model = YOLO(models_[0])
######################################################################################
files = [f for f in listdir(workspacePaths_["testImages"]) if isfile(join(workspacePaths_["testImages"], f))]

def decimal_coords(coords, ref):
	decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
	if ref == "S" or ref == "W":
		decimal_degrees = -decimal_degrees
	return decimal_degrees

def mins_to_coords(degrees, minutes):
    decimal_value = int(degrees) + (float(minutes) / 60)
    return decimal_value
######################################################################################
imageStats_ = {}
counter = 0
accepted_extensions = ["jpg","jpeg"]
#for f in range(0,len(files),1):
for f in range(0,5,1):
	file = files[f]
	print(file)
	fileName = str(workspacePaths_["testImages"])+"%s" % (file)
	split_fileName = file.split(".")
	image_name = split_fileName[0]
	imageStats_[counter] = {
		"file_name":image_name,
		"image_shape":None,
		"date":{},
		"time":{},
		"focal_length":None,
		"depth_near":None,
		"depth_far":None,
		"gps_coords":[],
		"gps_image_direction_ref":None,
		"gps_image_direction":None,
		"gps_altitude_ref":None,
		"gps_altitude":None,
		"detections":{}
	}

	fExtension = str(split_fileName[-1]).lower()
	image = None
	path_image = "%s%s" % (workspacePaths_["testImages"],file)
	if fExtension in ["jpg","jpeg"]:
		image = Image.open(path_image)
	if image != None:
		try:
			exif_data = image._getexif()
			if extractMetaData == True:
				for tag_id, value in exif_data.items():
					tag_name = TAGS.get(tag_id, tag_id)
					if tag_name == 'GPSInfo':
						coordKeys_ = [[1,2],[3,4]]

						for key in coordKeys_:
							#sub_tag_name = TAGS.get(key, key)
							gps_info = value[key[1]]
							#coords = decimal_coords(gps_info, value[key-1])
							#coords = mins_to_coords(gps_info[1], gps_info[2])
							coord = decimal_coords(gps_info, value[key[0]])
							imageStats_[counter]["gps_coords"].append(coord)
						imageStats_[counter]["gps_image_direction_ref"] = str(value[16])
						imageStats_[counter]["gps_image_direction"] = float(value[17])
						print(value[16],value[17])

					elif tag_name == "DateTime":
						image_datetime = str(value).split(" ")
						parse_date = image_datetime[0].split(":")
						imageStats_[counter]["date"]["Y"] = int(parse_date[0])
						imageStats_[counter]["date"]["m"] = int(parse_date[1])
						imageStats_[counter]["date"]["d"] = int(parse_date[2])

						parse_time = image_datetime[1].split(":")
						imageStats_[counter]["time"]["H"] = int(parse_time[0])
						imageStats_[counter]["time"]["M"] = int(parse_time[1])
						imageStats_[counter]["time"]["S"] = int(parse_time[2])

					elif tag_name == "FocalLength":
						imageStats_[counter]["focal_length"] = float(value)
					elif tag_name == "DepthNear":
						imageStats_[counter]["depth_near"] = float(value)
					elif tag_name == "DepthFar":
						imageStats_[counter]["depth_far"] = float(value)

		except Exception as e:
			print(e)
			pass
		
		try:
			max_dimension = 640
			width, height = image.size
			if width > height:
				new_width = max_dimension
				new_height = int(height * (max_dimension / width))
			else:
				new_height = max_dimension
				new_width = int(width * (max_dimension / height))
			imageStats_[counter]["image_shape"] = [new_width,new_height]
			imageResized = image.resize((new_width,new_height))
			imageResized = ImageOps.exif_transpose(imageResized)
			path_imageResized = "%s%s%s%s%s" % (workspacePaths_["output_test_resized"],image_name,"_resized_",max_dimension,".jpg")
			imageResized.save(path_imageResized)

			'''if extractFeatures == True:
				model_keras = VGG16(weights='imagenet', include_top=False)
				img = tf_image.load_img(path_imageResized, target_size=(224, 224))
				x = tf_image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				features = model_keras.predict(x)
				np.save("%s%s%s" % (workspacePaths_["output_features"],image_name,"_extractedFeatures.npy"), features)'''

			imageRseized = cv2.imread(path_imageResized)
			results = model.predict(path_imageResized, save=True)
			result = results[0]
			box = result.boxes[0]
			detectionId = 0
			for box in result.boxes:
				class_id = result.names[box.cls[0].item()]
				cords = box.xyxy[0].tolist()
				cords = [round(x) for x in cords]
				conf = round(box.conf[0].item(), 2)

				imageStats_[counter]["detections"][detectionId] = {
					"class_id":class_id,
					"coords":cords,
					"conf":conf
				}
				detectionId+=1
		except Exception as e:
			print(e)
			pass

		counter+=1
######################################################################################
with open(str("%s%s" % (workspacePaths_["02_output"],"imageStats.json")), "w", encoding='utf-8') as json_output:
	json_output.write(json.dumps(imageStats_, indent=4, ensure_ascii=False))
######################################################################################
print("DONE")
