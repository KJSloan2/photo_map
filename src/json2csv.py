import json
import csv
'''This script converts a given JSON file to CSV'''
######################################################################################
json_imageStats = json.load(open("%s%s" % (r"output/","imageStats.json")))
######################################################################################
with open("%s%s" % (r"output/","imageStats.csv"),'w',newline='', encoding='utf-8') as write_dataOut:
	writer_dataOut = csv.writer(write_dataOut)
	writer_dataOut.writerow([
		"IMAGE_ID","IMAGE_NAME","DATE",
		"HOUR","MIN","LAT","LON",
		"DETECTION_ID","CLASS_ID","CONF"
		])
	
	for imageStatsKey, imageStats_ in json_imageStats.items():
		file_name = imageStats_["file_name"]
		gps_coords = imageStats_["gps_coords"]

		if len(gps_coords) ==2:
			date_year = str(imageStats_["date"]["Y"])
			date_month = str(imageStats_["date"]["m"])
			date_day = str(imageStats_["date"]["d"])
			date_formated = date_year+"/"+date_month+"/"+date_day
			
			time_hour = str(imageStats_["time"]["H"])
			time_min = str(imageStats_["time"]["M"])
			time_formated = time_hour+":"+time_min
			gps_bearing = imageStats_["gps_bearing"]
			for detectionKey,detection_ in imageStats_["detections"].items():
				writer_dataOut.writerow([
					imageStatsKey,
					file_name,
					date_formated,
					time_hour,
					time_min,
					gps_coords[0],
					gps_coords[1],
					detectionKey,
					detection_["class_id"],
					detection_["conf"],
				])
######################################################################################
write_dataOut.close()
print("DONE")

