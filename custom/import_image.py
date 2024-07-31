# 2024 custom code
# python file for importing pcap file and convert to image
# https://ouster.com/blog/object-detection-and-tracking-using-deep-learning-and-ouster-python-sdk/

from ouster import client
from ouster import pcap
import cv2
import numpy as np
from contextlib import closing
import os

# metadata_path = 'Ouster-YOLOv5-sample.json' # location of json file (sample)
# pcap_path = 'Ouster-YOLOv5-sample.pcap' # location of pcap file (sample)
# img_path = 'extract_image_sample/' # location of extracted images (sample)

metadata_path = 'data-collected/current/json' # location of JSON files (real data)
pcap_path = 'data-collected/current/pcap' # location of PCAP files (real data)
img_path = 'extract_image_data/' # location of extracted images (real data)

# THE FOLLOWING CODE IS FOR EXTRACTING IMAGES FROM A SINGLE PCAP FILE
#
# with open(metadata_path, 'r') as f:
#     metadata = client.SensorInfo(f.read())

# pcap_file = pcap.Pcap(pcap_path, metadata)
# source = pcap.Pcap(pcap_path, metadata)

# counter = 0
# with closing(client.Scans(source)) as scans:
#     for scan in scans:
#         counter += 1
#         ref_field = scan.field(client.ChanField.REFLECTIVITY)
#         ref_val = client.destagger(source.metadata, ref_field) #the destagger function is to adjust for the pixel staggering that is inherent to Ouster lidar sensor raw data
#         ref_img = ref_val.astype(np.uint8) #convert to numpy array for image processing later

#         filename = 'extract'+str(counter)+'.jpg'
#         cv2.imwrite(img_path + filename, ref_img)

# metadata_files = os.listdir(metadata_path) # Get a list of JSON files in the folder

# ----------------------------
# THE FOLLOWING CODE IS FOR EXTRACTING IMAGES FROM MULTIPLE PCAP & JSON FILES
# The JSON files and PCAP files must have the same name
# 
# This updated code will process each JSON file with its corresponding PCAP file, 
# and save the extracted images using unique names that include the metadata file name.
# ----------------------------
metadata_files = os.listdir(metadata_path) # Get a list of JSON files in the folder

for metadata_file in metadata_files:
    pcap_file = metadata_file.replace('.json', '.pcap')
    metadata_file_path = os.path.join(metadata_path, metadata_file)
    pcap_file_path = os.path.join(pcap_path, pcap_file)

    with open(metadata_file_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    # print(pcap_file_path) # debug pcap
    # print(metadata_file_path) # debug json
    source = pcap.Pcap(pcap_file_path, metadata)

    counter = 0
    with closing(client.Scans(source)) as scans:
        for scan in scans:
            counter += 1
            ref_field = scan.field(client.ChanField.REFLECTIVITY)
            ref_val = client.destagger(source.metadata, ref_field)
            ref_img = ref_val.astype(np.uint8)

            filename = 'extract_' + metadata_file[:-5] + '_' + str(counter) + '.jpg'
            cv2.imwrite(os.path.join(img_path, filename), ref_img)