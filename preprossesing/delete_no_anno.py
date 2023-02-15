import glob, os
import shutil
import json
import numpy as np
import cv2
from tqdm import tqdm
from itertools import groupby

# basic settings
# files = glob.glob("/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset/*/*/*/*/*/*")
# dir = glob.glob("/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset/*/*/*/*/*")
json_path = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/annotation/data.json'
root_dir = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset'
base = None
gt_db = []
crop_path = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/preproccessed_img'

with open(json_path, "r") as json_file:
    json_data = json.load(json_file)
    
print(json_data)

# # make json then crop img
# for (root, dirs, files) in os.walk(root_dir):
#     for file_name in tqdm(files):
#         if os.path.splitext(file_name)[1] in ['.json']:
#             label_path = root + '/' + file_name
#             with open(label_path, 'r') as f:
#                 label = json.load(f)















#             label_path = root + '/' + file_name
#             label_path = label_path.replace('.json', '.jpg')
#             label_path = label_path.replace('라벨링데이터', '원천데이터')
#             label_path = label_path.replace('VL', 'VS')

#             try:
#                 anno = list(filter(lambda x: x['classId']=='P00.차량전체', label['learningDataInfo']['objects']))[0]
#                 img = cv2.imdecode(np.fromfile(label_path, dtype = np.uint8), cv2.IMREAD_UNCHANGED)
#                 cropped_img = img[int(anno['coords']['tl']['y']):int(anno['coords']['br']['y']),
#                 int(anno['coords']['tl']['x']):int(anno['coords']['br']['x'])]
#                 cv2.imwrite(crop_path + "/" + file_name, cropped_img)
#             except:
#                 pass