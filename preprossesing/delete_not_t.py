import glob, os
import shutil
import json
import numpy as np
import cv2
from tqdm import tqdm
from itertools import groupby

# basic settings
files = glob.glob("/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset/*/*/*/*/*/*")
dir = glob.glob("/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset/*/*/*/*/*")
root_dir = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/dataset'
base = None
gt_db = []
crop_path = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/preproccessed_img'

# delete which is not T
for file in files:  
    if file.split("_")[-3] != "T":
        os.chmod(file, 0o777)
        os.remove(file)

# make json then crop img
for (root, dirs, files) in os.walk(root_dir):
    for file_name in tqdm(files):
        if os.path.splitext(file_name)[1] in ['.json']:
            label_path = root + '/' + file_name
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            L_Category = label['rawDataInfo']['LargeCategoryId']
            M_Category = label['rawDataInfo']['MediumCategoryId']
            S_Category = label['rawDataInfo']['SmallCategoryId']
            year = label['rawDataInfo']['yearId'],
            color = label['rawDataInfo']['colorId']
            # anno = label['learningDataInfo']['objects'][0]

            parts_list = list(map(lambda x: x['classId'], label['learningDataInfo']['objects']))
            front = True if ("P10.헤드램프" in parts_list) or ("P01.프론트범퍼" in parts_list) else False
            back = True if ("P11.리어램프" in parts_list) or ("P02.리어범퍼" in parts_list) else False

            rec = []
            file_name = file_name.replace('.json', '.jpg')
            direction = 0

            i = 0
            id_num = {}
            cls_id = L_Category + '_' + M_Category + '_' + S_Category + '_' + str(year) + '_' + color

            if id_num.get(cls_id, -1) == -1:
                id_num[cls_id] = i
                i += 1

            if front and not back:
                direction = 'front'
            elif not front and back:
                direction = 'back'
            else:
                direction = 'side'

            rec = [{
                'cls_id': cls_id,
                'L_Category': L_Category,
                'M_Category': M_Category,
                'S_Category': S_Category,
                'year': year,
                'color': color,
                'direction': direction,
                'file_name': file_name,
                'id_num': id_num
            }]

            label_path = root + '/' + file_name
            label_path = label_path.replace('.json', '.jpg')
            label_path = label_path.replace('라벨링데이터', '원천데이터')
            label_path = label_path.replace('VL', 'VS')

            try:
                anno = list(filter(lambda x: x['classId']=='P00.차량전체', label['learningDataInfo']['objects']))[0]
                img = cv2.imdecode(np.fromfile(label_path, dtype = np.uint8), cv2.IMREAD_UNCHANGED)
                cropped_img = img[int(anno['coords']['tl']['y']):int(anno['coords']['br']['y']),
                int(anno['coords']['tl']['x']):int(anno['coords']['br']['x'])]
                cv2.imwrite(crop_path + "/" + file_name, cropped_img)

                # gt_db.append(rec)
                gt_db += rec

                rec.sort(key=lambda content: content['cls_id'])
                groups = groupby(rec, lambda content: content['cls_id'])

                with open("data.json", "w") as f:
                    json.dump(gt_db, f, ensure_ascii=False)

            except:
                pass

# integrate A, B, C to A + delete B, C
for directory in dir: 
    if directory[-1] == "A":
        base = directory
    else:
        file_list = os.listdir(directory)
        for file in file_list:
            shutil.move(os.path.join(directory, file), os.path.join(base, file))

        if os.path.exists(directory):
            os.rmdir(directory)