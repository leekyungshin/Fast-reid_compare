import numpy as np
import cv2
import settings
import pandas as pd
import torch
import onnxruntime
from reid_onnx_helper import ReidHelper


df = pd.read_csv('/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/annotation/data.csv')
df = df.groupby('cls_id').apply(lambda group:group if len(group) > 1 else None).reset_index(drop=True)

front_query = df[(df.direction == 'front')]
back_query = df[(df.direction == 'back')]
side_query = df[(df.direction == 'side')]

front_queries = front_query.groupby('cls_id').first()
back_queries = back_query.groupby('cls_id').first()
side_queries = side_query.groupby('cls_id').first()

front_gallaries = front_query.groupby('cls_id').apply(lambda group:group.iloc[1:]).reset_index(drop=True)
back_gallaries = back_query.groupby('cls_id').apply(lambda group:group.iloc[1:]).reset_index(drop=True)
side_gallaries = side_query.groupby('cls_id').apply(lambda group:group.iloc[1:]).reset_index(drop=True)

def cosine_sim(query_img_feat, gallary_img_feat):
    cos_sim = np.squeeze(query_img_feat @ gallary_img_feat.T)
    return cos_sim

if __name__ == "__main__":

    helper = ReidHelper(settings.ReID)
    root_dir = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/preproccessed_img/'
    gallary_similarity = []

    rank1, rank5, rank10, i = 0, 0, 0, 0

    # Front rank-k result
    for cid, query in front_queries.iterrows():
        img_array = np.fromfile(root_dir + query['file_name'], np.uint8)
        image_np = cv2.imdecode(img_array,  cv2.IMREAD_COLOR)
        query_img = helper.preprocess(image_np)
        query_img_feat = helper.infer(image_np)
        
        for _, gallary in front_gallaries.sample(1).iterrows():
            img_array2 = np.fromfile(root_dir + gallary['file_name'], np.uint8)
            image_np2 = cv2.imdecode(img_array2,  cv2.IMREAD_COLOR)
            gallary_img = helper.preprocess(image_np2)
            gallary_img_feat = helper.infer(image_np2)
            
            similarity = cosine_sim(query_img_feat, gallary_img_feat)
            gallary_similarity.append(tuple([similarity, gallary['cls_id']]))

        gallary_similarity.sort(reverse=True)

        rank1 += 1 if list(filter(lambda x: x[1] == cid, gallary_similarity[:1])) else 0
        rank5 += 1 if list(filter(lambda x: x[1] == cid, gallary_similarity[:5])) else 0
        rank10 += 1 if list(filter(lambda x: x[1] == cid, gallary_similarity[:10])) else 0
        i += 1

        
    print(f"rank1: {rank1}/{i}>{rank1/i}%, rank5: {rank5}/{i}>{rank5/i}%, rank10: {rank10}/{i}>{rank10/i}%")