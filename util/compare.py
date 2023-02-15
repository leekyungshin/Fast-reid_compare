import numpy as np
import cv2
import pandas as pd
import torch
import onnxruntime
from tqdm import tqdm

# infer, preprocess, normalize는 reid_onnx_helper.py의 것을 사용
class compare:
    size = (256, 256)
    def __init__(self):
        # Test models by changing the path

        MODEL_PATH = "/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/models/veriwild.onnx"   
        # MODEL_PATH = "/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/models/vehicleid.onnx"
        # MODEL_PATH = "/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/models/veri.onnx"
        # self.size = (256, 256)
        
        self.sess = onnxruntime.InferenceSession(MODEL_PATH, providers = ['CUDAExecutionProvider'])

        df = pd.read_csv('/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/annotation/data.csv')

        self.df = {}
        
        self.df['front'] = df[(df.direction == "front")]
        self.df['back'] = df[(df.direction == "back")]
        self.df['side'] = df[(df.direction == "side")]


    def infer(self, image_np):
        input_name = self.sess.get_inputs()[0].name

        image = self.preprocess(image_np)

        feat = self.sess.run(None, {input_name: image})[0]
        feat = self.normalize(feat, axis=1)
        
        return feat


    def preprocess(self, image_np):
        # the model expects RGB inputs
        original_image = image_np[:, :, ::-1]

        # Apply pre-processing to image.
        resize_width = self.size[0]
        resize_height = self.size[1]
        
        img = cv2.resize(original_image, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        
        return img


    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)


    def cosine_sim(self, query_img_feat, gallary_img_feat):
        cos_similarity = np.squeeze(query_img_feat @ gallary_img_feat.T)
        return cos_similarity


    def evaluate(self, direction):
        root_dir = '/home/leekyungshin/Desktop/Nextlab/reid_onnx_deploy/preproccessed_img/'
        gallary_similarity = []
        rank1, rank5, rank10 = 0, 0, 0
        i = 0

        self.df[direction] = self.df[direction].groupby("cls_id").apply(lambda group:group if len(group) >= 10 else None).reset_index(drop=True)

        queries = self.df[direction].groupby("cls_id").first()
        gallaries = self.df[direction].groupby("cls_id").apply(lambda group:group.iloc[1:]).reset_index(drop=True)

        pbar = tqdm(queries.iterrows(), total=queries.shape[0]) # 보류
        for cid, query in pbar:
            img_path = root_dir + query['file_name']
            image_np = cv2.imread(img_path)
            # query_img = self.preprocess(image_np)
            query_img_feat = self.infer(image_np)

            for _, gallary in gallaries.groupby("cls_id").sample(1).iterrows():
                img_path2 = root_dir + gallary["file_name"]
                image_np2 = cv2.imread(img_path2)
                # gallary_img = self.preprocess(image_np2)
                gallary_img_feat = self.infer(image_np2)

                similarity = self.cosine_sim(query_img_feat, gallary_img_feat)
                gallary_similarity.append(tuple([similarity, gallary["cls_id"]]))

            gallary_similarity.sort(reverse=True)

            rank1 += 1 if list(filter(lambda x: x[1]== cid, gallary_similarity[:1])) else 0
            rank5 += 1 if list(filter(lambda x: x[1] == cid, gallary_similarity[:5])) else 0
            rank10 += 1 if list(filter(lambda x: x[1]== cid, gallary_similarity[:10])) else 0
            i += 1
            
        print(f"{direction} comparation result -- rank1: {rank1}/{i}>{rank1/i}%, rank5: {rank5}/{i}>{rank5/i}%, rank10: {rank10}/{i}>{rank10/i}%")

        return (rank1, rank5, rank10, i)



eval = compare()
eval.evaluate("front")
eval.evaluate("back")
eval.evaluate("side") # Evaluate by changing front, back, side