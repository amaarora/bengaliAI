import glob 
import pandas as pd
import joblib
from tqdm import tqdm

if __name__ == '__main__':
    files = glob.glob("../data/train_image*.parquet")
    for file in files: 
        df = pd.read_parquet(file)
        img_ids = df['image_id'].values
        df = df.drop("image_id", axis=1)
        imgs = df.values

        for (i, img_id) in tqdm(enumerate(img_ids), total=len(imgs)):
            joblib.dump(imgs[i, :], f"../data/image_pickles/{img_id}.pkl")