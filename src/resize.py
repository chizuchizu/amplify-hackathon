from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2

train = pd.read_csv("data/train.csv")
img_size = 20
pixels = img_size ** 2  # ピクセル数

save = np.zeros(
    (
        train.shape[0],
        1 + pixels
    ),
    dtype=int
)

for i in tqdm(range(train.shape[0])):
    image = train.iloc[i, 1:].values.reshape(28, 28).astype(np.uint8)

    image = cv2.resize(image, dsize=(img_size, img_size)).flatten()

    save[i, 0] = train.iloc[i, 0]
    save[i, 1:] = image

save = pd.DataFrame(save)

save.to_csv(f"data/train_{pixels}.csv", index=False)
