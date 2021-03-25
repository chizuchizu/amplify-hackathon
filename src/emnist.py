import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from utils import *

conf = """
base:
  api_key_path: "token/token.json"
  train_path: "../data/emnist_train_400.csv"
  seed: 67
  
dataset:
  img_size: 20 
  features: 400  # 特徴量の数
  val_size: 300  # validationに使うデータ数
  target: 5  # ラベル
  
model:
  timeout: 3000  # ms  計算時間
  # 訓練に使うデータは(batch_size * n_iter)個
  batch_size: 20  # バッチサイズ
  n_iter: 5  # ループ数
  l: 2  # 正則化項
  each_weight: 1  # 重み係数
  length_weight: 3  # 重みの層の数
  multiprocessing: true
"""

cfg = OmegaConf.create(conf)

init_client(cfg)


# PyTorch形式
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = (data / 255).astype(np.float16)
        self.label = label

        self.label = np.where(self.label == cfg.dataset.target, 1, 0)
        self.data[self.data == 0] = -1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.label[idx]


def get_ds(n, seed):
    true_ = dataset.loc[dataset.iloc[:, 0] == cfg.dataset.target, :].sample(n // 2)
    false_ = dataset.loc[dataset.iloc[:, 0] != cfg.dataset.target, :].sample(n // 2)
    return pd.concat(
        [
            true_,
            false_
        ]
    ).sample(
        frac=1,
        random_state=seed
    ).values


dataset = pd.read_csv(cfg.base.train_path)  # .sample(100).values
for i in range(6, 20):
    init_client(cfg)
    cfg.dataset.target = i
    train = get_ds(
        int(
            cfg.model.n_iter * cfg.model.batch_size
        ),
        cfg.base.seed
    )

    val = get_ds(
        cfg.dataset.val_size,
        cfg.base.seed + 1
    )

    train_ds = MyDataset(train[:, 1:], train[:, 0])
    valid_ds = MyDataset(val[:, 1:], val[:, 0])
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.model.batch_size,
        shuffle=True
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=64,
        shuffle=False
    )
    weight = np.zeros(
        (
            cfg.dataset.features,
            cfg.model.length_weight
        )
    )

    weight = run_model(cfg, train_dl, weight, cfg.model.multiprocessing)
    pred, label = eval_model(cfg, valid_dl, weight)

    print("=" * 20, i, "=" * 20)
    print("AUC:", roc_auc_score(label, pred))
    print("ACC:", accuracy_score(label, np.round(pred)))
    print("=" * 43)
    weight = weight.sum(axis=1) * cfg.model.each_weight

    plt.imshow(
        weight.reshape(
            cfg.dataset.img_size,
            cfg.dataset.img_size
        )
    )
    plt.axis('tight')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(
        f"../img/EMNIST_{cfg.dataset.target}_weight.png"
    )
