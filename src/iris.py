import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score, accuracy_score
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from utils import *

conf = """
base:
  api_key_path: "token/token.json"
  seed: 67
  
dataset:
  features: 4  # 特徴量の数
  val_size: 100  # validationに使うデータ数
  target: 0  # ラベル
  
model: timeout: 3000  # ms  計算時間
  # 訓練に使うデータは(batch_size * n_iter)個
  batch_size: 5  # バッチサイズ
  n_iter: 3  # ループ数
  l: 0.3  # 正則化項
  each_weight: 1 # 重み係数
  length_weight: 3  # 重みの層の数
"""

cfg = OmegaConf.create(conf)
init_client(cfg)


# PyTorch形式
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.astype(np.float16)
        self.label = label

        self.label = np.where(self.label == cfg.dataset.target, 1, 0)
        self.data[self.data == 0] = -1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.label[idx]


def get_ds(n, seed):
    true_ = dataset.loc[dataset["target"] == cfg.dataset.target, :].sample(n // 2)
    false_ = dataset.loc[dataset["target"] != cfg.dataset.target, :].sample(n // 2)
    return pd.concat(
        [
            true_,
            false_
        ]
    ).sample(
        frac=1,
        random_state=seed
    ).values


iris = load_iris()
dataset = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
).apply(minmax, axis=1)
dataset["target"] = iris.target
for i in range(3):
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

    train_ds = MyDataset(train[:, :-1], train[:, -1])
    valid_ds = MyDataset(val[:, :-1], val[:, -1])
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

    print("=" * 20, cfg.dataset.target, "=" * 20)
    print("AUC:", roc_auc_score(label, pred))
    print("ACC:", accuracy_score(label, np.round(pred)))
    print("=" * 43)
    weight = weight.sum(axis=1) * cfg.model.each_weight

    true_list = pred[label == 1]
    false_list = pred[label != 1]
    plt.title(f"n = {cfg.dataset.val_size}")
    plt.hist(true_list, color="red", alpha=0.5)
    plt.hist(false_list, color="blue", alpha=0.5)
    plt.show()

