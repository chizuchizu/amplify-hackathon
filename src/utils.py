import numpy as np
from multiprocessing import Pool

from amplify.client import FixstarsClient
from amplify import Solver, IsingQuadraticModel
from amplify import gen_symbols, decode_solution, IsingPoly
import os
import json
import time

client = FixstarsClient()

cfg = None


def init_client(_cfg):
    global cfg
    cfg = _cfg.copy()
    # APIキーの取得
    with open(_cfg.base.api_key_path) as f:
        client.token = json.load(f)["AMPLIFY_TOKEN"]
    client.parameters.timeout = _cfg.model.timeout


def get_weight(w, k):
    # 重みに係数をかける
    return sum(w) * k


# 正規化
def minmax(x):
    return (x - min(x)) / (max(x) - min(x))


def standardization(x):
    return (x - x.mean()) / x.std()


def is_int(x):
    """
    学習されないBitが存在してInt変換時にエラーが出ることがあるので作った
    :param x: Any
    :return: Int
    """
    try:
        return int(x)
    except RuntimeError:
        return 0


def forward(data, label, weight, cfg, inf=False):
    if not inf:
        f1 = sum(
            (
                    sum(
                        get_weight(weight[j], cfg.model.each_weight) * data[i, j] - label[i]
                        for j in range(data.shape[1])  # pixels
                    ) / data.shape[1]
            ) ** 2
            for i in range(data.shape[0])  # index
        )

        memo = lambda x: sum(x) ** 2
        f2 = sum(map(memo, weight))
        f = f1 + cfg.model.l * f2
        return f,  # ,を打っておけばタプルになり、制約式もつけることが可能になる

    else:
        """
        np array
        """
        w = np.array(
            [
                get_weight(x, cfg.model.each_weight)
                for x in weight
            ]
        )
        f1 = np.array(
            [
                sum(
                    w * data[i, :].numpy()
                ) / data.shape[1]
                for i in range(data.shape[0])
            ]
        )
        # f1 /= sum(map(sum, weight)) * cfg.model.each_weight
        f1 = minmax(f1)
        #         f1 /= f1.max()

        return f1


def solve_model(expression):
    # 制約もつけられる
    model = IsingQuadraticModel(*expression)
    solver = Solver(client)
    result = solver.solve(model)
    print("energy: ", result[0].energy)
    # print("time(ms): ", solver.client_result.annealing_time_ms)
    return result[0].values


def model(cfg, data, label):
    w = gen_symbols(IsingPoly, cfg.dataset.features, cfg.model.length_weight)
    f = forward(data, label, w, cfg)

    q_result = decode_solution(w, solve_model(f))
    q_result = np.vectorize(is_int)(q_result)
    return q_result


def train_fn(ds):
    data = ds[0]
    label = ds[1]

    start = time.time()
    w = np.array(model(cfg, data, label), dtype=float)
    finish = time.time()
    print(f"time: {round(finish - start, 3)} s")
    return w


def run_model(cfg, ds, weight, multiprocessing=True):
    _weight = weight.copy()
    if multiprocessing:
        num_cores = np.min([os.cpu_count(), len(ds)])
        pool = Pool(num_cores)
        output = list(pool.map(train_fn, ds))

        for i, w in enumerate(output):
            _weight += w / cfg.model.n_iter

        pool.close()
        pool.join()
    else:
        for d in ds:
            out = train_fn(d)
            _weight += out / cfg.model.n_iter
    return _weight


def val_fn(cfg, ds, weight):
    data = ds[0]
    label = ds[1]

    result = forward(
        data,
        label,
        weight,
        cfg,
        inf=True
    )
    return result, label


_weight = None
_cfg = None


def val_fn(_ds):
    data = _ds[0]
    label = _ds[1]

    result = forward(
        data,
        label,
        _weight,
        _cfg,
        inf=True
    )
    return result, label


def eval_model(cfg, ds, weight):
    global _cfg
    global _weight
    _weight = weight.copy()
    _cfg = cfg.copy()

    num_cores = np.min([os.cpu_count(), len(ds)])
    pool = Pool(num_cores)
    output = list(pool.map(val_fn, ds))
    _weight = weight.copy()

    preds = []
    labels = []

    for p, l in output:
        preds += p.tolist()
        labels += l.numpy().tolist()

    pool.close()
    pool.join()

    preds, labels = np.array(preds), np.array(labels)

    return preds, labels
