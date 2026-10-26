import os, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, average_precision_score
from src.data import load_clean
from src.utils import set_seed, log_result
from src.train_plm import prepare_texts, best_threshold

NAME = "aubmindlab/bert-base-arabertv02-twitter"
MAXLEN = 96
RES = "results/results_kaggle.csv"


@torch.no_grad()
def extract(texts, tok, enc, dev, bs=64):
    cls, mean, states, masks = [], [], [], []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding="max_length", truncation=True,
                max_length=MAXLEN, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            h = enc(**b).last_hidden_state
        h = h.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        cls.append(h[:, 0].cpu())
        mean.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        states.append(h.half().cpu()); masks.append(b["attention_mask"].cpu())
    return dict(cls=torch.cat(cls), mean=torch.cat(mean),
                states=torch.cat(states), mask=torch.cat(masks))


def build_features():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test, _ = load_clean(verbose=False)
    tok = AutoTokenizer.from_pretrained(NAME)
    enc = AutoModel.from_pretrained(NAME).to(dev).eval()
    for p in enc.parameters():
        p.requires_grad = False
    F = {}
    for n, d in [("train", train), ("val", val), ("test", test)]:
        F[n] = extract(prepare_texts(d, NAME), tok, enc, dev)
        F[n]["y"] = d["sarcasm"].values
        F[n]["id"] = d["id"].values
        F[n]["texts"] = prepare_texts(d, NAME)
    return F, tok


def save_and_log(name, seed, F, pv, pt, thr, dur, n_params):
    yv, yt = F["val"]["y"], F["test"]["y"]
    os.makedirs("predictions", exist_ok=True)
    pd.DataFrame(dict(id=F["test"]["id"], y_true=yt, y_pred=(pt >= thr).astype(int),
                      prob=pt)).to_csv(f"predictions/{name}_seed{seed}.csv", index=False)
    pd.DataFrame(dict(id=F["val"]["id"], y_true=yv, prob=pv)).to_csv(
        f"predictions/{name}_seed{seed}_val.csv", index=False)
    log_result(dict(model=name, preprocessing="arabert_pre", seed=seed, threshold=round(thr, 4),
                    f1_val=round(f1_score(yv, pv >= thr), 4),
                    prauc_val=round(average_precision_score(yv, pv), 4),
                    macrof1_val=round(f1_score(yv, pv >= thr, average="macro"), 4),
                    f1_test=round(f1_score(yt, pt >= thr), 4),
                    prauc_test=round(average_precision_score(yt, pt), 4),
                    macrof1_test=round(f1_score(yt, pt >= thr, average="macro"), 4),
                    train_time_s=round(dur, 1), trained_params=n_params), path=RES)
    print(f"{name} seed{seed} | F1 val={f1_score(yv, pv >= thr):.4f} | "
          f"F1 test={f1_score(yt, pt >= thr):.4f} | {dur:.0f}s")


def run_d1(F, pooling="cls"):
    import time; t0 = time.time()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=3000))
    clf.fit(F["train"][pooling].numpy(), F["train"]["y"])
    pv = clf.predict_proba(F["val"][pooling].numpy())[:, 1]
    pt = clf.predict_proba(F["test"][pooling].numpy())[:, 1]
    thr, _ = best_threshold(F["val"]["y"], pv)          # seuil choisi sur val
    save_and_log(f"frozen_{pooling}_lr", 0, F, pv, pt, thr, time.time() - t0, 769)


class AttnHead(nn.Module):
    def __init__(self, d=768, h=64):
        super().__init__()
        self.rnn = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.W = nn.Linear(2 * h, h)
        self.v = nn.Linear(h, 1, bias=False)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * h, 2)
        self.last_a = None

    def forward(self, x, mask):
        L = x.size(1)
        pk = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True,
                                               enforce_sorted=False)
        h, _ = self.rnn(pk)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=L)
        s = self.v(torch.tanh(self.W(h))).squeeze(-1).masked_fill(mask == 0, -1e9)
        a = s.softmax(-1)                                # alpha_i = softmax(v^T tanh(W h_i))
        self.last_a = a.detach()
        return self.fc(self.drop((a.unsqueeze(-1) * h).sum(1)))   # c = sum alpha_i h_i


@torch.no_grad()
def _predict(head, S, M, dev, bs=128):
    head.eval(); out = []
    for i in range(0, len(S), bs):
        lg = head(S[i:i + bs].to(dev).float(), M[i:i + bs].to(dev))
        out.append(lg.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def run_d2(F, seed, epochs=10, patience=3, lr=1e-3, bs=32):
    import time; t0 = time.time()
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = AttnHead().to(dev)
    ytr = torch.tensor(F["train"]["y"])
    cnt = np.bincount(F["train"]["y"], minlength=2)
    w = torch.tensor(len(ytr) / (2 * cnt), dtype=torch.float).to(dev)
    lossfn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    S, M = F["train"]["states"], F["train"]["mask"]
    best = dict(f1=-1); bad = 0
    for ep in range(epochs):
        head.train()
        for idx in torch.randperm(len(ytr)).split(bs):
            opt.zero_grad()
            loss = lossfn(head(S[idx].to(dev).float(), M[idx].to(dev)), ytr[idx].to(dev))
            loss.backward(); opt.step()
        pv = _predict(head, F["val"]["states"], F["val"]["mask"], dev)
        thr, f1 = best_threshold(F["val"]["y"], pv)
        print(f"epoch {ep+1} | F1 val={f1:.4f}")
        if f1 > best["f1"]:
            best = dict(f1=f1, thr=thr, pv=pv, state={k: v.clone() for k, v in head.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best["state"])                   # meilleure époque choisie sur val
    pt = _predict(head, F["test"]["states"], F["test"]["mask"], dev)
    n_params = sum(p.numel() for p in head.parameters())
    save_and_log("frozen_bilstm_attn", seed, F, best["pv"], pt, best["thr"],
                 time.time() - t0, n_params)
    return head


def plot_attention(head, F, tok, n=5, path="figures/attention_D2.png"):
    import matplotlib.pyplot as plt
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        fix = lambda s: get_display(arabic_reshaper.reshape(s))
    except ImportError:
        fix = lambda s: s
    dev = next(head.parameters()).device
    os.makedirs("figures", exist_ok=True)
    idxs = np.flatnonzero(F["val"]["y"] == 1)[:n]
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    head.eval()
    for ax, i in zip(axes, idxs):
        with torch.no_grad():
            head(F["val"]["states"][i:i + 1].to(dev).float(), F["val"]["mask"][i:i + 1].to(dev))
        k = int(F["val"]["mask"][i].sum())
        a = head.last_a[0, :k].cpu().numpy()
        ids = tok(F["val"]["texts"][i], truncation=True, max_length=MAXLEN)["input_ids"]
        toks = [fix(t) for t in tok.convert_ids_to_tokens(ids)]
        ax.bar(range(k), a); ax.set_xticks(range(k)); ax.set_xticklabels(toks, rotation=90, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.show()