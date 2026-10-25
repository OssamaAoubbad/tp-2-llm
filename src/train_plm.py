import time, os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from src.data import load_clean
from src.utils import set_seed, log_result

SENT = {"NEG": 0, "NEU": 1, "POS": 2}


class PLMClassifier(nn.Module):
    def __init__(self, name, multitask=False):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.sarc = nn.Linear(h, 2)
        self.sent = nn.Linear(h, 3) if multitask else None

    def forward(self, input_ids, attention_mask):
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        cls = self.drop(cls)
        return self.sarc(cls), (self.sent(cls) if self.sent is not None else None)


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    i = f1s[:-1].argmax()
    return float(thr[i]), float(f1s[i])


def prepare_texts(df, model_name):
    """AraBERT : préprocesseur officiel (emojis gardés). MARBERT : texte 'clean' tel quel."""
    if "arabert" in model_name:
        from arabert.preprocess import ArabertPreprocessor
        try:   # on essaie de conserver aussi les allongements (indice du sarcasme)
            pre = ArabertPreprocessor(model_name=model_name, keep_emojis=True,
                                      remove_non_digit_repetition=False)
        except TypeError:
            pre = ArabertPreprocessor(model_name=model_name, keep_emojis=True)
        return [pre.preprocess(t) for t in df["clean"]]
    return df["clean"].tolist()


def make_loader(tok, texts, y_s, y_t, bs, shuffle):
    enc = tok(texts, truncation=True, max_length=128)
    items = [dict(input_ids=enc["input_ids"][i], y_s=int(y_s[i]), y_t=int(y_t[i]))
             for i in range(len(texts))]

    def collate(b):
        L = max(len(x["input_ids"]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id)
        for i, x in enumerate(b):
            ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
        return (ids, (ids != tok.pad_token_id).long(),
                torch.tensor([x["y_s"] for x in b]), torch.tensor([x["y_t"] for x in b]))

    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); out = []
    for ids, mask, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            ls, _ = model(ids.to(dev), mask.to(dev))
        out.append(torch.softmax(ls.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def run(model_name, seed, multitask=False, lam=0.5, epochs=4, lr=2e-5, bs=32,
        patience=2, n_sub=None, tag=None,
        res_path="results/results_kaggle.csv", pred_dir="predictions"):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test, _ = load_clean(verbose=False)
    if n_sub:
        train, val, test = train.sample(n_sub, random_state=0), val.head(n_sub), test.head(n_sub)

    tok = AutoTokenizer.from_pretrained(model_name)
    mk = lambda d, sh: make_loader(tok, prepare_texts(d, model_name), d["sarcasm"].values,
                                   d["sentiment"].map(SENT).values, bs, sh)
    tr_l, va_l, te_l = mk(train, True), mk(val, False), mk(test, False)

    model = PLMClassifier(model_name, multitask).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnt = np.bincount(train["sarcasm"].values, minlength=2)
    w = torch.tensor(len(train) / (2 * cnt), dtype=torch.float).to(dev)
    ce_s, ce_t = nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    total = epochs * len(tr_l)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    yv, yt = val["sarcasm"].values, test["sarcasm"].values
    best = dict(f1=-1); bad = 0; t0 = time.time()
    for ep in range(epochs):
        model.train()
        for ids, mask, ys, ytt in tr_l:
            ids, mask, ys, ytt = ids.to(dev), mask.to(dev), ys.to(dev), ytt.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                ls, lt = model(ids, mask)
                loss = ce_s(ls.float(), ys)
                if multitask:
                    loss = loss + lam * ce_t(lt.float(), ytt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        pv = predict(model, va_l, dev)
        thr, f1v = best_threshold(yv, pv)
        print(f"epoch {ep+1} | F1 val={f1v:.4f} | seuil={thr:.3f} | loss={loss.item():.3f}")
        if f1v > best["f1"]:
            # la sélection se fait UNIQUEMENT sur val ; les probas test sont seulement stockées
            best = dict(f1=f1v, thr=thr, pv=pv, pt=predict(model, te_l, dev), ep=ep + 1)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    dur = time.time() - t0

    pv, pt, thr = best["pv"], best["pt"], best["thr"]
    name = tag or (("mt" if multitask else "st") + "_" + model_name.split("/")[-1])
    if multitask:
        name += f"_lam{lam}"
    os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame(dict(id=test["id"].values, y_true=yt, y_pred=(pt >= thr).astype(int),
                      prob=pt)).to_csv(f"{pred_dir}/{name}_seed{seed}.csv", index=False)
    pd.DataFrame(dict(id=val["id"].values, y_true=yv, prob=pv)).to_csv(
        f"{pred_dir}/{name}_seed{seed}_val.csv", index=False)
    log_result(dict(model=name, preprocessing="arabert_pre" if "arabert" in model_name else "clean",
                    seed=seed, threshold=round(thr, 4), f1_val=round(best["f1"], 4),
                    prauc_val=round(average_precision_score(yv, pv), 4),
                    macrof1_val=round(f1_score(yv, pv >= thr, average="macro"), 4),
                    f1_test=round(f1_score(yt, pt >= thr), 4),
                    prauc_test=round(average_precision_score(yt, pt), 4),
                    macrof1_test=round(f1_score(yt, pt >= thr, average="macro"), 4),
                    train_time_s=round(dur, 1), trained_params=n_params), path=res_path)
    print(f"{name} seed{seed} | F1 val={best['f1']:.4f} | F1 test={f1_score(yt, pt >= thr):.4f} "
          f"| meilleure époque={best['ep']} | {dur:.0f}s")
    return best