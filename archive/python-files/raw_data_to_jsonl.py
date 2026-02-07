# Build a grouped-by-conversation training set from /mnt/data/human_convo.csv
# - Detect conversation boundaries by "Hi!/Hi./Hi" in left column (case-insensitive)
# - Keep strict alternation: human1 -> user, human2 -> assistant for every row
# - Emit one JSONL object per conversation: {"messages":[...], "domain":"general", "style":"neutral", "source":"human_convo"}
# - Also produce train/val/test splits by conversation (90/5/5)
# - Additionally produce a lightly filtered variant (drops rows with coarse toxic words), same splits
#
# Outputs (all under /mnt/data/processed):
#   human_convo_conversations.jsonl
#   human_convo_train.jsonl
#   human_convo_val.jsonl
#   human_convo_test.jsonl
#   human_convo_conversations_clean.jsonl
#   human_convo_train_clean.jsonl
#   human_convo_val_clean.jsonl
#   human_convo_test_clean.jsonl
#
import json, pathlib, re, random, os
import pandas as pd

random.seed(42)

IN_PATH = pathlib.Path(os.getcwd().replace('python-files', 'raw_datasets\\'))
OUT_DIR = pathlib.Path(os.getcwd().replace('python-files', 'jsonl_datasets\\'))

# ---- Load with '#' delimiter ----
df = pd.read_csv(IN_PATH + "human_convo.csv", sep="#", engine="python", header=None, names=["human1","human2"])

def norm_cell(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    if len(s) >= 2 and ((s[0]==s[-1]=='"') or (s[0]==s[-1]=="'")):
        s = s[1:-1]
    return s

df["human1"] = df["human1"].map(norm_cell)
df["human2"] = df["human2"].map(norm_cell)

# ---- Group rows into conversations using "Hi!/Hi./Hi" in human1 ----
start_re = re.compile(r"^\s*hi[!.]?\s*$", re.IGNORECASE)

conversations = []
current = []  # list of dicts with keys human1, human2
for _, row in df.iterrows():
    u, a = row["human1"], row["human2"]
    # skip fully empty rows
    if not (u or a):
        continue
    if start_re.match(u):
        if current:
            conversations.append(current)
            current = []
        current.append({"human1": u, "human2": a})
    else:
        if not current:
            # ignore preamble until first "Hi"
            continue
        current.append({"human1": u, "human2": a})
if current:
    conversations.append(current)

# ---- Toxicity filter (coarse) for a "clean" variant ----
toxic_words = set([
    "kill","suicide","die","hate","stupid","idiot","dumb","loser","racist","sex","rape",
    "self-harm","slit","kys","fuck","shit","bitch","bastard","moron","asshole"
])
def is_toxic(s: str) -> bool:
    s_l = s.lower()
    return any(w in s_l for w in toxic_words)

def to_messages(conv_rows):
    msgs = []
    for t in conv_rows:
        u, a = norm_cell(t["human1"]), norm_cell(t["human2"])
        if u: msgs.append({"role":"user","content":u})
        if a: msgs.append({"role":"assistant","content":a})
    return msgs

def conv_to_obj(conv_rows):
    msgs = to_messages(conv_rows)
    if len(msgs) < 2:  # at least one exchange
        return None
    return {"messages": msgs, "domain":"general", "style":"neutral", "source":"human_convo"}

def conv_to_obj_clean(conv_rows):
    # Drop any row that contains toxicity on either side to maintain alternation
    kept = [t for t in conv_rows if not (is_toxic(t["human1"]) or is_toxic(t["human2"]))]
    msgs = to_messages(kept)
    if len(msgs) < 2:
        return None
    return {"messages": msgs, "domain":"general", "style":"neutral", "source":"human_convo"}

raw_objs = []
clean_objs = []
for conv in conversations:
    o = conv_to_obj(conv)
    if o: raw_objs.append(o)
    oc = conv_to_obj_clean(conv)
    if oc: clean_objs.append(oc)

# ---- Save helpers ----
def write_jsonl(path: pathlib.Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- Shuffle and split by conversation ----
def split_convs(objs, train_ratio=0.90, val_ratio=0.05):
    idx = list(range(len(objs)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train = [objs[i] for i in idx[:n_train]]
    val   = [objs[i] for i in idx[n_train:n_train+n_val]]
    test  = [objs[i] for i in idx[n_train+n_val:]]
    return train, val, test

train_raw, val_raw, test_raw = split_convs(raw_objs)
train_clean, val_clean, test_clean = split_convs(clean_objs)

# ---- Write all files ----
paths = {
    "conversations": OUT_DIR/"human_convo_conversations.jsonl",
    "train": OUT_DIR/"human_convo_train.jsonl",
    "val": OUT_DIR/"human_convo_val.jsonl",
    "test": OUT_DIR/"human_convo_test.jsonl",
    "conversations_clean": OUT_DIR/"human_convo_conversations_clean.jsonl",
    "train_clean": OUT_DIR/"human_convo_train_clean.jsonl",
    "val_clean": OUT_DIR/"human_convo_val_clean.jsonl",
    "test_clean": OUT_DIR/"human_convo_test_clean.jsonl",
}
write_jsonl(paths["conversations"], raw_objs)
write_jsonl(paths["train"], train_raw)
write_jsonl(paths["val"], val_raw)
write_jsonl(paths["test"], test_raw)
write_jsonl(paths["conversations_clean"], clean_objs)
write_jsonl(paths["train_clean"], train_clean)
write_jsonl(paths["val_clean"], val_clean)
write_jsonl(paths["test_clean"], test_clean)

# ---- Quick preview back to user ----
def preview_objs(objs, k=5):
    rows = []
    for i, o in enumerate(objs[:k]):
        first_user = next((m["content"] for m in o["messages"] if m["role"]=="user"), "")[:80]
        first_asst = next((m["content"] for m in o["messages"] if m["role"]=="assistant"), "")[:80]
        rows.append({
            "conv_id": i,
            "num_messages": len(o["messages"]),
            "first_user": first_user,
            "first_assistant": first_asst
        })
    return pd.DataFrame(rows)

{
    "counts": {
        "conversations_raw": len(raw_objs),
        "conversations_clean": len(clean_objs),
        "train_raw": len(train_raw),
        "val_raw": len(val_raw),
        "test_raw": len(test_raw),
        "train_clean": len(train_clean),
        "val_clean": len(val_clean),
        "test_clean": len(test_clean),
    },
    "outputs": {k: str(v) for k,v in paths.items()}
}
