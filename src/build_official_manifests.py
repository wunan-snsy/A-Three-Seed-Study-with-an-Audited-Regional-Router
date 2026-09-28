"""Build canonical official-split JSONL files; derive validation only from train."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
OUT=ROOT/"manifests"
OFFICIAL={
    "NJU2K_train": DATA/"TrainingSet/TrainingSet/NJU2K_TRAIN",
    "NJU2K_test": DATA/"TestingSet/TestingSet/NJU2K_TEST",
    "NLPR_train": DATA/"TrainingSet/TrainingSet/NLPR_TRAIN",
    "NLPR_test": DATA/"TestingSet/TestingSet/NLPR_TEST",
    "SIP_test": DATA/"TestingSet/TestingSet/SIP",
}


def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read_split(name: str, folder: Path) -> list[dict]:
    mods={key:{p.stem:p for p in (folder/sub).iterdir() if p.is_file()}
          for key,sub in (("rgb","RGB"),("depth","depth"),("gt","GT"))}
    ids=sorted(set.intersection(*(set(x) for x in mods.values())))
    union=set.union(*(set(x) for x in mods.values()))
    if len(ids)!=len(union):
        raise RuntimeError(f"Unpaired files in {name}: matched={len(ids)} union={len(union)}")
    out=[]
    for sid in ids:
        row={"dataset":name.split("_")[0],"source_split":name.split("_")[1],"id":sid,
             "group":f"{name}:{sid}","official_split":name.split("_")[1],
             "partition":"test" if name.endswith("_test") else "official_train"}
        row["paths"]={}
        row["sha256"]={}
        for mod in mods:
            p=mods[mod][sid]
            row["paths"][mod]=p.relative_to(ROOT).as_posix()
            row["sha256"][mod]=hash_file(p)
        out.append(row)
    return out


def main():
    all_rows={name:read_split(name,folder) for name,folder in OFFICIAL.items()}
    # Split only the official 2,185-image training pool into fitting, validation,
    # and calibration groups; official test images remain untouched.
    pool=all_rows["NJU2K_train"]+all_rows["NLPR_train"]
    rng=np.random.Generator(np.random.PCG64(2026))
    ids=np.arange(len(pool));rng.shuffle(ids)
    n_cal=round(len(pool)*0.10);n_val=round(len(pool)*0.10)
    cal=set(ids[:n_cal].tolist());val=set(ids[n_cal:n_cal+n_val].tolist())
    for i,row in enumerate(pool):
        row["partition"]="calibration" if i in cal else "validation" if i in val else "fit"
    for ds in ("NJU2K","NLPR"):
        all_rows[f"{ds}_train"]= [r for r in pool if r["dataset"]==ds]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"combined_train_official.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in pool),encoding="utf-8")
    for name,rows in all_rows.items():
        path=OUT/f"{name.lower()}_official.jsonl"
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
        print(name,len(rows),{p:sum(r["partition"]==p for r in rows) for p in ("fit","validation","calibration","test","official_train")})


if __name__=="__main__":main()
