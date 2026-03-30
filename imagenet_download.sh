#!/bin/bash
#SBATCH --job-name=imagenet_prep
#SBATCH -p parallel 
#SBATCH --output=/scratch/hpc/07/zhang303/tecoa/%x_%j.out
#SBATCH --error=/scratch/hpc/07/zhang303/tecoa/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

########################################
# paths
########################################
PROJECT_DIR=/scratch/hpc/07/zhang303/tecoa
DATA_ROOT=$PROJECT_DIR/data
IMAGENET_BASE=$DATA_ROOT/imagenet1k
RAW_DIR=$IMAGENET_BASE/raw
IMAGENET_ROOT=$IMAGENET_BASE/ImageNet-clean
TRAIN_DIR=$IMAGENET_ROOT/train
VAL_DIR=$IMAGENET_ROOT/val
TMP_DIR=$IMAGENET_BASE/tmp

mkdir -p "$RAW_DIR" "$TRAIN_DIR" "$VAL_DIR" "$TMP_DIR"

########################################
# fill in your real links here
########################################
TRAIN_URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

TRAIN_TAR="$RAW_DIR/ILSVRC2012_img_train.tar"
VAL_TAR="$RAW_DIR/ILSVRC2012_img_val.tar"
DEVKIT_TAR="$RAW_DIR/ILSVRC2012_devkit_t12.tar.gz"

########################################
# optional cookie file
# leave empty if not needed
########################################
COOKIE_FILE=""

########################################
# helpers
########################################
timestamp() {
    date "+%F %T"
}

say() {
    echo "[$(timestamp)] $1"
}

download_with_resume() {
    local url="$1"
    local out="$2"
    local done_mark="${out}.download.done"

    if [[ -f "$done_mark" ]]; then
        say "跳过下载，已完成: $out"
        return 0
    fi

    say "开始下载: $out"

    if [[ -n "$COOKIE_FILE" ]]; then
        curl -L -C - --progress-bar -b "$COOKIE_FILE" -c "$COOKIE_FILE" "$url" -o "$out"
    else
        curl -L -C - --progress-bar "$url" -o "$out"
    fi

    if file "$out" | grep -qiE 'HTML|XML|ASCII text'; then
        say "错误: 下载下来的像是网页，不是压缩包: $out"
        exit 1
    fi

    touch "$done_mark"
    say "下载完成: $out"
}
########################################
# 1) download archives
########################################
download_with_resume "$TRAIN_URL" "$TRAIN_TAR"
download_with_resume "$VAL_URL" "$VAL_TAR"
download_with_resume "$DEVKIT_URL" "$DEVKIT_TAR"

########################################
# 2) extract devkit
########################################
DEVKIT_EXTRACT_DONE="$RAW_DIR/.devkit.extract.done"
if [[ ! -f "$DEVKIT_EXTRACT_DONE" ]]; then
    say "开始解压 devkit"
    tar -xzf "$DEVKIT_TAR" -C "$RAW_DIR"
    touch "$DEVKIT_EXTRACT_DONE"
    say "devkit 解压完成"
else
    say "跳过 devkit 解压，已完成"
fi

########################################
# 3) extract train outer tar
########################################
TRAIN_OUTER_DONE="$TRAIN_DIR/.outer.extract.done"
if [[ ! -f "$TRAIN_OUTER_DONE" ]]; then
    say "开始解压 train 外层 tar"
    tar --skip-old-files -xf "$TRAIN_TAR" -C "$TRAIN_DIR"
    touch "$TRAIN_OUTER_DONE"
    say "train 外层 tar 解压完成"
else
    say "跳过 train 外层解压，已完成"
fi

########################################
# 4) extract each train class tar
########################################
say "开始按类别解压 train 数据"
shopt -s nullglob
total_train_tars=0
done_train_tars=0
for tarfile in "$TRAIN_DIR"/*.tar; do
    total_train_tars=$((total_train_tars + 1))
done

for tarfile in "$TRAIN_DIR"/*.tar; do
    cls=$(basename "$tarfile" .tar)
    cls_dir="$TRAIN_DIR/$cls"
    cls_done="$TRAIN_DIR/.${cls}.done"

    if [[ -f "$cls_done" ]]; then
        done_train_tars=$((done_train_tars + 1))
        echo "[train extract] $done_train_tars / $total_train_tars 已完成，跳过 $cls"
        continue
    fi

    mkdir -p "$cls_dir"
    tar --skip-old-files -xf "$tarfile" -C "$cls_dir"
    touch "$cls_done"
    done_train_tars=$((done_train_tars + 1))
    echo "[train extract] $done_train_tars / $total_train_tars 已完成: $cls"
done
shopt -u nullglob
say "train 按类别解压完成"

########################################
# 5) extract val flat images
########################################
VAL_FLAT_DIR="$TMP_DIR/val_flat"
mkdir -p "$VAL_FLAT_DIR"

VAL_OUTER_DONE="$VAL_FLAT_DIR/.outer.extract.done"
if [[ ! -f "$VAL_OUTER_DONE" ]]; then
    say "开始解压 val 外层 tar"
    tar --skip-old-files -xf "$VAL_TAR" -C "$VAL_FLAT_DIR"
    touch "$VAL_OUTER_DONE"
    say "val 外层 tar 解压完成"
else
    say "跳过 val 外层解压，已完成"
fi

########################################
# 6) build val mapping from devkit
########################################
VALMAP="$TMP_DIR/valprep_map.txt"
if [[ ! -f "$VALMAP" ]]; then
    say "开始生成 validation 映射文件"
    python - <<'PY'
from pathlib import Path
from scipy.io import loadmat

project_dir = Path("/scratch/hpc/07/zhang303/tecoa")
raw_dir = project_dir / "data" / "imagenet1k" / "raw"
tmp_dir = project_dir / "data" / "imagenet1k" / "tmp"

meta_path = None
gt_path = None
for p in raw_dir.rglob("meta.mat"):
    meta_path = p
    break
for p in raw_dir.rglob("ILSVRC2012_validation_ground_truth.txt"):
    gt_path = p
    break

if meta_path is None or gt_path is None:
    raise FileNotFoundError("meta.mat or ILSVRC2012_validation_ground_truth.txt not found")

meta = loadmat(meta_path, squeeze_me=True)["synsets"]

id2synset = {}
for row in meta:
    try:
        ilsvrc_id = int(row[0])
        wnid = str(row[1])
        num_children = int(row[4])
    except Exception:
        continue
    if 1 <= ilsvrc_id <= 1000 and num_children == 0:
        id2synset[ilsvrc_id] = wnid

if len(id2synset) != 1000:
    raise RuntimeError(f"expected 1000 classes, got {len(id2synset)}")

with open(gt_path, "r") as f:
    labels = [int(x.strip()) for x in f if x.strip()]

if len(labels) != 50000:
    raise RuntimeError(f"expected 50000 val labels, got {len(labels)}")

out_path = tmp_dir / "valprep_map.txt"
with open(out_path, "w") as f:
    for i, cls_id in enumerate(labels, start=1):
        fname = f"ILSVRC2012_val_{i:08d}.JPEG"
        wnid = id2synset[cls_id]
        f.write(f"{fname}\t{wnid}\n")

print(f"wrote {out_path}")
PY
    say "validation 映射文件生成完成"
else
    say "跳过 validation 映射文件生成，已完成"
fi

########################################
# 7) reorganize val into class folders
########################################
VAL_REORG_DONE="$VAL_DIR/.reorg.done"
if [[ ! -f "$VAL_REORG_DONE" ]]; then
    say "开始重组 val 目录结构"

    total_val=50000
    moved_val=0

    while IFS=$'\t' read -r fname synset; do
        src="$VAL_FLAT_DIR/$fname"
        dst_dir="$VAL_DIR/$synset"
        dst="$dst_dir/$fname"

        mkdir -p "$dst_dir"

        if [[ -f "$dst" ]]; then
            moved_val=$((moved_val + 1))
            if (( moved_val % 1000 == 0 )); then
                echo "[val reorganize] $moved_val / $total_val"
            fi
            continue
        fi

        if [[ -f "$src" ]]; then
            mv "$src" "$dst"
            moved_val=$((moved_val + 1))
            if (( moved_val % 1000 == 0 )); then
                echo "[val reorganize] $moved_val / $total_val"
            fi
        fi
    done < "$VALMAP"

    touch "$VAL_REORG_DONE"
    say "val 目录重组完成"
else
    say "跳过 val 重组，已完成"
fi

########################################
# 8) final checks
########################################
train_class_count=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
val_class_count=$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

say "最终检查:"
say "train 类别文件夹数: $train_class_count"
say "val 类别文件夹数: $val_class_count"

say "全部完成。ImageNet 已整理到:"
say "$IMAGENET_ROOT"