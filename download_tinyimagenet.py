import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

# BASE_DIR = Path("/scratch/hpc/07/zhang303/tecoa")
# ZIP_PATH = BASE_DIR / "tiny-imagenet-200.zip"
# EXTRACT_ROOT = BASE_DIR
# TARGET_DIR = BASE_DIR / "tiny-imagenet-200"

BASE_DIR = Path("/scratch/hpc/07/zhang303/tecoa/data")
ZIP_PATH = BASE_DIR / "tiny-imagenet-200.zip"
EXTRACT_ROOT = BASE_DIR
TARGET_DIR = BASE_DIR / "tiny-imagenet-200"

CHUNK_SIZE = 1024 * 1024  # 1MB


def touch_path(path: Path, ts: Optional[float] = None):
    if ts is None:
        ts = time.time()
    try:
        os.utime(str(path), (ts, ts))
    except FileNotFoundError:
        pass


def touch_tree(root: Path, ts: Optional[float] = None):
    if ts is None:
        ts = time.time()

    if not root.exists():
        return

    for dirpath, dirnames, filenames in os.walk(str(root)):
        for name in filenames:
            touch_path(Path(dirpath) / name, ts)
        for name in dirnames:
            touch_path(Path(dirpath) / name, ts)

    touch_path(root, ts)


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "{:.2f}{}".format(num, unit)
        num /= 1024.0
    return "{:.2f}PB".format(num)


def print_progress(prefix, current, total, extra=""):
    if total and total > 0:
        percent = current * 100.0 / total
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = "\r{} [{}] {:6.2f}% ({}/{}) {}".format(
            prefix,
            bar,
            percent,
            sizeof_fmt(current),
            sizeof_fmt(total),
            extra
        )
    else:
        msg = "\r{} {} {}".format(prefix, sizeof_fmt(current), extra)

    sys.stdout.write(msg)
    sys.stdout.flush()


def get_remote_file_size(url: str) -> Optional[int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        r.raise_for_status()
        size = r.headers.get("Content-Length")
        if size is None:
            return None
        return int(size)
    except Exception:
        return None


def download_with_resume(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remote_size = get_remote_file_size(url)
    local_size = output_path.stat().st_size if output_path.exists() else 0

    if remote_size is not None and local_size == remote_size:
        print("[DOWNLOAD] File already complete, skip: {}".format(output_path))
        touch_path(output_path)
        return

    headers = {}
    mode = "wb"

    if local_size > 0:
        headers["Range"] = "bytes={}-".format(local_size)
        mode = "ab"
        print("[DOWNLOAD] Resume from {}".format(sizeof_fmt(local_size)))
    else:
        print("[DOWNLOAD] Start full download")

    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        if r.status_code == 200 and local_size > 0:
            print("[DOWNLOAD] Server does not support resume, restart full download")
            mode = "wb"
            local_size = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()

        with open(str(output_path), mode) as f:
            downloaded = local_size
            total = remote_size if remote_size is not None else 0
            last_print = 0.0

            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print > 0.2:
                    print_progress("[DOWNLOAD]", downloaded, total)
                    last_print = now

            print_progress("[DOWNLOAD]", downloaded, total)

    print("\n[DOWNLOAD] Done")
    touch_path(output_path)


def extract_with_resume(zip_path: Path, extract_root: Path):
    if not zip_path.exists():
        raise FileNotFoundError("Zip file not found: {}".format(zip_path))

    extract_root.mkdir(parents=True, exist_ok=True)
    now_ts = time.time()

    print("[EXTRACT] Check and continue extraction ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.infolist()

        file_members = [m for m in members if not m.is_dir()]
        total_files = len(file_members)
        total_uncompressed = sum(m.file_size for m in file_members)

        done_files = 0
        done_bytes = 0
        last_print = 0.0

        for member in members:
            target_path = extract_root / member.filename

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                touch_path(target_path, now_ts)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            touch_path(target_path.parent, now_ts)

            file_done = False
            if target_path.exists() and target_path.stat().st_size == member.file_size:
                touch_path(target_path, now_ts)
                file_done = True
            else:
                with zf.open(member, "r") as src, open(str(target_path), "wb") as dst:
                    while True:
                        chunk = src.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                touch_path(target_path, now_ts)
                file_done = True

            if file_done:
                done_files += 1
                done_bytes += member.file_size

            now = time.time()
            if now - last_print > 0.1:
                extra = "files {}/{} | {}".format(
                    done_files,
                    total_files,
                    member.filename[-40:]
                )
                print_progress("[EXTRACT]", done_bytes, total_uncompressed, extra=extra)
                last_print = now

        extra = "files {}/{}".format(done_files, total_files)
        print_progress("[EXTRACT]", done_bytes, total_uncompressed, extra=extra)

    print("\n[EXTRACT] All done")

    if TARGET_DIR.exists():
        touch_tree(TARGET_DIR, time.time())


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    download_with_resume(URL, ZIP_PATH)
    extract_with_resume(ZIP_PATH, EXTRACT_ROOT)

    touch_path(ZIP_PATH)
    if TARGET_DIR.exists():
        touch_path(TARGET_DIR)

    print("[DONE] zip: {}".format(ZIP_PATH))
    print("[DONE] data dir: {}".format(TARGET_DIR))


if __name__ == "__main__":
    main()