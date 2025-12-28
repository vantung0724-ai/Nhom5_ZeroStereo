import os
import cv2
import csv
import torch
import numpy as np

from model import fetch_model
from util.padder import InputPadder


# =========================
# CONFIG
# =========================
DEVICE = "cpu"   # đổi thành "cuda" nếu có GPU
DATA_ROOT = "dataset"
OUTPUT_DIR = "outputs"

MAX_SCENES = 3     # test trước
MAX_IMAGES = 5


# =========================
# INIT
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Loading model...")
model = fetch_model("stereogen", None)
model.to(DEVICE)
model.eval()

scenes = sorted(os.listdir(os.path.join(DATA_ROOT, "frames_cleanpass")))

results = []


# =========================
# RUN EVAL
# =========================
with torch.no_grad():
    for scene in scenes[:MAX_SCENES]:
        left_dir = os.path.join(DATA_ROOT, "frames_cleanpass", scene, "left")
        right_dir = os.path.join(DATA_ROOT, "frames_cleanpass", scene, "right")
        gt_dir = os.path.join(DATA_ROOT, "disparity", scene, "left")

        if not (os.path.exists(left_dir) and os.path.exists(gt_dir)):
            print(f"[SKIP] {scene} (missing data)")
            continue

        image_names = sorted(os.listdir(left_dir))[:MAX_IMAGES]

        for img_name in image_names:
            left_path = os.path.join(left_dir, img_name)
            right_path = os.path.join(right_dir, img_name)
            gt_path = os.path.join(gt_dir, img_name.replace(".png", ".npy"))

            if not os.path.exists(gt_path):
                print(f"[SKIP] missing GT {img_name}")
                continue

            # ---- load images ----
            left = cv2.imread(left_path)
            right = cv2.imread(right_path)

            left = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float()
            right = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float()

            left = left.to(DEVICE)
            right = right.to(DEVICE)

            # ---- pad ----
            padder = InputPadder(left.shape)
            left, right = padder.pad(left, right)

            # ---- predict disparity ----
            disp_pred = model(left, right)
            disp_pred = padder.unpad(disp_pred)[0, 0]

            # ---- load GT ----
            disp_gt = torch.from_numpy(np.load(gt_path)).to(DEVICE)

            # ---- EPE ----
            epe = torch.abs(disp_pred - disp_gt).mean().item()

            results.append([scene, img_name, epe])
            print(f"{scene} | {img_name} | EPE = {epe:.4f}")


# =========================
# SAVE CSV
# =========================
csv_path = os.path.join(OUTPUT_DIR, "epe.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["scene", "image", "epe"])
    writer.writerows(results)

print("\n==============================")
print("DONE ✔")
print(f"CSV saved to: {csv_path}")
print("==============================")
