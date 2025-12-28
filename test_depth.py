import torch
import cv2
import matplotlib.pyplot as plt
from model.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ⚠️ Vì bạn dùng checkpoint vits → encoder phải là vits
model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
)

model.load_state_dict(
    torch.load("checkpoints/depth_anything_v2_vits.pth", map_location=DEVICE)
)

model = model.to(DEVICE).eval()

# Đọc ảnh test
img = cv2.imread("test.png")
if img is None:
    raise FileNotFoundError("Không tìm thấy test.jpg trong thư mục hiện tại")

depth = model.infer_image(img)

plt.imshow(depth, cmap="inferno")
plt.colorbar()
plt.title("Depth Anything V2")
plt.show()