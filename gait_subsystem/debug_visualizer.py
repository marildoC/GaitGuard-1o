import cv2
import numpy as np
from ultralytics import YOLO
import os

VIDEO_PATH = "data/gait_videos/francesco_id/f_1_045_l.mp4" # Il tuo video a 90 gradi
SAVE_NPY_PATH = "debug_local_francesco_90.npy"
TARGET_LEN = 60

def process_video_locally():
    print(f"🎥 Analisi Video: {VIDEO_PATH}")
    
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(VIDEO_PATH)
    raw_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, verbose=False, conf=0.3)
        if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
            kps = results[0].keypoints.data[0].cpu().numpy()
            if kps.shape[1] == 2:
                kps = np.concatenate([kps, np.ones((17, 1))], axis=1)
            raw_keypoints.append(kps)
    cap.release()
    
    if len(raw_keypoints) < 20:
        print("❌ Troppi pochi frame!")
        return

    kp_all = np.array(raw_keypoints).astype(np.float32)
    np.save(SAVE_NPY_PATH, kp_all)
    print(f"💾 NPY Salvato: {SAVE_NPY_PATH}")
    print(f"   Raw Shape: {kp_all.shape}")
    print(f"   Raw Values -> Min: {kp_all[:,:,:2].min():.2f}, Max: {kp_all[:,:,:2].max():.2f}")

    for t in range(1, kp_all.shape[0]):
        mask_zeros = (kp_all[t, :, 0] == 0) & (kp_all[t, :, 1] == 0)
        if np.any(mask_zeros):
            kp_all[t, mask_zeros] = kp_all[t-1, mask_zeros]

    T_orig = kp_all.shape[0]
    idxs = np.linspace(0, T_orig-1, TARGET_LEN)
    kp_interp = np.zeros((TARGET_LEN, 17, 3), dtype=np.float32)
    for k in range(17):
        for c in range(3):
            kp_interp[:, k, c] = np.interp(idxs, np.arange(T_orig), kp_all[:, k, c])
    
    kp_final = kp_interp[:, :, :2]

    hips = (kp_final[:, 11, :2] + kp_final[:, 12, :2]) / 2.0
    ankles = (kp_final[:, 15, :2] + kp_final[:, 16, :2]) / 2.0
    
    heights = np.linalg.norm(kp_final[:, 0, :2] - ankles, axis=1)
    valid_heights = heights[heights > 10]
    if len(valid_heights) > 0:
        avg_height = np.median(valid_heights)
    else:
        avg_height = 1.0

    center_seq = np.mean(hips, axis=0)
    
    kp_xy = (kp_final - center_seq) / (avg_height + 1e-6)
    vel = np.diff(kp_xy, axis=0, prepend=kp_xy[:1])

    final_tensor = np.concatenate([kp_xy, vel], axis=2).transpose(2, 0, 1)

    print("\n📊 STATISTICHE TENSORE (Input Rete Locale)")
    print(f"   Shape: {final_tensor.shape}")
    print(f"   MEAN Totale: {np.mean(final_tensor):.6f}")
    print(f"   STD  Totale: {np.std(final_tensor):.6f}")
    print(f"   MIN  Totale: {np.min(final_tensor):.6f}")
    print(f"   MAX  Totale: {np.max(final_tensor):.6f}")
    print("-" * 30)
    print("Primi 5 valori del canale 0 (Pos X):")
    print(final_tensor[0, :5, 0])
    print("-" * 30)
    print("👉 CONFRONTA QUESTI NUMERI CON L'OUTPUT DI KAGGLE!")

if __name__ == "__main__":
    process_video_locally()