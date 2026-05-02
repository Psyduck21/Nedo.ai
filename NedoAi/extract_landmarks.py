import os
# Suppress MediaPipe/TF C++ backend warnings
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_MIN_LEVEL'] = '3'

import sys
import cv2
import numpy as np
import mediapipe as mp
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "NedoAi/models/holistic.task")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
SINGLE_DIR = os.path.join(DATASET_DIR, "single_person")
DUAL_DIR = os.path.join(DATASET_DIR, "dual_person")
OUT_DIR = os.path.join(DATASET_DIR, "processed_sequences")

os.makedirs(OUT_DIR, exist_ok=True)

@contextmanager
def suppress_stderr():
    """Silences C++ stderr output directly at the OS level during a block."""
    fd = sys.stderr.fileno()
    def_pad = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(def_pad, fd)
        os.close(devnull)
        os.close(def_pad)

def get_landmarks_array(landmark_list, num_expected):
    """Converts MediaPipe landmark list to numpy array of shape (num_expected, 3). Zero-fills if missing."""
    if not landmark_list or len(landmark_list) == 0:
        return np.zeros((num_expected, 3), dtype=np.float32)
    
    # Handle both single list or list of lists
    lms = landmark_list[0] if isinstance(landmark_list[0], list) else landmark_list
    
    arr = np.zeros((num_expected, 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        if i >= num_expected: break
        arr[i] = [lm.x, lm.y, lm.z]
    return arr

def extract_from_frame(frame, landmarker, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    with suppress_stderr():
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Extract arrays
    face = get_landmarks_array(result.face_landmarks, 468)
    pose = get_landmarks_array(result.pose_landmarks, 33)
    lh = get_landmarks_array(result.left_hand_landmarks, 21)
    rh = get_landmarks_array(result.right_hand_landmarks, 21)
    
    # Combine: Face(468) + Pose(33) + LH(21) + RH(21) = 543
    all_landmarks = np.concatenate([face, pose, lh, rh], axis=0)
    
    # Hip-centric Normalization
    if result.pose_landmarks and len(result.pose_landmarks) > 0 and getattr(result, 'pose_world_landmarks', None) and len(result.pose_world_landmarks) > 0:
        lms = result.pose_landmarks[0] if isinstance(result.pose_landmarks[0], list) else result.pose_landmarks
        world_lms = result.pose_world_landmarks[0] if isinstance(result.pose_world_landmarks[0], list) else result.pose_world_landmarks
        if len(lms) >= 25 and len(world_lms) >= 25:
            left_hip = np.array([lms[23].x, lms[23].y, lms[23].z])
            right_hip = np.array([lms[24].x, lms[24].y, lms[24].z])
            
            # Calculate shoulder width using world landmarks for 3D consistency
            world_left_shoulder = np.array([world_lms[11].x, world_lms[11].y, world_lms[11].z])
            world_right_shoulder = np.array([world_lms[12].x, world_lms[12].y, world_lms[12].z])
            
            mid_hip = (left_hip + right_hip) / 2.0
            shoulder_width = np.linalg.norm(world_left_shoulder - world_right_shoulder)
            
            if shoulder_width > 0:
                # Apply only to valid (non-zero) landmarks
                non_zero_mask = np.any(all_landmarks != 0, axis=1)
                all_landmarks[non_zero_mask] = (all_landmarks[non_zero_mask] - mid_hip) / shoulder_width
                
    return all_landmarks

def process_video(video_path, out_path_prefix, is_dual=False):
    # CPU Delegation for stability with high-parallelism
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.VIDEO
    )
    
    # Create landmarker instances and suppress initialization noise
    with suppress_stderr():
        if is_dual:
            landmarker_a = HolisticLandmarker.create_from_options(options)
            landmarker_b = HolisticLandmarker.create_from_options(options)
            seq_a = []
            seq_b = []
        else:
            landmarker = HolisticLandmarker.create_from_options(options)
            seq = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        
        if is_dual:
            # Overlap crop to prevent hand occlusion at center-line (60% / 40% split)
            h, w, _ = frame.shape
            mid_left_end = int(w * 0.60)
            mid_right_start = int(w * 0.40)
            frame_a = frame[:, :mid_left_end]
            frame_b = frame[:, mid_right_start:]
            
            lms_a = extract_from_frame(frame_a, landmarker_a, timestamp_ms)
            lms_b = extract_from_frame(frame_b, landmarker_b, timestamp_ms)
            
            seq_a.append(lms_a)
            seq_b.append(lms_b)
        else:
            lms = extract_from_frame(frame, landmarker, timestamp_ms)
            seq.append(lms)
            
        frame_idx += 1
        
    cap.release()
    
    if is_dual:
        landmarker_a.close()
        landmarker_b.close()
        np.save(f"{out_path_prefix}_PersonA.npy", np.array(seq_a))
        np.save(f"{out_path_prefix}_PersonB.npy", np.array(seq_b))
    else:
        landmarker.close()
        np.save(f"{out_path_prefix}.npy", np.array(seq))

def main():
    # Gather all tasks
    tasks = []
    
    # Process Single Person
    if os.path.exists(SINGLE_DIR):
        single_videos = [f for f in os.listdir(SINGLE_DIR) if f.endswith('.mp4')]
        for video in single_videos:
            in_path = os.path.join(SINGLE_DIR, video)
            out_prefix = os.path.join(OUT_DIR, os.path.splitext(video)[0])
            tasks.append((in_path, out_prefix, False))
        
    # Process Dual Person
    if os.path.exists(DUAL_DIR):
        dual_videos = [f for f in os.listdir(DUAL_DIR) if f.endswith('.mp4')]
        for video in dual_videos:
            in_path = os.path.join(DUAL_DIR, video)
            out_prefix = os.path.join(OUT_DIR, os.path.splitext(video)[0])
            tasks.append((in_path, out_prefix, True))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No videos found to process.")
        return

    # Fixed 8 workers as requested
    max_workers = 8
    
    # Single clean print statement before the progress bar
    print(f"🚀 Initializing {max_workers} Workers for {total_tasks} videos...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Distribute tasks
        futures = {executor.submit(process_video, *task): task for task in tasks}
        
        # Single master progress bar
        with tqdm(total=total_tasks, desc="Processing Batch", unit="video") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    task = futures[future]
                    video_name = os.path.basename(task[0])
                    tqdm.write(f"❌ Skipping {video_name}: {e}")
                pbar.update(1)

    print("✅ Extraction Complete!")

if __name__ == "__main__":
    main()
