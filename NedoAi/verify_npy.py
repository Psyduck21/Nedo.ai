import numpy as np
import cv2
import os

def verify_sequence(npy_path):
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return
        
    # Load the sequence (Frames, 543, 3)
    data = np.load(npy_path)
    print(f"Loaded shape: {data.shape}")

    # Create a black canvas (800x800 for normalized view)
    for frame_idx in range(len(data)):
        canvas = np.zeros((800, 800, 3), dtype=np.uint8)
        landmarks = data[frame_idx]

        # Draw dots for each landmark
        for i, lm in enumerate(landmarks):
            # If coordinates are 0, skip (occluded)
            if np.all(lm == 0): continue

            # Convert normalized/hip-centric coords to pixel space for viewing
            # We add 400 to center it and multiply by 200 to scale it up
            x = int(lm[0] * 200 + 400)
            y = int(lm[1] * 200 + 400)

            # Color code: Face (White), Pose (Blue), Hands (Green/Red)
            color = (255, 255, 255)
            if 468 <= i < 501: color = (255, 0, 0) # Pose (OpenCV uses BGR, so 255,0,0 is Blue)
            if i >= 501: color = (0, 255, 0) # Hands

            cv2.circle(canvas, (x, y), 2, color, -1)

        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Nedo.ai Verification", canvas)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # We will pick the first dual person video from the processed sequences to verify.
    PROCESSED_DIR = os.path.join(BASE_DIR, "Dataset", "processed_sequences")
    
    # Just grab any PersonA npy to test
    npy_files = [f for f in os.listdir(PROCESSED_DIR) if "PersonA" in f]
    if npy_files:
        test_file = os.path.join(PROCESSED_DIR, npy_files[0])
        print(f"Verifying {test_file}")
        verify_sequence(test_file)
    else:
        print("No PersonA .npy files found in processed_sequences. Please extract first.")
