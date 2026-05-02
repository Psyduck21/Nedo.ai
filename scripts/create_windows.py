import os
import numpy as np
from tqdm import tqdm

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(BASE_DIR, "Dataset", "processed_sequences")
OUT_DIR = os.path.join(BASE_DIR, "Dataset", "training_chunks")

# Hyperparameters
WINDOW_SIZE = 60
STEP_SIZE = 30
MAX_ZERO_FRAMES_RATIO = 0.20 # Max 20% zero-filled frames

def create_windows():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(IN_DIR):
        print(f"Input directory does not exist: {IN_DIR}")
        return
        
    files = [f for f in os.listdir(IN_DIR) if f.endswith('.npy')]
    
    if not files:
        print("No .npy files found to process.")
        return
        
    print(f"Found {len(files)} files. Starting sliding window extraction...")
    
    total_windows_created = 0
    total_windows_discarded = 0
    
    for filename in tqdm(files, desc="Processing Files"):
        in_path = os.path.join(IN_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # Load with mmap_mode to prevent loading the entire array into RAM
            data = np.load(in_path, mmap_mode='r')
            num_frames = data.shape[0]
            
            # Skip if file has fewer frames than WINDOW_SIZE
            if num_frames < WINDOW_SIZE:
                continue
                
            # Iterate with sliding window
            win_index = 0
            for start_idx in range(0, num_frames - WINDOW_SIZE + 1, STEP_SIZE):
                end_idx = start_idx + WINDOW_SIZE
                window = data[start_idx:end_idx] # shape: (60, 543, 3)
                
                # Check for zero-filled frames
                # A frame is completely zero if all its values are 0
                # We check the sum of absolute values or use np.all
                # sum over axis 1 and 2: if sum is 0, frame is zero-filled
                # To be safer with floats, we check if all values are 0
                frame_is_zero = np.all(window == 0, axis=(1, 2))
                num_zero_frames = np.sum(frame_is_zero)
                
                if num_zero_frames / WINDOW_SIZE > MAX_ZERO_FRAMES_RATIO:
                    total_windows_discarded += 1
                    continue
                    
                # Save valid window
                # We copy it into a new array to save it properly without mmap links
                out_path = os.path.join(OUT_DIR, f"{base_name}_win{win_index}.npy")
                np.save(out_path, np.array(window))
                
                win_index += 1
                total_windows_created += 1
                
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            
    print("\n--- Summary ---")
    print(f"Windows created: {total_windows_created}")
    print(f"Windows discarded (too many occlusions): {total_windows_discarded}")

if __name__ == "__main__":
    create_windows()
