import mediapipe as mp
import cv2
import numpy as np

def test_gpu():
    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    
    try:
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='NedoAi/models/holistic.task',
                delegate=mp.tasks.BaseOptions.Delegate.GPU
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=False
        )
        with HolisticLandmarker.create_from_options(options) as landmarker:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
            result = landmarker.detect_for_video(mp_image, 0)
            print("GPU Test Success!")
    except Exception as e:
        print(f"GPU Test Failed: {e}")

if __name__ == "__main__":
    test_gpu()
