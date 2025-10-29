from ultralytics import YOLO
import torch # Import torch to check for GPU

if __name__ == '__main__': # Important for multiprocessing used by YOLO

    # Check if GPU is available, otherwise use CPU
    device_to_use = 0
    print(f"--- Starting Training ---")
    print(f"Using device: {device_to_use}")

    try:
        # 1. Load the base model (yolov8n.pt)
        #    Make sure yolov8n.pt is in the same folder or downloadable
        model = YOLO('yolov8n.pt')

        # 2. Train the model using your dataset configuration
        print("Starting model training...")
        results = model.train(
            data='license_plate_dataset.yaml', # Path to your dataset config
            epochs=50,
            imgsz=320,
            device=device_to_use,
            batch=16,
            name='yolov8n_license_plate_custom' # Name for the output folder
        )
        print("--- Training Finished ---")
        print(f"Results saved to: {results.save_dir}") # Print where the results are

    except Exception as e:
        print(f"\n--- An Error Occurred During Training ---")
        print(e)
        import traceback
        traceback.print_exc()

    print("Script finished.")