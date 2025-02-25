from ultralytics import YOLO

# Using a prebuilt YOLO model until a custom one is ready.
MODEL_PATH = 'yolov8n.pt'  # If your brother later provides a custom model, update this path.

model = YOLO(MODEL_PATH)

def detect_enemies(frame):
    results = model(frame)
    enemy_coords = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            enemy_coords.append(((x1 + x2) // 2, (y1 + y2) // 2))
    return enemy_coords
