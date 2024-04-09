from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='custom_data.yaml', epochs=200, device='mps', batch=1)  # train the model