import tkinter as tk
from tkinter import ttk
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define the predict function
def predict(frame, prev_boxes=None, alpha=0.7):
    image_tensor = transform(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Get bounding boxes from the prediction
    current_boxes = prediction['boxes'].cpu().numpy().astype(int)

    # Apply temporal smoothing
    if prev_boxes is not None and len(prev_boxes) > 0:
        smoothed_boxes = alpha * current_boxes + (1 - alpha) * prev_boxes
        smoothed_boxes = smoothed_boxes.astype(int)
        for box in smoothed_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    else:
        for box in current_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        smoothed_boxes = current_boxes

    return frame, smoothed_boxes

# Load the trained model
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # Background + Spaghetti
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('/content/drive/MyDrive/print fails.v1i.voc/train/spaghetti_detection_model.pth'))
model.eval()

# Define the transform to preprocess the image
transform = T.Compose([T.ToTensor()])

# Define input video and output video paths
input_video_path = "/content/drive/MyDrive/print fails.v1i.voc/3dprintfail.mp4"
output_video_path = "/content/drive/MyDrive/print fails.v1i.voc/detected_with_temporal_smoothing.mp4"

# Initialize VideoCapture object
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Define input_video() function
def input_video():
    ret, frame = cap.read()
    prev_boxes = None
    while ret:
        frame_with_boxes, prev_boxes = predict(frame, prev_boxes)
        out.write(frame_with_boxes)
        ret, frame = cap.read()

# Define input_camera_feed() function
def input_camera_feed():
    # Add your code to handle inputting camera feed here
    pass

# Create the main window
root = tk.Tk()
root.title("Print Probe")

# Set window size and position
window_width = 400
window_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (window_width / 2)
y = (screen_height / 2) - (window_height / 2)
root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')

# Create a frame for the title
title_frame = tk.Frame(root, bg="light blue", width=window_width, height=50)
title_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents
title_frame.pack(side=tk.TOP, fill=tk.X)

# Create label for the title
title_label = tk.Label(title_frame, text="Print Probe", font=("Helvetica", 20), bg="light blue")
title_label.pack(fill=tk.BOTH, expand=True)

# Create a frame for the buttons
button_frame = tk.Frame(root, width=window_width, height=window_height - 50)
button_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents
button_frame.pack(side=tk.TOP, fill=tk.BOTH)

# Create buttons with custom styles
button_style = ttk.Style()
button_style.configure("TButton", font=("Helvetica", 12), background="light gray", borderwidth=5, relief="raised", highlightthickness=5)

video_button = ttk.Button(button_frame, text="Input Video", command=input_video, style="TButton")
video_button.pack(pady=10, padx=20, ipadx=20, ipady=10)

camera_button = ttk.Button(button_frame, text="Input Camera Feed", command=input_camera_feed, style="TButton")
camera_button.pack(pady=10, padx=20, ipadx=20, ipady=10)

root.mainloop()
