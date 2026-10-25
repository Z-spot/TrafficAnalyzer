from ultralytics import YOLO

# 1. Load the model
model = YOLO("yolov8n.pt")

# 2. Run prediction to get Results objects
results = model("main.jpeg")

# 3. Use the plot() method to customize label font size and line thickness
# Set pil=True if font_size modification requires PIL rendering
annotated_frame = results[0].plot(
    font_size=10,  # Decrease or increase label text size (default is usually larger)
    line_width=2,  # Adjust bounding box line width if needed
    pil=True,  # Optional: use PIL if font_size doesn't scale properly with OpenCV ndarray
)

# 4. Save or display the result
annotated_frame.show()  # Display image
# annotated_frame.save("smaller_labels.jpg") # Save to disk
