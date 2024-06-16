import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = './ssd_mobilenet_v2_coco_2018_03_29/saved_model'
model = tf.saved_model.load(model_path)

# Load the labels
category_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 7: 'train',
    8: 'truck',  # Add more classes as necessary
}

def detect_objects(image):
    # Prepare the input tensor
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

    # Convert input tensor to a format expected by the model
    input_tensor = tf.image.convert_image_dtype(input_tensor, tf.uint8)

    # Run inference
    detections = model.signatures['serving_default'](input_tensor)

    return detections

# Open the video file
video_path = './15 minutes of heavy traffic noise in India  14-08-2022.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video opened successfully. Dimensions: {width}x{height}, FPS: {fps}")

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    detections = detect_objects(frame)

    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            class_id = classes[i]
            if class_id in category_index:
                class_name = category_index[class_id]

                # Draw bounding box and label
                box = boxes[i]
                startY, startX, endY, endX = box
                startY, startX, endY, endX = int(startY * height), int(startX * width), int(endY * height), int(endX * width)

                # Draw bounding box and label based on class
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with detection boxes
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
