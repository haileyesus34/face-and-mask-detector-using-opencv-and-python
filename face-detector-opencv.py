import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect if a mask is present based on color
def is_mask_present(face_roi):
    # Convert the ROI (Region of Interest) to HSV color space
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting mask (e.g., blue or white mask)
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])

    # Create a mask for the color detection
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_blue, mask_white)

    # Check the percentage of the area covered by the mask color
    mask_ratio = cv2.countNonZero(mask) / (face_roi.size / 3)
    
    # If the mask color occupies more than a certain threshold of the face ROI, assume a mask is present
    return mask_ratio > 0.3

# Capture video from the webcam (or you can use a video file by providing the file path)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Extract the Region of Interest (ROI) for the face
        face_roi = frame[y:y+h, x:x+w]
        
        # Determine if a mask is present
        if is_mask_present(face_roi):
            label = "Mask"
            color = (0, 255, 0)  # Green for mask
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red for no mask
        
        # Draw rectangles around the face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the frame with the detected faces and mask/no mask labels
    cv2.imshow('Face and Mask Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
