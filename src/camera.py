import cv2

# Open the default webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read one frame from the webcam
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Sign Language Recognition - Camera", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()