import cv2
import matplotlib.pyplot as plt

# Step 1: Image Acquisition
image_path = 'image.jpg.jpg'
image = cv2.imread(image_path)

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()
import cv2
import matplotlib.pyplot as plt

# Step 1: Image Acquisition
image_path = 'image.jpg.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Step 2: Preprocessing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

    # Display the preprocessed image
    plt.imshow(normalized_image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()
import cv2
import matplotlib.pyplot as plt

# Assume 'normalized_image' is already defined from previous steps
# For example, normalized_image = cv2.normalize(...)

# Step 4: Image Segmentation
_, thresholded_image = cv2.threshold(normalized_image, 127, 255, cv2.THRESH_BINARY)

# Display the segmented image
plt.imshow(thresholded_image, cmap='gray')
plt.title("Segmented Image")
plt.axis('off')
plt.show()
import cv2
import matplotlib.pyplot as plt

# Step 1: Image Acquisition
image_path = 'image.jpg.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Step 2: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Step 4: Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Step 5: Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Step 6: Convert the image from BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 7: Display the result with detected faces
    plt.imshow(image_rgb)
    plt.title("Detected Faces")
    plt.axis('off')
    plt.show()
