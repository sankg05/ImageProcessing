import cv2
import numpy as np
import skfuzzy as fuzz
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray, image

def apply_morphological_operations(image):
    # Use morphological operations to enhance structures in the image
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closing

def apply_fuzzy_logic(image):
    # Fuzzy logic-based thresholding
    image_flat = image.flatten()
    x = np.arange(0, 256, 1)

    # Fuzzify the image
    fuzzy_image = fuzz.trapmf(x, [0, 50, 100, 255])
    fuzzy_membership = fuzz.interp_membership(x, fuzzy_image, image_flat)

    # Threshold the image based on fuzzy membership
    thresholded_image = np.zeros_like(image_flat)
    thresholded_image[fuzzy_membership > 0.9] = 255

    # Reshape the thresholded image
    result_image = thresholded_image.reshape(image.shape)

    return result_image

# Example usage
image_path = "img3.jpg"
gray_image, original_image = preprocess_image(image_path)
morphological_image = apply_morphological_operations(gray_image)
segmented_image = apply_fuzzy_logic(morphological_image)

# Display the result using Matplotlib
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')

plt.show()

psnr = cv2.PSNR(original_image, segmented_image)
print('The PSNR value is: ', psnr)

