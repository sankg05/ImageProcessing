import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image
image = cv2.imread('img3.jpg', cv2.IMREAD_COLOR)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define lower and upper intensity thresholds
lower_threshold = 30
upper_threshold = 125

# Threshold the grayscale image to create a binary mask for the desired intensity range
binary_mask = cv2.inRange(gray_image, lower_threshold, upper_threshold)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process the image if contours are found
if contours:
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    intensity_range_mask = np.zeros_like(binary_mask)
    cv2.drawContours(intensity_range_mask, [largest_contour], 0, (255), thickness=cv2.FILLED)

    # Create a black background
    black_background = np.zeros_like(image)

    # Create a result image with parts within the intensity range only
    result_image = np.where(intensity_range_mask[:, :, None].astype(bool), image, black_background)

    # Split the result image into channels
    b, g, r = cv2.split(result_image)

    # Apply histogram equalization to each channel
    enhanced_b = cv2.equalizeHist(b)
    enhanced_g = cv2.equalizeHist(g)
    enhanced_r = cv2.equalizeHist(r)

    # Merge the enhanced channels back into an image
    enhanced_image = cv2.merge((enhanced_b, enhanced_g, enhanced_r))

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    clahe_r = clahe.apply(r)
    clahe_g = clahe.apply(g)
    clahe_b = clahe.apply(b)

    # Merge the CLAHE enhanced channels back into an image
    clahe_image = cv2.merge((clahe_r, clahe_g, clahe_b))

    # Segmentation: Draw contours on a black background
    segmented_image = np.zeros_like(image)
    cv2.drawContours(segmented_image, [largest_contour], 0, (0, 255, 0), thickness=2)

    # Display the original image, enhanced image, CLAHE image, and segmented image side by side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('CLAHE Image')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Segmented Image (Intensity Range)')
    axes[3].axis('off')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    psnr = cv2.PSNR(image, enhanced_image)
    print('The PSNR value is: ', psnr)

