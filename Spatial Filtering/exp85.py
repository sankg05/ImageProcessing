import cv2
import numpy as np
import matplotlib.pyplot as plt 

def enhance_image_quality(image, alpha=1.5, sigma=1.0):
    """
    Enhance the quality of an image using Gaussian filters.

    Parameters:
        image (numpy.ndarray): Input image.
        alpha (float): Weighting factor for the enhanced image.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: Enhanced image.
    """
    # Step 1: Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)

    # Step 2: Subtract the smoothed image from the original
    subtracted_image = cv2.subtract(image, blurred_image)

    # Step 3: Add the result back to the original image
    enhanced_image = cv2.addWeighted(image, 1.0 + alpha, subtracted_image, -alpha, 0)

    # Step 4: Scale the enhanced image to maintain intensity levels
    enhanced_image = enhanced_image - (np.mean(enhanced_image) - np.mean(image))

    return enhanced_image

# Read the image
cimage = cv2.imread('img13.jpg', cv2.IMREAD_COLOR)

# Convert the image to float32 for processing
image = cimage.astype(np.float32)

# Split the image into channels (for color images)
b, g, r = cv2.split(image)

# Enhance the quality of each channel
enhanced_b = enhance_image_quality(b)
enhanced_g = enhance_image_quality(g)
enhanced_r = enhance_image_quality(r)

# Merge the enhanced channels back into an image
enhanced_image = cv2.merge([enhanced_b, enhanced_g, enhanced_r])

# Convert the result back to uint8 for display
enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

# Display the original and enhanced images using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axes[0].imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the enhanced image
axes[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Enhanced Image')
axes[1].axis('off')

plt.show()

psnr = cv2.PSNR(cimage, enhanced_image)
print('The PSNR value is: ', psnr)


# Display the original and enhanced images
# cv2.imshow('Original Image', image.astype(np.uint8))
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.subplot(1, 2, 1), plt.imshow(cimage, cmap='gray'), plt.title('Original Image')
# plt.subplot(1, 2, 2), plt.imshow(enhanced_image, cmap='gray'), plt.title('Filtered Image')
# plt.show()
