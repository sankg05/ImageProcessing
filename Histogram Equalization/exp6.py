import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def exposure_based_multi_histogram_equalization(image):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply histogram equalization to the L channel
    l_channel_eq = cv2.equalizeHist(l_channel)

    # Merge the equalized L channel with the original A and B channels
    lab_image_eq = cv2.merge([l_channel_eq, a_channel, b_channel])

    # Convert the LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_Lab2BGR)

    return enhanced_image

# Load your image
input_image = cv2.imread('img13.jpg')

# Apply the exposure-based multi-histogram equalization
enhanced_image = exposure_based_multi_histogram_equalization(input_image)

# Display the original and enhanced images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')

plt.show()

directory = r'C:\Sanika\spit\3rd Year\SEM V\FOSIP\Experiments'
filename = 'snip62A.jpg'
cv2.imwrite(filename, enhanced_image)
print('Saving complete...')

cv2.imshow('Original image', input_image)
cv2.imshow('Enhanced image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

psnr = cv2.PSNR(input_image, enhanced_image)
print('The PSNR value is: ', psnr)




