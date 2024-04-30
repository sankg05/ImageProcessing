import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is not None:
        # Resize image to a consistent size (target_size)
        img = cv2.resize(img, target_size)
    return img

def generate_centroid_image(images):
    if not images:
        raise ValueError("No valid images found.")

    # Convert the list of images to a NumPy array
    images_array = np.array(images)

    # Calculate the mean without specifying dtype for higher precision
    centroid_image = np.mean(images_array, axis=0)

    # Normalize to [0, 255] and convert to uint8
    centroid_image = np.clip(centroid_image, 0, 255).astype(np.uint8)

    return centroid_image

def encode_image_with_centroid(image_path, centroid_image):
    img = load_and_resize_image(image_path, target_size=centroid_image.shape[:2][::-1])

    if img is not None:
        # Compute the difference between the image and the centroid
        diff_image = cv2.absdiff(centroid_image, img)

        # Save the difference image
        diff_filename = f'diff_{os.path.basename(image_path)}'
        cv2.imwrite(diff_filename, diff_image)
        print(f"Difference image saved as {diff_filename}")

        # Calculate PSNR
        psnr = cv2.PSNR(img, diff_image)
        print('The PSNR value is: ', psnr)


        # Print image sizes
        original_size = os.path.getsize(image_path)
        compressed_size = os.path.getsize(diff_filename)
        print(f"Original Image Size: {original_size} bytes")
        #print(f"Compressed Image Size: {compressed_size} bytes")

        # Calculate bit rate
        image_size_in_pixels = img.size
        bit_rate = (compressed_size) / image_size_in_pixels
        print('No of pixels:', image_size_in_pixels)
        print(f"Bit Rate: {bit_rate} bits per pixel")

        # Calculate compression ratio
        compression_ratio = (original_size - compressed_size ) * 100 / original_size
        print(f"Compression Ratio: {compression_ratio:.2f}")



        return psnr, bit_rate, compression_ratio

# Specify the target image path
image_path = r'C:\Sanika\spit\3rd Year\SEM V\FOSIP\Experiments\img13.jpg'  # Replace with your image path

# Load and resize the target image
target_image = load_and_resize_image(image_path)

# Generate centroid image using the target image
centroid_image = generate_centroid_image([target_image])

# Encode the target image with the centroid
psnr, bit_rate, compression_ratio = encode_image_with_centroid(image_path, centroid_image)

# Plot PSNR vs Bit Rate
plt.scatter(bit_rate, psnr, label='PSNR vs Bit Rate')
plt.title('PSNR vs Bit Rate')
plt.xlabel('Bit Rate (bits per pixel)')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()
