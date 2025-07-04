from typing import Counter
import cv2
import hashlib
import math
from skimage.metrics import mean_squared_error
from matplotlib import pyplot as plt
import numpy as np
#Conculations


def calculate_entropy(image_matrix):
    """
    Calculate the entropy of an image matrix.
    
    :param image_matrix: 2D numpy array representing the image matrix
    :return: Entropy value of the image matrix
    """
    # Flatten the image matrix into a 1D array
    flattened_image = image_matrix.flatten()

    # Count the occurrences of each pixel value
    pixel_counts = Counter(flattened_image)

    # Total number of pixels
    total_pixels = len(flattened_image)

    # Calculate entropy
    entropy = 0.0
    for count in pixel_counts.values():
        probability = count / total_pixels
        entropy -= probability * math.log2(probability)

    return entropy



def calculate_psnr(original, distorted):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
        original (ndarray): The reference image (grayscale or color).
        distorted (ndarray): The distorted image (same dimensions as original).
    
    Returns:
        float: The PSNR value in decibels (dB).
    """
    # Ensure inputs are numpy arrays
    original = np.array(original, dtype=np.uint8)
    distorted = np.array(distorted, dtype=np.uint8)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:  # Images are identical
        return float('inf')
    
    # Maximum possible pixel value
    max_pixel = 255.0  # For 8-bit images
   # psnr = 10 * np.log10((max_pixel ** 2) / mse)
    psnr = 20*np.log10(max_pixel)-10*np.log10(mse)
    return psnr





def plot_histogram(image_matrix, title="Histogram"):
    """
    Plot the histogram of an image matrix.
    
    :param image_matrix: 2D numpy array representing the image matrix
    :param title: Title of the histogram plot
    """
    # Flatten the image matrix into a 1D array
    flattened_image = image_matrix.flatten()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_image, bins=256, range=(0, 255), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.grid(True)
    plt.show()




def calculate_npcr(plaintext_image, ciphertext_image):
    
    # Ensure the images have the same dimensions
    if plaintext_image.shape != ciphertext_image.shape:
        raise ValueError("Plaintext image and ciphertext image must have the same dimensions.")
    
    # Get the dimensions of the image
    M, N = plaintext_image.shape[:2]
    
    # NPCR calculation
    C = np.zeros_like(plaintext_image, dtype=np.uint8)
    C[plaintext_image != ciphertext_image] = 1
    npcr = np.sum(C) / (M * N) * 100
    
    return npcr




def uaci(img1, img2):
    height, width = img1.shape
    value = 0
    for y in range(height):
        for x in range(width):
            value += abs(int(img1[y, x]) - int(img2[y, x]))

    value = value * 100 / (width * height * 255)
    return value



def calculate_local_entropy(image_matrix, block_size=4):
    height, width = image_matrix.shape
    local_entropies = []

    # Iterate through the image in non-overlapping blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract the block from the image
            block = image_matrix[y:y+block_size, x:x+block_size]

            # Calculate the entropy of the block
            block_entropy = calculate_entropy(block)
            local_entropies.append(block_entropy)

    # Calculate the average local entropy
    avg_local_entropy = np.mean(local_entropies)
    return avg_local_entropy



def calculate_ssim(image1, image2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    :param image1: First image (grayscale)
    :param image2: Second image (grayscale)
    :return: SSIM value
    """
    # Constants for SSIM formula
    K1, K2 = 0.01, 0.03
    L = 255  # Dynamic range for 8-bit images
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate mean (μ)
    mu_x = np.mean(image1)
    mu_y = np.mean(image2)

    # Calculate variance (σ^2) and covariance (σ_xy)
    sigma_x2 = np.var(image1)
    sigma_y2 = np.var(image2)
    sigma_xy = np.mean((image1 - mu_x) * (image2 - mu_y))

    # SSIM calculation
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_value = numerator / denominator

    return ssim_value



def calculate_test(gray_image, encrypted_image, decrypted_image,  block_size=64):
    entropy = calculate_entropy(encrypted_image)
    print(f"Global Entropy value: {entropy:.6f}")

    local_entropy = calculate_local_entropy(encrypted_image, block_size=block_size)
    print(f"Local Entropy value (block size {block_size}x{block_size}): {local_entropy:.6f}")

    npcr_value = calculate_npcr(gray_image, encrypted_image)
    uaci_value = uaci( encrypted_image , gray_image)
    print(f"NPCR value: {npcr_value:.4f}%")
    print(f"UACI value: {uaci_value:.4f}")

    plot_histogram(gray_image, title="Original Image Histogram")
    plot_histogram(encrypted_image, title="Encrypted Image Histogram")

    psnr_value1 = calculate_psnr(gray_image, encrypted_image)
    psnr_value2 = calculate_psnr(encrypted_image, decrypted_image)
    psnr_value3 = calculate_psnr(gray_image, decrypted_image)


    mse = mean_squared_error(gray_image, decrypted_image)

    print("MSE value : ",mse)

    # if mse == 0:

    
    #     print("PSNR between original and decrypted image: inf dB (perfect decryption)")
    # else:
    #     psnr_decryption = calculate_psnr(gray_image, decrypted_image)
    #     print(f"PSNR between original and decrypted image: {psnr_decryption} dB")

    # Calculate SSIM for each PRNG-encrypted and decrypted pair
    ssim1 = calculate_ssim(gray_image, encrypted_image)
    ssim2 = calculate_ssim(encrypted_image, decrypted_image)
    ssim3 = calculate_ssim(gray_image, decrypted_image)

    preference1 = psnr_value1 / ssim1 if ssim1 != 0 else float('inf')
    print(f"PSNR1: {psnr_value1}, SSIM1: {ssim1}, Preference1: {preference1}")

    preference2 = psnr_value2 / ssim2 if ssim2 != 0 else float('inf')
    print(f"PSNR2: {psnr_value2}, SSIM2: {ssim2}, Preference2: {preference2}")

    preference3 = psnr_value3 / ssim3 if ssim3 != 0 else float('inf')
    print(f"PSNR3: {psnr_value3}, SSIM3: {ssim3}, Preference3: {preference3}")
# You can then call this function to test
# calculate_test_with_local_entropy(gray_image, encrypted_image, decrypted_image, block_size=8)