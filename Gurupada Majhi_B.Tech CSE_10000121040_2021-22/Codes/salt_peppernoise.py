import numpy as np
import cv2
from test_val import calculate_test


# Arnold Cat Map for forward and reverse transformations
def arnold_cat_map(block, iterations):
    N = block.shape[0]
    permuted_block = block.copy()
    
    for _ in range(iterations):
        temp_block = np.zeros_like(permuted_block)
        for i in range(N):
            for j in range(N):
                x_new = (i + j) % N
                y_new = (i + 2 * j) % N
                temp_block[x_new, y_new] = permuted_block[i, j]
        permuted_block = temp_block.copy()
    
    return permuted_block

def reverse_arnold_cat_map(block, iterations):
    N = block.shape[0]
    permuted_block = block.copy()
    
    for _ in range(iterations):
        temp_block = np.zeros_like(permuted_block)
        for i in range(N):
            for j in range(N):
                x_new = (2 * i - j) % N
                y_new = (-i + j) % N
                temp_block[x_new, y_new] = permuted_block[i, j]
        permuted_block = temp_block.copy()
    
    return permuted_block



# Function to add Salt and Pepper noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)

    # Add Salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Add Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image




# Load the grayscale medical image and divide it into blocks
image = cv2.imread('lungs.png', cv2.IMREAD_GRAYSCALE)
block_size = 16   
image = cv2.resize(image, (256, 256))  # Resize image to fit block division
cv2.imshow('Original_Image',image)


# Save noisy image for reference
cv2.imwrite('noisy_image.png', image)
n_blocks = (image.shape[0] // block_size, image.shape[1] // block_size)

# Divide image into blocks
blocks = [image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
         for i in range(n_blocks[0]) for j in range(n_blocks[1])]



# Set up PRNG seed using a PIN
pin = '123456'
base_seed = int.from_bytes(pin.encode(), 'little')


# Define PRNG classes with unique behaviors
class MRG:  
    def __init__(self, seed, variant):
        self.state = seed + variant
        self.modulus = 2**31 - 1
        self.multiplier = 16807 + variant
        self.increment = variant

    def randint(self, low, high):
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return low + int(self.state * (high - low) / self.modulus)
    

class MLFG:
    def __init__(self, seed, variant):
        self.state = [seed + variant]
        self.modulus = 2**32 - 1
        self.coefficient = 0x6C8E9CF7 + variant
        self.offset = 0xFACAC10B + variant

    def randint(self, low, high):
        next_state = (self.coefficient * self.state[-1] + self.offset) % self.modulus
        self.state.append(next_state)
        return low + int(next_state * (high - low) / self.modulus)


class MersenneTwister:        
    def __init__(self, seed, variant):
        self.prng = np.random.Generator(np.random.MT19937(seed + variant))

    def randint(self, low, high):
        return self.prng.integers(low, high)
    

class dSMFT:
    def __init__(self, seed, variant):
        self.prng = np.random.Generator(np.random.MT19937(seed + variant))

    def randint(self, low, high):
        return self.prng.integers(low, high)
    

# Initialize PRNGs
prng_mt = MersenneTwister(base_seed, 1)
prng_dsmft = dSMFT(base_seed, 2)
prng_mrg = MRG(base_seed, 3)
prng_mlfg = MLFG(base_seed, 4)

# Function to permute and substitute blocks
def permute_blocks(blocks, prng):
    permuted_blocks = []
    arnold_iterations = prng.randint(1, 10)
    
    for block in blocks:
        permuted_block = arnold_cat_map(block, arnold_iterations)
        permuted_blocks.append(permuted_block)
        
    permuted_indices = np.arange(len(blocks))
    return permuted_blocks, permuted_indices

def xor_substitution_block(block, prng):
    flat_block = block.flatten()
    random_bytes = np.array([prng.randint(0, 256) for _ in range(len(flat_block))], dtype=np.uint8)
    substituted_block = np.bitwise_xor(flat_block, random_bytes)
    return substituted_block.reshape(block.shape), random_bytes

# Encrypt and decrypt blocks
def encrypt_blocks(blocks, prng):
    permuted_blocks = []
    arnold_iterations = []
    random_keys = []
    
    for block in blocks:
        iterations = prng.randint(1, 10)
        arnold_iterations.append(iterations)
        permuted_block = arnold_cat_map(block, iterations)
        encrypted_block, random_key = xor_substitution_block(permuted_block, prng)
        permuted_blocks.append(encrypted_block)
        random_keys.append(random_key)
    
    return permuted_blocks, arnold_iterations, random_keys


def decrypt_blocks(blocks, prng, arnold_iterations, random_keys):
    decrypted_blocks = []
    
    for i, block in enumerate(blocks):
        decrypted_block = np.bitwise_xor(block.flatten(), random_keys[i]).reshape(block.shape)
        decrypted_block = reverse_arnold_cat_map(decrypted_block, arnold_iterations[i])
        decrypted_blocks.append(decrypted_block)
    
    return decrypted_blocks

# Reassemble image from blocks
def reassemble_image(blocks, n_blocks):
    block_size = blocks[0].shape[0]
    image = np.zeros((n_blocks[0] * block_size, n_blocks[1] * block_size), dtype=blocks[0].dtype)
    idx = 0
    for i in range(n_blocks[0]):
        for j in range(n_blocks[1]):
            image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = blocks[idx]
            idx += 1
    return image

# Encrypt image using each PRNG
encrypted_mt, arnold_iterations_mt, random_keys_mt = encrypt_blocks(blocks, prng_mt)
encrypted_dsmft, arnold_iterations_dsmft, random_keys_dsmft = encrypt_blocks(blocks, prng_dsmft)
encrypted_mrg, arnold_iterations_mrg, random_keys_mrg = encrypt_blocks(blocks, prng_mrg)
encrypted_mlfg, arnold_iterations_mlfg, random_keys_mlfg = encrypt_blocks(blocks, prng_mlfg)

# Reassemble encrypted images
encrypted_image_mt = reassemble_image(encrypted_mt, n_blocks)
encrypted_image_dsmft = reassemble_image(encrypted_dsmft, n_blocks)
encrypted_image_mrg = reassemble_image(encrypted_mrg, n_blocks)
encrypted_image_mlfg = reassemble_image(encrypted_mlfg, n_blocks)

# Add Salt and Pepper Noise
salt_prob =0.01  # Probability of salt noise
pepper_prob = 0.01 # Probability of pepper noise

noisy_image_mt = add_salt_and_pepper_noise(encrypted_image_mt, salt_prob, pepper_prob)
noisy_image_dsmft = add_salt_and_pepper_noise(encrypted_image_dsmft, salt_prob, pepper_prob)
noisy_image_mrg = add_salt_and_pepper_noise(encrypted_image_mrg, salt_prob, pepper_prob)
noisy_image_mlfg = add_salt_and_pepper_noise(encrypted_image_mlfg, salt_prob, pepper_prob)

# Save encrypted images
cv2.imwrite('encrypted_mt.png', noisy_image_mt)
cv2.imwrite('encrypted_dsmft.png', noisy_image_dsmft)
cv2.imwrite('encrypted_mrg.png', noisy_image_mrg)
cv2.imwrite('encrypted_mlfg.png', noisy_image_mlfg)

# Divide image into blocks
blocks_mt = [noisy_image_mt[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
         for i in range(n_blocks[0]) for j in range(n_blocks[1])]
blocks_dsmft = [noisy_image_dsmft[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
         for i in range(n_blocks[0]) for j in range(n_blocks[1])]
blocks_mrg = [noisy_image_mrg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
         for i in range(n_blocks[0]) for j in range(n_blocks[1])]
blocks_mlfg = [noisy_image_mlfg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
         for i in range(n_blocks[0]) for j in range(n_blocks[1])]

# Decrypt images
decrypted_mt = decrypt_blocks(blocks_mt,prng_mt,arnold_iterations_mt,random_keys_mt)
decrypted_dsmft = decrypt_blocks(blocks_dsmft, prng_dsmft, arnold_iterations_dsmft, random_keys_dsmft)
decrypted_mrg = decrypt_blocks(blocks_mrg, prng_mrg, arnold_iterations_mrg, random_keys_mrg)
decrypted_mlfg = decrypt_blocks(blocks_mlfg, prng_mlfg, arnold_iterations_mlfg, random_keys_mlfg)

# Reassemble decrypted images
decrypted_noise_image_mt = reassemble_image(decrypted_mt, n_blocks)
decrypted_noise_image_dsmft = reassemble_image(decrypted_dsmft, n_blocks)
decrypted_noise_image_mrg = reassemble_image(decrypted_mrg, n_blocks)
decrypted_noise_image_mlfg = reassemble_image(decrypted_mlfg, n_blocks)

# Save decrypted images
cv2.imwrite('decrypted_mt.png', decrypted_noise_image_mt)
cv2.imwrite('decrypted_dsmft.png', decrypted_noise_image_dsmft)
cv2.imwrite('decrypted_mrg.png', decrypted_noise_image_mrg)
cv2.imwrite('decrypted_mlfg.png', decrypted_noise_image_mlfg)

# Display encrypted and Decrypted images
cv2.imshow('Encrypted MT', noisy_image_mt)
cv2.imshow('Encrypted dSMFT', noisy_image_dsmft)
cv2.imshow('Encrypted MRG', noisy_image_mrg)
cv2.imshow('Encrypted MLFG', noisy_image_mlfg)

cv2.imshow('Decrypted MT', decrypted_noise_image_mt)
cv2.imshow('Decrypted dSMFT', decrypted_noise_image_dsmft)
cv2.imshow('Decrypted MRG', decrypted_noise_image_mrg)
cv2.imshow('Decrypted MLFG', decrypted_noise_image_mlfg)

# Call calculate_test with full images
calculate_test(gray_image=image, 
               encrypted_image=noisy_image_mt, 
               decrypted_image=decrypted_noise_image_mt)


calculate_test(gray_image=image, 
               encrypted_image=noisy_image_dsmft, 
               decrypted_image=decrypted_noise_image_dsmft)


calculate_test(gray_image=image, 
               encrypted_image=noisy_image_mrg, 
               decrypted_image=decrypted_noise_image_mrg)


calculate_test(gray_image=image, 
               encrypted_image=noisy_image_mlfg, 
               decrypted_image=decrypted_noise_image_mlfg)


cv2.waitKey(0)
cv2.destroyAllWindows()
