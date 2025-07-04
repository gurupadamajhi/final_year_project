import numpy as np
import cv2
from matplotlib import pyplot as plt
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

# Apply Cropping Attack
def crop_image(image, top_left, bottom_right):
    """Simulate a cropping attack by removing a portion of the image."""
    cropped_image = image.copy()
    cropped_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0  # Fill cropped area with black
    return cropped_image

# Load the grayscale medical image and divide it into blocks
image = cv2.imread('lungs.png', cv2.IMREAD_GRAYSCALE)
block_size = 16   
image = cv2.resize(image, (256, 256))  # Resize image to fit block division
n_blocks = (image.shape[0] // block_size, image.shape[1] // block_size)
# cv2.imshow('original_image', image)


# Divide image into blocks
blocks = [image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] 
          for i in range(n_blocks[0]) for j in range(n_blocks[1])]

# Select secret PIN as a seed for PRNGs
pin = '123456'
base_seed = int.from_bytes(pin.encode(), 'little')

# Define different PRNG classes with unique behavior
class MRG:  
    def __init__(self, seed, variant):
        self.state = seed + variant
        self.modulus = 2**31 - 1
        self.multiplier = 16807 + variant  # Slight variation for uniqueness
        self.increment = variant  # Small variant

    def randint(self, low, high):
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return low + int(self.state * (high - low) / self.modulus)

    def shuffle(self, x):
        np.random.shuffle(x)

class MLFG:
    def __init__(self, seed, variant):
        self.state = [seed + variant]
        self.modulus = 2**32 - 1
        self.coefficient = 0x6C8E9CF7 + variant  # Slight variation for uniqueness
        self.offset = 0xFACAC10B + variant  # Small variant

    def randint(self, low, high):
        next_state = (self.coefficient * self.state[-1] + self.offset) % self.modulus
        self.state.append(next_state)
        return low + int(next_state * (high - low) / self.modulus)

    def shuffle(self, x):
        np.random.shuffle(x)

class MersenneTwister:        
    def __init__(self, seed, variant):
        self.prng = np.random.Generator(np.random.MT19937(seed + variant))  # Slight variation for uniqueness

    def randint(self, low, high):
        return self.prng.integers(low, high)

    def shuffle(self, x):
        self.prng.shuffle(x)

class dSMFT:
    def __init__(self, seed, variant):
        self.prng = np.random.Generator(np.random.MT19937(seed + variant))  

    def randint(self, low, high):
        return self.prng.integers(low, high)

    def shuffle(self, x):
        self.prng.shuffle(x)

# Initialize PRNGs
prng_mt = MersenneTwister(base_seed, 1)  
prng_dsmft = dSMFT(base_seed, 2)  
prng_mrg = MRG(base_seed, 3)  
prng_mlfg = MLFG(base_seed, 4)

# Function to permute blocks using PRNG (Block Transposition)
def permute_blocks(blocks, prng):
    n_blocks = len(blocks)
    permuted_indices = np.arange(n_blocks)
    prng.shuffle(permuted_indices)  # Shuffle block indices using PRNG
    permuted_blocks = [blocks[i] for i in permuted_indices]
    return permuted_blocks, permuted_indices

# Function to reverse permute blocksmn"""
def reverse_permute_blocks(blocks, permuted_indices):
    inverse_indices = np.argsort(permuted_indices)
    restored_blocks = [blocks[i] for i in inverse_indices]
    return restored_blocks

# Function to apply XOR substitution (Substitution XORing) on blocks
def xor_substitution_block(block, prng):
    flat_block = block.flatten()
    random_bytes = np.array([prng.randint(0, 256) for _ in range(len(flat_block))], dtype=np.uint8)
    substituted_block = np.bitwise_xor(flat_block, random_bytes)
    return substituted_block.reshape(block.shape), random_bytes

# Encrypt blocks using Transposition + XOR
def encrypt_blocks(blocks, prng):
    # Permute blocks
    permuted_blocks, permuted_indices = permute_blocks(blocks, prng)
    
    # Apply XOR substitution for each block
    encrypted_blocks = []
    random_keys = []
    for block in permuted_blocks:
        encrypted_block, random_key = xor_substitution_block(block, prng)
        encrypted_blocks.append(encrypted_block)
        random_keys.append(random_key)
    
    return encrypted_blocks, permuted_indices, random_keys

# Decrypt blocks using XOR + Reverse Permutation
def decrypt_blocks(blocks, prng, permuted_indices, random_keys):
    decrypted_blocks = []
    
    # Reverse XOR substitution
    for i, block in enumerate(blocks):
        decrypted_block = np.bitwise_xor(block.flatten(), random_keys[i]).reshape(block.shape)
        decrypted_blocks.append(decrypted_block)
    
    # Reverse permutation
    restored_blocks = reverse_permute_blocks(decrypted_blocks, permuted_indices)
    
    return restored_blocks

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
encrypted_mt, permuted_indices_mt, random_keys_mt = encrypt_blocks(blocks, prng_mt)
encrypted_dsmft, permuted_indices_dsmft, random_keys_dsmft = encrypt_blocks(blocks, prng_dsmft)
encrypted_mrg, permuted_indices_mrg, random_keys_mrg = encrypt_blocks(blocks, prng_mrg)
encrypted_mlfg, permuted_indices_mlfg, random_keys_mlfg = encrypt_blocks(blocks, prng_mlfg)

print(image,prng_mt,)
print(image,prng_dsmft)
print(image,prng_mrg)
print(image,prng_mlfg)

# Reassemble encrypted images
encrypted_image_mt = reassemble_image(encrypted_mt, n_blocks)
encrypted_image_dsmft = reassemble_image(encrypted_dsmft, n_blocks)
encrypted_image_mrg = reassemble_image(encrypted_mrg, n_blocks)
encrypted_image_mlfg = reassemble_image(encrypted_mlfg, n_blocks)

encrypted_image_mt = arnold_cat_map(encrypted_image_mt,20)
encrypted_image_dsmft = arnold_cat_map(encrypted_image_dsmft,20)
encrypted_image_mrg = arnold_cat_map(encrypted_image_mrg,20)
encrypted_image_mlfg = arnold_cat_map(encrypted_image_mlfg,20)

# Define cropping area
top_left = (0, 0)  # Start of the cropping area
bottom_right = (128, 128)  # End of the cropping area

encrypted_image_mt = crop_image(encrypted_image_mt, top_left, bottom_right)
encrypted_image_dsmft = crop_image(encrypted_image_dsmft, top_left, bottom_right)
encrypted_image_mrg = crop_image(encrypted_image_mrg, top_left, bottom_right)
encrypted_image_mlfg = crop_image(encrypted_image_mlfg, top_left, bottom_right)


#Display encrypted and decrypted images
cv2.imshow('Encrypted MT', encrypted_image_mt)
cv2.imshow('Encrypted dSMFT', encrypted_image_dsmft)
cv2.imshow('Encrypted MRG', encrypted_image_mrg)
cv2.imshow('Encrypted MLFG', encrypted_image_mlfg)

# plt.figure(figsize=(12, 4))


# plt.subplot(1, 4, 1)
# plt.imshow(encrypted_image_mt, cmap='gray')
# plt.title('Encrypted MT')
# plt.axis('off')

# plt.subplot(1, 4, 2)
# plt.imshow(encrypted_image_dsmft, cmap='gray')
# plt.title('Encrypted dSMFT')
# plt.axis('off')


# plt.subplot(1, 4, 3)
# plt.imshow(encrypted_image_mrg, cmap='gray')
# plt.title('Encrypted MRG')
# plt.axis('off')


# plt.subplot(1, 4, 4)
# plt.imshow(encrypted_image_mlfg, cmap='gray')
# plt.title('Encrypted MLFG')
# plt.axis('off')

# plt.show()

# Save encrypted images
cv2.imwrite('encrypted_cropped_mt.png', encrypted_image_mt)
cv2.imwrite('encrypted_cropped_dsmft.png', encrypted_image_dsmft)
cv2.imwrite('encrypted_cropped_mrg.png', encrypted_image_mrg)
cv2.imwrite('encrypted_cropped_mlfg.png', encrypted_image_mlfg)

encrypted_image_mt = reverse_arnold_cat_map(encrypted_image_mt,20)
encrypted_image_dsmft = reverse_arnold_cat_map(encrypted_image_dsmft,20)
encrypted_image_mrg = reverse_arnold_cat_map(encrypted_image_mrg,20)
encrypted_image_mlfg = reverse_arnold_cat_map(encrypted_image_mlfg,20)

encrypted_mt_blocks = [encrypted_image_mt[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] 
          for i in range(n_blocks[0]) for j in range(n_blocks[1])]
encrypted_dsmft_blocks = [encrypted_image_dsmft[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] 
          for i in range(n_blocks[0]) for j in range(n_blocks[1])]
encrypted_mrg_blocks = [encrypted_image_mrg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] 
          for i in range(n_blocks[0]) for j in range(n_blocks[1])]
encrypted_mlfg_blocks = [encrypted_image_mlfg[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] 
          for i in range(n_blocks[0]) for j in range(n_blocks[1])]



# Decrypt images
decrypted_mt = decrypt_blocks(encrypted_mt_blocks, prng_mt, permuted_indices_mt, random_keys_mt)
decrypted_dsmft = decrypt_blocks(encrypted_dsmft_blocks, prng_dsmft, permuted_indices_dsmft, random_keys_dsmft)
decrypted_mrg = decrypt_blocks(encrypted_mrg_blocks, prng_mrg, permuted_indices_mrg, random_keys_mrg)
decrypted_mlfg = decrypt_blocks(encrypted_mlfg_blocks, prng_mlfg, permuted_indices_mlfg, random_keys_mlfg)

# Reassemble decrypted images
decrypted_image_mt = reassemble_image(decrypted_mt, n_blocks)
decrypted_image_dsmft = reassemble_image(decrypted_dsmft, n_blocks)
decrypted_image_mrg = reassemble_image(decrypted_mrg, n_blocks)
decrypted_image_mlfg = reassemble_image(decrypted_mlfg, n_blocks)

# Save decrypted images
cv2.imwrite('decrypted_mt.png', decrypted_image_mt)
cv2.imwrite('decrypted_dsmft.png', decrypted_image_dsmft)
cv2.imwrite('decrypted_mrg.png', decrypted_image_mrg)
cv2.imwrite('decrypted_mlfg.png', decrypted_image_mlfg)


# plt.figure(figsize=(12, 4))


# plt.subplot(1, 4, 1)
# plt.imshow(decrypted_image_mt, cmap='gray')
# plt.title('Decrypted MT')
# plt.axis('off')

# plt.subplot(1, 4, 2)
# plt.imshow(decrypted_image_dsmft, cmap='gray')
# plt.title('Decrypted dSMFT')
# plt.axis('off')


# plt.subplot(1, 4, 3)
# plt.imshow(decrypted_image_mrg, cmap='gray')
# plt.title('Decrypted MRG')
# plt.axis('off')


# plt.subplot(1, 4, 4)
# plt.imshow(decrypted_image_mlfg, cmap='gray')
# plt.title('Decrypted MLFG')
# plt.axis('off')

cv2.imshow('Decrypted MT', decrypted_image_mt)
cv2.imshow('Decrypted dSMFT', decrypted_image_dsmft)
cv2.imshow('Decrypted MRG', decrypted_image_mrg)
cv2.imshow('Decrypted MLFG', decrypted_image_mlfg)


# Reassemble encrypted and decrypted images for calculation
encrypted_image_mt_full = reassemble_image(encrypted_mt, n_blocks)
decrypted_image_mt_full = reassemble_image(decrypted_mt, n_blocks)

encrypted_image_dsmft_full= reassemble_image(encrypted_dsmft,n_blocks)
decrypted_image_dsmft_full = reassemble_image(decrypted_dsmft,n_blocks)

encrypted_image_mrg_full= reassemble_image(encrypted_mrg,n_blocks)
decrypted_image_mrg_full = reassemble_image(decrypted_mrg,n_blocks)

encrypted_image_mlfg_full= reassemble_image(encrypted_mlfg,n_blocks)
decrypted_image_mlfg_full = reassemble_image(decrypted_mlfg,n_blocks)



# Call calculate_test with full images
calculate_test(gray_image=image, 
               encrypted_image=encrypted_image_mt_full, 
               decrypted_image=decrypted_image_mt_full)


calculate_test(gray_image=image, 
               encrypted_image=encrypted_image_dsmft_full, 
               decrypted_image=decrypted_image_dsmft_full)


calculate_test(gray_image=image, 
               encrypted_image=encrypted_image_mrg_full, 
               decrypted_image=decrypted_image_mrg_full)


calculate_test(gray_image=image, 
               encrypted_image=encrypted_image_mlfg_full, 
               decrypted_image=decrypted_image_mlfg_full)


cv2.waitKey(0)
cv2.destroyAllWindows()
