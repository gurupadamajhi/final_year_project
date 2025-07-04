import cv2
import numpy as np
import matplotlib.pyplot as plt

def arnold_cat_map(image, iterations):

    #Apply the Arnold Cat Map to an image for a specified number of iterations.
    
    h, w = image.shape[:2]
    if h != w:
        raise ValueError("Arnold Cat Map requires a square image.")

    transformed_image = image.copy()

    for _ in range(iterations):
        new_image = np.zeros_like(transformed_image)

        for y in range(h):
            for x in range(w):
                new_x = (x + y) % w
                new_y = (x + 2 * y) % h
                new_image[new_y, new_x] = transformed_image[y, x]

        transformed_image = new_image

    return transformed_image

def main():
    # Load an image
    image_path = "brain side.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Ensure the image is square
    size = min(image.shape[:2])
    image = cv2.resize(image, (size, size))

    # Convert to grayscale for simplicity
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Iterations to display
    iterations_to_display = [1, 3, 5, 7, 10]

    # Create a figure to show iterations
    plt.figure(figsize=(15, 3))

    for i, iteration in enumerate(iterations_to_display):
        transformed_image = arnold_cat_map(gray_image, iteration)

        plt.subplot(1, len(iterations_to_display), i + 1)
        plt.imshow(transformed_image, cmap='gray')
        plt.title(f"Iteration {iteration}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
