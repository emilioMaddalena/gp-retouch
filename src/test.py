import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import GPy

# Step 1: Load a grayscale image
def load_grayscale_image():
    # Using a sample image from sklearn for demonstration
    china = load_sample_image("china.jpg")
    grayscale = np.mean(china, axis=2)  # Convert to grayscale by averaging color channels
    grayscale = grayscale[::10, ::10]  # Downsample for faster processing
    return grayscale

# Step 2: Delete some pixels (set them as missing values)
def delete_pixels(image, fraction_missing=0.3):
    mask = np.random.rand(*image.shape) > fraction_missing
    corrupted_image = np.where(mask, image, np.nan)
    return corrupted_image, mask

# Step 3: Reconstruct the image with Gaussian Processes
def reconstruct_image(image, mask):
    # Get coordinates of known pixels
    x_coords, y_coords = np.where(mask)
    known_coords = np.vstack((x_coords, y_coords)).T
    known_values = image[mask]

    # Define the grid for reconstruction
    x_grid, y_grid = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    grid_coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    # Fit Gaussian Process
    kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=5.0)
    gp = GPy.models.GPRegression(known_coords, known_values[:, None], kernel)
    gp.optimize(messages=True)

    # Predict values for all pixels
    mean_prediction, _ = gp.predict(grid_coords)
    reconstructed_image = mean_prediction.reshape(image.shape)

    return reconstructed_image

# Main script
if __name__ == "__main__":
    # Load and preprocess the image
    image = load_grayscale_image()
    corrupted_image, mask = delete_pixels(image)

    # Reconstruct the image
    reconstructed_image = reconstruct_image(corrupted_image, mask)

    # Plot the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Corrupted Image")
    plt.imshow(corrupted_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")

    plt.show()
