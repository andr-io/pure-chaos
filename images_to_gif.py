import imageio.v2 as imageio
from PIL import Image
import os
import numpy as np


def create_gif_from_images(image_folder, output_gif, duration=0.5):
    # Get all image filenames from the folder and sort them
    images = []
    first_image = None

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # You can add other image formats if needed
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)

            # Store the size of the first image for resizing
            if first_image is None:
                first_image = img.size

            # Resize image to match the first image's size
            img_resized = img.resize(first_image)

            # Convert the resized image to a numpy array
            img_array = np.array(img_resized)

            # Append the numpy array to the images list
            images.append(img_array)

    # Create a GIF from the images
    imageio.mimsave(output_gif, images, duration=duration)
    print(f"GIF saved to {output_gif}")


if __name__ == "__main__":
    image_folder = "."  # Replace with your folder containing the images
    output_gif = "output.gif"  # Path to save the output GIF
    duration = 1000  # Duration for each frame in seconds (adjust as needed)

    # Call the function to create the GIF
    create_gif_from_images(image_folder, output_gif, duration)
