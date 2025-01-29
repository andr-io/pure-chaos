from PIL import Image
import imageio


def crop_gif(input_gif, output_gif, left_crop, right_crop):
    # Read the GIF
    gif = imageio.mimread(input_gif)

    # Prepare a list to hold the cropped frames
    cropped_frames = []

    for frame in gif:
        # Convert frame to PIL Image for cropping
        pil_frame = Image.fromarray(frame)

        # Calculate the crop box (left, top, right, bottom)
        width, height = pil_frame.size
        crop_box = (left_crop, 0, width - right_crop, height)

        # Crop the image and append to the list
        cropped_frame = pil_frame.crop(crop_box)
        cropped_frames.append(cropped_frame)

    # Save the cropped GIF
    cropped_frames[0].save(output_gif, save_all=True, append_images=cropped_frames[1:], loop=0, duration=100)


# Example usage
input_gif = 'in.gif'  # Path to your input GIF
output_gif = 'cropped_output.gif'  # Path to save the cropped GIF
left_crop = 50  # Amount to crop from the left
right_crop = 50  # Amount to crop from the right

crop_gif(input_gif, output_gif, left_crop, right_crop)
