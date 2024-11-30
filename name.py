import os
from PIL import Image, ImageChops

# Paths for the original and augmented images
input_folder = "./dataset"
output_folder = "./final_set"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def get_next_number(base_name):
    """Find the next available number for the given base name in the output folder."""
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(base_name)]
    numbers = [int(f.rsplit("-", 1)[1].split(".")[0]) for f in existing_files]
    return max(numbers, default=0) + 1

def save_image(img, base_name, number):
    """Save the image with the correct name based on the naming convention."""
    new_name = f"{base_name}-{number}.jpg"
    img.save(os.path.join(output_folder, new_name))
    print(f"Saved: {new_name}")

def shift_image(img, x_offset, y_offset):
    """Shift the image slightly using translation."""
    return ImageChops.offset(img, x_offset, y_offset)

def augment_image(image_path, base_name):
    """Create augmented images including rotations and shifts."""
    with Image.open(image_path) as img:
        # Get the next available number for this base name
        next_number = get_next_number(base_name)

        # Save the original image
        save_image(img, base_name, next_number)
        
        # Create a slightly shifted version of the image
        next_number += 1
        shifted_img = shift_image(img, x_offset=10, y_offset=10)
        save_image(shifted_img, base_name, next_number)
        
        # Apply rotations to the original image
        for angle in [90, 180, 270]:
            next_number += 1
            rotated_img = img.rotate(angle, expand=True)
            save_image(rotated_img, base_name, next_number)

        # Apply rotations to the shifted image
        for angle in [90, 180, 270]:
            next_number += 1
            rotated_img = shifted_img.rotate(angle, expand=True)
            save_image(rotated_img, base_name, next_number)

# Process all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        base_name, _ = filename.rsplit("-", 1)  # Extract the base name
        image_path = os.path.join(input_folder, filename)
        augment_image(image_path, base_name)

print("Image augmentation complete!")
