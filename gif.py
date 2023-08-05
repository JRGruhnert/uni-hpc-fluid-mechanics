import imageio.v2 as imio
import os
import re

SOURCE_DIR = 'results/sliding_lid'
TARGET_DIR = 'results/sliding_lid/gifs/gif.gif'

def extract_numeric_part(file_name):
    # Extract the numeric part from the file name using regular expression
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else -1

# Get a list of all PNG files in the directory
png_files = [file_name for file_name in os.listdir(SOURCE_DIR) if file_name.endswith('.png')]

# Sort the PNG files based on the numeric part of their names
sorted_png_files = sorted(png_files, key=extract_numeric_part)

gif_images = []
for file_name in sorted_png_files:
    file_path = os.path.join(SOURCE_DIR, file_name)
    print(file_path)
    gif_images.append(imio.imread(file_path))


# Make it pause at the end so that the viewers can ponder
for _ in range(10):
    gif_images.append(imio.imread(file_path))



imio.mimsave(TARGET_DIR, gif_images,duration=5, loop=0)
