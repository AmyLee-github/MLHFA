import os
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def high_pass_filter(image, cutoff_ratio=6):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    cutoff = int(min(rows, cols) * (1 / cutoff_ratio))

    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    fshift = fftshift(fft2(image))
    fshift_filtered = fshift * mask
    f_ishift = ifftshift(fshift_filtered)
    img_back = np.abs(ifft2(f_ishift))

    return img_back

def process_images(input_dir, output_dir):
    error_log_path = os.path.join(output_dir, 'error_log.txt')
    total_files = sum(len(files) for _, _, files in os.walk(input_dir) if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) for file in files))
    processed_files = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)

                output_file = os.path.join(output_path, file.split('.')[0] + '_f.' + file.split('.')[1])
                
                if os.path.exists(output_file):
                    processed_files += 1
                    print(f"Skipping already processed file: {output_file}")
                    continue

                try:
                    img = Image.open(img_path).convert('L')
                    img_np = np.array(img)

                    img_filtered = high_pass_filter(img_np)

                    Image.fromarray(np.uint8(img_filtered)).save(output_file)

                    processed_files += 1
                    print(f"Processed {processed_files}/{total_files}: {file}")
                except Exception as e:
                    with open(error_log_path, 'a') as log_file:
                        processed_files += 1
                        log_file.write(f"Error processing {img_path}: {str(e)}\n")
                    print(f"Error processing {img_path}, logged to {error_log_path}")

    print("Processing complete.")

if __name__ == "__main__":
    input_directory = '/hexp/data/GenImage/genImage'
    output_directory = '/hexp/data/GenImage/genImage_filtered'
    process_images(input_directory, output_directory)