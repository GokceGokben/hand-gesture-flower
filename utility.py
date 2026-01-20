from PIL import Image
import os

def extract_frames(gif_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with Image.open(gif_path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            # Kareyi şeffaflığı koruyarak kaydet
            frame = im.convert("RGBA")
            frame.save(f"{output_folder}/frame_{i:03d}.png")
    print(f"{im.n_frames} kare başarıyla çıkarıldı!")

# Kullanımı:
extract_frames("flower.gif", "frames")