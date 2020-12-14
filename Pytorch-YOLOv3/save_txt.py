import os

path = "C:/Users/IVCL/Desktop/crane/PyTorch-YOLOv3/data/custom"
val_path = "C:/Users/IVCL/Desktop/crane/real_image"
image_path = "data/custom/images"
val_images = os.listdir(val_path)
os.chdir(path)

with open("valid.txt", 'a') as file: # append lines to file
        for image in val_images:
            new_line = image_path + "/" + image + "\n"
            file.write(new_line)