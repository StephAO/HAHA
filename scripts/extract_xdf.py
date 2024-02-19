import os
import shutil


parent_dir = 'path/to/directory/containing/the/datafolders'
new_dir_name = 'Name_of_the_xdffolder'
new_dir_path = os.path.join(parent_dir, new_dir_name)


if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)


for root, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.endswith('.xdf'):

            file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_dir_path, file)


            if not os.path.exists(new_file_path):
                shutil.copy(file_path, new_dir_path)
            else:
                print(f"File {file} already exists in the destination directory.")

print("All .xdf files have been copied to the new directory.")
