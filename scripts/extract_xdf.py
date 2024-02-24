import os
import shutil

import os
import zipfile

## unzip the folders
# def unzip_all_in_folder(root_folder):
    
#     for root, dirs, files in os.walk(root_folder):
#         for file in files:
#             if file.endswith('.zip'):
                
#                 zip_path = os.path.join(root, file)
                
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    
#                     zip_ref.extractall(root)
#                 print(f"Extracted {file} in {root}")

# root_folder = '/home/stephane/HAHA/eye_data'
# unzip_all_in_folder(root_folder)

## extract all .xdf files

# parent_dir = '/home/stephane/HAHA/eye_data/Data'
# new_dir_name = 'xdf_files'
# new_dir_path = os.path.join(parent_dir, new_dir_name)


# if not os.path.exists(new_dir_path):
#     os.makedirs(new_dir_path)


# for root, dirs, files in os.walk(parent_dir):
#     for file in files:
#         if file.endswith('.xdf'):

#             file_path = os.path.join(root, file)
#             new_file_path = os.path.join(new_dir_path, file)


#             if not os.path.exists(new_file_path):
#                 shutil.copy(file_path, new_dir_path)
#             else:
#                 print(f"File {file} already exists in the destination directory.")

# print("All .xdf files have been copied to the new directory.")



######## The below code counts the number of .xdf files.



# parent_dir = '/home/stephane/HAHA/eye_data/Data'
# new_dir_name = 'xdf_files'
# new_dir_path = os.path.join(parent_dir, new_dir_name)

# # Check if the new directory already exists
# if os.path.exists(new_dir_path):
#     xdf_file_count = 0  # Initialize the counter for .xdf files
    
#     # Only count the .xdf files in the new directory
#     for file in os.listdir(new_dir_path):
#         if file.endswith('.xdf'):
#             xdf_file_count += 1
    
#     # Display the total count of .xdf files found in the new directory
#     print(f"Total number of .xdf files in '{new_dir_name}': {xdf_file_count}")
# else:
#     print(f"The directory '{new_dir_name}' does not exist. No .xdf file count can be returned.")




import pandas as pd





# rows_above_30_percent = data[data['Percent Loss'] > 0.40].shape[0]
# print(f"Number of rows with 'Percent Loss' above 40%: {rows_above_30_percent}")


# unique_participants_above_30_percent = data[data['Percent Loss'] > 0.40]['Participant'].nunique()
# print(f"Number of unique participants with 'Percent Loss' above 40%: {unique_participants_above_30_percent}")


# unique_participants_list = data[data['Percent Loss'] > 0.40]['Participant'].unique()
# print("Unique participants with 'Percent Loss' above 40%:")
# for participant in unique_participants_list:
#     print(participant)


# print(data['Participant'].nunique())




file_path = '/home/stephane/HAHA/eye_data/DropoutMetrics_EachTrial.csv' 
data = pd.read_csv(file_path)

filtered_data = data[data['Percent Loss'] > 50]


participant_trials = filtered_data.groupby('Participant')['Trial'].apply(list)


print("Participant and their corresponding Trial IDs with 'Percent Loss' above 40%:")
for participant, trials in participant_trials.items():
    print(f"{participant}: {trials}")

print(len(participant_trials))
