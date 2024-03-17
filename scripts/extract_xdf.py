import os
import shutil

# import os
# import zipfile

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




# import pandas as pd





# rows_above_30_percent = data[data['Percent Loss'] > 0.40].shape[0]
# print(f"Number of rows with 'Percent Loss' above 40%: {rows_above_30_percent}")


# unique_participants_above_30_percent = data[data['Percent Loss'] > 0.40]['Participant'].nunique()
# print(f"Number of unique participants with 'Percent Loss' above 40%: {unique_participants_above_30_percent}")


# unique_participants_list = data[data['Percent Loss'] > 0.40]['Participant'].unique()
# print("Unique participants with 'Percent Loss' above 40%:")
# for participant in unique_participants_list:
#     print(participant)


# print(data['Participant'].nunique())




# file_path = '/home/stephane/HAHA/eye_data/DropoutMetrics_EachTrial.csv' 
# data = pd.read_csv(file_path)

# filtered_data = data[data['Percent Loss'] > 50]


# participant_trials = filtered_data.groupby('Participant')['Trial'].apply(list)


# print("Participant and their corresponding Trial IDs with 'Percent Loss' above 40%:")
# for participant, trials in participant_trials.items():
#     print(f"{participant}: {trials}")

# print(len(participant_trials))

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV files into DataFrames
# df_with_gaze = pd.read_csv('test_metrics_gd+eg_q3_3e-05_128.csv')
# df_without_gaze = pd.read_csv('test_metrics_gd_q3_3e-05_128.csv')

# # Ensure 'Epoch' and 'Validation F1' are of appropriate data types for both DataFrames
# df_with_gaze['Epoch'] = df_with_gaze['Epoch'].astype(int)
# df_with_gaze['Validation F1'] = df_with_gaze['Validation F1'].astype(float)

# df_without_gaze['Epoch'] = df_without_gaze['Epoch'].astype(int)
# df_without_gaze['Validation F1'] = df_without_gaze['Validation F1'].astype(float)

# # Group the data by 'Epoch' and calculate the mean 'Validation F1' for each epoch in both datasets
# average_val_f1_by_epoch_with_gaze = df_with_gaze.groupby('Epoch')['Validation F1'].mean().reset_index()
# average_val_f1_by_epoch_without_gaze = df_without_gaze.groupby('Epoch')['Validation F1'].mean().reset_index()

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(average_val_f1_by_epoch_with_gaze['Epoch'], average_val_f1_by_epoch_with_gaze['Validation F1'], marker='o', label='With Eye Gaze')
# plt.plot(average_val_f1_by_epoch_without_gaze['Epoch'], average_val_f1_by_epoch_without_gaze['Validation F1'], marker='o', label='Without Eye Gaze')

# plt.title('Average Validation F1 Score per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Average Validation F1 Score')
# plt.legend()
# plt.grid(True)

# plt.savefig('plot_q3_test.png')  # Save the figure as an image file
# plt.show()  # Show the plot as well
# plt.close() 


##########################################################################

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df_with_gaze = pd.read_csv('/home/stephane/HAHA/scripts/gd+eg metrics/test_metrics_gd+eg_q3_3e-05_128.csv')
df_without_gaze = pd.read_csv('/home/stephane/HAHA/scripts/gd_metrics/test_metrics_gd_q3_3e-05_128.csv')
# df_just_gaze = pd.read_csv('/home/stephane/HAHA/scripts/eg_metrics/test_metrics_eg_q3_3e-05_128.csv')
df_just_gaze = pd.read_csv('/home/stephane/HAHA/scripts/eg_metrics/test_metrics_eg_q3_3e-05_128.csv')
df_majority_f1 = pd.read_csv('q3_majority_class_f1_per_timestep.csv')
df_MLP_f1_ceg = pd.read_csv('mlp_testing_detailed_results_ceg_q3.csv')
df_MLP_f1_go = pd.read_csv('mlp_testing_detailed_results_go_q3.csv')
# Assuming 'F1_score' needs to be averaged across timesteps and agents for each layout.
# Group by 'Layout' and calculate the mean 'F1_score' for each layout in both datasets.

avg_f1_with_gaze = df_with_gaze.groupby('Timestep')['F1_score'].mean().reset_index()
avg_f1_without_gaze = df_without_gaze.groupby('Timestep')['F1_score'].mean().reset_index()
avg_f1_with_just_gaze = df_just_gaze.groupby('Timestep')['F1_score'].mean().reset_index()
avg_f1_majority = df_majority_f1.groupby('timestep')['majority_class_f1'].mean().reset_index()
avg_f1_MLP_ceg = df_MLP_f1_ceg['Test F1 Score'].mean()
avg_f1_MLP_go = df_MLP_f1_go['Test F1 Score'].mean()


# Now plot the data
plt.figure(figsize=(10, 6))



plt.plot(avg_f1_with_gaze['Timestep'], avg_f1_with_gaze['F1_score'], marker='o', label='With Eye Gaze')
plt.plot(avg_f1_without_gaze['Timestep'], avg_f1_without_gaze['F1_score'], marker='o', label='Without Eye Gaze')
plt.plot(avg_f1_with_just_gaze['Timestep'], avg_f1_with_just_gaze['F1_score'], marker='o', label='Only Eye Gaze')
plt.plot(avg_f1_majority['timestep'], avg_f1_majority['majority_class_f1'], marker='o', label='Majority F1')
plt.axhline(y=avg_f1_MLP_ceg, color='r', linestyle='--', label='Avg MLP F1 CEG')
plt.axhline(y=avg_f1_MLP_go, color='b', linestyle='--', label='Avg MLP F1 GO')


plt.xlabel('Timestep')
plt.ylabel('Average F1 Score')
plt.title('Comparison of Average F1 Scores Over Timesteps With and Without Eye Gaze for trust')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plot_trust_test_majority_F1_MLP_1.png')  # Save the figure as an image file
plt.close() 

##########################################################################################################################################################################

