import yaml
# Define the exact paths based on your file scan
path_root = 'D:\\Users\\Name\\.cache\\kagglehub\\datasets\\sujaymann\\car-number-plate-dataset-yolo-format\\versions\\3\\License-Plate-Data'
data_config = {
    'path': path_root,                         # Root directory
    'train': 'train/images',                   # Path to training images (relative to path)
    'val': 'test/images',                      # Path to validation images (using test set for val)
    'test': 'test/images',                     # Path to test images (optional)
    # Class configuration
    'nc': 1,                                   # Number of classes
    'names': ['license_plate']                 # Class name
}
# Write the file
with open('data_yolo_friendly.yaml', 'w') as f:
    yaml.dump(data_config, f)
print("data_yolo_friendly.yaml created successfully!")