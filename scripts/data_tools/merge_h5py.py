import h5py

def copy_group(source_group, target_group):
    """Recursively copy all datasets from the source group to the target group."""
    for name, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            # Copy dataset if it does not exist in the target group
            if name not in target_group:
                target_group.copy(item, name)
            else:
                print(f"Dataset '{name}' already exists in the target group. Skipping...")
        elif isinstance(item, h5py.Group):
            # Create a new group in the target if it does not exist and copy recursively
            if name not in target_group:
                new_group = target_group.create_group(name)
            else:
                new_group = target_group[name]
            copy_group(item, new_group)

def merge_hdf5(source_file_path, target_file_path):
    """Merge HDF5 file at source_file_path into the HDF5 file at target_file_path."""
    with h5py.File(target_file_path, 'a') as target_file, h5py.File(source_file_path, 'r') as source_file:
        copy_group(source_file, target_file)

# Paths to your HDF5 files
source_file_path = 'data/eva_features/web_obj_prog_crop_p1_EVA02-CLIP-L-14-336.hdf5'
target_file_path = 'data/eva_features/web_obj_prog_p1_EVA02-CLIP-L-14-336.hdf5'

# Merge source_file.h5 into target_file.h5
merge_hdf5(source_file_path, target_file_path)