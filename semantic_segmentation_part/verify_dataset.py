import os
import yaml
from pathlib import Path

def verify_dataset():
    """
    Verify the dataset structure and files
    """
    print("=== Dataset Verification ===")
    
    # Check data.yaml
    data_yaml_path = 'data.yaml'
    if not os.path.exists(data_yaml_path):
        print(f"ERROR: data.yaml not found at {data_yaml_path}")
        return False
    
    print(f"✓ data.yaml found")
    
    # Load and check data.yaml content
    try:
        with open(data_yaml_path, 'r') as file:
            data_config = yaml.safe_load(file)
        
        print("Data configuration:")
        print(f"  Train: {data_config.get('train', 'NOT FOUND')}")
        print(f"  Val: {data_config.get('val', 'NOT FOUND')}")
        print(f"  Test: {data_config.get('test', 'NOT FOUND')}")
        print(f"  Classes: {data_config.get('nc', 'NOT FOUND')}")
        print(f"  Class names: {data_config.get('names', 'NOT FOUND')}")
        
    except Exception as e:
        print(f"ERROR reading data.yaml: {e}")
        return False
    
    # Check if directories exist
    directories_to_check = [
        data_config.get('train'),
        data_config.get('val'), 
        data_config.get('test')
    ]
    
    for dir_path in directories_to_check:
        if dir_path and os.path.exists(dir_path):
            # Count images in directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_count = sum(1 for file in os.listdir(dir_path) 
                            if any(file.lower().endswith(ext) for ext in image_extensions))
            print(f"✓ {dir_path}: {image_count} images")
        else:
            print(f"✗ Directory not found: {dir_path}")
    
    # Check for label directories
    train_dir = data_config.get('train')
    if train_dir:
        label_dir = train_dir.replace('images', 'labels')
        if os.path.exists(label_dir):
            label_count = sum(1 for file in os.listdir(label_dir) if file.endswith('.txt'))
            print(f"✓ Labels found: {label_dir} ({label_count} files)")
        else:
            print(f"✗ Label directory not found: {label_dir}")
    
    return True

def check_sample_labels():
    """
    Check sample label files to ensure they are in correct format
    """
    print("\n=== Checking Sample Labels ===")
    
    # Find a label file to inspect
    label_files = list(Path('.').glob('**/*.txt'))
    if label_files:
        sample_label = label_files[0]
        print(f"Sample label file: {sample_label}")
        
        try:
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                print(f"Number of annotations: {len(lines)}")
                if lines:
                    print(f"First annotation: {lines[0].strip()}")
        except Exception as e:
            print(f"Error reading label file: {e}")
    else:
        print("No label files found!")

if __name__ == "__main__":
    if verify_dataset():
        check_sample_labels()
        print("\n✓ Dataset verification completed")
    else:
        print("\n✗ Dataset verification failed")