import os
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

def load_dataset_config():
    """Load dataset configuration from data.yaml"""
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_image_statistics(dataset_paths):
    """Analyze image dimensions, formats, and basic statistics"""
    print("📊 IMAGE STATISTICS ANALYSIS")
    print("="*60)
    
    all_stats = []
    image_formats = Counter()
    resolution_counter = Counter()
    
    for split_name, image_dir in dataset_paths.items():
        if not os.path.exists(image_dir):
            continue
            
        print(f"\n🔍 Analyzing {split_name} set...")
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(Path(image_dir).glob(f'*{ext}'))
            images.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        print(f"  Found {len(images)} images")
        
        # Analyze each image
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width, channels = img.shape
                    
                    # Collect statistics
                    stats = {
                        'split': split_name,
                        'filename': img_path.name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height,
                        'channels': channels,
                        'format': img_path.suffix.lower(),
                        'size_kb': os.path.getsize(img_path) / 1024
                    }
                    all_stats.append(stats)
                    
                    # Count formats
                    image_formats[img_path.suffix.lower()] += 1
                    # Count common resolutions
                    resolution_counter[f"{width}x{height}"] += 1
                    
            except Exception as e:
                print(f"  Error reading {img_path}: {e}")
    
    return pd.DataFrame(all_stats), image_formats, resolution_counter

def analyze_labels(dataset_paths):
    """Analyze label distribution and annotations"""
    print("\n📊 LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    label_data = []
    instances_per_image = defaultdict(list)
    
    for split_name, image_dir in dataset_paths.items():
        label_dir = str(image_dir).replace('images', 'labels')
        
        if not os.path.exists(label_dir):
            continue
            
        print(f"\n🔍 Analyzing {split_name} labels...")
        
        label_files = list(Path(label_dir).glob('*.txt'))
        print(f"  Found {len(label_files)} label files")
        
        total_instances = 0
        empty_files = 0
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            num_instances = len(lines)
            total_instances += num_instances
            
            if num_instances == 0:
                empty_files += 1
            
            instances_per_image[split_name].append(num_instances)
            
            # Parse each annotation
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO format: class x_center y_center width height
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    
                    label_data.append({
                        'split': split_name,
                        'file': label_file.name,
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width_norm': w,
                        'height_norm': h,
                        'area_norm': w * h  # Normalized area (0-1)
                    })
        
        print(f"  Total instances: {total_instances}")
        print(f"  Average instances per image: {total_instances/len(label_files):.2f}")
        print(f"  Empty label files: {empty_files}")
    
    return pd.DataFrame(label_data), instances_per_image

def analyze_class_distribution(label_df):
    """Analyze distribution across classes"""
    print("\n📊 CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    if label_df.empty:
        print("No label data found!")
        return None
    
    class_counts = label_df['class_id'].value_counts().sort_index()
    
    print(f"Total classes: {len(class_counts)}")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} instances ({count/len(label_df)*100:.1f}%)")
    
    return class_counts

def create_visualizations(img_stats_df, label_df, instances_per_image, image_formats, resolution_counter):
    """Create comprehensive visualizations"""
    print("\n📈 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Image Size Distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Exploratory Data Analysis: Blind Road Dataset', fontsize=16, fontweight='bold')
    
    # 1a. Image Width Distribution
    axes[0, 0].hist(img_stats_df['width'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Image Width Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 1b. Image Height Distribution
    axes[0, 1].hist(img_stats_df['height'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Image Height Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 1c. Aspect Ratio Distribution
    axes[0, 2].hist(img_stats_df['aspect_ratio'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Aspect Ratio Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Aspect Ratio (Width/Height)')
    axes[0, 2].set_ylabel('Frequency')
    
    # 2. Split Distribution
    if 'split' in img_stats_df.columns:
        split_counts = img_stats_df['split'].value_counts()
        axes[1, 0].bar(split_counts.index, split_counts.values, color=['blue', 'green', 'red'])
        axes[1, 0].set_title('Dataset Split Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Dataset Split')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 3. Image Format Distribution
    if image_formats:
        formats, counts = zip(*image_formats.items())
        axes[1, 1].bar(formats, counts, color='purple')
        axes[1, 1].set_title('Image Format Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Image Format')
        axes[1, 1].set_ylabel('Count')
    
    # 4. Instances per Image Distribution
    all_instances = []
    for split, instances in instances_per_image.items():
        all_instances.extend(instances)
    
    if all_instances:
        axes[1, 2].hist(all_instances, bins=range(0, max(all_instances)+2), 
                       edgecolor='black', alpha=0.7, align='left')
        axes[1, 2].set_title('Instances per Image', fontweight='bold')
        axes[1, 2].set_xlabel('Number of Blind Road Instances')
        axes[1, 2].set_ylabel('Number of Images')
        axes[1, 2].set_xticks(range(0, max(all_instances)+1))
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Object Size Distribution (if labels exist)
    if not label_df.empty:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # Normalized width and height
        axes2[0].scatter(label_df['width_norm'], label_df['height_norm'], 
                        alpha=0.5, s=10)
        axes2[0].set_title('Normalized Object Dimensions', fontweight='bold')
        axes2[0].set_xlabel('Normalized Width')
        axes2[0].set_ylabel('Normalized Height')
        axes2[0].grid(True, alpha=0.3)
        
        # Normalized area distribution
        axes2[1].hist(label_df['area_norm'], bins=30, edgecolor='black', alpha=0.7)
        axes2[1].set_title('Normalized Object Area Distribution', fontweight='bold')
        axes2[1].set_xlabel('Normalized Area (width * height)')
        axes2[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('object_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. Resolution heatmap
    if resolution_counter:
        resolutions = list(resolution_counter.keys())
        counts = list(resolution_counter.values())
        
        # Get top 10 resolutions
        if len(resolutions) > 10:
            top_indices = np.argsort(counts)[-10:]
            resolutions = [resolutions[i] for i in top_indices]
            counts = [counts[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(resolutions, counts, color='teal')
        plt.title('Top Image Resolutions', fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Resolution (Width x Height)')
        plt.tight_layout()
        plt.savefig('resolution_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_summary_report(img_stats_df, label_df, instances_per_image):
    """Generate comprehensive summary report"""
    print("\n📋 SUMMARY REPORT")
    print("="*60)
    
    # Basic statistics
    print("\n📁 DATASET OVERVIEW:")
    print(f"  Total Images: {len(img_stats_df)}")
    print(f"  Total Annotations: {len(label_df)}")
    
    if not img_stats_df.empty:
        print(f"\n📏 IMAGE DIMENSIONS:")
        print(f"  Width Range: {img_stats_df['width'].min()} - {img_stats_df['width'].max()} pixels")
        print(f"  Height Range: {img_stats_df['height'].min()} - {img_stats_df['height'].max()} pixels")
        print(f"  Mean Resolution: {img_stats_df['width'].mean():.0f}x{img_stats_df['height'].mean():.0f}")
        print(f"  Aspect Ratio Range: {img_stats_df['aspect_ratio'].min():.2f} - {img_stats_df['aspect_ratio'].max():.2f}")
        print(f"  Mean Aspect Ratio: {img_stats_df['aspect_ratio'].mean():.2f}")
    
    if not label_df.empty:
        print(f"\n🎯 ANNOTATION STATISTICS:")
        print(f"  Total Instances: {len(label_df)}")
        print(f"  Normalized Width Mean: {label_df['width_norm'].mean():.4f}")
        print(f"  Normalized Height Mean: {label_df['height_norm'].mean():.4f}")
        print(f"  Normalized Area Mean: {label_df['area_norm'].mean():.4f}")
    
    # Instances per image statistics
    all_instances = []
    for split, instances in instances_per_image.items():
        if instances:
            print(f"\n📊 {split.upper()} SET:")
            print(f"  Images: {len(instances)}")
            print(f"  Total Instances: {sum(instances)}")
            print(f"  Instances per Image: {np.mean(instances):.2f} ± {np.std(instances):.2f}")
            print(f"  Min Instances: {min(instances)}")
            print(f"  Max Instances: {max(instances)}")
            all_instances.extend(instances)
    
    if all_instances:
        print(f"\n📈 OVERALL INSTANCE DENSITY:")
        print(f"  Average instances per image: {np.mean(all_instances):.2f}")
        print(f"  Images with 0 instances: {sum(1 for x in all_instances if x == 0)}")
        print(f"  Images with 1 instance: {sum(1 for x in all_instances if x == 1)}")
        print(f"  Images with >1 instances: {sum(1 for x in all_instances if x > 1)}")

def main():
    """Main EDA function"""
    print("🚀 EXPLORATORY DATA ANALYSIS - BLIND ROAD DATASET")
    print("="*60)
    
    # Load dataset configuration
    config = load_dataset_config()
    
    # Define dataset paths
    dataset_paths = {
        'train': config['train'],
        'val': config['val'],
        'test': config['test']
    }
    
    # Analyze images
    img_stats_df, image_formats, resolution_counter = analyze_image_statistics(dataset_paths)
    
    # Analyze labels
    label_df, instances_per_image = analyze_labels(dataset_paths)
    
    # Analyze class distribution
    class_counts = analyze_class_distribution(label_df)
    
    # Create visualizations
    create_visualizations(img_stats_df, label_df, instances_per_image, image_formats, resolution_counter)
    
    # Generate summary report
    generate_summary_report(img_stats_df, label_df, instances_per_image)
    
    # Save dataframes for further analysis
    img_stats_df.to_csv('image_statistics.csv', index=False)
    if not label_df.empty:
        label_df.to_csv('label_statistics.csv', index=False)
    
    print("\n✅ EDA completed successfully!")
    print("📁 Output files created:")
    print("   - image_statistics.csv")
    print("   - label_statistics.csv")
    print("   - eda_visualizations.png")
    print("   - object_size_analysis.png")
    print("   - resolution_distribution.png")

if __name__ == "__main__":
    main()