#!/usr/bin/env python3
"""
Advanced YOLO Dataset Merger Tool
Supports flexible class mapping, renaming, and multi-dataset merging
Example usage:
python merge_yolo_datasets.py ^
  --base ".\Street View.v3i.yolov8" ^
  --output ".\blind_merged.v2" ^
  --config ".\blind_merge_config.json"

"""

import os
import json
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import logging
from datetime import datetime



class AdvancedYOLOMerger:
    def __init__(self, base_dataset: str, output_path: str, config_file: Optional[str] = None):
        """
        Initialize advanced merger
        
        Args:
            base_dataset: Base dataset path
            output_path: Output path
            config_file: Configuration file path (optional)
        """
        self.base_dataset = Path(base_dataset)
        self.output_path = Path(output_path)
        self.config_file = config_file
        
        # Setup logging
        self.setup_logging()
        
        # Class mapping table
        self.class_mapping = {}
        self.final_classes = []
        self.merge_configs = []
        
        # Statistics
        self.stats = {
            'base': {'train': 0, 'valid': 0, 'test': 0},
            'merged': {'train': 0, 'valid': 0, 'test': 0},
            'class_counts': Counter()
        }
        
    def setup_logging(self):
        """Setup logging system"""
        log_file = f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_yaml(self, yaml_path: Path) -> Dict:
        """Load YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_yaml(self, data: Dict, yaml_path: Path):
        """Save YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def load_config(self, config_file: str) -> Dict:
        """Load merge configuration file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                return json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError("Configuration file must be .json or .yaml/.yml format")
    
    def add_merge_source(self, 
                        source_dataset: str,
                        classes_to_merge: List[str] = None,
                        class_mapping: Dict[str, str] = None,
                        merge_all: bool = False,
                        prefix: str = None):
        """
        Add dataset source to merge
        
        Args:
            source_dataset: Source dataset path
            classes_to_merge: List of classes to merge
            class_mapping: Class renaming mapping {'old_name': 'new_name'}
            merge_all: Whether to merge all classes
            prefix: Filename prefix (to avoid conflicts)
        """
        config = {
            'path': Path(source_dataset),
            'classes': classes_to_merge,
            'mapping': class_mapping or {},
            'merge_all': merge_all,
            'prefix': prefix or f"ds{len(self.merge_configs)+2}_"
        }
        self.merge_configs.append(config)
        self.logger.info(f"Added merge source: {source_dataset}")
        
    def setup_output_structure(self):
        """Create output directory structure"""
        self.logger.info("Creating output directory structure...")
        for split in ['train', 'valid', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def build_class_mapping(self):
        """Build complete class mapping table"""
        self.logger.info("="*60)
        self.logger.info("Building class mapping table...")
        
        # 1. Load base dataset classes
        base_yaml = self.load_yaml(self.base_dataset / 'data.yaml')
        base_classes = base_yaml.get('names', [])
        self.final_classes = base_classes.copy()
        
        self.logger.info(f"Base dataset classes ({len(base_classes)}): {base_classes}")
        
        # 为基础数据集创建映射（保持原索引）
        for i, cls in enumerate(base_classes):
            self.class_mapping[f"base_{cls}"] = i
        
        # 2. Process each dataset to merge
        for idx, config in enumerate(self.merge_configs):
            source_yaml = self.load_yaml(config['path'] / 'data.yaml')
            source_classes = source_yaml.get('names', [])
            
            self.logger.info(f"\nProcessing source {idx+1}: {config['path'].name}")
            self.logger.info(f"  Source classes ({len(source_classes)}): {source_classes}")
            
            # Determine classes to merge
            if config['merge_all']:
                classes_to_merge = source_classes
            else:
                classes_to_merge = config['classes'] or []
            
            self.logger.info(f"  Classes to merge: {classes_to_merge}")
            
            # Process each class to merge
            for cls in classes_to_merge:
                if cls not in source_classes:
                    self.logger.warning(f"  Warning: Class '{cls}' not in source dataset, skipping")
                    continue
                
                old_idx = source_classes.index(cls)
                
                # Check if renaming is needed
                new_cls_name = config['mapping'].get(cls, cls)
                
                # Check if new class name already exists in final class list
                if new_cls_name in self.final_classes:
                    new_idx = self.final_classes.index(new_cls_name)
                    self.logger.info(f"  '{cls}' → '{new_cls_name}' (exists, index: {new_idx})")
                else:
                    # Add new class
                    new_idx = len(self.final_classes)
                    self.final_classes.append(new_cls_name)
                    self.logger.info(f"  '{cls}' → '{new_cls_name}' (new class, index: {new_idx})")
                
                # Record mapping
                mapping_key = f"{config['prefix']}{old_idx}"
                self.class_mapping[mapping_key] = new_idx
        
        self.logger.info(f"\nFinal class list ({len(self.final_classes)}): {self.final_classes}")
        self.logger.info(f"Class mapping table: {json.dumps(self.class_mapping, indent=2)}")
        
    def copy_base_dataset(self):
        """Copy base dataset"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Copying base dataset...")
        
        for split in ['train', 'valid', 'test']:
            src_img_dir = self.base_dataset / split / 'images'
            src_lbl_dir = self.base_dataset / split / 'labels'
            dst_img_dir = self.output_path / split / 'images'
            dst_lbl_dir = self.output_path / split / 'labels'
            
            if not src_img_dir.exists():
                self.logger.warning(f"  {src_img_dir} does not exist, skipping")
                continue
            
            # Copy images
            img_count = 0
            for img_file in src_img_dir.glob('*'):
                if img_file.is_file():
                    shutil.copy2(img_file, dst_img_dir / img_file.name)
                    img_count += 1
            
            # Copy labels
            lbl_count = 0
            if src_lbl_dir.exists():
                for lbl_file in src_lbl_dir.glob('*.txt'):
                    shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)
                    lbl_count += 1
            
            self.stats['base'][split] = img_count
            self.logger.info(f"  {split}: Copied {img_count} images, {lbl_count} labels")
    
    def merge_source_dataset(self, config: Dict):
        """Merge single source dataset"""
        self.logger.info(f"\nMerging source: {config['path'].name}")
        
        source_yaml = self.load_yaml(config['path'] / 'data.yaml')
        source_classes = source_yaml.get('names', [])
        
        # Determine classes to merge
        if config['merge_all']:
            classes_to_merge = source_classes
        else:
            classes_to_merge = config['classes'] or []
        
        # Get indices of classes to merge
        merge_indices = set()
        for cls in classes_to_merge:
            if cls in source_classes:
                merge_indices.add(source_classes.index(cls))
        
        self.logger.info(f"  Class indices to merge: {merge_indices}")
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            src_img_dir = config['path'] / split / 'images'
            src_lbl_dir = config['path'] / split / 'labels'
            dst_img_dir = self.output_path / split / 'images'
            dst_lbl_dir = self.output_path / split / 'labels'
            
            if not src_img_dir.exists():
                self.logger.warning(f"  {src_img_dir} does not exist, skipping")
                continue
            
            merged_count = 0
            
            if src_lbl_dir.exists():
                for lbl_file in src_lbl_dir.glob('*.txt'):
                    # Read labels
                    with open(lbl_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Filter and convert labels
                    new_lines = []
                    has_target_class = False
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            
                            # Check if it's a class to merge
                            if class_id in merge_indices:
                                # Get new class ID
                                mapping_key = f"{config['prefix']}{class_id}"
                                new_class_id = self.class_mapping.get(mapping_key)
                                
                                if new_class_id is not None:
                                    parts[0] = str(new_class_id)
                                    new_lines.append(' '.join(parts) + '\n')
                                    has_target_class = True
                                    
                                    # Update statistics
                                    class_name = self.final_classes[new_class_id]
                                    self.stats['class_counts'][class_name] += 1
                    
                    # If contains target classes, copy files
                    if has_target_class:
                        # Generate new filename
                        base_name = lbl_file.stem
                        new_lbl_name = f"{config['prefix']}{base_name}.txt"
                        
                        # Find corresponding image
                        image_copied = False
                        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                            img_file = src_img_dir / f"{base_name}{ext}"
                            if img_file.exists():
                                new_img_name = f"{config['prefix']}{base_name}{ext}"
                                shutil.copy2(img_file, dst_img_dir / new_img_name)
                                image_copied = True
                                break
                        
                        if image_copied:
                            # Save label file
                            with open(dst_lbl_dir / new_lbl_name, 'w') as f:
                                f.writelines(new_lines)
                            merged_count += 1
            
            self.stats['merged'][split] += merged_count
            self.logger.info(f"  {split}: Merged {merged_count} samples")
    
    def create_final_yaml(self):
        """Create final configuration file"""
        self.logger.info("\nCreating final configuration file...")
        
        # Load base configuration
        base_yaml = self.load_yaml(self.base_dataset / 'data.yaml')
        
        # Create new configuration
        new_yaml = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',
            'nc': len(self.final_classes),
            'names': self.final_classes
        }
        
        # Keep roboflow info if exists
        if 'roboflow' in base_yaml:
            new_yaml['roboflow'] = base_yaml['roboflow']
        
        # Add merge information
        new_yaml['merge_info'] = {
            'base_dataset': str(self.base_dataset),
            'merge_sources': [str(config['path']) for config in self.merge_configs],
            'merge_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_classes': len(self.final_classes)
        }
        
        # Save configuration
        self.save_yaml(new_yaml, self.output_path / 'data.yaml')
        
        # Save detailed merge configuration
        merge_detail = {
            'class_mapping': self.class_mapping,
            'final_classes': self.final_classes,
            'merge_configs': [
                {
                    'path': str(config['path']),
                    'classes': config['classes'],
                    'mapping': config['mapping'],
                    'merge_all': config['merge_all'],
                    'prefix': config['prefix']
                }
                for config in self.merge_configs
            ]
        }
        
        with open(self.output_path / 'merge_details.json', 'w', encoding='utf-8') as f:
            json.dump(merge_detail, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"  Configuration saved: {self.output_path / 'data.yaml'}")
        self.logger.info(f"  Merge details saved: {self.output_path / 'merge_details.json'}")
    
    def print_summary(self):
        """Print merge summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Merge Summary")
        self.logger.info("="*60)
        
        # Dataset statistics
        self.logger.info("\n📊 Dataset Statistics:")
        total_base = sum(self.stats['base'].values())
        total_merged = sum(self.stats['merged'].values())
        total_final = total_base + total_merged
        
        self.logger.info(f"  Base dataset: {total_base} samples")
        for split, count in self.stats['base'].items():
            if count > 0:
                self.logger.info(f"    - {split}: {count}")
        
        self.logger.info(f"  Newly merged: {total_merged} samples")
        for split, count in self.stats['merged'].items():
            if count > 0:
                self.logger.info(f"    - {split}: {count}")
        
        self.logger.info(f"  Final total: {total_final} samples")
        
        # Class distribution
        if self.stats['class_counts']:
            self.logger.info("\n📈 Class distribution of new samples:")
            for cls, count in sorted(self.stats['class_counts'].items(), 
                                    key=lambda x: x[1], reverse=True):
                self.logger.info(f"  - {cls}: {count} bounding boxes")
        
        self.logger.info(f"\n✅ Merge completed!")
        self.logger.info(f"Output path: {self.output_path}")
        self.logger.info(f"Final number of classes: {len(self.final_classes)}")
    
    def run(self):
        """Execute merge workflow"""
        self.logger.info("="*60)
        self.logger.info("Advanced YOLO Dataset Merger")
        self.logger.info("="*60)
        
        # Load from config file if provided
        if self.config_file:
            self.load_from_config()
        
        # 1. Create output directories
        self.setup_output_structure()
        
        # 2. Build class mapping
        self.build_class_mapping()
        
        # 3. Copy base dataset
        self.copy_base_dataset()
        
        # 4. Merge each source dataset
        for config in self.merge_configs:
            self.merge_source_dataset(config)
        
        # 5. Create final configuration file
        self.create_final_yaml()
        
        # 6. Print summary
        self.print_summary()
    
    def load_from_config(self):
        """Load merge settings from configuration file"""
        config = self.load_config(self.config_file)
        
        # Process each merge source
        for source in config.get('sources', []):
            self.add_merge_source(
                source_dataset=source['path'],
                classes_to_merge=source.get('classes'),
                class_mapping=source.get('mapping'),
                merge_all=source.get('merge_all', False),
                prefix=source.get('prefix')
            )


def main():
    parser = argparse.ArgumentParser(
        description='Advanced YOLO Dataset Merger Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

1. Command line with single source:
   python advanced_yolo_merger.py \\
       --base ./dataset1 \\
       --output ./merged \\
       --source ./dataset2 \\
       --classes car person \\
       --mapping '{"car": "vehicle", "person": "human"}'

2. Using configuration file:
   python advanced_yolo_merger.py \\
       --base ./dataset1 \\
       --output ./merged \\
       --config merge_config.json

3. Merge all classes:
   python advanced_yolo_merger.py \\
       --base ./dataset1 \\
       --output ./merged \\
       --source ./dataset2 \\
       --merge-all
        """
    )
    
    parser.add_argument('--base', type=str, required=True,
                        help='Base dataset path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output dataset path')
    
    # Configuration file mode
    parser.add_argument('--config', type=str,
                        help='Configuration file path (JSON or YAML format)')
    
    # Single source mode
    parser.add_argument('--source', type=str,
                        help='Source dataset path to merge')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='List of classes to merge')
    parser.add_argument('--mapping', type=str,
                        help='Class renaming mapping (JSON format string)')
    parser.add_argument('--merge-all', action='store_true',
                        help='Merge all classes')
    parser.add_argument('--prefix', type=str,
                        help='Filename prefix')
    
    args = parser.parse_args()
    
    # Create merger
    merger = AdvancedYOLOMerger(args.base, args.output, args.config)
    
    # If no config file, use command line arguments
    if not args.config and args.source:
        mapping = {}
        if args.mapping:
            mapping = json.loads(args.mapping)
        
        merger.add_merge_source(
            source_dataset=args.source,
            classes_to_merge=args.classes,
            class_mapping=mapping,
            merge_all=args.merge_all,
            prefix=args.prefix
        )
    elif not args.config and not args.source:
        parser.error("Must provide either --config or --source parameter")
    
    # Execute merge
    merger.run()


if __name__ == '__main__':
    main()