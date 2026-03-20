"""
Master Preprocessing Script
Preprocesses all datasets (vision and audio)
"""

import sys
import os
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import VideoPreprocessor, AudioPreprocessor


def preprocess_vision_data(config, dataset_root='datasets'):
    """Preprocess all vision datasets"""
    print("\n" + "="*70)
    print("PREPROCESSING VISION DATA")
    print("="*70 + "\n")
    
    preprocessor = VideoPreprocessor(config)
    
    categories = ['falls', 'normal', 'bed_exit', 'unusual_movement']
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        input_dir = os.path.join(dataset_root, 'vision/raw', category)
        output_dir = os.path.join(dataset_root, 'vision/processed', category)
        
        if not os.path.exists(input_dir):
            print(f"  ⚠️  Directory not found: {input_dir}")
            continue
        
        results = preprocessor.process_directory(
            input_dir, 
            output_dir, 
            extract_frames=True
        )
        
        print(f"  ✓ Processed {len(results)} videos")
    
    # Create train/val/test splits
    print("\n📊 Creating train/val/test splits...")
    splits_output = os.path.join(dataset_root, 'vision/splits/splits.json')
    stats = preprocessor.create_train_val_test_split(
        os.path.join(dataset_root, 'vision/processed'),
        splits_output
    )
    print(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")


def preprocess_audio_distress(config, dataset_root='datasets'):
    """Preprocess audio distress sounds"""
    print("\n" + "="*70)
    print("PREPROCESSING AUDIO DISTRESS DATA")
    print("="*70 + "\n")
    
    preprocessor = AudioPreprocessor(config)
    
    categories = ['moan', 'gasp', 'cough', 'cry', 'wheeze', 'normal']
    
    for category in categories:
        print(f"\nProcessing category: {category}")
        input_dir = os.path.join(dataset_root, 'audio/distress/raw', category)
        output_dir = os.path.join(dataset_root, 'audio/distress/processed', category)
        
        if not os.path.exists(input_dir):
            print(f"  ⚠️  Directory not found: {input_dir}")
            continue
        
        results = preprocessor.process_directory(input_dir, output_dir)
        successful = sum(1 for r in results if r['success'])
        print(f"  ✓ Processed {successful}/{len(results)} audio files")
    
    # Create splits
    print("\n📊 Creating train/val/test splits...")
    splits_output = os.path.join(dataset_root, 'audio/distress/splits/splits.json')
    stats = preprocessor.create_train_val_test_split(
        os.path.join(dataset_root, 'audio/distress/processed'),
        splits_output
    )
    print(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")


def preprocess_audio_keywords(config, dataset_root='datasets'):
    """Preprocess audio keywords"""
    print("\n" + "="*70)
    print("PREPROCESSING AUDIO KEYWORDS")
    print("="*70 + "\n")
    
    preprocessor = AudioPreprocessor(config)
    
    languages = {
        'english': ['help', 'pain', 'nurse', 'doctor', 'emergency'],
        'urdu': ['madad', 'dard', 'nurse', 'doctor']
    }
    
    for lang, keywords in languages.items():
        print(f"\n{lang.upper()} Keywords:")
        for keyword in keywords:
            print(f"  Processing: {keyword}")
            input_dir = os.path.join(
                dataset_root, 
                'audio/keywords/raw', 
                lang, 
                keyword
            )
            output_dir = os.path.join(
                dataset_root,
                'audio/keywords/processed',
                lang,
                keyword
            )
            
            if not os.path.exists(input_dir):
                print(f"    ⚠️  Directory not found: {input_dir}")
                continue
            
            results = preprocessor.process_directory(input_dir, output_dir)
            successful = sum(1 for r in results if r['success'])
            print(f"    ✓ Processed {successful}/{len(results)} audio files")
    
    # Create splits
    print("\n📊 Creating train/val/test splits...")
    splits_output = os.path.join(dataset_root, 'audio/keywords/splits/splits.json')
    stats = preprocessor.create_train_val_test_split(
        os.path.join(dataset_root, 'audio/keywords/processed'),
        splits_output
    )
    print(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*70)
    print("VITAL GUARDIAN - DATASET PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = 'datasets'
    
    # Preprocess all data
    preprocess_vision_data(config, dataset_root)
    preprocess_audio_distress(config, dataset_root)
    preprocess_audio_keywords(config, dataset_root)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Verify processed data in datasets/*/processed/")
    print("2. Check split files in datasets/*/splits/")
    print("3. Begin model training")
    print("\n")


if __name__ == '__main__':
    main()

