"""
Setup Dataset Structure
Creates the complete directory structure for datasets
"""

import os


def create_dataset_structure(base_dir='datasets'):
    """
    Create complete dataset directory structure
    
    Args:
        base_dir: Root directory for datasets
    """
    directories = [
        # Vision datasets
        'vision/raw/falls',
        'vision/raw/normal',
        'vision/raw/bed_exit',
        'vision/raw/unusual_movement',
        'vision/processed/frames',
        'vision/processed/annotations',
        'vision/splits',
        
        # Audio distress datasets
        'audio/distress/raw/moan',
        'audio/distress/raw/gasp',
        'audio/distress/raw/cough',
        'audio/distress/raw/cry',
        'audio/distress/raw/wheeze',
        'audio/distress/raw/normal',  # Background sounds
        'audio/distress/processed',
        'audio/distress/splits',
        
        # Audio keywords - English
        'audio/keywords/raw/english/help',
        'audio/keywords/raw/english/pain',
        'audio/keywords/raw/english/nurse',
        'audio/keywords/raw/english/doctor',
        'audio/keywords/raw/english/emergency',
        
        # Audio keywords - Urdu
        'audio/keywords/raw/urdu/madad',      # help
        'audio/keywords/raw/urdu/dard',       # pain
        'audio/keywords/raw/urdu/nurse',      # nurse
        'audio/keywords/raw/urdu/doctor',     # doctor
        
        # Processed keywords
        'audio/keywords/processed',
        'audio/keywords/splits',
        
        # Metadata
        'metadata',
        
        # Models (for saving trained models)
        'models/vision',
        'models/audio',
        'models/llm'
    ]
    
    print(f"Creating dataset structure in: {base_dir}\n")
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"[OK] Created: {directory}")
    
    # Create README files in key directories
    readme_content = {
        'vision/raw/falls': 'Place fall detection videos here (people falling or in fallen positions)',
        'vision/raw/normal': 'Place videos of people in normal bed positions here',
        'vision/raw/bed_exit': 'Place videos of people getting out of bed here',
        'vision/raw/unusual_movement': 'Place videos of unusual movements (seizures, erratic motion) here',
        'audio/distress/raw/moan': 'Place moaning sound recordings here',
        'audio/distress/raw/gasp': 'Place gasping sound recordings here',
        'audio/distress/raw/cough': 'Place coughing sound recordings here',
        'audio/distress/raw/cry': 'Place crying/whimpering sound recordings here',
        'audio/keywords/raw/english/help': 'Place recordings of "help" keyword here',
        'audio/keywords/raw/urdu/madad': 'Place recordings of "madad" (help) keyword here',
    }
    
    for path, content in readme_content.items():
        readme_path = os.path.join(base_dir, path, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(content + '\n')
    
    print(f"\n[SUCCESS] Dataset structure created successfully!")
    print(f"\nNext steps:")
    print("1. Download datasets from Kaggle")
    print("2. Place files in appropriate raw/ directories")
    print("3. Run: python data_preprocessing/dataset_validator.py")


if __name__ == '__main__':
    create_dataset_structure()

