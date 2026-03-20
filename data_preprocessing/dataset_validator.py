"""
Dataset Validator
Validates dataset structure and provides statistics
"""

import os
from pathlib import Path
import json


class DatasetValidator:
    """Validates dataset structure and completeness"""
    
    def __init__(self, dataset_root='datasets'):
        """
        Initialize validator
        
        Args:
            dataset_root: Root directory of datasets
        """
        self.dataset_root = dataset_root
        
    def validate_structure(self):
        """
        Validate expected directory structure
        
        Returns:
            dict: Validation results
        """
        required_dirs = [
            'vision/raw/falls',
            'vision/raw/normal',
            'vision/raw/bed_exit',
            'vision/raw/unusual_movement',
            'audio/distress/raw/moan',
            'audio/distress/raw/gasp',
            'audio/distress/raw/cough',
            'audio/keywords/raw/english/help',
            'audio/keywords/raw/english/pain',
            'audio/keywords/raw/urdu/madad',
            'audio/keywords/raw/urdu/dard',
        ]
        
        results = {
            'valid': True,
            'missing_dirs': [],
            'existing_dirs': []
        }
        
        for dir_path in required_dirs:
            full_path = os.path.join(self.dataset_root, dir_path)
            if os.path.exists(full_path):
                results['existing_dirs'].append(dir_path)
            else:
                results['missing_dirs'].append(dir_path)
                results['valid'] = False
        
        return results
    
    def count_files(self):
        """
        Count files in each category
        
        Returns:
            dict: File counts
        """
        counts = {
            'vision': {},
            'audio': {
                'distress': {},
                'keywords': {'english': {}, 'urdu': {}}
            }
        }
        
        # Video extensions
        video_ext = {'.mp4', '.avi', '.mov', '.mkv'}
        # Audio extensions
        audio_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        # Count vision files (recursive search for videos)
        vision_categories = ['falls', 'normal', 'bed_exit', 'unusual_movement']
        for category in vision_categories:
            path = os.path.join(self.dataset_root, 'vision/raw', category)
            if os.path.exists(path):
                video_files = [f for f in Path(path).rglob('*') if f.suffix.lower() in video_ext]
                counts['vision'][category] = len(video_files)
            else:
                counts['vision'][category] = 0
        
        # Count audio distress files (recursive search for audio)
        distress_categories = ['moan', 'gasp', 'cough', 'cry']
        for category in distress_categories:
            path = os.path.join(self.dataset_root, 'audio/distress/raw', category)
            if os.path.exists(path):
                audio_files = [f for f in Path(path).rglob('*') if f.suffix.lower() in audio_ext]
                counts['audio']['distress'][category] = len(audio_files)
            else:
                counts['audio']['distress'][category] = 0
        
        # Count keyword files (recursive search for audio)
        english_keywords = ['help', 'pain', 'nurse', 'doctor']
        for keyword in english_keywords:
            path = os.path.join(self.dataset_root, 'audio/keywords/raw/english', keyword)
            if os.path.exists(path):
                audio_files = [f for f in Path(path).rglob('*') if f.suffix.lower() in audio_ext]
                counts['audio']['keywords']['english'][keyword] = len(audio_files)
            else:
                counts['audio']['keywords']['english'][keyword] = 0
        
        urdu_keywords = ['madad', 'dard', 'nurse']
        for keyword in urdu_keywords:
            path = os.path.join(self.dataset_root, 'audio/keywords/raw/urdu', keyword)
            if os.path.exists(path):
                audio_files = [f for f in Path(path).rglob('*') if f.suffix.lower() in audio_ext]
                counts['audio']['keywords']['urdu'][keyword] = len(audio_files)
            else:
                counts['audio']['keywords']['urdu'][keyword] = 0
        
        return counts
    
    def check_minimum_requirements(self, counts):
        """
        Check if dataset meets minimum requirements
        
        Args:
            counts: File counts from count_files()
            
        Returns:
            dict: Requirements check results
        """
        requirements = {
            'vision': {
                'falls': 100,
                'normal': 100,
                'bed_exit': 50,
                'unusual_movement': 50
            },
            'audio_distress': 250,  # Total distress sounds
            'audio_keywords_per_word': 50
        }
        
        results = {
            'meets_requirements': True,
            'vision': {},
            'audio_distress': {},
            'audio_keywords': {}
        }
        
        # Check vision
        for category, min_count in requirements['vision'].items():
            actual = counts['vision'].get(category, 0)
            results['vision'][category] = {
                'required': min_count,
                'actual': actual,
                'met': actual >= min_count
            }
            if actual < min_count:
                results['meets_requirements'] = False
        
        # Check distress audio
        total_distress = sum(counts['audio']['distress'].values())
        results['audio_distress'] = {
            'required': requirements['audio_distress'],
            'actual': total_distress,
            'met': total_distress >= requirements['audio_distress']
        }
        if total_distress < requirements['audio_distress']:
            results['meets_requirements'] = False
        
        # Check keywords
        for lang in ['english', 'urdu']:
            for keyword, count in counts['audio']['keywords'][lang].items():
                key = f"{lang}_{keyword}"
                results['audio_keywords'][key] = {
                    'required': requirements['audio_keywords_per_word'],
                    'actual': count,
                    'met': count >= requirements['audio_keywords_per_word']
                }
                if count < requirements['audio_keywords_per_word']:
                    results['meets_requirements'] = False
        
        return results
    
    def generate_report(self):
        """
        Generate comprehensive validation report
        
        Returns:
            str: Formatted report
        """
        print("\n" + "="*70)
        print("DATASET VALIDATION REPORT")
        print("="*70)
        
        # Validate structure
        structure = self.validate_structure()
        print("\n[*] Directory Structure:")
        if structure['valid']:
            print("  [OK] All required directories exist")
        else:
            print(f"  [X] Missing {len(structure['missing_dirs'])} directories:")
            for dir_path in structure['missing_dirs']:
                print(f"     - {dir_path}")
        
        # Count files
        counts = self.count_files()
        print("\n[*] File Counts:")
        
        print("\n  Vision Dataset:")
        for category, count in counts['vision'].items():
            print(f"    {category:20s}: {count:4d} files")
        
        print("\n  Audio Distress Dataset:")
        for category, count in counts['audio']['distress'].items():
            print(f"    {category:20s}: {count:4d} files")
        
        print("\n  Audio Keywords (English):")
        for keyword, count in counts['audio']['keywords']['english'].items():
            print(f"    {keyword:20s}: {count:4d} files")
        
        print("\n  Audio Keywords (Urdu):")
        for keyword, count in counts['audio']['keywords']['urdu'].items():
            print(f"    {keyword:20s}: {count:4d} files")
        
        # Check requirements
        requirements = self.check_minimum_requirements(counts)
        print("\n[*] Minimum Requirements Check:")
        
        if requirements['meets_requirements']:
            print("  [OK] Dataset meets all minimum requirements!")
        else:
            print("  [X] Dataset does NOT meet minimum requirements\n")
            
            print("  Vision Module:")
            for category, status in requirements['vision'].items():
                icon = "[OK]" if status['met'] else "[X]"
                print(f"    {icon} {category:20s}: {status['actual']:4d} / {status['required']:4d}")
            
            print("\n  Audio Distress:")
            status = requirements['audio_distress']
            icon = "[OK]" if status['met'] else "[X]"
            print(f"    {icon} Total distress sounds: {status['actual']:4d} / {status['required']:4d}")
            
            print("\n  Audio Keywords:")
            for keyword, status in requirements['audio_keywords'].items():
                icon = "[OK]" if status['met'] else "[X]"
                print(f"    {icon} {keyword:20s}: {status['actual']:4d} / {status['required']:4d}")
        
        print("\n" + "="*70)
        print("\n[!] Next Steps:")
        if not structure['valid']:
            print("  1. Create missing directories")
            print("  2. Download/collect datasets")
        if not requirements['meets_requirements']:
            print("  3. Collect more data to meet minimum requirements")
        else:
            print("  [OK] Dataset is ready for preprocessing!")
        print("\n" + "="*70)


def main():
    """Run dataset validation"""
    validator = DatasetValidator()
    validator.generate_report()


if __name__ == '__main__':
    main()

