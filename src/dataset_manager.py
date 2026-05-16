"""
Dataset Manager — Download, Filter, and Prepare Training Datasets
==================================================================
This module handles:
  - Downloading FACTIFY dataset from Hugging Face
  - Downloading IndicGLUE Telugu splits
  - Filtering Telugu content using Unicode range detection
  - Balancing datasets via oversampling/undersampling
  - Creating stratified train/validation splits
  - Generating dataset statistics reports

Usage:
    from src.dataset_manager import DatasetManager
    
    manager = DatasetManager()
    train_dataset, eval_dataset = manager.prepare_training_data()
    stats = manager.generate_statistics_report()
"""

import os
import re
import json
import logging
import random
from typing import Tuple, Dict, Any, List, Optional
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.config import DATA_DIR, RESULTS_DIR
from src.preprocess import is_telugu

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset acquisition, filtering, balancing, and preparation."""
    
    def __init__(self, data_dir: str = DATA_DIR, results_dir: str = RESULTS_DIR):
        """
        Initialize DatasetManager.
        
        Args:
            data_dir: Directory to store downloaded datasets
            results_dir: Directory to store statistics reports
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.datasets = {}  # Store loaded datasets
        self.statistics = {}  # Store dataset statistics
    
    def is_telugu(self, text: str) -> bool:
        """Delegate to the canonical is_telugu in preprocess.py."""
        return is_telugu(text)
    
    def download_factify(self) -> Optional[pd.DataFrame]:
        """
        Download FACTIFY dataset from Hugging Face.
        
        The FACTIFY dataset contains 50,000+ claim-verification pairs.
        This function attempts to load it from Hugging Face datasets library.
        
        Returns:
            DataFrame with columns: text, label
            None if download fails
        """
        logger.info("📥 Downloading FACTIFY dataset from Hugging Face...")
        
        try:
            from datasets import load_dataset
            
            # Try to load FACTIFY dataset
            # Note: The actual dataset name may vary - adjust as needed
            dataset = load_dataset("newsmediabias/FACTIFY", split="train")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)
            
            logger.info(f"✅ Downloaded FACTIFY dataset: {len(df):,} samples")
            return df
            
        except Exception as e:
            logger.error(f"❌ FACTIFY download failed: {e}")
            logger.warning("⚠️  Continuing without FACTIFY dataset")
            return None
    
    def filter_telugu_factify(self, input_df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract Telugu-only samples from FACTIFY dataset using Unicode range detection.
        
        Maps FACTIFY labels to our 3-class schema:
          - 'Fake' / 'Refuted' / 'False' → 1 (Fake)
          - 'True' / 'Supported' / 'Real' → 0 (Real)
          - 'Other' / 'NEI' / 'Unverifiable' → 2 (Unverifiable)
        
        Args:
            input_df: Input DataFrame with text and label columns
            output_path: Optional path to save filtered CSV
            
        Returns:
            Filtered DataFrame with Telugu-only samples
        """
        logger.info("🔍 Filtering FACTIFY for Telugu content...")
        
        original_count = len(input_df)
        
        # Detect text column
        text_col = None
        for col in ['text', 'claim', 'sentence', 'content']:
            if col in input_df.columns:
                text_col = col
                break
        
        if text_col is None:
            logger.error("❌ No text column found in FACTIFY dataset")
            return pd.DataFrame()
        
        # Filter to Telugu rows
        telugu_mask = input_df[text_col].apply(self.is_telugu)
        telugu_df = input_df[telugu_mask].copy()
        
        logger.info(f"   Original rows: {original_count:,}")
        logger.info(f"   Telugu rows: {len(telugu_df):,}")
        logger.info(f"   Removed: {original_count - len(telugu_df):,} non-Telugu rows")
        
        # Normalize text column name
        if text_col != 'text':
            telugu_df = telugu_df.rename(columns={text_col: 'text'})
        
        # Map labels if present
        label_col = None
        for col in ['label', 'Label', 'verdict', 'Verdict', 'class']:
            if col in telugu_df.columns:
                label_col = col
                break
        
        if label_col:
            telugu_df['label'] = telugu_df[label_col].apply(self._map_factify_label)
            if label_col != 'label':
                telugu_df = telugu_df.drop(columns=[label_col])
            
            # Log class distribution
            logger.info("📊 Class distribution after mapping:")
            for label_id, label_name in {0: "Real", 1: "Fake", 2: "Unverifiable"}.items():
                n = (telugu_df['label'] == label_id).sum()
                logger.info(f"   {label_name:15s}: {n:,}")
        
        # Keep only essential columns
        keep_cols = ['text', 'label'] if 'label' in telugu_df.columns else ['text']
        telugu_df = telugu_df[keep_cols]
        
        # Save if output path provided
        if output_path:
            telugu_df.to_csv(output_path, index=False)
            logger.info(f"✅ Saved filtered FACTIFY → {output_path}")
        
        return telugu_df
    
    def _map_factify_label(self, raw_label) -> int:
        """
        Map FACTIFY label to our 3-class schema.
        
        Args:
            raw_label: Original label (string or int)
            
        Returns:
            Mapped label: 0 (Real), 1 (Fake), or 2 (Unverifiable)
        """
        label_map = {
            "fake": 1, "refuted": 1, "false": 1,
            "true": 0, "real": 0, "supported": 0, "verified": 0,
            "unverifiable": 2, "nei": 2, "not enough info": 2, "other": 2,
        }
        
        if isinstance(raw_label, int):
            return raw_label if raw_label in (0, 1, 2) else 2
        
        return label_map.get(str(raw_label).strip().lower(), 2)
    
    def download_indicglue(self) -> Optional[pd.DataFrame]:
        """
        Download IndicGLUE Telugu splits from Hugging Face.
        
        IndicGLUE contains Telugu NLI and sentiment classification data.
        This function loads the Telugu splits and converts them to our format.
        
        Available Telugu configs: actsa-sc.te, csqa.te, inltkh.te, wiki-ner.te, wstp.te
        We use csqa.te (Commonsense QA) for classification tasks.
        
        Returns:
            DataFrame with columns: text, label
            None if download fails
        """
        logger.info("📥 Downloading IndicGLUE Telugu dataset from Hugging Face...")
        
        try:
            from datasets import load_dataset
            
            # Load IndicGLUE Telugu - using csqa.te (Commonsense QA in Telugu)
            # Note: wnli.te is not available, but csqa.te is a good alternative
            # csqa.te only has 'test' split available
            dataset = load_dataset("indic_glue", "csqa.te", split="test")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)
            
            # Map IndicGLUE format to our format
            # CSQA has 'question', 'choices', 'answerKey' columns
            # We'll use the question as text and map answer to our labels
            if 'question' in df.columns:
                df['text'] = df['question']
            elif 'sentence1' in df.columns and 'sentence2' in df.columns:
                # Combine sentences for classification
                df['text'] = df['sentence1'] + " " + df['sentence2']
            elif 'sentence' in df.columns:
                df['text'] = df['sentence']
            else:
                logger.warning("⚠️  Unexpected IndicGLUE format, using first text column")
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'question' in col.lower()]
                if text_cols:
                    df['text'] = df[text_cols[0]]
                else:
                    logger.error("❌ No text column found in IndicGLUE dataset")
                    return None
            
            # Map labels to our schema
            # CSQA is factual Q&A — use as "Real" class (label 0).
            # We MUST NOT label everything as 0; Fake/Unverifiable will come
            # from synthetic generation below.
            df['label'] = 0
            
            # Keep only Telugu text
            df = df[df['text'].apply(self.is_telugu)]
            
            # Keep only essential columns
            df = df[['text', 'label']]
            
            logger.info(f"✅ Downloaded IndicGLUE Telugu dataset: {len(df):,} samples")
            return df
            
        except Exception as e:
            logger.error(f"❌ IndicGLUE download failed: {e}")
            logger.warning("⚠️  Continuing without IndicGLUE dataset")
            return None
    
    def balance_dataset(self, df: pd.DataFrame, strategy: str = "oversample") -> pd.DataFrame:
        """
        Balance dataset classes via oversampling or undersampling.
        
        Args:
            df: Input DataFrame with 'text' and 'label' columns
            strategy: 'oversample' (duplicate minority classes) or 
                     'undersample' (reduce majority classes)
        
        Returns:
            Balanced DataFrame with approximately equal samples per class
        """
        logger.info(f"⚖️  Balancing dataset using {strategy} strategy...")
        
        if 'label' not in df.columns:
            logger.warning("⚠️  No label column found, skipping balancing")
            return df
        
        # Get class distribution
        class_counts = df['label'].value_counts()
        logger.info("📊 Original class distribution:")
        for label, count in class_counts.items():
            label_name = {0: "Real", 1: "Fake", 2: "Unverifiable"}.get(label, f"Class {label}")
            logger.info(f"   {label_name:15s}: {count:,}")
        
        if strategy == "oversample":
            # Oversample minority classes to match majority
            max_count = class_counts.max()
            balanced_dfs = []
            
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) < max_count:
                    # Oversample with replacement
                    label_df = resample(label_df, 
                                       replace=True, 
                                       n_samples=max_count, 
                                       random_state=42)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif strategy == "undersample":
            # Undersample majority classes to match minority
            min_count = class_counts.min()
            balanced_dfs = []
            
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) > min_count:
                    # Undersample without replacement
                    label_df = resample(label_df, 
                                       replace=False, 
                                       n_samples=min_count, 
                                       random_state=42)
                balanced_dfs.append(label_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            logger.warning(f"⚠️  Unknown strategy '{strategy}', returning original dataset")
            return df
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log new distribution
        new_counts = balanced_df['label'].value_counts()
        logger.info("📊 Balanced class distribution:")
        for label, count in new_counts.items():
            label_name = {0: "Real", 1: "Fake", 2: "Unverifiable"}.get(label, f"Class {label}")
            logger.info(f"   {label_name:15s}: {count:,}")
        
        return balanced_df
    
    def create_train_val_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation split maintaining class distribution.
        
        Args:
            df: Input DataFrame with 'text' and 'label' columns
            test_size: Fraction of data to use for validation (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info(f"✂️  Creating train/validation split (test_size={test_size})...")
        
        if 'label' not in df.columns:
            logger.warning("⚠️  No label column found, using random split")
            train_df = df.sample(frac=1-test_size, random_state=random_state)
            val_df = df.drop(train_df.index)
            return train_df, val_df
        
        # Stratified split to maintain class distribution
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'], 
            random_state=random_state
        )
        
        logger.info(f"   Train samples: {len(train_df):,}")
        logger.info(f"   Val samples: {len(val_df):,}")
        
        # Log class distribution in splits
        logger.info("📊 Train set class distribution:")
        for label, count in train_df['label'].value_counts().items():
            label_name = {0: "Real", 1: "Fake", 2: "Unverifiable"}.get(label, f"Class {label}")
            logger.info(f"   {label_name:15s}: {count:,}")
        
        logger.info("📊 Validation set class distribution:")
        for label, count in val_df['label'].value_counts().items():
            label_name = {0: "Real", 1: "Fake", 2: "Unverifiable"}.get(label, f"Class {label}")
            logger.info(f"   {label_name:15s}: {count:,}")
        
        return train_df, val_df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate samples from dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        removed = original_count - len(df)
        
        if removed > 0:
            logger.info(f"🗑️  Removed {removed:,} duplicate samples")
        
        return df
    
    def filter_by_length(self, df: pd.DataFrame, min_length: int = 15) -> pd.DataFrame:
        """
        Filter out samples shorter than minimum length.
        
        Args:
            df: Input DataFrame
            min_length: Minimum character length (default: 15)
            
        Returns:
            Filtered DataFrame
        """
        original_count = len(df)
        df = df[df['text'].str.len() >= min_length]
        removed = original_count - len(df)
        
        if removed > 0:
            logger.info(f"🗑️  Removed {removed:,} samples shorter than {min_length} characters")
        
        return df
    
    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that all labels are in the correct range (0, 1, 2).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only valid labels
        """
        if 'label' not in df.columns:
            return df
        
        original_count = len(df)
        df = df[df['label'].isin([0, 1, 2])]
        removed = original_count - len(df)
        
        if removed > 0:
            logger.warning(f"⚠️  Removed {removed:,} samples with invalid labels")
        
        return df
    
    def prepare_training_data(self, balance_strategy: str = "oversample", 
                             test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete pipeline: download, filter, balance, and split datasets.
        
        Args:
            balance_strategy: 'oversample' or 'undersample'
            test_size: Fraction for validation set
            
        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info("🚀 Starting dataset preparation pipeline...")
        
        all_data = []
        
        # 1. Download and filter FACTIFY
        factify_df = self.download_factify()
        if factify_df is not None:
            factify_telugu = self.filter_telugu_factify(
                factify_df, 
                output_path=os.path.join(self.data_dir, "factify_telugu.csv")
            )
            if len(factify_telugu) > 0:
                factify_telugu['source'] = 'FACTIFY'
                all_data.append(factify_telugu)
        
        # 2. Download IndicGLUE
        indicglue_df = self.download_indicglue()
        if indicglue_df is not None and len(indicglue_df) > 0:
            indicglue_df['source'] = 'IndicGLUE'
            indicglue_df.to_csv(os.path.join(self.data_dir, "indicglue_telugu.csv"), index=False)
            all_data.append(indicglue_df)
        
        # 3. Generate synthetic Fake & Unverifiable samples
        #    This is essential because IndicGLUE only provides Real-class data.
        synthetic_df = self.generate_synthetic_samples()
        if len(synthetic_df) > 0:
            synthetic_df['source'] = 'Synthetic'
            all_data.append(synthetic_df)
        
        # 4. Combine all datasets
        if not all_data:
            logger.error("❌ No datasets available! Cannot proceed.")
            return pd.DataFrame(), pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"📦 Combined dataset: {len(combined_df):,} samples")
        
        # 5. Data quality checks
        combined_df = self.remove_duplicates(combined_df)
        combined_df = self.filter_by_length(combined_df, min_length=15)
        combined_df = self.validate_labels(combined_df)
        
        # Log class distribution before balancing
        logger.info("📊 Pre-balance class distribution:")
        for label_id in [0, 1, 2]:
            count = (combined_df['label'] == label_id).sum()
            name = {0: 'Real', 1: 'Fake', 2: 'Unverifiable'}[label_id]
            logger.info(f"   {name:15s}: {count:,}")
        
        # 6. Balance dataset
        balanced_df = self.balance_dataset(combined_df, strategy=balance_strategy)
        
        # 7. Create train/val split
        train_df, val_df = self.create_train_val_split(balanced_df, test_size=test_size)
        
        # 8. Save prepared datasets
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        logger.info(f"✅ Saved training data → {train_path}")
        logger.info(f"✅ Saved validation data → {val_path}")
        
        # Store for statistics
        self.datasets['train'] = train_df
        self.datasets['val'] = val_df
        
        return train_df, val_df
    
    def generate_statistics_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics report.
        
        Returns:
            Dictionary containing dataset statistics
        """
        logger.info("📊 Generating dataset statistics report...")
        
        stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'class_distribution': {},
            'sources': {},
            'avg_text_length': 0,
            'min_text_length': 0,
            'max_text_length': 0,
        }
        
        # Combine train and val for overall stats
        if 'train' in self.datasets and 'val' in self.datasets:
            all_data = pd.concat([self.datasets['train'], self.datasets['val']], ignore_index=True)
            
            stats['total_samples'] = len(all_data)
            stats['train_samples'] = len(self.datasets['train'])
            stats['val_samples'] = len(self.datasets['val'])
            
            # Class distribution
            if 'label' in all_data.columns:
                class_counts = all_data['label'].value_counts().to_dict()
                stats['class_distribution'] = {
                    'Real': class_counts.get(0, 0),
                    'Fake': class_counts.get(1, 0),
                    'Unverifiable': class_counts.get(2, 0),
                }
            
            # Source distribution
            if 'source' in all_data.columns:
                stats['sources'] = all_data['source'].value_counts().to_dict()
            
            # Text length statistics
            text_lengths = all_data['text'].str.len()
            stats['avg_text_length'] = float(text_lengths.mean())
            stats['min_text_length'] = int(text_lengths.min())
            stats['max_text_length'] = int(text_lengths.max())
        
        # Save report
        report_path = os.path.join(self.results_dir, "dataset_stats.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Statistics report saved → {report_path}")
        
        # Log summary
        logger.info("📊 Dataset Statistics Summary:")
        logger.info(f"   Total samples: {stats['total_samples']:,}")
        logger.info(f"   Train samples: {stats['train_samples']:,}")
        logger.info(f"   Val samples: {stats['val_samples']:,}")
        logger.info(f"   Class distribution: {stats['class_distribution']}")
        logger.info(f"   Sources: {stats['sources']}")
        logger.info(f"   Avg text length: {stats['avg_text_length']:.1f} chars")
        
        return stats


    def generate_synthetic_samples(self, n_per_class: int = 500) -> pd.DataFrame:
        """
        Generate synthetic Fake and Unverifiable Telugu samples using
        template-based generation. Essential because available Telugu
        datasets (IndicGLUE) only provide Real-class data.
        
        Args:
            n_per_class: Number of samples to generate per class
            
        Returns:
            DataFrame with synthetic samples
        """
        logger.info(f"🔨 Generating {n_per_class} synthetic samples per minority class...")
        
        fake_templates = [
            "ఈ లింక్ క్లిక్ చేస్తే ఉచితంగా {item} వస్తుంది. ఫార్వర్డ్ చేయండి!",
            "ప్రభుత్వం ప్రతి పౌరుడికి రూ. {amount} ఇస్తోంది. వెంటనే అప్లై చేయండి!",
            "ఈ మెసేజ్ {n} మందికి ఫార్వర్డ్ చేస్తే రూ. {amount} వస్తుంది. నిజం!",
            "రేపటి నుండి {item} ధర తగ్గించారు! షేర్ చేయండి!",
            "షాకింగ్ న్యూస్! {event}. వెంటనే షేర్ చేయండి!",
            "బ్రేకింగ్: {event}. ఫార్వర్డ్ చేయండి!",
            "{org} ప్రతి భారతీయుడికి రూ. {amount} ఇస్తోంది. లింక్ క్లిక్ చేయండి!",
            "ఈ వాట్సాప్ మెసేజ్ డిలీట్ చేస్తే మీ ఫోన్ {threat}!",
            "100% గ్యారంటీ! ఈ {remedy} వాడితే {disease} తగ్గిపోతుంది!",
            "ఇప్పుడే క్లిక్ చేయండి! ఉచితంగా {item} పొందండి! చివరి అవకాశం!",
        ]
        
        fake_fills = {
            'item': ['జియో ఫోన్', 'ల్యాప్‌టాప్', 'బైక్', 'కారు', 'టీవీ', 'గోల్డ్', 'సైకిల్'],
            'amount': ['10,000', '15,000', '25,000', '50,000', '1,00,000', '5,000', '2,000'],
            'n': ['5', '10', '15', '20', '25', '50', '100'],
            'event': ['ఎన్నికలు రద్దు', 'కర్ఫ్యూ విధించారు', 'స్కూళ్లు మూసివేత', 'పెట్రోల్ ఉచితం'],
            'org': ['గూగుల్', 'ఫేస్‌బుక్', 'వాట్సాప్', 'టాటా', 'రిలయన్స్'],
            'threat': ['హ్యాక్ అవుతుంది', 'పేలిపోతుంది', 'లాక్ అవుతుంది', 'డేటా పోతుంది'],
            'remedy': ['ఆకు రసం', 'మందు', 'పొడి', 'నూనె', 'చూర్ణం'],
            'disease': ['డయాబెటిస్', 'కేన్సర్', 'హార్ట్ ప్రాబ్లమ్', 'కరోనా', 'జ్వరం'],
        }
        
        unverifiable_templates = [
            "ఈ {remedy} వాడితే {condition} తగ్గుతుందని నిపుణులు చెప్తున్నారు.",
            "ప్రముఖ {person} రహస్యంగా {action} అని వార్తలు.",
            "{item} వల్ల {effect} వస్తుందని కొందరు వైద్యులు అంటున్నారు.",
            "వచ్చే సంవత్సరం {event} వస్తుందని వాతావరణ నిపుణులు.",
            "ఈ {food} తింటే {benefit} పెరుగుతుందని పరిశోధనలు చెప్తున్నాయి.",
            "{person} పార్టీ మారనున్నారని సోషల్ మీడియాలో ప్రచారం.",
            "కొత్త {item} పూర్తిగా సురక్షితం కాదని కొందరు అంటున్నారు.",
            "{food} తాగితే {benefit} అవుతుందని సోషల్ మీడియాలో ప్రచారం.",
        ]
        
        unverifiable_fills = {
            'remedy': ['తేయాకు', 'పసుపు నీళ్ళు', 'వేప ఆకు', 'అల్లం రసం', 'తులసి నీళ్ళు'],
            'condition': ['బరువు', 'జుట్టు రాలడం', 'చర్మ వ్యాధులు', 'నిద్రలేమి'],
            'person': ['నటుడు', 'నేత', 'క్రికెటర్', 'వ్యాపారవేత్త', 'మంత్రి'],
            'action': ['రాజకీయ పార్టీలో చేరారు', 'సినిమా విరమణ', 'దేశం విడిచారు'],
            'item': ['వ్యాక్సిన్', 'మందు', 'టెక్నాలజీ', 'యాప్', 'డ్రింక్'],
            'effect': ['తలనొప్పి', 'నీరసం', 'కడుపు నొప్పి', 'అలర్జీ'],
            'event': ['భారీ కరువు', 'వరదలు', 'భూకంపం', 'తుఫాను'],
            'food': ['పండు', 'కూరగాయ', 'గింజలు', 'తేనె'],
            'benefit': ['జ్ఞాపకశక్తి', 'రోగనిరోధక శక్తి', 'ఎముకల బలం'],
        }
        
        samples = []
        random.seed(42)
        
        for _ in range(n_per_class):
            # Fake sample
            tmpl = random.choice(fake_templates)
            text = tmpl
            for key, values in fake_fills.items():
                text = text.replace('{' + key + '}', random.choice(values), 1)
            samples.append({'text': text, 'label': 1})
            
            # Unverifiable sample
            tmpl = random.choice(unverifiable_templates)
            text = tmpl
            for key, values in unverifiable_fills.items():
                text = text.replace('{' + key + '}', random.choice(values), 1)
            samples.append({'text': text, 'label': 2})
        
        df = pd.DataFrame(samples)
        logger.info(f"✅ Generated {n_per_class} Fake + {n_per_class} Unverifiable synthetic samples")
        return df


if __name__ == "__main__":
    # Example usage
    manager = DatasetManager()
    train_df, val_df = manager.prepare_training_data()
    stats = manager.generate_statistics_report()
