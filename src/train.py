"""
Model Trainer — Fine-tune MuRIL for Telugu Fake News Classification
====================================================================
This module handles:
  - Loading and preparing datasets (FACTIFY Telugu, IndicGLUE Telugu, Telugu news)
  - Initializing MuRIL model for 3-class classification (Real/Fake/Unverifiable)
  - Implementing training loop with metrics tracking
  - Saving trained model and tokenizer
  - Generating training report with F1 score comparison to GPT-4 baseline (0.61)

Usage:
    from src.train import ModelTrainer
    
    trainer = ModelTrainer()
    metrics = trainer.train()
    trainer.save_model()
    trainer.generate_training_report()
"""

# Workaround for torchvision VideoReader import error
try:
    import torchvision.io
    if not hasattr(torchvision.io, 'VideoReader'):
        class MockVideoReader:
            pass
        torchvision.io.VideoReader = MockVideoReader
except Exception:
    pass

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate

from src.config import (
    BASE_MODEL_NAME,
    NUM_LABELS,
    LABELS,
    LABEL2ID,
    ID2LABEL,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    TARGET_F1,
    MODEL_DIR,
    DATA_DIR,
    RESULTS_DIR,
    LOGS_DIR,
)
from src.dataset_manager import DatasetManager

# Ensure logs directory exists before setting up logging
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Manages model training pipeline for Telugu fake news classification.
    
    This class handles the complete training workflow:
    1. Load and prepare datasets using DatasetManager
    2. Initialize MuRIL model with 3-class classification head
    3. Execute training loop with metrics tracking
    4. Save trained model and tokenizer
    5. Generate comprehensive training report
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Use config from src/config.py by default
        self.model_name = config.get('model_name', BASE_MODEL_NAME) if config else BASE_MODEL_NAME
        self.num_labels = config.get('num_labels', NUM_LABELS) if config else NUM_LABELS
        self.batch_size = config.get('batch_size', BATCH_SIZE) if config else BATCH_SIZE
        self.epochs = config.get('epochs', EPOCHS) if config else EPOCHS
        self.learning_rate = config.get('learning_rate', LEARNING_RATE) if config else LEARNING_RATE
        self.weight_decay = config.get('weight_decay', WEIGHT_DECAY) if config else WEIGHT_DECAY
        self.warmup_ratio = config.get('warmup_ratio', WARMUP_RATIO) if config else WARMUP_RATIO
        self.max_length = config.get('max_length', MAX_LENGTH) if config else MAX_LENGTH
        self.target_f1 = config.get('target_f1', TARGET_F1) if config else TARGET_F1
        
        self.model_dir = config.get('model_dir', MODEL_DIR) if config else MODEL_DIR
        self.data_dir = config.get('data_dir', DATA_DIR) if config else DATA_DIR
        self.results_dir = config.get('results_dir', RESULTS_DIR) if config else RESULTS_DIR
        self.logs_dir = config.get('logs_dir', LOGS_DIR) if config else LOGS_DIR
        
        # Ensure directories exist
        for directory in [self.model_dir, self.data_dir, self.results_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_results = {}
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 ModelTrainer initialized")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Target F1: > {self.target_f1}")
    
    def load_and_prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load and prepare training datasets using DatasetManager.
        
        This method:
        1. Uses DatasetManager to download and prepare datasets
        2. Loads prepared train.csv and val.csv if they exist
        3. Converts pandas DataFrames to Hugging Face Dataset objects
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info("📥 Loading and preparing datasets...")
        
        # Check if prepared datasets already exist
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            logger.info("✅ Found existing prepared datasets")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
        else:
            logger.info("📦 Preparing datasets from scratch...")
            dataset_manager = DatasetManager(data_dir=self.data_dir, results_dir=self.results_dir)
            train_df, val_df = dataset_manager.prepare_training_data()
            
            # Generate statistics report
            dataset_manager.generate_statistics_report()
        
        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
        eval_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
        
        logger.info(f"✅ Datasets loaded: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
        
        # Log class distribution
        logger.info("📊 Training set class distribution:")
        for label_id, label_name in ID2LABEL.items():
            count = (train_df['label'] == label_id).sum()
            logger.info(f"   {label_name:15s}: {count:,}")
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        return train_dataset, eval_dataset
    
    def initialize_model(self):
        """
        Initialize MuRIL model with 3-class classification head.
        
        This method:
        1. Loads the tokenizer from Hugging Face
        2. Loads the model with a classification head for 3 classes
        3. Configures label mappings (id2label, label2id)
        """
        logger.info(f"🧠 Initializing model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ Tokenizer loaded")
            
            # Load model with classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            logger.info(f"✅ Model loaded with {self.num_labels} classification labels")
            logger.info(f"   Label mapping: {ID2LABEL}")
            
            # Move model to device
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def tokenize_datasets(self):
        """
        Tokenize train and eval datasets using the loaded tokenizer.
        
        This method applies tokenization to both datasets and sets the format
        for PyTorch training.
        """
        logger.info("🔤 Tokenizing datasets...")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call initialize_model() first.")
        
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError("Datasets not loaded. Call load_and_prepare_datasets() first.")
        
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(tokenize_fn, batched=True)
        self.eval_dataset = self.eval_dataset.map(tokenize_fn, batched=True)
        
        # Set format for PyTorch
        self.train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        self.eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        logger.info("✅ Tokenization complete")
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics (accuracy, precision, recall, F1).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary with computed metrics
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Load metrics
        f1_metric = evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        
        # Compute metrics
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
        
        return {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    
    def train(self) -> Dict[str, float]:
        """
        Execute training loop with metrics tracking.
        
        This method:
        1. Configures TrainingArguments with hyperparameters
        2. Creates Trainer with compute_metrics callback
        3. Executes training loop
        4. Evaluates on validation set
        5. Returns final metrics
        
        Returns:
            Dictionary with final evaluation metrics
        """
        logger.info("🏋️ Starting training...")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Learning rate: {self.learning_rate}")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError("Datasets not prepared. Call load_and_prepare_datasets() and tokenize_datasets() first.")
        
        try:
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.results_dir, "checkpoints"),
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_steps=10,
                logging_dir=self.logs_dir,
                report_to="none",
                push_to_hub=False,
                fp16=torch.cuda.is_available(),  # Mixed precision on GPU
                save_total_limit=2,  # Keep only 2 best checkpoints
            )
            
            # Create Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
            
            logger.info("✅ Trainer configured")
            
            # Execute training
            logger.info("🚀 Training started...")
            train_result = self.trainer.train()
            
            logger.info("✅ Training completed")
            logger.info(f"   Training loss: {train_result.training_loss:.4f}")
            
            # Evaluate on validation set
            logger.info("📊 Evaluating on validation set...")
            eval_results = self.trainer.evaluate()
            
            logger.info("✅ Evaluation completed")
            logger.info(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
            logger.info(f"   F1 Score: {eval_results['eval_f1']:.4f}")
            logger.info(f"   Precision: {eval_results['eval_precision']:.4f}")
            logger.info(f"   Recall: {eval_results['eval_recall']:.4f}")
            
            # Store results
            self.training_results = {
                'training_loss': train_result.training_loss,
                'eval_accuracy': eval_results['eval_accuracy'],
                'eval_f1': eval_results['eval_f1'],
                'eval_precision': eval_results['eval_precision'],
                'eval_recall': eval_results['eval_recall'],
                'train_samples': len(self.train_dataset),
                'eval_samples': len(self.eval_dataset),
            }
            
            # Check if target F1 achieved
            if eval_results['eval_f1'] >= self.target_f1:
                logger.info(f"🎉 SUCCESS — F1 target ({self.target_f1}) achieved!")
            else:
                logger.warning(f"⚠️  F1 score ({eval_results['eval_f1']:.4f}) below target ({self.target_f1})")
                logger.warning("   Recommendations:")
                logger.warning("   - Collect more training data")
                logger.warning("   - Increase number of epochs")
                logger.warning("   - Try hyperparameter tuning (learning rate, batch size)")
            
            return self.training_results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("❌ Out of memory error!")
                logger.error("   Try reducing batch size or using a smaller model")
                logger.error(f"   Current batch size: {self.batch_size}")
                logger.error("   Suggestion: Reduce batch size to 8 or 4")
            raise
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_model(self, path: Optional[str] = None):
        """
        Save trained model, tokenizer, and configuration.
        
        Args:
            path: Optional custom path to save model (defaults to MODEL_DIR)
        """
        save_path = path if path else self.model_dir
        
        logger.info(f"💾 Saving model to {save_path}...")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized. Train the model first.")
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save model card with training details
            model_card = {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "id2label": ID2LABEL,
                "label2id": LABEL2ID,
                "training_config": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "warmup_ratio": self.warmup_ratio,
                    "max_length": self.max_length,
                },
                "training_results": self.training_results,
                "trained_on": datetime.now().isoformat(),
            }
            
            model_card_path = os.path.join(save_path, "model_card.json")
            with open(model_card_path, 'w', encoding='utf-8') as f:
                json.dump(model_card, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Model saved successfully")
            logger.info(f"   Model weights: {save_path}")
            logger.info(f"   Tokenizer: {save_path}")
            logger.info(f"   Model card: {model_card_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report with metrics and comparison to baseline.
        
        This method creates a detailed report including:
        - Final metrics (accuracy, F1, precision, recall)
        - Comparison to GPT-4 baseline (0.61 F1)
        - Training configuration
        - Recommendations if target not met
        
        Returns:
            Dictionary containing the complete training report
        """
        logger.info("📊 Generating training report...")
        
        if not self.training_results:
            raise ValueError("No training results available. Train the model first.")
        
        # GPT-4 baseline F1 score
        gpt4_baseline_f1 = 0.61
        
        # Calculate improvement over baseline
        improvement = self.training_results['eval_f1'] - gpt4_baseline_f1
        improvement_pct = (improvement / gpt4_baseline_f1) * 100
        
        # Create report
        report = {
            "model_info": {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "label_mapping": ID2LABEL,
            },
            "training_config": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "max_length": self.max_length,
            },
            "dataset_info": {
                "train_samples": self.training_results['train_samples'],
                "eval_samples": self.training_results['eval_samples'],
                "total_samples": self.training_results['train_samples'] + self.training_results['eval_samples'],
            },
            "final_metrics": {
                "accuracy": self.training_results['eval_accuracy'],
                "f1_score": self.training_results['eval_f1'],
                "precision": self.training_results['eval_precision'],
                "recall": self.training_results['eval_recall'],
                "training_loss": self.training_results['training_loss'],
            },
            "baseline_comparison": {
                "gpt4_baseline_f1": gpt4_baseline_f1,
                "our_f1": self.training_results['eval_f1'],
                "improvement": round(improvement, 4),
                "improvement_percentage": round(improvement_pct, 2),
            },
            "target_achievement": {
                "target_f1": self.target_f1,
                "achieved": self.training_results['eval_f1'] >= self.target_f1,
                "gap": round(self.target_f1 - self.training_results['eval_f1'], 4) if self.training_results['eval_f1'] < self.target_f1 else 0,
            },
            "recommendations": [],
            "generated_at": datetime.now().isoformat(),
        }
        
        # Add recommendations if target not met
        if self.training_results['eval_f1'] < self.target_f1:
            report["recommendations"] = [
                "Collect more training data from diverse sources",
                f"Increase training epochs from {self.epochs} to {self.epochs + 2}",
                "Try hyperparameter tuning (learning rate, batch size)",
                "Consider data augmentation techniques",
                "Experiment with different model architectures (XLM-RoBERTa)",
            ]
        else:
            report["recommendations"] = [
                "Model achieved target F1 score",
                "Consider deploying to production",
                "Monitor performance on real-world data",
                "Collect user feedback for continuous improvement",
            ]
        
        # Save report
        report_path = os.path.join(self.results_dir, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Training report saved to {report_path}")
        
        # Log summary
        logger.info("=" * 60)
        logger.info("📊 TRAINING REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Training samples: {report['dataset_info']['train_samples']:,}")
        logger.info(f"Validation samples: {report['dataset_info']['eval_samples']:,}")
        logger.info("-" * 60)
        logger.info("Final Metrics:")
        logger.info(f"  Accuracy:  {report['final_metrics']['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {report['final_metrics']['f1_score']:.4f}")
        logger.info(f"  Precision: {report['final_metrics']['precision']:.4f}")
        logger.info(f"  Recall:    {report['final_metrics']['recall']:.4f}")
        logger.info("-" * 60)
        logger.info("Baseline Comparison:")
        logger.info(f"  GPT-4 Baseline F1: {report['baseline_comparison']['gpt4_baseline_f1']:.4f}")
        logger.info(f"  Our F1:            {report['baseline_comparison']['our_f1']:.4f}")
        logger.info(f"  Improvement:       +{report['baseline_comparison']['improvement']:.4f} ({report['baseline_comparison']['improvement_percentage']:.2f}%)")
        logger.info("-" * 60)
        logger.info(f"Target F1: {report['target_achievement']['target_f1']:.4f}")
        logger.info(f"Achieved: {'✅ YES' if report['target_achievement']['achieved'] else '❌ NO'}")
        if not report['target_achievement']['achieved']:
            logger.info(f"Gap: {report['target_achievement']['gap']:.4f}")
        logger.info("=" * 60)
        
        return report


def main():
    """
    Main function to run the complete training pipeline.
    
    This function demonstrates the full workflow:
    1. Initialize ModelTrainer
    2. Load and prepare datasets
    3. Initialize model
    4. Tokenize datasets
    5. Train model
    6. Save model
    7. Generate training report
    """
    logger.info("🚀 Starting Telugu Fake News Classifier Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load and prepare datasets
        trainer.load_and_prepare_datasets()
        
        # Initialize model
        trainer.initialize_model()
        
        # Tokenize datasets
        trainer.tokenize_datasets()
        
        # Train model
        metrics = trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Generate training report
        report = trainer.generate_training_report()
        
        logger.info("=" * 60)
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("=" * 60)
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
