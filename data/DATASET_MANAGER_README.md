# Dataset Manager Module

## Overview

The `dataset_manager.py` module provides comprehensive functionality for downloading, filtering, balancing, and preparing Telugu fake news datasets for training. It handles data from multiple sources including FACTIFY, IndicGLUE, and RSS feeds.

## Features

### 1. Telugu Content Detection
- **Unicode-based detection**: Uses Telugu Unicode range (U+0C00-U+0C7F) to identify Telugu text
- **Mixed script handling**: Correctly processes text containing both Telugu and English

### 2. Dataset Download
- **FACTIFY**: Downloads claim-verification pairs from Hugging Face
- **IndicGLUE**: Downloads Telugu NLI and sentiment classification data
- **Error handling**: Gracefully handles download failures and continues with available data

### 3. Data Filtering
- **Telugu extraction**: Filters datasets to keep only Telugu-language samples
- **Label mapping**: Converts various label formats to standardized 3-class schema:
  - 0 = Real (true, supported, verified)
  - 1 = Fake (fake, refuted, false)
  - 2 = Unverifiable (nei, other, not enough info)

### 4. Data Quality Checks
- **Duplicate removal**: Removes duplicate text samples
- **Length filtering**: Filters out samples shorter than minimum length (default: 15 chars)
- **Label validation**: Ensures all labels are in valid range (0, 1, 2)

### 5. Dataset Balancing
- **Oversampling**: Duplicates minority classes to match majority class
- **Undersampling**: Reduces majority classes to match minority class
- **Maintains class distribution**: Ensures balanced training data

### 6. Train/Validation Split
- **Stratified splitting**: Maintains class distribution in both splits
- **Configurable ratio**: Default 80/20 train/val split
- **Reproducible**: Uses fixed random seed for consistency

### 7. Statistics Reporting
- **Comprehensive metrics**: Total samples, class distribution, source breakdown
- **Text statistics**: Average, min, and max text length
- **JSON export**: Saves report to `results/dataset_stats.json`

## Usage

### Basic Usage

```python
from src.dataset_manager import DatasetManager

# Create manager instance
manager = DatasetManager()

# Prepare complete training dataset
train_df, val_df = manager.prepare_training_data(
    balance_strategy="oversample",
    test_size=0.2
)

# Generate statistics report
stats = manager.generate_statistics_report()
```

### Advanced Usage

```python
# Step-by-step pipeline
manager = DatasetManager()

# 1. Download datasets
factify_df = manager.download_factify()
indicglue_df = manager.download_indicglue()

# 2. Filter Telugu content
telugu_factify = manager.filter_telugu_factify(
    factify_df,
    output_path="data/factify_telugu.csv"
)

# 3. Quality checks
cleaned = manager.remove_duplicates(telugu_factify)
filtered = manager.filter_by_length(cleaned, min_length=15)
validated = manager.validate_labels(filtered)

# 4. Balance dataset
balanced = manager.balance_dataset(validated, strategy="oversample")

# 5. Create splits
train_df, val_df = manager.create_train_val_split(balanced, test_size=0.2)
```

### Filtering Existing Data

```python
import pandas as pd
from src.dataset_manager import DatasetManager

manager = DatasetManager()

# Load existing dataset
df = pd.read_csv("data/raw_data.csv")

# Filter Telugu content
telugu_only = df[df['text'].apply(manager.is_telugu)]

# Map labels
telugu_only['label'] = telugu_only['label'].apply(manager._map_factify_label)
```

## API Reference

### DatasetManager Class

#### Methods

**`__init__(data_dir, results_dir)`**
- Initialize DatasetManager with custom directories
- Default: Uses `DATA_DIR` and `RESULTS_DIR` from config

**`is_telugu(text: str) -> bool`**
- Check if text contains Telugu Unicode characters
- Returns: True if Telugu detected, False otherwise

**`download_factify() -> Optional[pd.DataFrame]`**
- Download FACTIFY dataset from Hugging Face
- Returns: DataFrame or None if download fails

**`filter_telugu_factify(input_df, output_path=None) -> pd.DataFrame`**
- Extract Telugu-only samples from FACTIFY
- Maps labels to 3-class schema
- Optionally saves to CSV

**`download_indicglue() -> Optional[pd.DataFrame]`**
- Download IndicGLUE Telugu splits
- Returns: DataFrame or None if download fails

**`balance_dataset(df, strategy="oversample") -> pd.DataFrame`**
- Balance class distribution
- Strategies: "oversample" or "undersample"
- Returns: Balanced DataFrame

**`create_train_val_split(df, test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]`**
- Create stratified train/validation split
- Returns: (train_df, val_df)

**`remove_duplicates(df) -> pd.DataFrame`**
- Remove duplicate text samples
- Returns: Deduplicated DataFrame

**`filter_by_length(df, min_length=15) -> pd.DataFrame`**
- Filter out short samples
- Returns: Filtered DataFrame

**`validate_labels(df) -> pd.DataFrame`**
- Ensure labels are in valid range (0, 1, 2)
- Returns: Validated DataFrame

**`prepare_training_data(balance_strategy="oversample", test_size=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]`**
- Complete pipeline: download, filter, balance, split
- Returns: (train_df, val_df)

**`generate_statistics_report() -> Dict[str, Any]`**
- Generate comprehensive dataset statistics
- Saves to `results/dataset_stats.json`
- Returns: Statistics dictionary

## Label Mapping

The module standardizes various label formats to a 3-class schema:

| Original Label | Mapped Value | Class Name |
|---------------|--------------|------------|
| true, real, supported, verified | 0 | Real |
| fake, false, refuted | 1 | Fake |
| unverifiable, nei, other, not enough info | 2 | Unverifiable |

## Data Quality Checks

The module performs several quality checks:

1. **Duplicate Detection**: Removes exact duplicate text samples
2. **Length Filtering**: Removes samples shorter than 15 characters (configurable)
3. **Label Validation**: Ensures all labels are 0, 1, or 2
4. **Telugu Detection**: Filters out non-Telugu content

## Output Files

The module creates the following files:

- `data/factify_telugu.csv`: Filtered FACTIFY Telugu samples
- `data/indicglue_telugu.csv`: IndicGLUE Telugu samples
- `data/train.csv`: Training dataset
- `data/val.csv`: Validation dataset
- `results/dataset_stats.json`: Statistics report

## Testing

The module includes comprehensive tests:

- **Unit tests**: 20 tests covering individual functions
- **Integration tests**: 5 tests covering complete pipelines
- **Code coverage**: 71% (243 statements, 71 missed)

Run tests:
```bash
pytest tests/test_dataset_manager.py -v
pytest tests/test_dataset_manager_integration.py -v
```

## Examples

See `examples/dataset_manager_example.py` for complete usage examples:

```bash
python examples/dataset_manager_example.py
```

## Requirements

The module requires the following dependencies:

- pandas
- scikit-learn
- datasets (Hugging Face)
- numpy

Install with:
```bash
pip install pandas scikit-learn datasets numpy
```

## Error Handling

The module handles errors gracefully:

- **Download failures**: Logs error and continues with available data
- **Missing columns**: Detects and adapts to different column names
- **Invalid labels**: Maps unknown labels to "Unverifiable" (2)
- **Empty datasets**: Returns empty DataFrames with proper structure

## Logging

The module uses Python's logging module with INFO level:

- Dataset download progress
- Filtering statistics
- Class distribution changes
- File save confirmations

## Implementation Details

### Requirements Satisfied

This implementation satisfies the following requirements from the spec:

- **Requirement 1.1**: Downloads FACTIFY dataset (50,000+ samples)
- **Requirement 1.2**: Filters Telugu content using Unicode detection
- **Requirement 1.3**: Downloads IndicGLUE Telugu splits
- **Requirement 1.5**: Balances dataset with equal class samples
- **Requirement 1.6**: Saves datasets with clear naming conventions
- **Requirement 18.6**: Creates stratified train/val split

### Design Alignment

The implementation follows the design document specifications:

- **Module Architecture**: Separate DatasetManager class with clear responsibilities
- **Data Flow**: Download → Filter → Clean → Balance → Split
- **Error Handling**: Graceful degradation with logging
- **Testing**: Comprehensive unit and integration tests

## Future Enhancements

Potential improvements for future versions:

1. **Parallel downloads**: Download multiple datasets concurrently
2. **Caching**: Cache downloaded datasets to avoid re-downloading
3. **Progress bars**: Add progress indicators for long operations
4. **Data augmentation**: Add text augmentation for minority classes
5. **Cross-validation**: Support k-fold cross-validation splits
6. **Custom sources**: Easy integration of additional data sources

## Support

For issues or questions:
- Check the test files for usage examples
- Review the example script in `examples/dataset_manager_example.py`
- Refer to the main project README for general information
