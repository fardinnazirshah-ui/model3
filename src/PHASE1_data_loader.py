x`"""
src/data_loader.py
Load economic data from FRED, World Bank, and NBER sources.
Handles encoding variations, missing values, and data validation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fred_data(
    filename: str,
    required_columns: List[str],
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Load FRED (Federal Reserve Economic Data) CSV with robust error handling.
    
    Handles:
    - Multiple encodings (UTF-8, Latin-1, ISO-8859-1)
    - Missing files
    - Empty data
    - Invalid date formats
    - Missing required columns
    - NaN validation
    
    Args:
        filename: Path to FRED CSV file
        required_columns: List of expected column names
        date_column: Name of date column (default: 'date')
    
    Returns:
        pd.DataFrame: Loaded and validated FRED data
        
    Raises:
        FileNotFoundError: If file not found
        ValueError: If data validation fails
        
    Example:
        >>> df = load_fred_data(
        ...     'data/raw/fred_raw_indicators.csv',
        ...     required_columns=['UNRATE', 'PAYEMS', 'DGS10', 'DGS2']
        ... )
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    df = None
    
    # Try multiple encodings
    for encoding in encodings:
        try:
            logger.info(f"Attempting to load {filename} with encoding {encoding}")
            df = pd.read_csv(filename, encoding=encoding)
            logger.info(f"✓ Successfully loaded {filename} with {encoding}")
            break
        except UnicodeDecodeError:
            logger.warning(f"✗ Encoding {encoding} failed, trying next...")
            continue
        except FileNotFoundError:
            logger.error(f"✗ File not found: {filename}")
            raise FileNotFoundError(f"Data file not found: {filename}")
        except pd.errors.EmptyDataError:
            logger.error(f"✗ File is empty: {filename}")
            raise ValueError(f"File is empty: {filename}")
        except Exception as e:
            logger.warning(f"✗ Loading with {encoding} failed: {e}")
            continue
    
    if df is None:
        raise ValueError(
            f"Could not load {filename} with any known encoding. "
            f"Tried: {encodings}"
        )
    
    # Validate required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"✗ Missing columns: {missing_cols}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        raise ValueError(
            f"Required columns missing: {missing_cols}. "
            f"Available: {df.columns.tolist()}"
        )
    
    # Convert date column
    if date_column in df.columns:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            logger.info(f"✓ Date range: {df[date_column].min()} to {df[date_column].max()}")
        except ValueError as e:
            logger.error(f"✗ Invalid date format in {date_column}: {e}")
            raise ValueError(f"Cannot parse {date_column}: {e}")
    
    # NaN validation
    nan_count = df.isnull().sum().sum()
    nan_pct = (nan_count / (len(df) * len(df.columns))) * 100
    logger.info(f"NaN count: {nan_count} ({nan_pct:.2f}%)")
    
    if nan_pct > 50:
        logger.warning(f"⚠ High NaN percentage: {nan_pct:.2f}%")
    
    logger.info(f"✓ Data shape: {df.shape}")
    return df


def load_worldbank_data(filename: str, date_column: str = 'year') -> pd.DataFrame:
    """
    Load World Bank data (usually annual, converted to monthly).
    
    Args:
        filename: Path to World Bank CSV file
        date_column: Name of date/year column (default: 'year')
    
    Returns:
        pd.DataFrame: Loaded World Bank data
        
    Example:
        >>> df = load_worldbank_data('data/raw/worldbank_raw_indicators.csv')
    """
    try:
        logger.info(f"Loading World Bank data from {filename}")
        df = pd.read_csv(filename, encoding='utf-8')
        logger.info(f"✓ Loaded World Bank data: {df.shape}")
        
        # World Bank data is often annual
        if date_column in df.columns:
            logger.info(f"✓ Date column ({date_column}) found - data is annual")
        
        return df
    except FileNotFoundError:
        logger.error(f"✗ World Bank data file not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to load World Bank data: {e}")
        raise


def load_nber_data(filename: str) -> pd.DataFrame:
    """
    Load NBER official recession dates.
    
    Expected columns: start_date, end_date, recession_name
    
    Args:
        filename: Path to NBER recession dates CSV
    
    Returns:
        pd.DataFrame: Recession dates with parsed datetime
        
    Example:
        >>> df = load_nber_data('data/raw/nber_recession_dates.csv')
        >>> print(df)
           start_date    end_date recession_name
        0  2001-03-01  2001-11-30  2001 Recession
        1  2007-12-01  2009-06-30  Great Recession
    """
    try:
        logger.info(f"Loading NBER recession dates from {filename}")
        df = pd.read_csv(filename, encoding='utf-8')
        
        # Parse date columns
        for col in ['start_date', 'end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        logger.info(f"✓ Loaded NBER recession dates: {len(df)} recession periods")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to load NBER data: {e}")
        raise


def align_economic_data(
    fred_df: pd.DataFrame,
    worldbank_df: pd.DataFrame,
    nber_df: pd.DataFrame,
    output_path: str = 'data/processed/aligned_economic_indicators_monthly.csv'
) -> pd.DataFrame:
    """
    Align economic data from FRED (monthly), World Bank (annual), and NBER.
    Creates recession labels and handles missing values.
    
    Args:
        fred_df: FRED monthly data
        worldbank_df: World Bank annual data
        nber_df: NBER recession dates
        output_path: Path to save aligned data
    
    Returns:
        pd.DataFrame: Aligned data with recession labels
        
    Steps:
        1. Interpolate World Bank annual data to monthly
        2. Merge FRED and World Bank on date
        3. Create recession labels from NBER dates
        4. Forward fill remaining NaN values
        5. Validate alignment
    """
    logger.info("Starting data alignment...")
    
    # 1. Interpolate World Bank data to monthly
    logger.info("Interpolating World Bank data to monthly...")
    if 'year' in worldbank_df.columns:
        worldbank_df['date'] = pd.to_datetime(
            worldbank_df['year'].astype(str) + '-01-01'
        )
        worldbank_df = worldbank_df.set_index('date').asfreq('MS').interpolate()
        worldbank_df = worldbank_df.reset_index()
    
    # 2. Merge FRED and World Bank
    logger.info("Merging FRED and World Bank data...")
    merged = fred_df.merge(worldbank_df, on='date', how='left')
    
    # 3. Create recession labels
    logger.info("Creating recession labels from NBER dates...")
    merged['recession_label'] = 0
    
    for _, row in nber_df.iterrows():
        start = row['start_date']
        end = row['end_date']
        mask = (merged['date'] >= start) & (merged['date'] <= end)
        merged.loc[mask, 'recession_label'] = 1
        logger.info(f"  Labeled {mask.sum()} months as recession: {start.date()} to {end.date()}")
    
    # Count recessions
    recession_count = merged['recession_label'].sum()
    normal_count = len(merged) - recession_count
    logger.info(f"✓ Recession months: {recession_count}, Normal months: {normal_count}")
    
    # 4. Forward fill NaN values
    logger.info("Forward filling NaN values...")
    merged = merged.fillna(method='ffill')
    
    # 5. Validate alignment
    remaining_nan = merged.isnull().sum().sum()
    if remaining_nan > 0:
        logger.warning(f"⚠ {remaining_nan} NaN values remain after forward fill")
        # Drop rows with NaN if still present
        merged = merged.dropna()
    
    logger.info(f"✓ Final aligned data shape: {merged.shape}")
    
    # Save aligned data
    merged.to_csv(output_path, index=False)
    logger.info(f"✓ Saved aligned data to {output_path}")
    
    return merged


def load_engineered_features(filename: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load pre-engineered features and extract X, y arrays.
    
    Args:
        filename: Path to engineered features CSV
    
    Returns:
        Tuple of (DataFrame, X array, y array)
        
    Example:
        >>> df, X, y = load_engineered_features('data/processed/features_engineered_monthly.csv')
        >>> print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    """
    try:
        logger.info(f"Loading engineered features from {filename}")
        df = pd.read_csv(filename)
        
        # Extract features and target
        X = df.drop(['date', 'recession_label'], axis=1).values
        y = df['recession_label'].values
        
        logger.info(f"✓ Loaded features: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"✓ Recession distribution: {np.sum(y)} recessions, {len(y) - np.sum(y)} normal")
        
        return df, X, y
    except Exception as e:
        logger.error(f"✗ Failed to load engineered features: {e}")
        raise


# Main execution
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("RECESSION PREDICTOR: Data Loading Pipeline")
    logger.info("=" * 80)
    
    try:
        # Define FRED indicators
        fred_columns = [
            'date', 'UNRATE', 'PAYEMS', 'CIVPART', 'INDPRO', 'HOUST',
            'CPIAUCSL', 'CPILFESL', 'UMCSENT', 'ICSA',
            'DGS10', 'DGS2', 'DFF', 'VIXCLS'
        ]
        
        # Load data
        logger.info("\nStep 1: Loading FRED data...")
        fred_df = load_fred_data(
            'data/raw/fred_raw_indicators.csv',
            required_columns=fred_columns[1:]
        )
        
        logger.info("\nStep 2: Loading World Bank data...")
        wb_df = load_worldbank_data('data/raw/worldbank_raw_indicators.csv')
        
        logger.info("\nStep 3: Loading NBER recession dates...")
        nber_df = load_nber_data('data/raw/nber_recession_dates.csv')
        
        # Align data
        logger.info("\nStep 4: Aligning economic data...")
        aligned_df = align_economic_data(fred_df, wb_df, nber_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Data loading pipeline completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
