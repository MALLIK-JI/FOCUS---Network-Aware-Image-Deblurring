import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'full_gopro_evaluation_1111'
CSV_PATH = RESULTS_DIR / 'per_image_results.csv'
OUTPUT_DIR = RESULTS_DIR / 'final_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('=' * 65)
print('FULL GOPRO EVALUATION - FINAL STATISTICAL ANALYSIS')
print('=' * 65)
df = pd.read_csv(CSV_PATH)
required_columns = {'Sequence', 'Image', 'Model', 'MSE', 'PSNR', 'SSIM', 'Inference Time (seconds)'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f'Missing CSV columns: {missing}')
print('\n1. DATA VALIDATION')
print('Total rows:', len(df))
print('\nRows per model:')
print(df.groupby('Model').size())
df['Image_ID'] = df['Sequence'].astype(str) + '/' + df['Image'].astype(str)
print('\nUnique image pairs per model:')
print(df.groupby('Model')['Image_ID'].nunique())
print('\nDuplicate records:', df.duplicated(['Sequence', 'Image', 'Model']).sum())
expected_models = 4
expected_images = 1111
if len(df) != expected_models * expected_images:
    raise ValueError('Unexpected total row count.')
if (df.groupby('Model')['Image_ID'].nunique() != expected_images).any():
    raise ValueError('Image coverage is incomplete.')
if df.duplicated(['Sequence', 'Image', 'Model']).any():
    raise ValueError('Duplicate evaluation records found.')
print('\nCoverage validation passed.')
print('\n2. OVERALL MODEL STATISTICS')
metrics = ['MSE', 'PSNR', 'SSIM', 'Inference Time (seconds)']
summary = df.groupby('Model')[metrics].agg(['mean', 'std', 'median'])
print(summary.round(6))
summary.to_csv(OUTPUT_DIR / 'overall_model_statistics.csv')
print('\n3. SEQUENCE-WISE STATISTICS')
sequence_summary = df.groupby(['Sequence', 'Model'])[metrics].mean()
sequence_summary.to_csv(OUTPUT_DIR / 'sequence_wise_statistics.csv')
print(sequence_summary.round(5))
quality = df.groupby('Model')[['MSE', 'PSNR', 'SSIM']].mean()
quality['MSE_Rank'] = quality['MSE'].rank()
quality['PSNR_Rank'] = quality['PSNR'].rank(ascending=False)
quality['SSIM_Rank'] = quality['SSIM'].rank(ascending=False)
quality.to_csv(OUTPUT_DIR / 'model_rankings.csv')
print('\n4. MODEL RANKINGS')
print(quality.round(6))
plot_metrics = ['MSE', 'PSNR', 'SSIM']
for metric in plot_metrics:
    means = df.groupby('Model')[metric].mean()
    plt.figure(figsize=(10, 6))
    means.sort_values().plot(kind='bar')
    plt.title(f'Model Comparison - {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{metric.lower()}_comparison.png', dpi=200)
    plt.close()
times = df.groupby('Model')['Inference Time (seconds)'].mean()
plt.figure(figsize=(10, 6))
times.plot(kind='bar')
plt.title('Average Inference Time per Image')
plt.ylabel('Seconds')
plt.xlabel('Model')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'inference_time_comparison.png', dpi=200)
plt.close()
df.drop(columns=['Image_ID']).to_csv(OUTPUT_DIR / 'validated_full_results.csv', index=False)
print('\n' + '=' * 65)
print('ANALYSIS COMPLETED SUCCESSFULLY')
print('Output folder:', OUTPUT_DIR)
print('=' * 65)
