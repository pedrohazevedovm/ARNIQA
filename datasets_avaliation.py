import pandas as pd


hriq_df = pd.read_csv('hiq_mos_file.csv')
iqa_hriq_df = pd.read_csv('image_quality_scores_HRIQ.csv')

print(hriq_df.head())
print(iqa_hriq_df.head())
