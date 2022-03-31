import pandas as pd
import json
import numpy as np

with open('./output/roberta_zero_shot/best_model/predictions_test_phase_1_update.json', 'r') as f:
  data = json.load(f)

df = pd.DataFrame(data).transpose()

df['submission'] = None

df['submission'] = df['predicted_label'].map({'NEI': 0, 'SUPPORTED': 1, 'REFUTED':2})

np.savetxt(r'submission.txt', df.submission, fmt='%s')