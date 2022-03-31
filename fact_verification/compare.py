from tkinter.ttk import LabeledScale
import pandas as pd
import json
import numpy as np

with open('../data/test_phase_1_update_labelled.json', 'r') as f:
  gt = json.load(f)

with open('./output/roberta_zero_shot/best_model/predictions_test_phase_1_update.json', 'r') as f:
  test = json.load(f)

dfgt = pd.DataFrame(gt)['label'].to_numpy()

dftest = pd.DataFrame(test).transpose()['predicted_label'].to_numpy()

eq = np.sum(dftest==dfgt)

print(eq)

# print(df.keys())

# df['submission'] = None

# df['submission'] = df['predicted_label'].map({'NEI': 0, 'SUPPORTED': 1, 'REFUTED':2})

# np.savetxt(r'submission.txt', df.submission, fmt='%s')