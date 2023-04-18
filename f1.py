from sklearn.metrics import f1_score

import pandas as pd

lel = pd.read_excel('sciEnts2Way.xlsx')

gt = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-answers']['gt_score'].values]
pred_score = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-answers']['predicted_score'].values]
print(f1_score(gt, pred_score, average='micro', labels=[0, 1]))

gt = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-questions']['gt_score'].values]
pred_score = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-questions']['predicted_score'].values]
print(f1_score(gt, pred_score, average='micro', labels=[0, 1]))

gt = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-domains']['gt_score'].values]
pred_score = [0 if v == 'incorrect' else 1 for v in lel[lel['sub-set'] == 'test-unseen-domains']['predicted_score'].values]
print(f1_score(gt, pred_score, average='micro', labels=[0, 1]))

