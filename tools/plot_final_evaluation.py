"""Plot frozen final results and independently check saved prediction arithmetic."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'reports/metrics/final_evaluation.json').read_text(encoding='utf8'))
assert s['status']=='completed' and s['final_test_evaluated'] and not s['new_training']
with np.load(ROOT/'outputs/final-evaluation-2026-11-11/test_predictions.npz') as a:
    assert a['truth'].shape==(178,36) and len(set(a['dates']))==178
    np.testing.assert_allclose(a['combined'],(a['rf']+a['lstm_component'])/2)
    for n,k in [('RF','rf'),('RF_LSTM_50','combined')]:
        np.testing.assert_allclose(abs(a[k]-a['truth']).mean(),s['results'][n]['metrics']['overall']['mae'],rtol=1e-12)
        months=s['results'][n]['metrics']['month']
        assert sum(v['days'] for v in months.values())==178
        np.testing.assert_allclose(sum(v['mae']*v['days'] for v in months.values())/178,s['results'][n]['metrics']['overall']['mae'])
fig,axes=plt.subplots(1,3,figsize=(14,4))
names=['RF','RF_LSTM_50'];labels=['RF','Fixed 50:50'];x=np.arange(2)
for j,key in enumerate(['mae','board_mae','alight_mae']):axes[0].bar(x+j*.23,[s['results'][n]['metrics']['overall'][key] for n in names],.23,label=key)
axes[0].set_xticks(x+.23,labels);axes[0].set_title('Final MAE (people)');axes[0].legend(fontsize=8)
for n,label in zip(names,labels):
    axes[1].plot(range(7,13),[s['results'][n]['metrics']['month'][f'2025-{m:02}']['mae'] for m in range(7,13)],marker='o',label=label)
axes[1].set_title('Monthly final MAE');axes[1].set_xlabel('Month');axes[1].legend()
for j,d in enumerate(['board','alight']):axes[2].bar(x+j*.35,[s['results'][n]['diagnostics'][d]['high']['mae'] for n in names],.35,label=d)
axes[2].set_xticks(x+.175,labels);axes[2].set_title('High-demand MAE (fixed H1 threshold)');axes[2].legend()
for ax in axes:ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
fig.suptitle('Frozen seed42 models | 2025 H2 final evaluation | 178 days / 6408 values | no refit')
fig.tight_layout();fig.savefig(ROOT/'reports/figures/final_evaluation.png',dpi=160);plt.close(fig)
print('Saved predictions, 50:50 arithmetic, aggregate MAE and month-weighted MAE verified.')
