"""Safely flatten stored final author probabilities for per-pair comparison."""
import pickle,io,collections,csv,json
from pathlib import Path
import core
import torch
ROOT=Path(__file__).resolve().parent
class PredictionsUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch.storage','_load_from_bytes'):return lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return torch._utils._rebuild_tensor_v2
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        raise pickle.UnpicklingError(f'Unexpected reference global {module}.{name}')
def run():
    path=ROOT/'artifact_v2/_results/preds/codebert_token_diff_suite_outliers_cp_1024_test_preds.pkl'
    with path.open('rb') as f:suites=PredictionsUnpickler(f).load()
    rows=[]
    for i,suite in enumerate(suites):
        labels=suite.get('labels')
        for j,score in enumerate(suite['scores']):
            rows.append({'suite_index':i,'test_index':j,'p_killed':float(score[0,1]),
                         'label_killed':int(labels[j].item()) if labels is not None else ''})
    if len(suites)!=1040 or len(rows)!=42687:raise RuntimeError('Final author prediction count mismatch')
    out=ROOT/'results/v2';out.mkdir(parents=True,exist_ok=True)
    core.save_csv(out/'author_reference_pair_predictions.csv',rows)
    suite_rows=[{'suite_index':i,'label_killed':int(suite['label']),
                 'n_covering_tests':len(suite['scores']),
                 'max_p_killed':max(float(score[0,1]) for score in suite['scores'])}
                for i,suite in enumerate(suites)]
    core.save_csv(out/'author_reference_suite_predictions.csv',suite_rows)
    print('Flattened author reference',len(suites),len(rows),'has_pair_labels',labels is not None,flush=True)
if __name__=='__main__':run()
