"""Common final-v2 inference operations and project-local caches."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for key, directory in {
    'HF_HOME': 'hf-cache', 'TORCH_HOME': 'torch-cache',
    'CUDA_CACHE_PATH': 'cuda-cache', 'PIP_CACHE_DIR': 'pip-cache',
    'TEMP': 'tmp', 'TMP': 'tmp',
}.items():
    path = ROOT / 'work' / directory
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import csv, pickle, time
import numpy as np
import torch
from torch import nn
from transformers import RobertaModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError(f'Data shard contains a non-primitive global: {module}.{name}')

def load_data(path):
    with path.open('rb') as f: return PrimitiveUnpickler(f).load()

class MutationBERT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.trans=RobertaModel(config)
        self.linear=nn.Linear(config.hidden_size,2)
    def forward(self,ids,mask):
        cls=self.trans(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        return torch.softmax(self.linear(cls),dim=-1)[:,1]

def metrics(labels,predictions,positive=1):
    return {'positive_label':positive,'n':len(labels),
            'precision':float(precision_score(labels,predictions,pos_label=positive,zero_division=0)),
            'recall':float(recall_score(labels,predictions,pos_label=positive,zero_division=0)),
            'f1':float(f1_score(labels,predictions,pos_label=positive,zero_division=0)),
            'accuracy':float(accuracy_score(labels,predictions)),
            'confusion_matrix_labels_0_1':confusion_matrix(labels,predictions,labels=[0,1]).tolist()}

def predict(model,pairs,batch,device):
    output=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for pos in range(0,len(pairs),batch):
            part=pairs[pos:pos+batch]
            # Remove only right padding; actual token content and attention are retained.
            length=max(sum(p['mask']) for p in part)
            ids=torch.tensor([p['embed'][:length] for p in part],dtype=torch.long,device=device)
            mask=torch.tensor([p['mask'][:length] for p in part],dtype=torch.long,device=device)
            scores=model(ids,mask)
            output.extend(scores.float().cpu().tolist())
            if pos%(batch*100)==0:
                print('inference',pos,'/',len(pairs),'elapsed',round(time.perf_counter()-started,1),flush=True)
    return output,time.perf_counter()-started

def flatten(suites): return [pair for suite in suites for pair in suite['mutants']]

def aggregate(suites,scores):
    result=[]
    pos=0
    for i,suite in enumerate(suites):
        count=len(suite['mutants'])
        result.append({'sample_index':i,'label_killed':int(suite['label']),
                       'retained_tests_or_label':max(int(p['label']) for p in suite['mutants']),
                       'n_covering_tests':count,'max_p_killed':max(scores[pos:pos+count]),
                       'n_killing_tests':sum(int(p['label']) for p in suite['mutants'])})
        pos+=count
    if pos!=len(scores): raise RuntimeError('Aggregation did not consume every pair')
    return result

def suite_metrics(rows,threshold):
    labels=[r['label_killed'] for r in rows]
    preds=[int(r['max_p_killed']>threshold) for r in rows]
    out=metrics(labels,preds,positive=0)
    out.update({'threshold':threshold,'true_mutation_score':float(np.mean(labels)),
                'predicted_mutation_score':float(np.mean(preds)),
                'mutation_score_absolute_error':float(abs(np.mean(labels)-np.mean(preds)))})
    return out

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
