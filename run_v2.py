"""Local inference of the final 1024-token cross-project suite checkpoint."""
import argparse,csv,hashlib,json,random,time,sys
from collections import defaultdict
from pathlib import Path
import core
import torch
from model_v2 import load_v2,historical_reference
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/v2';OUT.mkdir(exist_ok=True)
def dataset():
    directory=ROOT/'artifact_v2/data/codebert_token_diff_suite_outliers_cp_1024/test'
    paths=[directory/f'test_{i}' for i in range(6)]
    regenerated=any(not p.exists() for p in paths)
    if regenerated:
        directory=ROOT/'work/v2_generated_input';paths=[directory/f'test_{i}' for i in range(6)]
        if not (directory/'manifest.json').exists() or any(not p.exists() for p in paths):raise RuntimeError('Six official or locally regenerated final shards required')
        generation_manifest=json.loads((directory/'manifest.json').read_text())
    suites=[];sources=[]
    for path in paths:
        part=core.load_data(path)
        suites.extend(part);sources.extend((path.name,i) for i in range(len(part)))
    pairs=core.flatten(suites)
    if len(suites)!=1040 or len(pairs)!=42687:raise RuntimeError('Final dataset counts do not match paper')
    for p in pairs:
        if len(p['embed'])!=1024 or len(p['mask'])!=1024 or p['label'] not in [0,1]:raise RuntimeError('Invalid encoded v2 input')
        if sum(p['mask'])<3 or p['mask']!=[1]*sum(p['mask'])+[0]*(1024-sum(p['mask'])):raise RuntimeError('Invalid right-padding mask')
    manifest=json.loads((ROOT/'artifact_v2_manifest.json').read_text(encoding='utf-8'))
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT).as_posix()
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        expected=generation_manifest['files'][rel]['sha256'] if regenerated else manifest[path.relative_to(ROOT/'artifact_v2').as_posix()]['sha256']
        if sha!=expected:raise RuntimeError('Dataset hash mismatch')
        hashes[rel]=sha
    raw_hashes={}
    for i in range(6):
        path=ROOT/f'artifact_v2/data/base_set_suite_outliers_cp/test/test_{i}'
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=manifest[path.relative_to(ROOT/'artifact_v2').as_posix()]['sha256']:raise RuntimeError('Raw input hash mismatch')
        raw_hashes[path.relative_to(ROOT).as_posix()]=sha
    origin='Regenerated locally from official final raw Java suites by executing unmodified author v2 preprocessing functions' if regenerated else 'Official preprocessed final v2 suite shards'
    return suites,sources,pairs,hashes,raw_hashes,origin
def parity_check(fresh,old,pairs,device):
    random_indices=random.Random(2026).sample(range(len(pairs)),48)
    # Include the longest retained inputs, so compatibility is checked at 1024.
    long_indices=sorted(range(len(pairs)),key=lambda i:sum(pairs[i]['mask']),reverse=True)[:16]
    indices=sorted(set(random_indices+long_indices));old=old.to(device).eval();rows=[]
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            selected=indices[start:start+4];part=[pairs[i] for i in selected]
            ids=torch.tensor([p['embed'] for p in part],device=device)
            mask=torch.tensor([p['mask'] for p in part],device=device)
            actual=fresh(ids,mask);expected=historical_reference(old,ids,mask)
            length=max(sum(p['mask']) for p in part);trimmed=fresh(ids[:,:length],mask[:,:length])
            for i,a,b,c in zip(selected,actual.cpu().tolist(),expected.cpu().tolist(),trimmed.cpu().tolist()):
                rows.append({'pair_index':i,'token_length':sum(pairs[i]['mask']),'modern':a,'legacy_formula':b,'trimmed':c})
    old.to('cpu');del old;torch.cuda.empty_cache()
    report={'n':len(rows),'max_abs_difference':max(abs(v['modern']-v['legacy_formula']) for v in rows),
            'trim_max_abs_difference':max(abs(v['modern']-v['trimmed']) for v in rows),
            'threshold_decision_disagreements':{str(t):sum((v['modern']>t)!=(v['legacy_formula']>t) for v in rows) for t in [.25,.5,.9]},
            'reference':'Independent historical HF 4.23.1 encoder formula on original serialized modules; current PyTorch FP32 CUDA',
            'records':rows}
    (OUT/'forward_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['max_abs_difference']>2e-5 or any(report['threshold_decision_disagreements'].values()):raise RuntimeError('V2 forward parity failed')
    print('v2 parity',report['n'],report['max_abs_difference'],flush=True)
def align_author_reference(suites,sources):
    with (OUT/'author_reference_suite_predictions.csv').open(encoding='utf-8-sig',newline='') as f:author=list(csv.DictReader(f))
    with (OUT/'author_reference_pair_predictions.csv').open(encoding='utf-8-sig',newline='') as f:author_pairs=list(csv.DictReader(f))
    by_suite=defaultdict(list)
    for row in author_pairs:by_suite[int(row['suite_index'])].append(row)
    signatures=[(int(row['label_killed']),tuple(int(v['label_killed']) for v in by_suite[i])) for i,row in enumerate(author)]
    mapping={};blocks=[]
    for shard in dict.fromkeys(v[0] for v in sources):
        local_indices=[i for i,v in enumerate(sources) if v[0]==shard]
        block=[(suites[i]['label'],tuple(p['label'] for p in suites[i]['mutants'])) for i in local_indices]
        starts=[i for i in range(len(signatures)-len(block)+1) if signatures[i:i+len(block)]==block]
        if len(starts)!=1:raise RuntimeError(f'Author block order cannot be uniquely aligned for {shard}: {starts}')
        start=starts[0]
        mapping.update({i:start+j for j,i in enumerate(local_indices)})
        blocks.append({'source_shard':shard,'local_global_start':local_indices[0],'author_global_start':start,'n_suites':len(block)})
    if len(mapping)!=1040 or set(mapping.values())!=set(range(1040)):raise RuntimeError('Author block alignment is not a complete bijection')
    report={'method':'Unique whole-shard matching on true suite labels and ordered pair-label vectors; probabilities are not used for alignment',
            'blocks':blocks,'author_index_for_local_suite':[mapping[i] for i in range(1040)]}
    (OUT/'author_alignment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Author shard order aligned',blocks,flush=True)
    return [author[mapping[i]] for i in range(1040)],[row for i in range(1040) for row in by_suite[mapping[i]]],mapping
def run(args):
    torch.manual_seed(10);torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this full local run')
    device=torch.device('cuda');suites,sources,pairs,hashes,raw_hashes,origin=dataset()
    author,author_pairs,author_map=align_author_reference(suites,sources)
    model,old,info=load_v2('suite',device)
    print('Loaded final checkpoint',json.dumps(info),'suites',len(suites),'pairs',len(pairs),flush=True)
    parity_check(model,old,pairs,device);del old
    if len(author)!=len(suites):raise RuntimeError('Author reference suite count mismatch')
    pilot=[]
    for i in [0,100,200,400,700,1039]:
        suite=suites[i];reference=author[i]
        if suite['label']!=int(reference['label_killed']) or len(suite['mutants'])!=int(reference['n_covering_tests']):raise RuntimeError(f'Pilot dataset alignment mismatch {i}')
        p,_=core.predict(model,suite['mutants'],args.batch_size,device)
        actual=max(p);expected=float(reference['max_p_killed'])
        pilot.append({'suite_index':i,'local_max_p_killed':actual,'author_max_p_killed':expected,'abs_difference':abs(actual-expected)})
    (OUT/'pilot_comparison.json').write_text(json.dumps(pilot,indent=2),encoding='utf-8')
    print('v2 pilot',json.dumps(pilot),flush=True)
    if max(v['abs_difference'] for v in pilot)>1e-3:raise RuntimeError('Pilot differs materially from stored author predictions; investigate before full run')
    scores,seconds=core.predict(model,pairs,args.batch_size,device)
    rows=core.aggregate(suites,scores)
    for i,(row,source) in enumerate(zip(rows,sources)):
        row['source_shard'],row['source_shard_suite_index']=source
        row['author_suite_index']=author_map[i]
    core.save_csv(OUT/'test_suite_predictions.csv',rows)
    pair_rows=[];offset=0
    for i,suite in enumerate(suites):
        for j,pair in enumerate(suite['mutants']):
            pair_rows.append({'suite_index':i,'source_shard':sources[i][0],'source_shard_suite_index':sources[i][1],
                              'author_suite_index':author_map[i],
                              'test_index':j,'label_killed':int(pair['label']),'p_killed':scores[offset],'pred_killed_0_5':int(scores[offset]>.5)})
            offset+=1
    core.save_csv(OUT/'test_pair_predictions.csv',pair_rows)
    if len(author)!=len(rows):raise RuntimeError('Author reference suite count mismatch')
    for i,(a,b) in enumerate(zip(rows,author)):
        if a['label_killed']!=int(b['label_killed']) or a['n_covering_tests']!=int(b['n_covering_tests']):raise RuntimeError(f'Author reference alignment mismatch at suite {i}')
    differences=[abs(a['max_p_killed']-float(b['max_p_killed'])) for a,b in zip(rows,author)]
    disagreements=sum((a['max_p_killed']>.25)!=(float(b['max_p_killed'])>.25) for a,b in zip(rows,author))
    if len(author_pairs)!=len(pair_rows):raise RuntimeError('Author pair count mismatch')
    pair_differences=[];pair_decisions={str(t):0 for t in [.25,.5,.9]}
    for actual,reference in zip(pair_rows,author_pairs):
        if (actual['author_suite_index'],actual['test_index'])!=(int(reference['suite_index']),int(reference['test_index'])):raise RuntimeError('Pair reference alignment mismatch')
        if reference['label_killed'] and actual['label_killed']!=int(reference['label_killed']):raise RuntimeError('Pair reference label mismatch')
        value=float(reference['p_killed']);pair_differences.append(abs(actual['p_killed']-value))
        for t in [.25,.5,.9]:pair_decisions[str(t)]+=int((actual['p_killed']>t)!=(value>t))
    labels=[p['label'] for p in pairs]
    summary={'scope':'Local fresh inference, final v2 cross-project suite checkpoint, all six 1024-token test shards',
             'fresh_inference':True,'checkpoint':info,'n_suites':len(rows),'n_pairs':len(scores),'device':str(device),
             'gpu':torch.cuda.get_device_name(0),'batch_size':args.batch_size,'inference_seconds':seconds,
             'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'python':sys.version},
             'input_dataset_origin':origin,'input_file_sha256':hashes,'raw_source_file_sha256':raw_hashes,
             'pair_level_suite_checkpoint':core.metrics(labels,[int(p>.5) for p in scores],1),
             'suite_level_threshold_0_25':core.suite_metrics(rows,.25),'suite_level_threshold_0_90':core.suite_metrics(rows,.9),
             'suite_label_vs_pair_or_mismatches':sum(v['label_killed']!=v['retained_tests_or_label'] for v in rows),
             'author_stored_prediction_comparison':{'alignment':'Unique whole-shard ground-label sequence matching; complete bijection, no probabilities used for alignment',
                  'max_suite_probability_abs_difference':max(differences),'mean_suite_probability_abs_difference':sum(differences)/len(differences),
                  'decision_disagreements_threshold_0_25':disagreements,
                  'pairs_compared':len(pair_differences),'max_pair_probability_abs_difference':max(pair_differences),
                  'mean_pair_probability_abs_difference':sum(pair_differences)/len(pair_differences),
                  'pair_threshold_decision_disagreements':pair_decisions},
             'limitations':['No retraining or Major mutation execution.','Pair metrics here use the suite-optimal checkpoint, not the paper matrix-optimal checkpoint.','Historical encoder formula parity checked under current PyTorch; historical training runtime not recreated.']}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL V2',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args();run(args)
