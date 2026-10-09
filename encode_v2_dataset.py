"""Regenerate v2 inputs from final raw Java data with the author's exact function."""
import ast,difflib,hashlib,json,pickle,time
from pathlib import Path
import core
from transformers import RobertaTokenizer
ROOT=Path(__file__).resolve().parent
def run():
    source=ROOT/'author_v2/src/preprocessing/codebert_token_diff_suite/mutants_to_dataset.py'
    expected=json.loads((ROOT/'provenance/artifact_files.json').read_text())['src/preprocessing/codebert_token_diff_suite/mutants_to_dataset.py']['sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise RuntimeError('Author preprocessing source hash mismatch')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    functions=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ['tokenize_str','subsample_mutants']]
    if len(functions)!=2:raise RuntimeError('Required author preprocessing functions missing')
    namespace={'difflib':difflib}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    tokenizer=RobertaTokenizer.from_pretrained(str(ROOT/'tokenizer'),local_files_only=True)
    out=ROOT/'work/v2_generated_input';out.mkdir(exist_ok=True)
    manifest={'method':'Executed the unmodified AST definitions tokenize_str and subsample_mutants from author v2 source; max_len=1024',
              'author_preprocessing_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'context':1024,'files':{}}
    total_suites=total_pairs=0;started=time.time()
    for i in range(6):
        raw_path=ROOT/f'artifact_v2/data/base_set_suite_outliers_cp/test/test_{i}'
        raw=core.load_data(raw_path);encoded=namespace['subsample_mutants'](raw,tokenizer,1024)
        path=out/f'test_{i}'
        with path.open('wb') as f:pickle.dump(encoded,f,protocol=4)
        with raw_path.open('rb') as f:raw_sha=hashlib.file_digest(f,'sha256').hexdigest()
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        count=sum(len(v['mutants']) for v in raw);total_suites+=len(raw);total_pairs+=count
        manifest['files'][path.relative_to(ROOT).as_posix()]={'sha256':sha,'raw_source':raw_path.relative_to(ROOT).as_posix(),
                    'raw_sha256':raw_sha,'suites':len(raw),'pairs':count}
        print('Encoded v2',i,len(raw),count,'seconds',round(time.time()-started,1),flush=True)
    if total_suites!=1040 or total_pairs!=42687:raise RuntimeError('Final raw data counts do not match publication')
    manifest.update(n_suites=total_suites,n_pairs=total_pairs,seconds=time.time()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':run()
