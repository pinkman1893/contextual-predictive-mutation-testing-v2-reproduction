import argparse,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=4);args=parser.parse_args()
if args.batch_size<1:parser.error('batch-size must be positive')
for file in ['encode_v2_dataset.py','extract_v2_author_pairs.py']:
    runpy.run_path(str(ROOT/file),run_name='__main__')
from run_v2 import run
run(args)
from verify_v2 import verify
print(verify())
runpy.run_path(str(ROOT/'make_report.py'),run_name='__main__')
