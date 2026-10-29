"""Simple launcher for bundled webcam-free cases. Run: python run_test.py"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CASES={
    '1':('hand_sweep','Large moving hand, rotation and scale'),
    '2':('perspective','Changing camera/page angle, no hand'),
    '3':('writing','Perspective + moving hand + new text (updates ON)'),
    '4':('stress','Large hand / major occlusion: deliberate tracking stress test'),
    '5':('original','Original NEW INK demo with the text-update fix'),
}
parser=argparse.ArgumentParser()
parser.add_argument('case',nargs='?',choices=list(CASES))
parser.add_argument('--headless',action='store_true')
args=parser.parse_args()
choice=args.case
if choice is None:
    for k,(_,label) in CASES.items():print(f'{k}. {label}')
    choice=input('Choose 1-5: ').strip()
if choice not in CASES:raise SystemExit('Choose a number from 1 to 5.')
name,_=CASES[choice]
folder=ROOT/('demo' if name=='original' else 'test_cases/'+name)
video=folder/('input.avi' if name=='original' else 'input.mp4')
config=folder/('demo_config.json' if name=='original' else 'config.json')
cmd=[sys.executable,'-m','paper_cloak','--source',str(video),'--corners',str(folder/'corners.json'),'--config',str(config)]
if args.headless:cmd+=['--headless','--output',str(ROOT/'recordings'/f'{name}.avi')]
raise SystemExit(subprocess.call(cmd,cwd=ROOT))
