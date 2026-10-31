"""Predict new recipes using a bundled inference-only SAC-Gen model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sacfactory.pretrained import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--model',default='SAC-Gen-GB')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    content=json.loads(args.recipe.read_text(encoding='utf-8'))
    predictions=load(args.model,args.device).predict(content if isinstance(content,list) else [content])
    args.output.write_text(json.dumps(predictions,indent=2,default=str)+'\n',encoding='utf-8')
    print('Saved forecasts:',len(predictions))


if __name__=='__main__':main()
