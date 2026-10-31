"""Register a fixed science control alongside the completed experimental-only model."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import read_source,dump,markdown_table
from scripts.refine_sac_gen import point_rows,summarize
from sacfactory.planning import fingerprint
from sacfactory.science_only import SciencePriorModel
from sacfactory.weighted_validation import collect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();app=Path(__file__).resolve().parents[1]
    base=app/'data'/'comparison_controls'/'real'
    registry=json.loads((base/'latest.json').read_text())
    samples,methods=read_source(app/'data')
    if fingerprint(samples)!=registry['source_fingerprint']:raise ValueError('Science and data-only snapshots differ.')
    records,_=collect(samples,methods,'REAL','periodic',use_thermal=False)
    model=SciencePriorModel('REAL');predictions=model.predict([r['sample'] for r in records])
    frame=pd.DataFrame(point_rows(records,predictions));frame['model']='Science prior (fixed)';frame['fold']='FIXED_PRIOR';frame['repeat']=0
    frame.to_csv(output/'science_prior_predictions.csv',index=False)
    result=summarize(frame);pd.DataFrame([result]).to_csv(output/'science_prior_metrics.csv',index=False)
    checkpoint=base/registry['run_id']/'science_prior'
    if checkpoint.exists():
        if SciencePriorModel.load(checkpoint).model_id!=model.model_id:raise ValueError('A different science control already exists in this run.')
    else:model.save(checkpoint)
    registry['science']={'model_id':model.model_id,'checkpoint':'science_prior','fitted_to_current_experiments':False}
    dump(checkpoint.parent/'registry.json',registry)
    current,_=read_source(app/'data')
    if fingerprint(current)!=registry['source_fingerprint']:raise ValueError('Experiments changed before science-control registration.')
    temp=base/'latest.pending.json';dump(temp,registry);temp.replace(base/'latest.json')
    dump(output/'trained_controls.json',registry)
    dump(output/'science_control_addendum.json',{'definition':'Fixed physical/chemical prior; no current-label optimization',
        'preference_assumption':'Recommended fixed-prior interpretation; descriptor-only supervised fitting is a different optional control',
        'empirical_anchor_disclosure':'User-reported carbonization, pulse-loss and Fe retention anchors remain fixed.',
        'model_id':model.model_id,'source_fingerprint':registry['source_fingerprint'],'metrics':result})
    comparison=pd.read_csv(output/'information_model_comparison.csv')
    comparison=comparison.loc[comparison.model!='Science prior (fixed)']
    comparison=pd.concat([comparison,pd.DataFrame([{'model':'Science prior (fixed)','scope':'Fixed evaluation; no fitting',**result}])],ignore_index=True)
    comparison.to_csv(output/'information_model_comparison.csv',index=False)
    report=output/'REPORT.md';text=report.read_text(encoding='utf-8')
    text+='\n\n## Fixed science-prior control\n\n'+markdown_table(pd.DataFrame([result]))+'\n\n'
    text+='This reference is frozen rather than trained on current labels. It uses mass conservation, '
    text+='thermal retention, mixing and kinetic-prior calculations. User-reported carbonization/pulse-loss/Fe-retention '
    text+='anchors are empirical assumptions, not first-principles constants. Predicted XRD probabilities and '
    text+='prior sensitivity ranges are uncalibrated. No original measurements or labels were changed.\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps({'science_model_id':model.model_id,'fitted_to_current_experiments':False,'metrics':result}))


if __name__=='__main__':main()
