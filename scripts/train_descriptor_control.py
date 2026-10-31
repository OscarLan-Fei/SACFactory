"""Match the chosen data-only backend while changing to scientific descriptors."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import read_source,dump,markdown_table
from scripts.refine_sac_gen import point_rows,summarize
from sacfactory.compute_runtime import execution
from sacfactory.planning import fingerprint
from sacfactory.scientific_descriptors_model import ScientificDescriptorModel
from sacfactory.validation_groups import RANDOM_RECIPE_CV,assert_mask
from sacfactory.weighted_validation import collect,split_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();app=Path(__file__).resolve().parents[1]
    base=app/'data'/'comparison_controls'/'real';registry=json.loads((base/'latest.json').read_text())
    config=json.loads((output/'configuration.json').read_text());samples,methods=read_source(app/'data')
    if fingerprint(samples)!=registry['source_fingerprint']:raise ValueError('Descriptor and raw-control snapshots differ.')
    records,_=collect(samples,methods,'REAL','periodic',use_thermal=False)
    targets=sorted({e for r in records for e in r['sample']['elements']})
    backend=registry['data_only']['backend'];points=[];audit=[]
    with execution({'device':'cpu','threads':4,'cpu_jobs':4}):
        for repeat,seed in enumerate(config['seeds']):
            for index,(train,test,fold,_) in enumerate(split_records(records,samples,'REAL',targets,RANDOM_RECIPE_CV,5,seed=seed),1):
                prefix=output/f'descriptor_repeat_{repeat+1}_fold_{index}'
                if prefix.with_suffix('.csv').exists() and prefix.with_suffix('.json').exists():
                    frame=pd.read_csv(prefix.with_suffix('.csv'));info=json.loads(prefix.with_suffix('.json').read_text())
                else:
                    assert_mask([r['sample'] for r in train],[r['sample'] for r in test],True)
                    model=ScientificDescriptorModel(backend,seed).fit_records(train,'REAL')
                    frame=pd.DataFrame(point_rows(test,model.predict([r['sample'] for r in test])))
                    frame['model']='Scientific descriptors '+backend;frame['repeat']=repeat;frame['fold']=fold
                    info={'repeat':repeat,'fold':fold,'model_id':model.model_id,'info':model.info,
                        'train_ids':[r['sample']['sample_id'] for r in train],'test_ids':[r['sample']['sample_id'] for r in test]}
                    frame.to_csv(prefix.with_suffix('.csv'),index=False);dump(prefix.with_suffix('.json'),info)
                points.append(frame);audit.append(info)
                print(json.dumps({'stage':'Scientific descriptor control','repeat':repeat+1,'fold':index}),flush=True)
        full=pd.concat(points,ignore_index=True);full.to_csv(output/'scientific_descriptor_out_of_fold_predictions.csv',index=False)
        summary=pd.DataFrame([{'repeat':repeat,**summarize(p)} for repeat,p in full.groupby('repeat')])
        summary.to_csv(output/'scientific_descriptor_repeat_metrics.csv',index=False)
        model=ScientificDescriptorModel(backend,42).fit_records(records,'REAL')
        folder=base/registry['run_id']/'scientific_descriptors';model.save(folder)
        loaded=ScientificDescriptorModel.load(folder);assert loaded.model_id==model.model_id
    registry['scientific_descriptors']={'model_id':model.model_id,'backend':backend,'checkpoint':'scientific_descriptors',
        'fitted_to_current_experiments':True,'input_profile':'Scientific descriptors, no nominal element identity'}
    current,_=read_source(app/'data')
    if fingerprint(current)!=registry['source_fingerprint']:raise ValueError('Experiments changed before descriptor-model registration.')
    temp=base/'latest.pending.json';dump(temp,registry);temp.replace(base/'latest.json')
    dump(base/registry['run_id']/'registry.json',registry);dump(output/'trained_controls.json',registry)
    dump(output/'scientific_descriptor_validation_methods.json',{'folds':audit,
        'definition':'Scientific inputs with supervised ICP/XRD fitting; distinct from the unfitted science prior'})
    comparison=pd.read_csv(output/'information_model_comparison.csv')
    comparison=comparison.loc[comparison.model!='Scientific descriptors '+backend]
    result={'model':'Scientific descriptors '+backend,'scope':'Mean of three matched repeats',**summary.mean(numeric_only=True).to_dict()}
    comparison=pd.concat([comparison,pd.DataFrame([result])],ignore_index=True)
    comparison.to_csv(output/'information_model_comparison.csv',index=False)
    report=output/'REPORT.md';text=report.read_text(encoding='utf-8')
    text+='\n\n## Matched-backend scientific-descriptor control\n\n'+markdown_table(pd.DataFrame([result]))+'\n\n'
    text+='This additional control retains the selected data-only backend, training eligibility, classifier-calibration scheme '
    text+='and outer masks. Nominal element identities are replaced by scientific descriptors and mechanistic features. '
    text+='It is supervised using the experimental ICP/XRD labels and must not be described as experiment-free. '
    text+='The fixed science-prior reference above is the separate control without current-label fitting.\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
