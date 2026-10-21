"""Audit CSV round trips, source design preservation and exact latent mass closure."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sacfactory.domain import screen
from sacfactory.importers import read_table,samples_from_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT.parent/'SACFactory-demo-data')
    args=parser.parse_args()
    content=json.loads((args.directory/'demo_seed.json').read_bytes())
    source=content['samples']
    samples=read_table((args.directory/'samples.csv').read_bytes(),'samples.csv')
    assays=read_table((args.directory/'icp.csv').read_bytes(),'icp.csv')
    imported=samples_from_rows(samples,assays)
    assert len(imported)==len(source)==1476
    maximum_residual=0.
    for original,roundtrip in zip(source,imported):
        for key in ('sample_id','origin','batch_id','elements','concentrations_mM','pulse_count','pulse_limit_C','temperature_C','precursors'):
            assert original[key]==roundtrip[key],(original['sample_id'],key)
        assert screen(original,{})['status']==screen(roundtrip,{})['status']
        b=original['demo_mass_balance']
        residual=abs(b['recovered_sample_g']-b['carrier_after_pulse_g']-sum(e['retained_metal_g'] for e in b['elements'].values()))
        maximum_residual=max(maximum_residual,residual)
        assert residual<1e-12
        for e in b['elements'].values():
            assert 0<=e['retained_metal_g']<=e['feed_metal_g']+1e-12
            assert abs(e['feed_metal_g']-e['lost_metal_g']-e['retained_metal_g'])<1e-12
            if e['true_w_wt_percent'] is not None:
                assert abs(e['true_w_wt_percent']-100*e['retained_metal_g']/b['recovered_sample_g'])<1e-10
    report={'samples_checked':len(source),'sample_element_rows':len(assays),
        'csv_roundtrip_controls_chemistry_status':'PASS','latent_mass_closure':'PASS',
        'maximum_mass_closure_residual_g':maximum_residual,
        'scope':'CSV preserves standard recipe and reported-result fields. JSON retains all nested simulation/forecast provenance.'}
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
