"""Serialize the current synthetic campaign into inspectable data-release files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sacfactory.domain import screen
from sacfactory.importers import SAMPLE_COLUMNS,ICP_COLUMNS
from sacfactory.precursors import to_columns as precursor_columns


def write_csv(path,rows,columns=None):
    if columns is None:
        columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'SACFactory-demo-data')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from sacfactory.bootstrap import load_demo_seed
    content=load_demo_seed(ROOT);raw=json.dumps(content,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    samples=[];icp_rows=[];ledgers=[];batch_counts={}
    for s in content['samples']:
        b=s['demo_mass_balance'];latent=s['demo_observation']['screening_latent'];outcome=screen(s,{})
        record={k:s.get(k,'') for k in SAMPLE_COLUMNS}
        record.update(target_elements=';'.join(s['elements']),xrd_state=s['xrd']['state'],xrd_mode='MANUAL',
            xrd_source=s['xrd']['source'],xrd_reviewer=s['xrd']['reviewer'],
            icp_state=s['icp_review']['state'],icp_source=s['icp_review']['source'],icp_reviewer=s['icp_review']['reviewer'],
            stem_state='NOT_TESTED',demo_stage=s['demo_stage'],demo_sequence=s['demo_sequence'],
            result_reported_at=s['result_reported_at'],screening_status=outcome['status'],
            screening_label=outcome['label'],saved_p_screen_pass=s.get('proposal',{}).get('p_screen_pass'),
            latent_recovered_sample_g=b['recovered_sample_g'],latent_carrier_after_pulse_g=b['carrier_after_pulse_g'],
            latent_carrier_loss_percent=100*b['carrier_loss_fraction'],
            latent_xrd_fail_probability=latent.get('xrd_fail_probability'),
            workflow_case=s['demo_observation']['workflow_case'])
        samples.append(record)
        assays={a['element']:a for a in s['icp']}
        for e,entry in b['elements'].items():
            a=assays.get(e,{})
            row={k:'' for k in ICP_COLUMNS}
            row.update(sample_id=s['sample_id'],element=e,w_wt_percent=a.get('value',''),unit='wt%',
                flag=a.get('flag','MISSING'),method_id='',source=a.get('source','SIMULATED no assay result'),
                precursor_mM=s['concentrations_mM'][e],**precursor_columns(s,e))
            icp_rows.append(row)
            ledgers.append({'sample_id':s['sample_id'],'element':e,'origin':'DEMO','batch_id':s['batch_id'],
                'temperature_C':b['used_temperature_C'],'pulse_count':s['pulse_count'],'workflow_case':s['demo_observation']['workflow_case'],
                'feed_amount_mmol':entry['amount_mmol'],'feed_metal_g':entry['feed_metal_g'],
                'latent_retained_metal_g':entry['retained_metal_g'],'latent_lost_metal_g':entry['lost_metal_g'],
                'latent_retention_percent':100*entry['retention_fraction'],'latent_true_w_wt_percent':entry['true_w_wt_percent'],
                'observed_icp_value':a.get('value',''),'observed_icp_flag':a.get('flag','MISSING'),
                'assay_available':e in assays,'model_version':b['loss_model_version']})
        batch_counts[s['batch_id']]=batch_counts.get(s['batch_id'],0)+1
    write_csv(args.output/'samples.csv',samples)
    write_csv(args.output/'icp.csv',icp_rows,ICP_COLUMNS)
    write_csv(args.output/'element_mass_balance.csv',ledgers)
    write_csv(args.output/'batches.csv',[{'batch_id':b,'sample_count':n} for b,n in batch_counts.items()])
    from plot_pulse_losses import build_data
    carrier_grid,metal_grid=build_data()
    carrier_grid.to_csv(args.output/'deterministic_carrier_grid.csv',index=False)
    metal_grid.to_csv(args.output/'deterministic_all50_metal_grid.csv',index=False)
    (args.output/'demo_seed.json').write_bytes(raw)
    for filename in ('empirical_loss_settings.json','demo_observation_settings.json'):
        shutil.copyfile(ROOT/'resources'/filename,args.output/filename)
    summary={'demo_version':content['demo_version'],'sample_count':len(samples),'sample_element_rows':len(icp_rows),
        'batches':len(batch_counts),'distinct_recipes':content['unique_recipes'],'status_counts':content['status_counts'],
        'seed_sha256':hashlib.sha256(raw).hexdigest(),'loss_model_version':content['loss_model_version'],
        'loss_settings_sha256':content['loss_settings_sha256'],'loss_reference_sha256':content['loss_reference_sha256'],
        'observation_settings_sha256':content['observation_settings']['settings_sha256'],
        'data_origin':'SIMULATED_NOT_EXPERIMENTAL','files':{}}
    readme='''# Physically linked synthetic SAC campaign

This release contains 1476 simulated syntheses, not experimental results. The sequence is Fe development, the other 49 single-element targets, then mixtures. Every recipe, material declaration, batch and synthetic sensor readback is preserved from the frozen design; ICP/XRD observations and source-only forecasts are regenerated.

Use demo_seed.json as the authoritative complete application fixture. samples.csv and icp.csv are inspectable standard bulk-import tables; CSV import preserves recipes, chemistry and reported screening but does not restore every nested timestamp, forecast or latent-provenance field. Preserve origin DEMO when inspecting/importing. MISSING/censored entries stay blank or <limit, never numeric zero. element_mass_balance.csv contains generative latent values, including unobserved workflow cases; these are not future model inputs. Mass balance is exact for latent masses. Noisy ICP readings need not equal the latent mass fraction exactly. The two deterministic reference grids add all three requested temperatures and pulse counts 1-16, with 2400 individual one-mmol metal conditions. These grids are model evaluations, not extra synthesis records or training observations. The preserved campaign itself uses its original pulse counts through ten.

The reference is 2 g PAN, 20 mL DMF and 6% pre-pulse carbonization loss. One mmol is 50 mM. Fe retains a nominal 2.5 wt% reference at 1000 C / one pulse. Other metal loadings are not forced equal. Temperature/pulse-dependent empirical loss uses sourced melting/boiling references and a gas-atom formation-energy escape proxy; independent sublimation temperatures are absent for all fifty targets. Weights and response parameters are assumptions, not fitted kinetics.

Recorded peak temperature drives latent physics. Controlled cutoff and pre-synthesis references drive the source-only demonstration forecaster; actual readbacks, observed ICP/XRD and latent ledgers are not feature inputs. A shared 1.5% log-SD batch retention factor and 1.2% log-SD process noise precede simulated ICP readings with 2% log-SD noise. Noise scales are illustrative.

Synthetic ICP LOD/LOQ are 0.0015/0.005 wt% points, not real instrumental specifications. All targets must be detected for the simulated ICP PASS. BLQ can pass this presence screen when the synthetic signal is explicitly above LOD, but it never supplies a numeric loading score. XRD events are probabilistic surrogates of retained effective molar burden per remaining carrier, thermal mobility and incomplete conversion; they are not measured spectra or proof of single atoms. The Fe overload reference is 2.4 mmol at 1000 C / one pulse. No severe metal loss is triggered by Sample ID. A burnout, equipment abort and pending row remain explicit workflow examples, independent of the normal thermal prior. All-single-element progress is checked without relabeling points to force success.

Forecasts use earlier stages only with a declared 144-source-record budget. They are illustrative captures, not evidence of BO efficiency or transfer accuracy. The latent observation model and fitted screening engine remain separate. Weights and detection/event settings are included. REAL data are not part of this release. All content is English.
'''
    (args.output/'README.md').write_text(readme,encoding='utf-8')
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name!='manifest.json':
            summary['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    archive_path=args.output.parent/'SACFactory-demo-data.zip'
    with ZipFile(archive_path,'w',ZIP_DEFLATED) as archive:
        for path in sorted(args.output.iterdir()):
            if path.is_file():
                archive.write(path,'SACFactory-demo-data/'+path.name)
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))


if __name__=='__main__':
    main()
