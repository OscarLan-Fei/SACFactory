"""Run the focused discovery and adjacent-workflow regression suite."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP));sys.path.insert(0,str(APP/'tests'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.chdir(APP)
    patterns=['test_discovery.py','test_discovery_ui.py','test_characterization.py','test_research_chat.py',
              'test_pending.py','test_pending_prediction.py','test_manual.py',
              'test_sac_gen_workflow.py','test_information_controls.py',
              'test_analysis_navigation.py','test_concentrations.py']
    suite=unittest.TestSuite()
    for pattern in patterns:suite.addTests(unittest.TestLoader().discover(str(APP/'tests'),pattern=pattern))
    with (out/'regression_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'passed':result.wasSuccessful(),'patterns':patterns}
    (out/'regression_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));sys.exit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
