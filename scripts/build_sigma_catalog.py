"""Serialize reviewed Sigma listing facts; no stock, price or actual-use claims."""
import json
import csv
from pathlib import Path

# Facts checked against the linked product/substance listings on 2026-10-13.
# Fields: element | formula | formal charge | CAS | catalog identifier | observed source.
FACTS = '''Mg|MgCl2·6H2O|2|7791-18-6|M0250|US/en/product/sigald/m0250
Al|AlCl3|3|7446-70-0|563919|US/en/product/aldrich/563919
K|KCl|1|7447-40-7|204099|US/en/product/aldrich/204099
Ca|CaCl2·2H2O|2|10035-04-8|223506|US/en/substance/calciumchloridedihydrate1470110035048
Sc|ScCl3·6H2O|3|20662-14-0|451274|US/en/product/aldrich/451274
Ti|TiCl2|2|10049-06-6|451738|US/en/product/aldrich/451738
V|VCl2|2|10580-52-6|422371|US/en/product/aldrich/422371
Cr|CrCl2|2|10049-05-5|802482|US/en/product/mm/802482
Mn|MnCl2·4H2O|2|13446-34-9|105927|US/en/product/mm/105927
Fe|FeCl2|2|7758-94-3|372870|US/en/product/aldrich/372870
Co|CoCl2·6H2O|2|7791-13-1|935743|US/en/product/aldrich/935743
Ni|NiCl2·6H2O|2|7791-20-0|203866|US/en/product/aldrich/203866
Cu|CuCl2|2|7447-39-4|222011|US/en/product/aldrich/222011
Zn|ZnCl2|2|7646-85-7|793523|US/en/product/sigald/793523
Ga|GaCl3|3|13450-90-3|427128|US/en/product/aldrich/427128
Sr|SrCl2·6H2O|2|10025-70-4|255521|SR/en/substance/strontiumchloridehexahydrate2666210025704
Y|YCl3·6H2O|3|10025-94-2|464317|US/en/product/aldrich/464317
Zr|ZrCl4|4|10026-11-6|808913|US/en/product/mm/808913
Nb|NbCl3·CH3OCH2CH2OCH3|3|110615-13-9|326356|US/en/product/aldrich/326356
Mo|MoCl3|3|13478-18-7|CH6H8C2C510A|US/en/product/chemimpexinternationalinc/ch6h8c2c510a
Ru|RuCl3·xH2O|3|14898-67-0|931578|US/en/product/aldrich/931578
Rh|RhCl3·xH2O|3|20765-98-4|520772|US/en/product/aldrich/520772
Pd|PdCl2|2|7647-10-1|76050|US/en/product/aldrich/76050
Ag|AgCl|1|7783-90-6|204382|US/en/product/aldrich/204382
In|InCl3|3|10025-82-8|334065|US/en/product/aldrich/334065
Sn|SnCl2·2H2O|2|10025-69-1|474762|US/en/product/aldrich/474762
Sb|SbCl3|3|10025-91-9|311375|US/en/substance/antimonyiiichloride2281210025919
Ba|BaCl2·2H2O|2|10326-27-9|101719|US/en/product/mm/101719
Hf|HfCl4|4|13499-05-3|258202|US/en/substance/hafniumivchloride3203013499053
Ta|TaCl5|5|7721-01-9|400475|US/en/substance/tantalumvchloride358217721019
W|WCl4|4|13470-13-8|263974|US/en/product/aldrich/263974
Ir|IrCl3·xH2O|3|14996-61-3|206245|US/en/substance/iridiumiiichloridehydrate29858anhydrousbasis14996613
Pt|PtCl2|2|10025-65-7|520632|US/en/product/aldrich/520632
Au|AuCl3|3|13453-07-1|334049|US/en/product/aldrich/334049
Pb|PbCl2|2|7758-95-4|268690|US/en/product/aldrich/268690
Bi|BiCl3|3|7787-60-2|112403|US/en/product/mm/112403
La|LaCl3·7H2O|3|10025-84-0|203521|CL/en/product/aldrich/203521
Ce|CeCl3·7H2O|3|18618-55-8|228931|US/en/substance/ceriumiiichlorideheptahydrate3725818618558
Pr|PrCl3·xH2O|3|19423-77-9|205141|US/en/product/aldrich/205141
Nd|NdCl3·6H2O|3|13477-89-9|289183|GI/en/product/aldrich/289183
Sm|SmCl3·6H2O|3|13465-55-9|248800|US/en/product/aldrich/248800
Eu|EuCl2|2|13769-20-5|431850|EC/en/product/aldrich/431850
Gd|GdCl3·6H2O|3|13450-84-5|203289|US/en/substance/gadoliniumiiichloridehexahydrate3717013450845
Tb|TbCl3·6H2O|3|13798-24-8|212903|US/en/substance/terbiumiiichloridehexahydrate3733813798248
Dy|DyCl3·6H2O|3|15059-52-6|289272|US/en/product/aldrich/289272
Ho|HoCl3·6H2O|3|14914-84-2|289213|US/en/substance/holmiumiiichloridehexahydrate3793814914842
Er|ErCl3·6H2O|3|10025-75-9|289256|US/en/substance/erbiumiiichloridehexahydrate3817110025759
Tm|TmCl3·6H2O|3|1331-74-4|204668|US/en/product/aldrich/204668
Yb|YbCl3·6H2O|3|10035-01-5|337927|US/en/product/aldrich/337927
Lu|LuCl3·6H2O|3|15230-79-2|542075|US/en/product/aldrich/542075'''

NOTES = {
    'Fe': 'FeCl2 follows the declared reagent formula. Product 372870 lists FeCl2, but Sigma technical support does not certify it as anhydrous; water content and purchased lot remain unconfirmed.',
    'V': 'Listed grade is 85%; formal label does not establish the purity/speciation of a purchased lot.',
    'Nb': 'Chloride donor-ligand adduct; retain DME identity. Not a bare-ion or solution-coordination assignment.',
    'Mo': 'Sigma-hosted Chem-Impex partner listing, not an Aldrich-branded product. Lower Mo(II) chloride was not verified in this lookup.',
    'Ga': 'A lower Ga(I) chloride product was not verified in this lookup.',
    'Zr': 'Lower Zr(II/III) chlorides were not verified in this lookup.',
    'Ta': 'A lower Ta(III) chloride product was not verified in this lookup.',
    'Sm': 'Sm(II) chloride was not verified; selected chloride is Sm(III). No claim that a lower-state chloride does not exist.',
    'Tm': 'Tm(II) chloride was not verified; selected chloride is Tm(III).',
    'Yb': 'Yb(II) chloride was not verified; selected chloride is Yb(III).',
    'Ag': 'Verify dissolution/speciation in the actual solvent; catalogue identity is not a solution-state measurement.',
    'Cu': 'CuCl2 retained according to the declared reagent preference; lower-valence default overridden. Actual product grade, lot and hydration were not provided.',
    'In': 'InCl3 retained according to the declared reagent preference; lower-valence default overridden. Actual product grade, lot and hydration were not provided.',
    'Au': 'AuCl3 retained according to the declared reagent preference; lower-valence default overridden. This is AuCl3, not HAuCl4 or its hydrate. Actual product grade, lot and hydration were not provided.',
}


def main():
    rows = []
    for line in FACTS.splitlines():
        element, formula, charge, cas, identifier, source = line.split('|')
        rows.append({'element': element, 'compound': formula, 'type': 'CHLORIDE',
                     'formal_oxidation_state': int(charge), 'cas': cas, 'catalog_number': identifier,
                     'source_url': 'https://www.sigmaaldrich.com/' + source,
                     'source_kind': 'PRODUCT_LISTING' if '/product/' in source else 'SUBSTANCE_LISTING_WITH_PRODUCTS',
                     'seller': 'Chem-Impex via Sigma marketplace' if element == 'Mo' else 'Sigma-Aldrich / MilliporeSigma',
                     'checked_date': '2026-10-13', 'status': 'CATALOG_CANDIDATE_NOT_ACTUAL_USAGE',
                     'coordination': None, 'spin': None, 'solution_state_verified': False,
                     'notes': NOTES.get(element, 'Candidate identity only; verify exact hydrate and purchased product before recording actual usage.')})
    if len(rows) != 50 or len({r['element'] for r in rows}) != 50:
        raise ValueError('The reviewed original target table must contain exactly 50 unique elements.')
    data = {'version': 'sigma-target50-reviewed-2026-10-13', 'checked_date': '2026-10-13',
            'scope': 'Original 50 SAC targets. Other periodic-table elements remain manually editable and unresolved.',
            'policy': 'Chlorides are the automatic per-element default; no automatic nitrate/other fallback. Explicit formulas FeCl2, CuCl2, InCl3 and AuCl3 take precedence; other entries prefer lower verified formal oxidation-state chlorides. User-edited defaults remain editable and preserved. Unreviewed elements remain unresolved. This bounded lookup does not prove globally minimum states or stock availability.',
            'actual_usage': 'Catalog candidates never overwrite experimental records. Planned compounds require user review; actual bottle/lot/hydrate can differ.',
            'species_note': 'Formal reagent oxidation state is not a measured solution-ion or final catalyst state. No coordination/spin is assigned from a salt formula.',
            'entries': rows}
    path = Path(__file__).resolve().parents[1] / 'resources/sigma_precursors.json'
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with path.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} sourced candidate records: {path.name}')


if __name__ == '__main__':
    main()
