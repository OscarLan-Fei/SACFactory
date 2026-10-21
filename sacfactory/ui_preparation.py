"""English preparation controls shared by manual input paths."""
import streamlit as st

from .preparation import DEFAULT_PAN_G, DEFAULT_DMF_ML, material_inputs


def preparation_inputs(key, sample=None):
    sample = sample or {}
    previous_unit = sample.get('concentration_input_unit')
    unit = st.radio('Precursor input unit', ['mM', 'mmol'], index=0 if previous_unit == 'mM' else 1,
                    horizontal=True, key=key + '_unit')
    cols = st.columns(2)
    pan = cols[0].number_input('PAN mass / g', min_value=.000001,
        value=float(sample.get('pan_mass_g') or DEFAULT_PAN_G), key=key + '_pan')
    volume = cols[1].number_input('DMF volume / mL', min_value=.000001,
        value=float(sample.get('solvent_volume_mL') or DEFAULT_DMF_ML), key=key + '_volume')
    st.caption('Default: 2 g PAN and 20 mL DMF. At 20 mL, 1 mmol = 50 mM.')
    return unit, material_inputs(pan, volume)
