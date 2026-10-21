"""Explicit material assumptions for optional physical ICP residual validation."""
import streamlit as st


def render_physics_options(key):
    enabled = st.checkbox('Compare physical ICP prior and learned residual', value=False, key=key + '_enable')
    if not enabled:
        return None
    with st.expander('Physical ICP prior: material assumptions', expanded=True):
        st.caption('Mass-balance calculations use recorded PAN mass and solvent volume. Retention parameters are model assumptions.')
        pan = st.number_input('PAN feed mass / g', min_value=.000001, value=None, key=key + '_pan')
        volume = st.number_input('Solvent volume / mL', min_value=.000001, value=None, key=key + '_volume')
        carbon = st.number_input('Carbonization mass loss / %', min_value=0., max_value=99.99, value=None, key=key + '_carbon')
        source = st.text_input('Material assumption source and experiment scope', key=key + '_source')
        confirmed = st.checkbox('These material inputs apply to all selected experiments', value=False, key=key + '_confirmed')
    if not confirmed or pan is None or volume is None or carbon is None or not source.strip():
        st.info('Complete and confirm the material assumptions before running the physical comparison.')
    return {'pan_mass_g': pan, 'solvent_volume_mL': volume,
            'carbonization_loss': None if carbon is None else carbon / 100.,
            'source': source, 'confirmed': confirmed}


def physical_comparison_ready(physics):
    return physics is None or (physics.get('confirmed') and physics.get('source', '').strip() and
        all(physics.get(k) is not None for k in ('pan_mass_g', 'solvent_volume_mL', 'carbonization_loss')))
