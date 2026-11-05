"""Retain analysis settings while inactive Streamlit tabs omit their widgets."""
import streamlit as st


ANALYSIS_CONTROL_PREFIXES = (
    'analysis_', 'correlation_', 'weighted_', 'loading_', 'pooled_loading_',
    'descriptor_', 'outcome_', 'benchmark_', 'element_', 'campaign_',
    'map_', 'surface_', 'transfer_analysis_', 'readiness_', 'recipe_precision_',
    'recipe_cap_', 'repeat_', 'pair_elements_', 'prediction_', 'evaluation_',
    'chemistry_', 'agreement_', 'research_', 'comparison_',
)


def preserve_analysis_controls(prefixes=ANALYSIS_CONTROL_PREFIXES):
    """Interrupt widget cleanup for settings, without writing button/editor state."""
    scalar = (str, int, float)
    for key in list(st.session_state):
        if not key.startswith(prefixes):
            continue
        value = st.session_state[key]
        # Streamlit forbids programmatic assignment to button-like widgets.
        if isinstance(value, bool):
            continue
        if isinstance(value, scalar) or (
            isinstance(value, (list, tuple)) and all(isinstance(item, scalar) for item in value)
        ):
            st.session_state[key] = value


def refresh_data_epoch(session,origin,epoch):
    """Invalidate derived views while retaining explicit chat permissions and run scope.

    Committing characterization is a data refresh, not a new conversation. Run
    authorization still checks namespace, selected IDs and stable record identities.
    """
    key='seen_data_epoch_'+origin
    if session.get(key)==epoch:return
    retained=('api_','seen_data_epoch_','record_action_','research_samples_',
              'research_permission_','research_discussion_','chat_workflow_state_')
    for candidate in list(session):
        if origin in candidate and not candidate.startswith(retained):del session[candidate]
    session[key]=epoch
