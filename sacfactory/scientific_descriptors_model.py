"""Database descriptors and recorded controls, without elicited heating priors."""
import hashlib
from pathlib import Path
import numpy as np
from .data_only import DataOnlyModel
from .planning import fingerprint


class ScientificDescriptorModel(DataOnlyModel):
    def encode(self,sample,element=None):
        """Build an allowlisted schema; never call a loss or retention model.

        No carbonization fraction, carrier-loss curve, metal-retention anchor,
        waveform, mechanistic screening prior or mass decoder enters this model.
        Thermal constants are sourced bulk properties, not fitted loss estimates.
        """
        from .chemical_descriptors import descriptor_matrix
        from .domain import ELEMENTS
        from .physics_features import thermal_matrix
        from .precursors import precursor_features
        from .mixing import feature_vector,component_features
        selected=[ELEMENTS.index(e) for e in sample['elements']]
        weights=np.asarray([sample['concentrations_mM'][e] for e in sample['elements']],dtype=float)
        weights/=weights.sum()
        positions=np.column_stack([descriptor_matrix('chemistry'),thermal_matrix()])
        local=positions[selected]
        mean=weights@local
        spread=np.sqrt(np.maximum(weights@(local*local)-mean*mean,0.))
        controls=[np.log1p(sum(sample['concentrations_mM'][e] for e in sample['elements'])),
            np.log1p(len(selected)),np.log1p(sample['pulse_count']),float(sample['pulse_limit_C'])/1000.,
            np.log(float(sample['pan_mass_g'])),np.log(float(sample['solvent_volume_mL']))]
        # Actual precursor type/oxidation are covariates; assumed coordination
        # and spin-dependent ionic radii are deliberately excluded.
        recipe=np.r_[controls,precursor_features(sample,include_state=False,for_training=True),
            mean,spread,feature_vector(sample)]
        result=recipe if element is None else np.r_[recipe,
            np.log1p(sample['concentrations_mM'][element]),positions[ELEMENTS.index(element)],
            component_features(sample,element)]
        if not np.isfinite(result).all():raise ValueError('Finite recorded controls and masked database descriptors required.')
        return result

    def fit_records(self,records,origin):
        super().fit_records(records,origin)
        self.info.update(input_profile='Recorded controls and sourced database descriptors; no elicited Joule-heating knowledge',
            feature_schema='database-descriptors-only-v2',
            excluded_information=['Nominal element one-hot identity','Carbonization and carbon-loss assumptions',
                'Metal-loss and retention assumptions','Fe loading anchor','Assumed waveform and kinetic rates',
                'Generated ICP predictions','Mechanistic screening probability','Mass-conservation decoder',
                'Assumed coordination/spin-dependent ionic radii'],
            label_disclosure='Uses measured ICP/XRD labels for supervised fitting; not an experiment-free model')
        return self

    @property
    def model_id(self):
        from .chemical_descriptors import reference
        from .precursors import state_reference
        from .physical_priors import thermal_reference
        from ._pulseicp.thermodynamics import reference_identity
        return fingerprint({'base':super().model_id,'input_mode':'science_descriptors',
            'element_reference':reference()['snapshot_sha256'],'state_reference':state_reference()['snapshot_sha256'],
            'thermal_reference':thermal_reference()['snapshot_sha256'],
            'mixing_reference':reference_identity(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def predict(self,samples):
        result=super().predict(samples)
        for row in result:row['model_name']='Scientific descriptors '+self.backend
        return result
