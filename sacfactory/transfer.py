"""Compatibility exports for the composition-transfer implementation."""
from .composition import (
    POSITIONS, descriptors, transfer_feature, CompositionTransferKernel,
    transfer_classifier, process_key, fit_transfer_model, source_evidence,
)

ElementTransferKernel = CompositionTransferKernel
