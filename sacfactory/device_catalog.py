"""Extensible laboratory equipment profiles, separate from installed instruments."""
import copy

CATEGORIES = ['Synthesis equipment', 'Characterization equipment', 'Performance testing', 'General equipment']
CAPABILITIES = ['synthesize', 'characterize', 'performance', 'prepare', 'separate', 'transfer', 'store', 'monitor', 'compute']
SIMULATED_MODALITIES = ['ICP', 'XRD', 'STEM', 'SEM']

# These are equipment types, not a claim that a laboratory owns connected instruments.
_PROFILES = {
    'Synthesis equipment': [
        ('pulsed_heater', 'Pulsed Joule heater', 'synthesize', []),
        ('tube_furnace', 'Tube furnace', 'synthesize', []),
        ('muffle_furnace', 'Muffle furnace', 'synthesize', []),
        ('rapid_thermal', 'Rapid thermal annealing system', 'synthesize', []),
        ('induction_heater', 'Induction heating system', 'synthesize', []),
        ('microwave_reactor', 'Microwave synthesis reactor', 'synthesize', []),
        ('hydrothermal_reactor', 'Hydrothermal / solvothermal reactor', 'synthesize', []),
        ('flow_reactor', 'Continuous-flow synthesis reactor', 'synthesize', []),
        ('cvd', 'Chemical vapor deposition system', 'synthesize', []),
        ('ald', 'Atomic layer deposition system', 'synthesize', []),
        ('pvd', 'Physical vapor deposition system', 'synthesize', []),
        ('sputter', 'Magnetron sputtering system', 'synthesize', []),
        ('plasma_reactor', 'Plasma synthesis system', 'synthesize', []),
        ('laser_synthesis', 'Laser synthesis system', 'synthesize', []),
        ('electrospinning', 'Electrospinning system', 'prepare', []),
        ('spray_dryer', 'Spray dryer / aerosol synthesis', 'prepare', []),
        ('ball_mill', 'Ball mill / mechanochemical reactor', 'prepare', []),
        ('electrodeposition', 'Electrodeposition system', 'synthesize', []),
        ('photochemical_reactor', 'Photochemical synthesis reactor', 'synthesize', []),
        ('impregnation', 'Automated impregnation workstation', 'prepare', []),
        ('electrospray', 'Electrospray deposition system', 'prepare', []),
    ],
    'Characterization equipment': [
        ('icp_oes', 'ICP-OES', 'characterize', ['ICP']),
        ('icp_ms', 'ICP-MS', 'characterize', ['ICP']),
        ('aas', 'Atomic absorption spectrometer', 'characterize', ['AAS']),
        ('xrd', 'X-ray diffractometer (XRD)', 'characterize', ['XRD']),
        ('stem', 'Aberration-corrected STEM', 'characterize', ['STEM']),
        ('sem', 'Scanning electron microscope (SEM)', 'characterize', ['SEM']),
        ('tem', 'Transmission electron microscope (TEM)', 'characterize', ['TEM']),
        ('eds', 'EDS / EDX microanalysis', 'characterize', ['EDS']),
        ('eels', 'Electron energy-loss spectrometer', 'characterize', ['EELS']),
        ('xps', 'X-ray photoelectron spectrometer', 'characterize', ['XPS']),
        ('xas', 'X-ray absorption spectroscopy (XANES / EXAFS)', 'characterize', ['XAS']),
        ('xrf', 'X-ray fluorescence spectrometer', 'characterize', ['XRF']),
        ('raman', 'Raman spectrometer', 'characterize', ['RAMAN']),
        ('ftir', 'FTIR spectrometer', 'characterize', ['FTIR']),
        ('uv_vis', 'UV-visible spectrometer', 'characterize', ['UV_VIS']),
        ('fluorescence', 'Fluorescence spectrometer', 'characterize', ['FLUORESCENCE']),
        ('bet', 'Gas adsorption / BET analyzer', 'characterize', ['BET']),
        ('porosimeter', 'Porosimeter', 'characterize', ['POROSIMETRY']),
        ('tga', 'Thermogravimetric analyzer', 'characterize', ['TGA']),
        ('dsc', 'Differential scanning calorimeter', 'characterize', ['DSC']),
        ('chns', 'CHNS elemental analyzer', 'characterize', ['CHNS']),
        ('epr', 'Electron paramagnetic resonance spectrometer', 'characterize', ['EPR']),
        ('mossbauer', 'Mossbauer spectrometer', 'characterize', ['MOSSBAUER']),
        ('nmr', 'Nuclear magnetic resonance spectrometer', 'characterize', ['NMR']),
        ('afm', 'Atomic force microscope', 'characterize', ['AFM']),
        ('stm', 'Scanning tunneling microscope', 'characterize', ['STM']),
        ('dls', 'Dynamic light scattering / zeta potential analyzer', 'characterize', ['DLS']),
        ('saxs', 'Small-angle X-ray scattering system', 'characterize', ['SAXS']),
        ('pdf', 'Total scattering / pair distribution function system', 'characterize', ['PDF']),
        ('tpr_tpd', 'Temperature-programmed reaction / desorption analyzer', 'characterize', ['TPR_TPD']),
        ('chemisorption', 'Chemisorption analyzer', 'characterize', ['CHEMISORPTION']),
        ('operando_cell', 'In situ / operando spectroscopy cell', 'characterize', ['OPERANDO']),
    ],
    'Performance testing': [
        ('potentiostat', 'Potentiostat / galvanostat', 'performance', ['ELECTROCHEMISTRY']),
        ('rrde', 'Rotating disk / ring-disk electrode system', 'performance', ['RDE_RRDE']),
        ('fuel_cell', 'Fuel-cell test station', 'performance', ['FUEL_CELL']),
        ('electrolyzer', 'Electrolyzer test station', 'performance', ['ELECTROLYSIS']),
        ('battery_cycler', 'Battery cycler', 'performance', ['BATTERY']),
        ('eis', 'Electrochemical impedance analyzer', 'performance', ['EIS']),
        ('gas_reactor', 'Gas-phase catalytic test reactor', 'performance', ['GAS_CATALYSIS']),
        ('liquid_reactor', 'Liquid-phase catalytic test reactor', 'performance', ['LIQUID_CATALYSIS']),
        ('photocatalysis', 'Photocatalytic performance station', 'performance', ['PHOTOCATALYSIS']),
        ('gc', 'Gas chromatograph', 'performance', ['GC']),
        ('gc_ms', 'Gas chromatography-mass spectrometer', 'performance', ['GC_MS']),
        ('hplc', 'High-performance liquid chromatograph', 'performance', ['HPLC']),
        ('lc_ms', 'Liquid chromatography-mass spectrometer', 'performance', ['LC_MS']),
        ('online_ms', 'Online mass spectrometer', 'performance', ['MS']),
        ('faradaic', 'Gas / liquid product quantification station', 'performance', ['PRODUCT_ANALYSIS']),
        ('durability', 'Accelerated durability test station', 'performance', ['DURABILITY']),
        ('membrane_electrode', 'Membrane electrode assembly test fixture', 'performance', ['MEA']),
        ('flow_battery', 'Flow-battery test station', 'performance', ['FLOW_BATTERY']),
    ],
    'General equipment': [
        ('balance', 'Analytical balance', 'prepare', []),
        ('microbalance', 'Microbalance', 'prepare', []),
        ('powder_dispenser', 'Automated powder dispenser', 'prepare', []),
        ('liquid_handler', 'Liquid-handling workstation', 'prepare', []),
        ('pipette', 'Pipette / electronic pipette', 'prepare', []),
        ('syringe_pump', 'Syringe pump', 'transfer', []),
        ('peristaltic_pump', 'Peristaltic pump', 'transfer', []),
        ('stirrer', 'Magnetic stirrer / hotplate', 'prepare', []),
        ('overhead_stirrer', 'Overhead stirrer', 'prepare', []),
        ('ultrasonic_bath', 'Ultrasonic bath', 'prepare', []),
        ('probe_sonicator', 'Probe sonicator', 'prepare', []),
        ('centrifuge', 'Centrifuge', 'separate', []),
        ('filtration', 'Vacuum filtration station', 'separate', []),
        ('rotary_evaporator', 'Rotary evaporator', 'separate', []),
        ('vacuum_oven', 'Vacuum drying oven', 'prepare', []),
        ('drying_oven', 'Drying oven', 'prepare', []),
        ('freeze_dryer', 'Freeze dryer', 'prepare', []),
        ('glovebox', 'Glovebox', 'store', []),
        ('fume_hood', 'Fume hood', 'prepare', []),
        ('gas_manifold', 'Gas manifold / mass-flow controller', 'monitor', []),
        ('vacuum_pump', 'Vacuum pump', 'prepare', []),
        ('chiller', 'Recirculating chiller', 'monitor', []),
        ('water_system', 'Ultrapure water system', 'prepare', []),
        ('ph_meter', 'pH / conductivity meter', 'monitor', []),
        ('robot_arm', 'Robotic arm / sample transfer', 'transfer', []),
        ('autosampler', 'Autosampler', 'transfer', []),
        ('sample_storage', 'Sample storage / vial rack', 'store', []),
        ('barcode', 'Barcode / sample tracking station', 'monitor', []),
        ('pyrometer', 'Optical pyrometer / thermal camera', 'monitor', []),
        ('daq', 'Data acquisition / electrical monitoring', 'monitor', []),
        ('power_supply', 'Programmable power supply', 'monitor', []),
        ('compute', 'Analysis workstation / compute node', 'compute', []),
    ],
}

CATALOG = [dict(id=key, name=name, category=category, capabilities=[capability], modalities=modalities)
           for category, rows in _PROFILES.items() for key, name, capability, modalities in rows]
PROFILES = {p['id']: p for p in CATALOG}


def category_of(device):
    if device.get('category') in CATEGORIES:
        return device['category']
    caps = device.get('capabilities', [])
    return CATEGORIES[0] if 'synthesize' in caps else CATEGORIES[1] if 'characterize' in caps else CATEGORIES[2] if 'performance' in caps else CATEGORIES[3]


def profile_device(profile_id, device_id=None, name=None, mode='INVENTORY_ONLY'):
    profile = copy.deepcopy(PROFILES[profile_id])
    return dict(id=device_id or 'inventory_' + profile_id, name=name or profile['name'],
                category=profile['category'], equipment_type=profile_id, capabilities=profile['capabilities'],
                modalities=profile['modalities'], mode=mode,
                limits={'cutoff_C': [100, 2600], 'pulse_count': [1, 16]} if profile_id == 'pulsed_heater' else {})


def modalities_of(device):
    # Old saved combined characterizers retain their frozen all-modality contract.
    return device.get('modalities', SIMULATED_MODALITIES if 'characterize' in device.get('capabilities', []) else [])
