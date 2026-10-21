"""Export deterministic pulse-loss curves from the current synthetic mass model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
from sacfactory.demo_physics import mass_balance, thermal_reference
from sacfactory.domain import TARGETS
from sacfactory.thermal_loss import loss_settings, loss_reference, elemental_loss_reference

TEMPERATURES = [1000, 1200, 1600]
PULSES = list(range(1, 17))
COLORS = ['#3E6C9D', '#208F8C', '#C47A39']
STYLES = ['-', '--', ':']
MARKERS = ['o', 's', '^']


def build_data():
    carrier_rows, metal_rows = [], []
    for temperature in TEMPERATURES:
        for pulse in PULSES:
            for element in TARGETS:
                # An individual 1 mmol metal feed in 20 mL; no random noise.
                balance = mass_balance({element: 50.0}, temperature, pulse)
                entry = balance['elements'][element]
                parameters = entry['empirical_loss_parameters']
                metal_rows.append({
                    'data_origin': 'SIMULATED_CURRENT_DEMO_MODEL',
                    'element': element, 'temperature_C': temperature,
                    'pulse_count': pulse, 'feed_amount_mmol': 1.0,
                    'feed_metal_g': entry['feed_metal_g'],
                    'retained_metal_g': entry['retained_metal_g'],
                    'lost_metal_g': entry['lost_metal_g'],
                    'metal_loss_percent_of_feed': 100 * (1 - entry['retention_fraction']),
                    'metal_retention_percent_of_feed': 100 * entry['retention_fraction'],
                    'recovered_sample_g': balance['recovered_sample_g'],
                    'W_wt_percent': entry['true_w_wt_percent'],
                    'loss_model_version': balance['loss_model_version'],
                    'thermal_risk': parameters['thermal_risk'],
                    'loss_first_percent': 100 * parameters['loss_first_fraction'],
                    'loss_plateau_percent': 100 * parameters['loss_plateau_fraction'],
                    'tau_pulses': parameters['tau_pulses'],
                    'escape_energy_proxy_used': parameters['escape_proxy_used'],
                    'sublimation_temperature_used': parameters['sublimation_temperature_used'],
                    'reference_weight_coverage': parameters['reference_weight_coverage'],
                })
                if element == 'Fe':
                    carrier_rows.append({
                        'data_origin': 'SIMULATED_CURRENT_DEMO_MODEL',
                        'temperature_C': temperature, 'pulse_count': pulse,
                        'carrier_before_pulse_g': balance['carrier_before_pulse_g'],
                        'carrier_after_pulse_g': balance['carrier_after_pulse_g'],
                        'pulse_loss_percent_of_carbonized_carrier': 100 * balance['carrier_loss_fraction'],
                        'total_loss_percent_of_initial_matrix': 100 * (1 - balance['carrier_after_pulse_g'] / balance['pan_mass_g']),
                    })
    return pd.DataFrame(carrier_rows), pd.DataFrame(metal_rows)


def plot_curves(carrier, metals, output, alignment_scripts=None):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.linewidth': .8, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'legend.frameon': False, 'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.08661417, 3.30708661))
    fig.subplots_adjust(left=.085, right=.98, bottom=.25, top=.76, wspace=.42)
    panels = [
        (carrier, 'pulse_loss_percent_of_carbonized_carrier', 'Carbonized carrier', 'Carrier pulse loss (%)'),
        (metals[metals.element == 'Fe'], 'metal_loss_percent_of_feed', 'Fe reference', 'Fe metal loss (%)'),
        (metals[metals.element == 'Zn'], 'metal_loss_percent_of_feed', 'Zn temperature contrast', 'Zn metal loss (%)'),
    ]
    for index, (axis, (frame, field, title, ylabel)) in enumerate(zip(axes, panels)):
        for j, temperature in enumerate(TEMPERATURES):
            part = frame[frame.temperature_C == temperature].sort_values('pulse_count')
            axis.plot(part.pulse_count, part[field], color=COLORS[j], linestyle=STYLES[j],
                      marker=MARKERS[j], markersize=3, markevery=(j, 3),
                      linewidth=1.2, label=f'{temperature} °C')
        axis.set(xlim=(.5, 16.5), ylim=(0, 30) if index == 0 else (0, 100),
                 xlabel='Pulse count', ylabel=ylabel)
        axis.set_xticks([1, 4, 8, 12, 16])
        axis.set_yticks(np.arange(0, 31, 5) if index == 0 else np.arange(0, 101, 20))
        axis.tick_params(length=3, width=.7, pad=2)
        axis.set_title(title, pad=8, fontsize=7)
        axis.text(-.19, 1.095, 'abc'[index], transform=axis.transAxes,
                  fontsize=8, fontweight='bold', va='bottom')
    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3,
                      linewidth=1.2, label=f'{temperature} °C')
               for temperature, color, style, marker in zip(TEMPERATURES, COLORS, STYLES, MARKERS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .93), ncol=3,
               columnspacing=2.4, handlelength=2.5, fontsize=7)
    fig.text(.085, .95, 'SIMULATED MODEL | Pulse-dependent mass loss', fontsize=8, fontweight='bold')
    fig.text(.085, .10, 'Carrier: relative to the carbonized matrix before pulsing. Metal: relative to metal feed mass.', fontsize=6.2)
    fig.text(.085, .055, 'Deterministic assumptions; no experimental points, fitted kinetics or uncertainty intervals.', fontsize=6.2, color='#555B62')
    if alignment_scripts:
        sys.path.insert(0, str(alignment_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=output / 'pulse_losses.alignment.json',
            overlay_svg=output / 'pulse_losses.alignment.svg', strict=True,
            require_panel_labels=True, panel_ids=['a', 'b', 'c'])
    fig.savefig(output / 'pulse_losses.png', dpi=600)
    fig.savefig(output / 'pulse_losses.pdf')
    fig.savefig(output / 'pulse_losses.svg')
    plt.close(fig)


def plot_element_overview(metals, output):
    order = metals[(metals.temperature_C == 1000) & (metals.pulse_count == 1)].sort_values('thermal_risk').element.tolist()
    columns = [(t, n) for t in TEMPERATURES for n in (1, 16)]
    values = np.array([[metals[(metals.element == e) & (metals.temperature_C == t) & (metals.pulse_count == n)].metal_retention_percent_of_feed.iloc[0]
                        for t, n in columns] for e in order])
    fig, axis = plt.subplots(figsize=(4.80314961, 8.66141732))
    fig.subplots_adjust(left=.13, right=.82, bottom=.07, top=.9)
    image = axis.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    axis.set_yticks(range(len(order)), labels=order, fontsize=6.2)
    axis.set_xticks(range(6), labels=[f'{t} °C\nn = {n}' for t, n in columns], fontsize=6.2)
    axis.xaxis.tick_top()
    axis.tick_params(axis='both', length=0, pad=4)
    for i in range(50):
        for j in range(6):
            axis.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', fontsize=5.8,
                      color='white' if values[i,j] >= 65 else '#263746')
    fig.text(.13, .97, 'SIMULATED MODEL | All 50 metals', fontsize=8, fontweight='bold')
    fig.text(.13, .942, 'Retained metal / initial metal feed (%)', fontsize=7)
    color_axis = fig.add_axes([.87, .44, .035, .24])
    colorbar = fig.colorbar(image, cax=color_axis, ticks=[0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=6.2, length=2)
    fig.text(.13, .022, 'Ranked by the declared thermal prior; no measured recovery data.', fontsize=6.2)
    fig.savefig(output / 'all50_metal_retention.png', dpi=600)
    fig.savefig(output / 'all50_metal_retention.pdf')
    fig.savefig(output / 'all50_metal_retention.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=APP_ROOT.parent / 'pulse_loss_curves')
    parser.add_argument('--alignment-scripts', type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    carrier, metals = build_data()
    assert len(carrier) == 48 and len(metals) == 2400
    assert metals.metal_loss_percent_of_feed.between(0, 100).all()
    assert np.allclose(metals.feed_metal_g, metals.retained_metal_g + metals.lost_metal_g)
    for _, part in carrier.groupby('temperature_C'):
        assert np.all(np.diff(part.pulse_loss_percent_of_carbonized_carrier) >= 0)
    for _, part in metals.groupby(['element', 'temperature_C']):
        assert np.all(np.diff(part.metal_loss_percent_of_feed) >= -1e-12)
    for element in TARGETS:
        values = [metals[(metals.element == element) & (metals.temperature_C == t)].metal_loss_percent_of_feed.to_numpy() for t in TEMPERATURES]
        assert np.all(values[0] < values[1]) and np.all(values[1] < values[2])
    carrier.to_csv(args.output / 'carrier_loss_curves.csv', index=False)
    metals.to_csv(args.output / 'metal_loss_curves_all50.csv', index=False)
    metals[metals.element.isin(['Fe', 'Zn'])].to_csv(args.output / 'plotted_metal_loss_curves.csv', index=False)
    pd.DataFrame([elemental_loss_reference(e) for e in TARGETS]).to_csv(args.output / 'thermal_reference_coverage_all50.csv', index=False)
    manifest = {
        'origin': 'SIMULATED_CURRENT_DEMO_MODEL', 'temperature_C': TEMPERATURES,
        'pulse_counts': PULSES, 'noise': False, 'carrier_rows': len(carrier),
        'all_element_metal_rows': len(metals), 'plotted_elements': ['Fe', 'Zn'],
        'model_source_sha256': hashlib.sha256((APP_ROOT / 'sacfactory/demo_physics.py').read_bytes()).hexdigest(),
        'shared_physical_priors_sha256': hashlib.sha256((APP_ROOT / 'sacfactory/physical_priors.py').read_bytes()).hexdigest(),
        'thermal_snapshot_sha256': thermal_reference()['snapshot_sha256'],
        'empirical_model_version': loss_settings()['version'],
        'settings': loss_settings(), 'thermochemical_reference_sha256': loss_reference()['snapshot_sha256'],
        'coverage_note': 'All fifty have melting, boiling and formation-energy references; independent sublimation temperatures are absent. Escape energy is an explicit proxy, not an imputed temperature.',
        'figure_contract': {
            'question': 'How do the current temperature and pulse assumptions approach mass-loss plateaus?',
            'claim': 'Continuous thermal risk changes first loss, plateau and pulse response; high thermal resistance lowers modeled losses.',
            'panels': {'a': 'Carrier trend anchored to reported single-pulse losses',
                       'b': 'Fe preserves the 1000 C anchor with continuous temperature-dependent loss',
                       'c': 'Zn illustrates element-dependent thermal sensitivity'},
            'archetype': 'quantitative grid', 'backend': 'Python / matplotlib',
            'dimensions_mm': [180, 84], 'statistics': 'Deterministic model values; no measured replicates or uncertainty',
        },
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    plot_curves(carrier, metals, args.output, args.alignment_scripts)
    plot_element_overview(metals, args.output)
    print(carrier[carrier.pulse_count.isin([1, 16])].to_string(index=False))
    print(metals[(metals.element.isin(['Fe', 'Zn'])) & metals.pulse_count.isin([1, 16])][
        ['element', 'temperature_C', 'pulse_count', 'metal_loss_percent_of_feed']].to_string(index=False))


if __name__ == '__main__':
    main()
