from .io.read import read_poscar, read_procar, extract_proj_for_plot
from .io.plot_band import plot_projected_bands

def generate_projected_band(poscar_path, procar_path, channel, fig_path,
                            width_ev=0.20, dE=0.01, norm='minmax', agg='sum',
                            overlay_lines=False, ef=None, e_window=None):
    """
    on step to generate figure in fig_path
    """
    atom_types, atom_numbers, idx_ranges = read_poscar(poscar_path)
    proj, elements, used_orbs, kpts, energies = read_procar(
        procar_path, atom_types, idx_ranges, orbital=('s', 'p', 'd')
    )
    data3, labels = extract_proj_for_plot(
        proj, elements, used_orbs, element_sel='all', orbital_sel='all'
    )
    if channel not in labels:
        raise ValueError(f"NOT FOUND '{channel}', AVAILABLE: {labels}")

    plot_projected_bands(
        energies, data3, labels,
        kpts=kpts,
        channel=channel,
        width_ev=width_ev,
        dE=dE,
        cmap='viridis',
        norm=norm,
        agg=agg,
        overlay_lines=overlay_lines,
        ef=ef,
        e_window=e_window,
        fig_path=fig_path
    )
    return fig_path

__all__ = [
    "read_poscar", "read_procar", "extract_proj_for_plot",
    "plot_projected_bands", "generate_projected_band","read_outcar",
]
