from vasp_bandplot.io.read import read_poscar,read_procar,read_outcar,extract_proj_for_plot
from vasp_bandplot.io.plot_band import plot_projected_bands_grid


def main(outcar_path,poscar_path, procar_path, element_orbital,fig_path):
    type , number ,ionrange= read_poscar(poscar_path)
    efermi = read_outcar(outcar_path)
    proj, elements, used_orbs, kpts, energies = read_procar(procar_path, type, ionrange, orbital=('s', 'p', 'd'))

    # all元素与轨道，返回 (Nk, Nbands, K)
    data3, labels = extract_proj_for_plot(proj, elements, used_orbs,
                                          element_sel='all', orbital_sel='all')

    fig, ax, im, cbar = plot_projected_bands_grid(
            energies, data3, labels,
            kpts=kpts,
            channels=element_orbital,          # 用标签选通道
            width_ev=0.20,              # 条带展宽（eV）
            dE=0.01,                    # 能量网格步长
            cmap='magma',
            agg='sum',
            overlay_lines=False,
            share_colorbar=False,
            ef=efermi,                    # 如需对齐费米能级，填入 ef 值
            e_window=None ,              # 指定能量窗口 (emin, emax)；None 自动
            fig_path=fig_path
        )

if __name__ == "__main__":

#================  input parameters  =======================  
    outcar_path = './OUTCAR'
    poscar_path = './POSCAR'
    procar_path = './PROCAR'
    element_orbital= ['S-p','W-s','W-p']
    fig_path='./test.png'

#===========================================================

    main(outcar_path=outcar_path,poscar_path=poscar_path , 
         procar_path=procar_path,
         element_orbital=element_orbital , fig_path=fig_path)
