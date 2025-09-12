import numpy as np
import matplotlib.pyplot as plt
from vasp_bandplot.io.read import extract_proj_for_plot,read_poscar,read_procar,read_outcar


def _centers_to_edges(x):
    """
    将一维中心点坐标转换为网格边界，长度变为 len(x)+1。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        # 退化或单点时，构造一个极小宽度的边界
        dx = 1.0 if x.size == 0 else 1e-6
        return np.array([x[0] - dx/2, x[0] + dx/2], dtype=float) if x.size == 1 else np.array([0.0, 1.0])
    mid = 0.5 * (x[:-1] + x[1:])
    left = x[0] - (mid[0] - x[0])
    right = x[-1] + (x[-1] - mid[-1])
    return np.concatenate(([left], mid, [right]))

def _kpath_distance(kpts):
    """
    将 (Nk,3) 的k点坐标转换为路径距离标量 (Nk,)
    直接用相邻点的欧氏距离累积(适用于已按路径给出的k点)
    """
    kpts = np.asarray(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError("kpts.shape should be (Nk, 3)")
    diffs = np.diff(kpts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])

def _build_band_weight_image(kx, energies, weights, e_window=None, dE=0.01, width_ev=0.15, agg='sum'):
    """
    将能带+权重离散到能量-路径网格，返回图像矩阵用于 imshow。
    - kx: (Nk,) 横坐标（路径距离）
    - energies: (Nk, Nbands)
    - weights:  (Nk, Nbands) 已选通道的权重
    - e_window: (emin, emax) 可选能量窗口；None 则由数据自动确定并留少量边距
    - dE: 能量栅格间距 (eV)
    - width_ev: 条带宽度（能量方向，总宽度，eV）
    - agg: 聚合方式，'sum' 或 'max'
    返回 (img, extent, Egrid)
    """
    Nk, Nb = energies.shape
    if weights.shape != energies.shape:
        raise ValueError("weights 形状必须与 energies 相同")
    if e_window is None:
        emin = float(np.nanmin(energies))
        emax = float(np.nanmax(energies))
        pad = 0.05 * max(1e-6, emax - emin)
        emin -= pad
        emax += pad
    else:
        emin, emax = e_window
    emin, emax = float(emin), float(emax)

    dE = float(dE)
    Ny = int(np.ceil((emax - emin) / dE)) + 1
    Egrid = np.linspace(emin, emax, Ny)
    img = np.zeros((Ny, Nk), dtype=float)

    half_bins = max(1, int(round(width_ev / dE / 2.0)))

    for b in range(Nb):
        Ek = energies[:, b]
        Wk = weights[:, b]
        # 将能量映射到栅格索引
        j = np.clip(np.round((Ek - emin) / dE).astype(int), 0, Ny - 1)
        for k in range(Nk):
            w = float(Wk[k])
            if not np.isfinite(w) or w <= 0.0:
                continue
            jl = max(0, j[k] - half_bins)
            jr = min(Ny, j[k] + half_bins + 1)
            if agg == 'max':
                img[jl:jr, k] = np.maximum(img[jl:jr, k], w)
            else:  # 'sum'
                img[jl:jr, k] += w

    extent = (float(kx.min()), float(kx.max()), emin, emax)
    return img, extent, Egrid

def plot_projected_bands(energies, data3, labels, kpts=None, channel=None, 
                         channel_index=None,
                         width_ev=0.15, dE=0.01, cmap='magma',  agg='sum',
                         overlay_lines=True, ef=None, e_window=None, ax=None, 
                         fig_path=None,
                         k_upsample=4,
                         weight_norm='all-channels',  # 'per-channel'|'all-channels'|'ref'|'none'
                         weight_vmax=0.5,            # 色标上限(固定映射到 0..weight_vmax)
                         weight_ref_max=None,       # weight_norm='ref' 时提供的参考最大值
                         show_colorbar=True):       
    """
    绘制带有投影权重的能带图（展宽条带+颜色强度表示权重），并附带色标。

    参数
    - energies: (Nk, Nbands) 能量（单位 eV）
    - data3, labels: 从 extract_proj_for_plot 得到的 (Nk, Nbands, K) 与标签
    - kpts: (Nk,3) 可选；若提供，用于生成横坐标路径距离；否则横坐标为 0..Nk-1
    - channel: 要显示的通道标签（如 'Mo-p'）。若提供则优先于 channel_index
    - channel_index: 要显示的通道索引（K 维的索引）
    - width_ev: 展宽条带的总宽度（沿能量方向，单位 eV）
    - dE: 能量栅格间距（越小越平滑，计算量越大）
    - cmap: 颜色映射
    - norm: 权重归一化方式：'global'（按所选通道全局最大值归一）或 None（不归一）
    - agg: 多条带相同 k 上的聚合方式：'sum' 或 'max'
    - overlay_lines: 是否叠加黑色能带曲线
    - ef: 费米能级，若提供则绘图时 energies-ef
    - e_window: 能量显示窗口 (emin, emax)，不传则自动
    - ax: 可选 Matplotlib Axes 不传则新建
    - cmap: 传入任意 Matplotlib 色图名，例如 'magma'/'inferno'/'plasma'/'viridis'/'cividis'/'turbo'/'coolwarm'
    - weight_norm:
        'per-channel'  按当前通道的全局最大值归一（默认）
        'all-channels' 使用 data3 全部通道的全局最大值（跨通道可比）
        'ref'          使用 weight_ref_max 作为归一化参考（不同图保持一致）
        'none'         不对权重做归一化（可能超过色标上限并被截断）
    - weight_vmax: 将归一化后的权重映射到 [0, weight_vmax]，默认 0.5
    返回 (fig, ax, im, cbar)
    """
    energies = np.asarray(energies, dtype=float)
    Nk, Nb = energies.shape
    data3 = np.asarray(data3, dtype=float)
    if data3.shape[:2] != (Nk, Nb):
        raise ValueError("data3 前两维应与 energies 一致 (Nk, Nbands)")
    K = data3.shape[2]

    # 横坐标
    if kpts is not None:
        kx = _kpath_distance(kpts)
    else:
        kx = np.arange(Nk, dtype=float)

    # 选择通道
    if channel is not None:
        if channel not in labels:
            raise ValueError(f"未找到通道 '{channel}'，可用通道：{labels}")
        ci = labels.index(channel)
    else:
        ci = 0 if channel_index is None else int(channel_index)
        if ci < 0 or ci >= K:
            raise IndexError(f"channel_index 越界：{ci}，有效范围 0..{K-1}")
    ch_label = labels[ci]

    weights = data3[:, :, ci]

    # (set efermi = 0)
    Eplot = energies.copy()
    if ef is not None:
        Eplot -= float(ef)

    # 先得到一个 0..1 的归一化权重
    if weight_norm == 'per-channel':
        base = float(np.nanmax(weights))
        if np.isfinite(base) and base > 0:
            weights = weights / base
        else:
            weights = np.zeros_like(weights)
    elif weight_norm == 'all-channels':
        base = float(np.nanmax(data3))
        if np.isfinite(base) and base > 0:
            weights = weights / base
        else:
            weights = np.zeros_like(weights)
    elif weight_norm == 'ref':
        if weight_ref_max is None or not np.isfinite(weight_ref_max) or weight_ref_max <= 0:
            raise ValueError("weight_norm='ref' 需要提供正的 weight_ref_max。")
        weights = weights / float(weight_ref_max)
    elif weight_norm == 'none':
        # 不缩放
        pass
    else:
        raise ValueError("weight_norm only support 'per-channel' | 'all-channels' | 'ref' | 'none'")

    # 将权重限定到 [0,1]，再映射到 [0, weight_vmax]，以便固定色度范围
    weights = np.clip(weights, 0.0, 1.0) * float(weight_vmax)

    # k 轴上采样，减少块状感
    kx_use, E_use, W_use = _upsample_along_k(kx, Eplot, weights, factor=k_upsample)

    # 构造图像（注意使用上采样后的数据）
    img, extent, Egrid = _build_band_weight_image(
        kx_use, E_use, W_use, e_window=e_window, dE=dE, width_ev=width_ev, agg=agg
    )

    # 不再按图像最大值二次归一化，直接在色图中以 vmin=0, vmax=weight_vmax 呈现
    img = np.clip(img, 0.0, float(weight_vmax))

    # 绘图
    imax = float(np.nanmax(img))
    if np.isfinite(imax) and imax > 0:
        img = img / imax
    else:
        img = np.zeros_like(img)

    # 绘图
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
        created_fig = True
    else:
        fig = ax.figure

    # 使用 pcolormesh 以真实 kx 间距绘制，固定色度范围到 [0, weight_vmax]
    k_edges = _centers_to_edges(kx_use)
    e_edges = _centers_to_edges(Egrid)
    KX, EE = np.meshgrid(k_edges, e_edges)
    pcm = ax.pcolormesh(KX, EE, img, cmap=cmap, shading='auto', vmin=0.0, vmax=float(weight_vmax))
    cbar = None
    if show_colorbar:
        cbar = fig.colorbar(pcm, ax=ax)

    # 叠加能带线
    if overlay_lines:
        for b in range(Nb):
            ax.plot(kx, Eplot[:, b], color='k', lw=0.5, alpha=0.6)
    
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Projected bands: {ch_label}")

    if fig_path:
        import os
        out_dir = os.path.dirname(fig_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    return fig, ax, pcm, cbar

def plot_projected_bands_grid(energies, data3, labels, channels,
                              kpts=None, width_ev=0.15, dE=0.01, cmap='magma', agg='sum',
                              overlay_lines=True, ef=None, e_window=None,
                              ncols=2, figsize=None, share_colorbar=True,
                              k_upsample=4,
                              weight_norm='all-channels', weight_vmax=0.5, weight_ref_max=None,
                              fig_path=None):
    """
    组图：在一张图里以子图呈现多个元素+轨道通道的投影权重。
    - channels: list[str]，例如 ['S-p','W-d','W-s']
    - weight_norm 建议用 'all-channels' 或 'ref'，可保证各子图色度一致可比
    - share_colorbar=True 时使用全局共享色标（范围 [0, weight_vmax]）
    返回 (fig, axes, pcms, cbar)
    """
    # 基本检查
    energies = np.asarray(energies, dtype=float)
    data3 = np.asarray(data3, dtype=float)
    Nk, Nb = energies.shape
    if data3.shape[:2] != (Nk, Nb):
        raise ValueError("data3 前两维应与 energies 一致 (Nk, Nbands)")
    if isinstance(channels, str):
        channels = [channels]
    missing = [ch for ch in channels if ch not in labels]
    if missing:
        raise ValueError(f"以下通道不存在于 labels: {missing}\n可用: {labels}")

    n = len(channels)
    ncols = int(ncols) if ncols else 2
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.5 * ncols, 3.6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=120, squeeze=False)
    pcms = []
    for idx, ch in enumerate(channels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        # 每个子图不单独画 colorbar；叠加线与色表范围保持一致
        _, _, pcm, _ = plot_projected_bands(
            energies, data3, labels,
            kpts=kpts,
            channel=ch,
            width_ev=width_ev, dE=dE, cmap=cmap, agg=agg,
            overlay_lines=overlay_lines, ef=ef, e_window=e_window,
            ax=ax, fig_path=None, k_upsample=k_upsample,
            weight_norm=weight_norm, weight_vmax=weight_vmax, weight_ref_max=weight_ref_max,
            show_colorbar=False
        )
        pcms.append(pcm)

        # 隐藏多余空轴
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis('off')

    cbar = None
    if share_colorbar and pcms:
        # 共享色标：范围固定为 [0, weight_vmax]
        cbar = fig.colorbar(pcms[0], ax=axes, fraction=0.025, pad=0.02)

    fig.tight_layout()
    if fig_path:
        import os
        out_dir = os.path.dirname(fig_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    return fig, axes, pcms, cbar


def _upsample_along_k(kx, energies, weights, factor=4):
    """
    在 k 轴按路径距离做线性插值上采样，以获得更连续的条带。
    返回 kx_fine, energies_fine, weights_fine
    """
    if factor is None or int(factor) <= 1:
        return kx, energies, weights
    factor = int(factor)
    Nk, Nb = energies.shape
    kx = np.asarray(kx, dtype=float)
    kx_fine = np.linspace(kx[0], kx[-1], (Nk - 1) * factor + 1)
    E_fine = np.empty((kx_fine.size, Nb), dtype=float)
    W_fine = np.empty((kx_fine.size, Nb), dtype=float)
    for b in range(Nb):
        E_fine[:, b] = np.interp(kx_fine, kx, energies[:, b])
        W_fine[:, b] = np.interp(kx_fine, kx, weights[:, b])
    return kx_fine, E_fine, W_fine

if __name__=="__main__":  

    outcar_path = "./test/OUTCAR"
    poscar_path = "./test/POSCAR"
    procar_path = "./test/PROCAR"
    fig_path = './figure_name.png'

    type , number ,ionrange= read_poscar(poscar_path)

    proj, elements, used_orbs, kpts, energies = read_procar(procar_path, type, ionrange, orbital=('s', 'p', 'd'))

    data3, labels = extract_proj_for_plot(proj, elements, used_orbs,
                                          element_sel='all', orbital_sel='all')

    efermi = read_outcar(outcar_path)

    chan_name = 'S-p'
    # === 调用绘图函数 ===
    # 若找到指定通道则用标签指定；否则回退到第0个通道
    if chan_name in labels:
        fig, ax, im, cbar = plot_projected_bands(
            energies, data3, labels,
            kpts=kpts,
            channel=chan_name,          # 用标签选通道
            width_ev=0.20,              # 条带展宽（eV）
            dE=0.01,                    # 能量网格步长
            cmap='magma',             # 'inferno','magma','turbo','cividis','coolwarm'
            agg='sum',
            overlay_lines=False,
            ef=efermi,                    # 如需对齐费米能级，填入 ef 值
            e_window=None ,              # 指定能量窗口 (emin, emax)；None 自动
            fig_path=fig_path
        )
    else:
        print("Fallback to channel_index=0 for plotting.")
        fig, ax, im, cbar = plot_projected_bands(
            energies, data3, labels,
            kpts=kpts,
            channel_index=0,            # 回退到第0个通道
            width_ev=0.20,
            dE=0.01,
            cmap='viridis',
            agg='sum',
            overlay_lines=True,
            ef=None,
            e_window=None,
            fig_path=fig_path
        )


    
