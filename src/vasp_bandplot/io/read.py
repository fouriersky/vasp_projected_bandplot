import numpy as np

def read_outcar(outcar_path):
    """
    从 OUTCAR 中读取费米能，返回 float ef。
    同时兼容 'E-fermi :' 与 'Fermi energy:' 两种格式，取文件中最后一次出现的值。
    """
    import re
    ef = None
    pat = re.compile(r'(?:E-fermi|Fermi\s*energy)\s*:?\s*([-\d.+Ee]+)', re.IGNORECASE)
    with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'fermi' in line.lower():
                m = pat.search(line)
                if m:
                    try:
                        ef = float(m.group(1))
                    except ValueError:
                        pass
    if ef is None:
        raise RuntimeError(" E-fermi/Fermi energy NOT FOUND in OUTCAR ")
    return ef


def read_poscar(poscar_file):
    with open(poscar_file, 'r') as file:
        lines = file.readlines()

    # Extracting relevant data from POSCAR file
    atom_types = lines[5].split()
    atom_numbers = list(map(int, lines[6].split()))
    ranges = atom_type_index_ranges(atom_numbers)

    return atom_types, atom_numbers,ranges

def atom_type_index_ranges(atom_numbers):
    ranges = []
    start = 1
    for cnt in atom_numbers:
        end = start + cnt - 1
        ranges.append((start, end))
        start = end + 1
    return ranges

def _build_ion2elem(atom_types, idx_ranges):
    """
    根据元素顺序与每个元素覆盖的 1-based 离子序号范围，构建 ion->element 映射列表(0-based 索引)
    """
    if not atom_types or not idx_ranges or len(atom_types) != len(idx_ranges):
        raise ValueError("atom_types length not consistent with idx_ranges ")
    n_ions = idx_ranges[-1][1]
    ion2elem = [None] * n_ions
    for t, (s, e) in zip(atom_types, idx_ranges):
        for ion_idx in range(s, e + 1):  # 1-based
            ion2elem[ion_idx - 1] = t
    if any(v is None for v in ion2elem):
        raise ValueError("element index out of range")
    return ion2elem


def read_procar(procar_path, atom_types, idx_ranges, orbital=('s', 'p', 'd')):
    import re
    """
    读取 VASP PROCAR，聚合到元素，并按给定轨道集合(s/p/d)求和。

    参数
    - procar_path: PROCAR 文件路径
    - atom_types: 从 POSCAR 得到的元素列表（如 ['Mo','S','Se','W']）
    - idx_ranges: 从 POSCAR 得到的各元素覆盖的 1-based 离子序号闭区间列表，如 [(1,2),(3,6),...]
    - orbital: 选取并聚合的轨道集合，支持：
        * 序列：('s','p','d')、['p','d'] 等
        * 字符串：'spd'、'p,d'、's p' 等

    返回
    - proj: np.ndarray, 形状 (Nk, Nbands, Nelements, Norbits)
    - elements: list[str], 元素列表（与 proj 的第3维一致，一般等于 atom_types）
    - used_orbs: list[str], 实际使用的轨道标签（与 proj 的第4维一致）
    - kpts: np.ndarray, 形状 (Nk, 3)
    - energies: np.ndarray, 形状 (Nk, Nbands)

    注：当前实现按非自旋（单分量）处理；若为自旋极化需扩展一维或选择分量。
    """
    # 规范化 orbital 参数
    if isinstance(orbital, str):
        s = orbital.replace(',', ' ').replace(';', ' ').split()
        if len(s) == 1 and set(s[0]).issubset(set('spd')) and len(s[0]) >= 1:
            used_orbs = [ch for ch in 'spd' if ch in s[0]]
        else:
            used_orbs = s
    else:
        used_orbs = list(orbital)
    used_orbs = [o.lower() for o in used_orbs]
    for o in used_orbs:
        if o not in ('s', 'p', 'd'):
            raise ValueError(f"orbital 仅支持 's','p','d'，收到: {o}")
        
    # PROCAR 列映射（忽略末尾 tot）
    # 行形如：ion s py pz px dxy dyz dz2 dxz dx2 tot
    orb_cols = {
        's': [0],
        'p': [1, 2, 3],           # py, pz, px
        'd': [4, 5, 6, 7, 8],     # dxy, dyz, dz2, dxz, dx2
    }

    # 读取整个文件
    with open(procar_path, 'r') as f:
        lines = f.read().splitlines()

    # 解析头部
    header_line = next((ln for ln in lines if '# of k-points' in ln), None)
    if header_line is None:
        raise RuntimeError(" '# of k-points' line NOT FOUND in PROCAR ")
    m = re.search(r'# of k-points:\s*(\d+)\s*# of bands:\s*(\d+)\s*# of ions:\s*(\d+)', header_line)
    if not m:
        raise RuntimeError("error in reading PROCAR  Nk/Nbands/Nions")
    Nk = int(m.group(1)); Nbands = int(m.group(2)); Nions = int(m.group(3))

    # 构建 ion->element 映射并校验
    ion2elem = _build_ion2elem(atom_types, idx_ranges)
    if len(ion2elem) != Nions:
        raise ValueError(f"POSCAR's ion_number({len(ion2elem)}) inconsistent with PROCAR's  Nions({Nions})")

    # 元素顺序使用 atom_types
    elements = list(atom_types)
    elem_index = {e: i for i, e in enumerate(elements)}
    Nelements = len(elements)

    # 预分配
    kpts = np.zeros((Nk, 3), dtype=float)
    energies = np.zeros((Nk, Nbands), dtype=float)
    proj = np.zeros((Nk, Nbands, Nelements, len(used_orbs)), dtype=float)

    # main loop
    i = 0
    k_idx = -1
    while i < len(lines):
        line = lines[i].strip()

        # k-point line
        if line.startswith('k-point'):
            # 例: k-point    1 :    -0.00000000-0.00000000 0.00000000     weight = 0.00285714
            # 兼容数字连写和可选的 weight 字段
            # 解析 k 点索引
            m_idx = re.search(r'k-point\s+(\d+)', line, flags=re.IGNORECASE)
            if not m_idx:
                raise RuntimeError(f"failed to parse k-point index from line: {line}")
            k_idx = int(m_idx.group(1)) - 1

            # 只在 'weight' 之前的子串中找坐标，避免把权重当作坐标
            pre = line.split('weight', 1)[0]

            # 先匹配带小数点/科学计数的数，避免把 '150' 这样的索引当作坐标
            float_pat_strict = r'[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?'
            nums = re.findall(float_pat_strict, pre)

            # 若异常情况仍少于3个，再退回到宽松匹配并取最后三个
            if len(nums) < 3:
                float_pat_loose = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
                nums = re.findall(float_pat_loose, pre)

            if len(nums) < 3:
                raise RuntimeError(f"failed to parse k-point coords from line: {line}")

            kx_, ky_, kz_ = map(float, nums[-3:])
            kpts[k_idx, :] = [kx_, ky_, kz_]

            i += 1
            continue
            
        # band line
        if line.startswith('band'):
            # 例: band   1 # energy  -12.59193244 # occ.  2.00000000
            b_tokens = line.split()
            b_idx = int(b_tokens[1]) - 1
            m_energy = re.search(r'energy\s+([-\d.Ee+]+)', line)
            if m_energy:
                energies[k_idx, b_idx] = float(m_energy.group(1))

            # 跳到 'ion' 表头
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('ion'):
                i += 1
            if i >= len(lines):
                break  # 文件结束

            # 跳过 'ion ...' 这一表头行
            i += 1

            # 读取 Nions 行
            for j in range(Nions):
                ion_line = lines[i + j].strip()
                if not ion_line or ion_line.lower().startswith('tot'):
                    break
                parts = ion_line.split()
                ion_id = int(parts[0]) - 1  # 转为 0-based
                # 取 s,py,pz,px,dxy,dyz,dz2,dxz,dx2 这 9 列
                vals = list(map(float, parts[1:1 + 9]))
                eidx = elem_index[ion2elem[ion_id]]
                for oidx, o in enumerate(used_orbs):
                    ssum = sum(vals[cid] for cid in orb_cols[o])
                    proj[k_idx, b_idx, eidx, oidx] += ssum

            # 跳过 Nions 行
            i += Nions
            # 可选：跳过 'tot' 汇总行
            if i < len(lines) and lines[i].strip().lower().startswith('tot'):
                i += 1
            continue

        i += 1

    return proj, elements, used_orbs, kpts, energies


def _resolve_element_indices(elements, elem_sel):
    """
    将用户输入的元素选择解析为索引列表与标签列表。
    支持：
    - 'all' 或 '*' 或 None
    - 单个元素名 'Mo' 或多个 'Mo,S'
    - 元素名列表 ['Mo','S'] 或索引/索引列表 0 / [0,1]
    """
    if elem_sel in (None, 'all', '*'):
        idx = list(range(len(elements)))
        return idx, [elements[i] for i in idx]

    if isinstance(elem_sel, (int, np.integer)):
        idx = [int(elem_sel)]
        return idx, [elements[idx[0]]]

    if isinstance(elem_sel, str):
        names = [s for s in elem_sel.replace(',', ' ').split() if s]
        idx = []
        for n in names:
            if n not in elements:
                raise ValueError(f"element '{n}' not exist in {elements}")
            idx.append(elements.index(n))
        return idx, names

    # 可迭代：索引或名称混合
    idx = []
    labels = []
    for x in elem_sel:
        if isinstance(x, (int, np.integer)):
            i = int(x)
            if i < 0 or i >= len(elements):
                raise IndexError(f"out of index {i}, effective range 0..{len(elements)-1}")
            idx.append(i)
            labels.append(elements[i])
        else:
            if x not in elements:
                raise ValueError(f"element '{x}' not in elements {elements}")
            idx.append(elements.index(x))
            labels.append(x)
    return idx, labels


def _resolve_orbital_indices(used_orbs, orb_sel):
    """
    将用户输入的轨道选择解析为索引列表与标签列表。
    支持：
    - 'all' 或 '*' 或 None
    - 's' / 'p' / 'd' 或组合字符串 'spd' / 'p,d' / 's p'
    - 列表 ['s','p'] 或索引/索引列表 0 / [0,1]
    """
    if orb_sel in (None, 'all', '*'):
        idx = list(range(len(used_orbs)))
        return idx, [used_orbs[i] for i in idx]

    # 字符串
    if isinstance(orb_sel, str):
        # 支持 'spd' 合写
        tokens = orb_sel.replace(',', ' ').replace(';', ' ').split()
        if len(tokens) == 1 and set(tokens[0]).issubset(set('spd')) and tokens[0] not in used_orbs:
            req = [ch for ch in 'spd' if ch in tokens[0]]
        else:
            req = tokens
        idx = []
        for r in req:
            rlow = r.lower()
            if rlow not in used_orbs:
                raise ValueError(f"orbital '{r}' not in used_orbs  {used_orbs}")
            idx.append(used_orbs.index(rlow))
        return idx, [used_orbs[i] for i in idx]

    # 可迭代：索引或标签混合
    idx = []
    labels = []
    for x in orb_sel:
        if isinstance(x, (int, np.integer)):
            i = int(x)
            if i < 0 or i >= len(used_orbs):
                raise IndexError(f"out of index {i}, effective range 0..{len(used_orbs)-1}")
            idx.append(i)
            labels.append(used_orbs[i])
        else:
            rlow = str(x).lower()
            if rlow not in used_orbs:
                raise ValueError(f"orbital '{x}' not in used_orbs {used_orbs}")
            idx.append(used_orbs.index(rlow))
            labels.append(rlow)
    return idx, labels


def extract_proj_for_plot(proj, elements, used_orbs, element_sel='all', orbital_sel='all'):
    """
    从 read_procar 返回的数据中按用户指定选择元素与轨道，恒定返回三维数组：
    - data: 形状 (Nk, Nbands, K)，其中 K = 选中元素数 * 选中轨道数
    - labels: 长度为 K 的标签列表，对应第三维的顺序（元素优先，再轨道）
    """
    e_idx, e_labels = _resolve_element_indices(elements, element_sel)
    o_idx, o_labels = _resolve_orbital_indices(used_orbs, orbital_sel)

    # 选择子集：形状 (Nk, Nbands, Ne_sel, No_sel)
    data4 = proj[:, :, e_idx, :][:, :, :, o_idx]
    Nk, Nb, Ne, No = data4.shape

    # 压平成第三维：元素优先，再轨道 => (Nk, Nbands, Ne*No)
    data3 = data4.reshape(Nk, Nb, Ne * No)

    # 组合标签（与上面展平顺序一致）
    labels = [f"{e}-{o}" for e in e_labels for o in o_labels]

    return data3, labels


if __name__ == "__main__":

    poscar_path = "./test/POSCAR"
    procar_path = "./test/PROCAR"

    type , number ,ionrange= read_poscar(poscar_path)
    #print(type,'\n',number,'\n',range)

    proj, elements, used_orbs, kpts, energies = read_procar(procar_path, type, ionrange, orbital=('s', 'p', 'd'))
    #print("proj shape:", proj.shape)
    #print("kpts shape:", kpts.shape, "energies shape:", energies.shape)
    #print("elements:", elements, "used_orbs:", used_orbs)

    # all元素与轨道，返回 (Nk, Nbands, K)
    data3, labels = extract_proj_for_plot(proj, elements, used_orbs,
                                          element_sel='all', orbital_sel='all')

    chan_name = 'W-s'

    if chan_name in labels:
        c = labels.index(chan_name)
        print(f"{chan_name} weight min/max/mean = ",
            float(np.min(data3[:, :, c])),
            float(np.max(data3[:, :, c])),
            float(np.mean(data3[:, :, c])))
    else:
        print(f"NOT FOUND: {chan_name}, AVAILABLE: {labels}")
