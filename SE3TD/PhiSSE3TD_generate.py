import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import traceback
import faulthandler 
faulthandler.enable()

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase 

# OpenBabel
from openbabel import openbabel as ob

# Local Imports
import PhiSSE3TD_config as config
from PhiSSE3TD_utils_chem import (
    decode_and_align_fragments,
    save_fragments_as_single_sdf_entry,
    combine_ligand_and_protein,
    get_largest_connected_component,                                       
    get_anchor_connected_component,                                        
    build_known_bonds_for_fixed_pairs,
    simulate_hydrogen_consumption_and_plan_bonds,
    build_molecule_from_blueprint, 
    refine_fragments_in_pocket                              
)
                    
from PhiSSE3TD_dataset import preprocess_pocket_data, preprocess_fixed_ligand_data, split_complex_data
from PhiSSE3TD_model import E3NNTransformerDiffusion
from PhiSSE3TD_diffusion import DiffusionProcess
from PhiSSE3TD_nodes_predicter import PocketToLigandSizePredictor

rdBase.DisableLog('rdApp.warning')  
rdBase.DisableLog('rdApp.error')

# =============================================================================
                            
# =============================================================================

def get_valid_tasks(mode):
    """
    根据模式扫描 TASK_A / TASK_B 目录，返回有效的任务列表。
    Survival 模式: 扫描 TASK_A/{id}/ 下的 complex.pt + pocket.pdb
    Creative 模式: 扫描 TASK_B/{id}/ 下的 {id}.pt + pocket.pdb
    """
    tasks = []

    if mode == 'SURVIVAL':
        scan_dir = config.TASK_A_DATA_DIR
        if not os.path.exists(scan_dir):
            print(f"[Error] TASK_A directory not found: {scan_dir}")
            return []

        print(f"正在扫描任务... (模式: {mode})")
        print(f"扫描目录: {scan_dir}")
        for task_id in sorted(os.listdir(scan_dir)):
            task_dir = os.path.join(scan_dir, task_id)
            if not os.path.isdir(task_dir):
                continue

            complex_pt = os.path.join(task_dir, 'complex.pt')
            pocket_pdb = os.path.join(task_dir, 'pocket.pdb')

            if os.path.exists(complex_pt) and os.path.exists(pocket_pdb):
                tasks.append({
                    'id': task_id,
                    'complex_pt_path': complex_pt,
                    'pocket_pdb_path': pocket_pdb
                })

    else:
        scan_dir = config.TASK_B_DATA_DIR
        if not os.path.exists(scan_dir):
            print(f"[Error] TASK_B directory not found: {scan_dir}")
            return []

        print(f"正在扫描任务... (模式: {mode})")
        print(f"扫描目录: {scan_dir}")

        for task_id in sorted(os.listdir(scan_dir)):
            task_dir = os.path.join(scan_dir, task_id)
            if not os.path.isdir(task_dir):
                continue

            pocket_pt = os.path.join(task_dir, f'{task_id}.pt')
            pocket_pdb = os.path.join(task_dir, 'pocket.pdb')

            if os.path.exists(pocket_pt) and os.path.exists(pocket_pdb):
                tasks.append({
                    'id': task_id,
                    'pocket_pt_path': pocket_pt,
                    'pocket_pdb_path': pocket_pdb
                })

    return tasks

@torch.no_grad()
def generate_ligand(model, diffusion_process, pocket_data, num_noise_nodes, ddim_steps, device,
                    mode,
                    initial_noise_feat=None,
                    initial_noise_pos=None,
                    initial_noise_ref=None,
                    fixed_ligand_data=None,
                    fixed_ligand_edges=None):
    """
    执行配体生成的迭代去噪过程。
    [修订版]: 删除 Survival 2.0，新增固定配体边注入 (策略 B)。
    """
    model.eval()
    pocket_data = pocket_data.to(device)
    num_fixed = 0
    if fixed_ligand_data is not None:
        fixed_ligand_data = fixed_ligand_data.to(device)
        num_fixed = fixed_ligand_data.num_nodes
        x0_fixed_feat_full = fixed_ligand_data.x[:, :config.LIGAND_FEATURE_DIM]
        x0_fixed_pos = fixed_ligand_data.pos
        x0_fixed_ref = fixed_ligand_data.ref_coords
        fixed_batch_idx = torch.zeros(num_fixed, dtype=torch.long, device=device)
    if initial_noise_pos is not None:
        xt_curr_free_pos = initial_noise_pos[num_fixed:]
        xt_curr_free_ref = initial_noise_ref[num_fixed:]
        xt_curr_free_feat = initial_noise_feat[num_fixed:]
    else:
        xt_curr_free_pos = torch.randn(num_noise_nodes, 3, device=device)
        xt_curr_free_ref = torch.randn(num_noise_nodes, 3, 3, device=device)
        xt_curr_free_feat = torch.randn(num_noise_nodes, config.LIGAND_FEATURE_DIM, device=device)
    dynamic_indices = config.DYNAMIC_CHEM_INDICES

                    
    if num_fixed > 0:
        if initial_noise_feat is not None:
            current_evolving_dynamic_feat = initial_noise_feat[:num_fixed, dynamic_indices]
        else:
            current_evolving_dynamic_feat = torch.randn(num_fixed, len(dynamic_indices), device=device)
    timesteps = torch.linspace(diffusion_process.num_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)
    pbar = tqdm(range(ddim_steps), desc=f"{mode} Sampling", leave=False)

    current_free_feat = xt_curr_free_feat
    current_free_pos = xt_curr_free_pos
    current_free_ref = xt_curr_free_ref

                    
    final_edge_logits = None
    final_filtered_edge_index = None

    for i in pbar:
        t_scalar = timesteps[i]
        t = t_scalar.view(1)
        t_prev = timesteps[i+1] if i < ddim_steps - 1 else torch.tensor(-1, device=device, dtype=torch.long)
        if num_fixed > 0:
                                                        
            input_fixed_feat = x0_fixed_feat_full.clone()
            # 保留动态特征（如允许特定特征随时间步演化）
            for idx_i, dim_idx in enumerate(dynamic_indices):
                input_fixed_feat[:, dim_idx] = current_evolving_dynamic_feat[:, idx_i]

            input_fixed_pos = x0_fixed_pos
            input_fixed_ref = x0_fixed_ref

            xt_curr_feat = torch.cat([input_fixed_feat, current_free_feat], dim=0)
            xt_curr_pos = torch.cat([input_fixed_pos, current_free_pos], dim=0)
            xt_curr_ref = torch.cat([input_fixed_ref, current_free_ref], dim=0)

                                        
            chem_start = config.EMBEDDING_DIM_IN
            chem_end = config.LIGAND_FEATURE_DIM
            current_fixed_chem = xt_curr_feat[:num_fixed, chem_start:chem_end]
            current_free_chem = current_free_feat[:, chem_start:chem_end]
            chem_props_input = torch.cat([current_fixed_chem, current_free_chem], dim=0)
        else:
            xt_curr_feat = current_free_feat
            xt_curr_pos = current_free_pos
            xt_curr_ref = current_free_ref
            chem_props_input = None

        xt_prev_tuple, edge_info_tuple = diffusion_process.p_sample_ddim(
            xt_tuple=(xt_curr_feat, xt_curr_pos, xt_curr_ref),
            protein_data=pocket_data,
            t=t,
            t_prev=t_prev,
            eta=config.DDIM_ETA,
            known_chem_props=chem_props_input,
            fixed_ligand_edges=fixed_ligand_edges
        )

        xt_prev_feat, xt_prev_pos, xt_prev_ref = xt_prev_tuple
        if i == ddim_steps - 1:
            final_edge_logits, final_filtered_edge_index = edge_info_tuple

        # =========================================================================
                            
        # =========================================================================
        if num_fixed > 0:
            current_free_feat = xt_prev_feat[num_fixed:]
            current_free_pos = xt_prev_pos[num_fixed:]
            current_free_ref = xt_prev_ref[num_fixed:]
            current_evolving_dynamic_feat = xt_prev_feat[:num_fixed, dynamic_indices]
        else:
            current_free_feat = xt_prev_feat
            current_free_pos = xt_prev_pos
            current_free_ref = xt_prev_ref

                        
    if num_fixed > 0:
        final_fixed_feat = x0_fixed_feat_full.clone()
        for idx_i, dim_idx in enumerate(dynamic_indices):
            final_fixed_feat[:, dim_idx] = current_evolving_dynamic_feat[:, idx_i]

        final_fixed_pos = x0_fixed_pos
        final_fixed_ref = x0_fixed_ref

        xt_curr_feat = torch.cat([final_fixed_feat, current_free_feat], dim=0)
        xt_curr_pos = torch.cat([final_fixed_pos, current_free_pos], dim=0)
        xt_curr_ref = torch.cat([final_fixed_ref, current_free_ref], dim=0)
    else:
        xt_curr_feat = current_free_feat
        xt_curr_pos = current_free_pos
        xt_curr_ref = current_free_ref

    return (
        (xt_curr_feat, xt_curr_pos, xt_curr_ref),
        final_edge_logits,
        final_filtered_edge_index
    )

def load_and_merge_vocabularies(vocab_keys, device):
    """
    接收指定的词汇表键名列表，加载并合并词汇表。
    """
    print(f"Loading vocabularies for keys: {vocab_keys}")
    merged_vocab_dict = {}
    
    for key in vocab_keys:
        path = config.VOCAB_FILES_MAP.get(key)
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    merged_vocab_dict.update(data)
                print(f"  -> Loaded '{key}': {len(data)} fragments")
            except Exception as e:
                print(f"  [Error] Failed to load '{key}' at {path}: {e}")
        else:
            print(f"  [Warn] Vocabulary file for '{key}' not found at: {path}")
            
    if not merged_vocab_dict:
        raise ValueError(f"No vocabulary data loaded for keys: {vocab_keys}")

    vocab_smiles = list(merged_vocab_dict.keys())
    vocab_embeds = torch.tensor(list(merged_vocab_dict.values()), dtype=torch.float32, device=device)
    print(f"Total Combined Vocabulary: {len(vocab_smiles)} fragments")
    
    return vocab_smiles, vocab_embeds


def _format_node_kind(node_idx, fixed_node_set):
    return "fixed" if node_idx in fixed_node_set else "free"


def _format_total_h(debug_info, key, node_idx):
    values = debug_info.get(key, {})
    return values.get(node_idx, values.get(str(node_idx), "?"))


def format_sand_table_reason(debug_info, lcc_smiles, sorted_lcc_nodes):
    """Return one concise, specific failure reason for the sand table."""
    if not debug_info:
        return "Unknown sand-table failure"

    reason = debug_info.get('reason', 'unknown')
    if 'edge' not in debug_info:
        return reason.replace('_', ' ')

    frag_A, frag_B = debug_info['edge']
    original_A = sorted_lcc_nodes[frag_A] if frag_A < len(sorted_lcc_nodes) else "?"
    original_B = sorted_lcc_nodes[frag_B] if frag_B < len(sorted_lcc_nodes) else "?"
    smiles_A = lcc_smiles[frag_A] if frag_A < len(lcc_smiles) else "?"
    smiles_B = lcc_smiles[frag_B] if frag_B < len(lcc_smiles) else "?"

    if reason == 'distance_exceeded':
        min_dist = debug_info.get('min_dist', float('nan'))
        threshold = debug_info.get('dist_threshold', '?')
        return (
            f"Distance exceeded on {debug_info.get('edge_kind', '?')} edge "
            f"{original_A}:{smiles_A} -- {original_B}:{smiles_B} "
            f"({min_dist:.2f}A > {threshold}A)"
        )

    if reason == 'no_available_h':
        return (
            f"No available H on {debug_info.get('edge_kind', '?')} edge "
            f"{original_A}:{smiles_A} -- {original_B}:{smiles_B}"
        )

    return (
        f"{reason.replace('_', ' ')} on {debug_info.get('edge_kind', '?')} edge "
        f"{original_A}:{smiles_A} -- {original_B}:{smiles_B}"
    )


def print_sand_table_debug(debug_info, lcc_smiles, sorted_lcc_nodes, fixed_node_set):
    """Print detailed failure diagnostics for the hydrogen sand table."""
    if not debug_info:
        print("        -> [SandDiag] No detailed diagnostics were returned.")
        return

    reason = debug_info.get('reason', 'unknown')
    print(
        "        -> [SandDiag] Reason: "
        f"{reason} | UniqueEdges={debug_info.get('total_unique_edges', '?')} "
        f"(fixed-fixed kept={debug_info.get('fixed_fixed_edges', '?')}, "
        f"fixed-free={debug_info.get('fixed_generated_edges', '?')}, "
        f"free-free={debug_info.get('generated_generated_edges', '?')}) | "
        f"KnownBonds={debug_info.get('known_bonds_count', '?')} | "
        f"PlannedBeforeFail={debug_info.get('planned_before_failure', 0)} | "
        f"SkippedFixedFree={debug_info.get('skipped_fixed_generated_count', 0)}"
    )

    if 'edge' not in debug_info:
        return

    frag_A, frag_B = debug_info['edge']
    original_A = sorted_lcc_nodes[frag_A] if frag_A < len(sorted_lcc_nodes) else "?"
    original_B = sorted_lcc_nodes[frag_B] if frag_B < len(sorted_lcc_nodes) else "?"
    smiles_A = lcc_smiles[frag_A] if frag_A < len(lcc_smiles) else "?"
    smiles_B = lcc_smiles[frag_B] if frag_B < len(lcc_smiles) else "?"

    print(
        "        -> [SandDiag] FailedEdge: "
        f"LCC {frag_A}({ _format_node_kind(frag_A, fixed_node_set) }, orig={original_A}, smi={smiles_A}) "
        f"-- LCC {frag_B}({ _format_node_kind(frag_B, fixed_node_set) }, orig={original_B}, smi={smiles_B}) "
        f"| kind={debug_info.get('edge_kind', '?')} | order={debug_info.get('edge_order', '?')}"
    )

    for label, frag_idx in [('A', frag_A), ('B', frag_B)]:
        print(
            f"        -> [SandDiag] Node{label} H: "
            f"initial={_format_total_h(debug_info, 'initial_total_h', frag_idx)} "
            f"afterKnown={_format_total_h(debug_info, 'after_known_total_h', frag_idx)} "
            f"remainingNow={debug_info.get(f'total_remaining_h_{label}', '?')} | "
            f"degrees total={_format_total_h(debug_info, 'degree_total', frag_idx)} "
            f"fixed-fixed={_format_total_h(debug_info, 'degree_fixed_fixed', frag_idx)} "
            f"fixed-free={_format_total_h(debug_info, 'degree_fixed_generated', frag_idx)} "
            f"free-free={_format_total_h(debug_info, 'degree_generated_generated', frag_idx)}"
        )
        print(
            f"        -> [SandDiag] Node{label} atom H ledger: "
            f"initial={debug_info.get(f'initial_h_{label}', {})} "
            f"remaining={debug_info.get(f'remaining_h_{label}', {})} "
            f"available={debug_info.get(f'avail_{label}', [])}"
        )

    if reason == 'distance_exceeded':
        print(
            "        -> [SandDiag] Distance: "
            f"min={debug_info.get('min_dist', '?'):.3f}A "
            f"> threshold={debug_info.get('dist_threshold', '?')}A | "
            f"best_atoms=({debug_info.get('best_atom_A', '?')}, {debug_info.get('best_atom_B', '?')})"
        )


def bond_blueprints_to_edge_index(bond_blueprints, device):
    """Convert atom-level bond blueprints to fragment-level edges."""
    fragment_pairs = [
        (int(frag_A), int(frag_B))
        for frag_A, _, frag_B, _ in bond_blueprints
        if int(frag_A) != int(frag_B)
    ]
    if not fragment_pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(fragment_pairs, dtype=torch.long, device=device).t().contiguous()


def validate_actual_fragment_connectivity(num_nodes, actual_edge_index, fixed_node_set):
    """Check connectivity after edge pruning, using only bonds that will be built."""
    if num_nodes <= 1:
        return True, num_nodes, "single fragment"

    if fixed_node_set:
        actual_component = get_anchor_connected_component(
            num_nodes, actual_edge_index,
            anchor_indices=fixed_node_set
        )
        if not actual_component:
            return False, 0, "fixed anchors disconnected after edge pruning"
    else:
        actual_component = get_largest_connected_component(num_nodes, actual_edge_index)

    if len(actual_component) != num_nodes:
        return (
            False,
            len(actual_component),
            f"assembled graph has multiple components ({len(actual_component)}/{num_nodes} fragments connected)"
        )

    return True, len(actual_component), "all fragments connected"


def _min_heavy_atom_distance(mol_a, mol_b):
    if (
        mol_a is None or mol_b is None
        or mol_a.GetNumConformers() == 0
        or mol_b.GetNumConformers() == 0
    ):
        return None

    conf_a = mol_a.GetConformer()
    conf_b = mol_b.GetConformer()
    coords_a = [
        np.array(conf_a.GetAtomPosition(atom.GetIdx()), dtype=float)
        for atom in mol_a.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    coords_b = [
        np.array(conf_b.GetAtomPosition(atom.GetIdx()), dtype=float)
        for atom in mol_b.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not coords_a or not coords_b:
        return None

    min_dist = float('inf')
    for pos_a in coords_a:
        dists = np.linalg.norm(np.asarray(coords_b) - pos_a, axis=1)
        min_dist = min(min_dist, float(np.min(dists)))
    return min_dist


def _pdb_residue_key(atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None
    return (
        info.GetChainId().strip(),
        int(info.GetResidueNumber()),
        info.GetInsertionCode().strip(),
        info.GetResidueName().strip(),
    )


def _copy_mol_subset_with_conformer(mol, atom_indices):
    atom_indices = sorted(set(int(idx) for idx in atom_indices))
    if not atom_indices:
        return None

    idx_map = {}
    rw_mol = Chem.RWMol()
    for old_idx in atom_indices:
        atom = Chem.Atom(mol.GetAtomWithIdx(old_idx))
        new_idx = rw_mol.AddAtom(atom)
        idx_map[old_idx] = new_idx

    selected = set(atom_indices)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in selected and end in selected:
            rw_mol.AddBond(idx_map[begin], idx_map[end], bond.GetBondType())

    subset = rw_mol.GetMol()
    if mol.GetNumConformers() > 0:
        src_conf = mol.GetConformer()
        conf = Chem.Conformer(subset.GetNumAtoms())
        for old_idx, new_idx in idx_map.items():
            conf.SetAtomPosition(new_idx, src_conf.GetAtomPosition(old_idx))
        subset.AddConformer(conf, assignId=True)

    try:
        subset.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return subset


def _select_local_protein_atoms(protein_mol, ligand_mol, cutoff):
    if (
        protein_mol is None or ligand_mol is None
        or protein_mol.GetNumConformers() == 0
        or ligand_mol.GetNumConformers() == 0
    ):
        return []

    protein_conf = protein_mol.GetConformer()
    ligand_conf = ligand_mol.GetConformer()
    ligand_coords = [
        np.array(ligand_conf.GetAtomPosition(atom.GetIdx()), dtype=float)
        for atom in ligand_mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not ligand_coords:
        return []
    ligand_coords = np.asarray(ligand_coords, dtype=float)

    selected_atoms = set()
    selected_residues = set()
    cutoff = float(cutoff)

    for atom in protein_mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        atom_idx = atom.GetIdx()
        pos = np.array(protein_conf.GetAtomPosition(atom_idx), dtype=float)
        if float(np.min(np.linalg.norm(ligand_coords - pos, axis=1))) <= cutoff:
            residue_key = _pdb_residue_key(atom)
            if residue_key is None:
                selected_atoms.add(atom_idx)
            else:
                selected_residues.add(residue_key)

    if selected_residues:
        for atom in protein_mol.GetAtoms():
            if _pdb_residue_key(atom) in selected_residues:
                selected_atoms.add(atom.GetIdx())

    expanded = set(selected_atoms)
    for atom_idx in selected_atoms:
        atom = protein_mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            expanded.add(neighbor.GetIdx())

    return sorted(expanded)


def optimize_ligand_with_fixed_pocket(ligand_mol, pocket_pdb_path, max_iters=100):
    """
    在固定的蛋白口袋中优化配体坐标。
    [修订版]：完全移除了原有的 UFF 逻辑。改为严格的二级串行机制：
    1. 第一阶段：Ligand-only MMFF94 优化，调整小分子自身由于组装造成的局部张力。
    2. 第二阶段：Pocket-aware MMFF94 优化（固定局部蛋白）。
    若第二阶段因参数缺失等原因失败，直接熔断并无损使用第一阶段的优化结果作为最终产物。
    """
    debug = {
        'method': 'skipped',
        'reason': None,
        'return_code': None,
        'protein_atom_count': 0,
        'local_protein_atom_count': 0,
        'local_cutoff': float(getattr(config, 'POCKET_AWARE_LOCAL_CUTOFF', 6.0)),
        'ligand_atom_count': ligand_mol.GetNumAtoms() if ligand_mol is not None else 0,
        'min_lp_dist_before': None,
        'min_lp_dist_after': None,
    }

    if ligand_mol is None:
        debug['reason'] = 'ligand_none'
        return ligand_mol, debug
    if ligand_mol.GetNumConformers() == 0:
        debug['reason'] = 'ligand_no_conformer'
        return ligand_mol, debug
    if not pocket_pdb_path or not os.path.exists(pocket_pdb_path):
        debug['reason'] = 'pocket_missing'
        return ligand_mol, debug

    protein_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False, sanitize=True)
    if protein_mol is None:
        protein_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False, sanitize=False)
    if protein_mol is None:
        debug['reason'] = 'pocket_read_failed'
        return ligand_mol, debug
    if protein_mol.GetNumConformers() == 0:
        debug['reason'] = 'pocket_no_conformer'
        return ligand_mol, debug

    debug['protein_atom_count'] = protein_mol.GetNumAtoms()
    debug['min_lp_dist_before'] = _min_heavy_atom_distance(protein_mol, ligand_mol)
    local_cutoff = float(getattr(config, 'POCKET_AWARE_LOCAL_CUTOFF', 6.0))
    local_atom_indices = _select_local_protein_atoms(protein_mol, ligand_mol, local_cutoff)
    debug['local_cutoff'] = local_cutoff
    debug['local_protein_atom_count'] = len(local_atom_indices)
    if not local_atom_indices:
        debug['reason'] = 'no_local_protein_atoms'
        return ligand_mol, debug

    local_protein_mol = _copy_mol_subset_with_conformer(protein_mol, local_atom_indices)
    if local_protein_mol is None or local_protein_mol.GetNumAtoms() == 0:
        debug['reason'] = 'local_pocket_subset_failed'
        return ligand_mol, debug

    protein_atom_count = local_protein_mol.GetNumAtoms()
    nonbonded_thresh = float(getattr(config, 'POCKET_AWARE_FF_NONBONDED_THRESH', 8.0))

    def atom_type_summary(mol):
        items = []
        for atom in mol.GetAtoms():
            items.append(
                f"{atom.GetSymbol()}(q={atom.GetFormalCharge()},rad={atom.GetNumRadicalElectrons()})"
            )
        return ", ".join(items[:80])

    def prepare_mmff_ligand(mol):
        prepared = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(prepared)
        except Exception:
            try:
                prepared.UpdatePropertyCache(strict=False)
            except Exception:
                pass

        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize
            prepared = rdMolStandardize.Cleanup(prepared)
            prepared = rdMolStandardize.Normalize(prepared)
            prepared = rdMolStandardize.Reionize(prepared)
            prepared = rdMolStandardize.Uncharger().uncharge(prepared)
            Chem.SanitizeMol(prepared)
        except Exception:
            try:
                prepared.UpdatePropertyCache(strict=False)
            except Exception:
                pass

        if prepared.GetNumConformers() == 0 and mol.GetNumConformers() > 0 and prepared.GetNumAtoms() == mol.GetNumAtoms():
            src_conf = mol.GetConformer()
            conf = Chem.Conformer(prepared.GetNumAtoms())
            for atom_idx in range(prepared.GetNumAtoms()):
                conf.SetAtomPosition(atom_idx, src_conf.GetAtomPosition(atom_idx))
            prepared.AddConformer(conf, assignId=True)

        try:
            prepared = Chem.AddHs(prepared, addCoords=True)
            Chem.SanitizeMol(prepared)
        except Exception:
            try:
                prepared.UpdatePropertyCache(strict=False)
            except Exception:
                pass
        return prepared

    def run_pocket_forcefield(input_ligand, method_name, stage_iters):
        opt_ligand = Chem.Mol(input_ligand)
        ff = None
        try:
            if method_name == 'MMFF94':
                if not getattr(config, 'POCKET_AWARE_TRY_MMFF', True):
                    return opt_ligand, {
                        'stage': 'pocket-MMFF94',
                        'ok': False,
                        'reason': 'disabled',
                        'return_code': None,
                    }
                mmff_ligand = prepare_mmff_ligand(opt_ligand)
                src_heavy = [atom.GetIdx() for atom in mmff_ligand.GetAtoms() if atom.GetAtomicNum() > 1]
                dst_heavy = [atom.GetIdx() for atom in opt_ligand.GetAtoms() if atom.GetAtomicNum() > 1]
                if len(src_heavy) != len(dst_heavy):
                    return opt_ligand, {
                        'stage': 'pocket-MMFF94',
                        'ok': False,
                        'reason': 'heavy_atom_count_changed_during_cleanup',
                        'atom_summary': atom_type_summary(mmff_ligand),
                        'return_code': None,
                    }
                complex_mol = Chem.CombineMols(local_protein_mol, mmff_ligand)
                try:
                    complex_mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass
                if not AllChem.MMFFHasAllMoleculeParams(complex_mol):
                    return opt_ligand, {
                        'stage': 'pocket-MMFF94',
                        'ok': False,
                        'reason': 'no_mmff_params_after_ligand_cleanup',
                        'atom_summary': atom_type_summary(mmff_ligand),
                        'return_code': None,
                    }
                props = AllChem.MMFFGetMoleculeProperties(complex_mol, mmffVariant='MMFF94')
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        complex_mol, props,
                        nonBondedThresh=nonbonded_thresh,
                        ignoreInterfragInteractions=False
                    )
                except TypeError:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        complex_mol, props,
                        nonbonded_thresh, -1, False
                    )
                stage = 'pocket-MMFF94'
            else:
                return opt_ligand, {
                    'stage': method_name,
                    'ok': False,
                    'reason': 'unknown_method',
                    'return_code': None,
                }

            if ff is None:
                return opt_ligand, {
                    'stage': stage,
                    'ok': False,
                    'reason': 'forcefield_init_failed',
                    'return_code': None,
                }

            for atom_idx in range(protein_atom_count):
                ff.AddFixedPoint(int(atom_idx))
            ff.Initialize()
            return_code = ff.Minimize(maxIts=int(stage_iters))

            complex_conf = complex_mol.GetConformer()
            ligand_conf = opt_ligand.GetConformer()
            for src_idx, dst_idx in zip(src_heavy, dst_heavy):
                ligand_conf.SetAtomPosition(
                    dst_idx,
                    complex_conf.GetAtomPosition(protein_atom_count + src_idx)
                )

            return opt_ligand, {
                'stage': stage,
                'ok': True,
                'reason': 'ok',
                'return_code': int(return_code),
            }
        except Exception as exc:
            return opt_ligand, {
                'stage': method_name,
                'ok': False,
                'reason': f'{type(exc).__name__}: {str(exc).splitlines()[0]}',
                'traceback': traceback.format_exc(),
                'return_code': None,
            }

    def run_ligand_only_mmff(input_ligand, stage_iters):
        opt_ligand = Chem.Mol(input_ligand)
        try:
            mmff_ligand = prepare_mmff_ligand(opt_ligand)
            if not AllChem.MMFFHasAllMoleculeParams(mmff_ligand):
                return opt_ligand, {
                    'stage': 'ligand-only-MMFF94',
                    'ok': False,
                    'reason': 'no_mmff_params_after_cleanup',
                    'atom_summary': atom_type_summary(mmff_ligand),
                    'return_code': None,
                }
            return_code = AllChem.MMFFOptimizeMolecule(mmff_ligand, maxIters=int(stage_iters))

            src_conf = mmff_ligand.GetConformer()
            dst_conf = opt_ligand.GetConformer()
            src_heavy = [atom.GetIdx() for atom in mmff_ligand.GetAtoms() if atom.GetAtomicNum() > 1]
            dst_heavy = [atom.GetIdx() for atom in opt_ligand.GetAtoms() if atom.GetAtomicNum() > 1]
            if len(src_heavy) != len(dst_heavy):
                return opt_ligand, {
                    'stage': 'ligand-only-MMFF94',
                    'ok': False,
                    'reason': 'heavy_atom_count_changed_during_cleanup',
                    'return_code': None,
                }
            for src_idx, dst_idx in zip(src_heavy, dst_heavy):
                dst_conf.SetAtomPosition(dst_idx, src_conf.GetAtomPosition(src_idx))

            return opt_ligand, {
                'stage': 'ligand-only-MMFF94',
                'ok': True,
                'reason': 'ok',
                'return_code': int(return_code),
            }
        except Exception as exc:
            return opt_ligand, {
                'stage': 'ligand-only-MMFF94',
                'ok': False,
                'reason': f'{type(exc).__name__}: {str(exc).splitlines()[0]}',
                'traceback': traceback.format_exc(),
                'return_code': None,
            }

    # 从配置加载两部分的优化步数
    pocket_mmff_steps = int(getattr(config, 'POCKET_AWARE_MMFF_STEPS', 100))
    ligand_mmff_steps = int(getattr(config, 'POCKET_AWARE_LIGAND_ONLY_MMFF_STEPS', 60))

    stage_logs = []

    # -------------------------------------------------------------------------
    # 【反转机制 - 步骤一】：先执行带口袋环境约束的联合优化（让口袋力场引导分子落位，锁死全局漂移）
    # -------------------------------------------------------------------------
    pocket_mmff_mol, pocket_mmff_info = run_pocket_forcefield(ligand_mol, 'MMFF94', pocket_mmff_steps)
    stage_logs.append(pocket_mmff_info)
    debug['stages'] = stage_logs

    if pocket_mmff_info.get('ok'):
        # -------------------------------------------------------------------------
        # 【反转机制 - 步骤二】：若步骤一成功，在其优化的结构上继续执行 Ligand-only 舒展
        # （切断蛋白硬原子碰撞造成的局部扭曲，让小分子键长键角回归物理正常值）
        # -------------------------------------------------------------------------
        ligand_only_mol, ligand_only_info = run_ligand_only_mmff(pocket_mmff_mol, ligand_mmff_steps)
        stage_logs.append(ligand_only_info)
        
        if ligand_only_info.get('ok'):
            final_ligand = ligand_only_mol
            final_method = 'pocket-MMFF94 + ligand-only-MMFF94'
            final_return_code = ligand_only_info.get('return_code')
        else:
            # 极罕见情况：如果第二步仅配体优化失败，直接无损回退到步骤一（带口袋优化）的产物
            final_ligand = pocket_mmff_mol
            final_method = 'pocket-MMFF94 (ligand-only-MMFF94 failed fallback)'
            final_return_code = pocket_mmff_info.get('return_code')
    else:
        # -------------------------------------------------------------------------
        # 【熔断机制】：若步骤一（带口袋优化）因为受体原子参数缺失直接失败
        # 则跳过步骤一，直接将【原始配体】送入步骤二进行单独优化，作为底层保底
        # -------------------------------------------------------------------------
        print(f"        [Warn] Pocket-aware MMFF94 failed ({pocket_mmff_info.get('reason')}). Falling back to optimize original ligand directly.")
        ligand_only_mol, ligand_only_info = run_ligand_only_mmff(ligand_mol, ligand_mmff_steps)
        stage_logs.append(ligand_only_info)
        
        if ligand_only_info.get('ok'):
            final_ligand = ligand_only_mol
            final_method = 'ligand-only-MMFF94 (pocket-MMFF94 failed fallback)'
            final_return_code = ligand_only_info.get('return_code')
        else:
            # 双重失败极值情况，直接返回原始未优化结构
            final_ligand = ligand_mol
            final_method = 'none (all MMFF94 stages failed)'
            final_return_code = None

    debug.update({
        'method': final_method,
        'reason': 'ok' if final_return_code is not None else 'all_stages_failed',
        'return_code': int(final_return_code) if final_return_code is not None else None,
        'min_lp_dist_after': _min_heavy_atom_distance(protein_mol, final_ligand),
    })
    return final_ligand, debug

if __name__ == '__main__':
    print("\n" + "="*60)
    print(f"--- [PhiSSE3TD Generator] Mode: {config.DEFAULT_GENERATION_MODE} ---")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device 
    print(f"Device: {device}")
    
    try:
        model = E3NNTransformerDiffusion().to(device)
        
                    
        model_choice = getattr(config, 'DEFAULT_MODEL_CHOICE', 'PRETRAIN')
        model_path = config.MODEL_WEIGHT_ROUTES.get(model_choice)
        
        if model_path is None:
            print(f"[警告] DEFAULT_MODEL_CHOICE '{model_choice}' 未在 MODEL_WEIGHT_ROUTES 中定义！将回退加载预训练模型。")
            model_path = getattr(config, 'BEST_MODEL_PATH_PRETRAIN', None)

        print(f"Loading Model [{model_choice}] from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
                    
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        diffusion_generator = DiffusionProcess(model=model, device=device)
        print("Model loaded and Diffusion Process initialized.")
        
    except Exception as e:
        print(f"[Critical Error] Model Initialization Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    size_predictor = None
    if getattr(config, 'USE_SIZE_PREDICTOR', False):
        print(f"\nInitializing Size Predictor from: {config.SIZE_PREDICTOR_MODEL_PATH}")
        try:
            size_predictor = PocketToLigandSizePredictor().to(device)
                                        
            checkpoint = torch.load(config.SIZE_PREDICTOR_MODEL_PATH, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            size_predictor.load_state_dict(new_state_dict)
            size_predictor.eval()
            print("✅ Size Predictor loaded successfully.")
        except Exception as e:
            print(f"⚠️ [Warn] Failed to load Size Predictor: {e}")
            print(f"   Falling back to fixed config.INPUT_NOISE_NODES = {config.INPUT_NOISE_NODES}")
            size_predictor = None

    try:
        current_mode = config.DEFAULT_GENERATION_MODE
        print("\n[Vocab] Loading Free Node Vocabularies...")
        free_vocab_keys = config.MODE_VOCAB_SETTINGS.get(current_mode, list(config.VOCAB_FILES_MAP.keys()))
        free_vocab_smiles, free_vocab_embeds = load_and_merge_vocabularies(free_vocab_keys, device)
        print("\n[Vocab] Loading Anchor Node Vocabularies...")
        anchor_vocab_smiles, anchor_vocab_embeds = load_and_merge_vocabularies(['ol'], device)
        
    except Exception as e:
        print(f"[Critical Error] Vocabulary Load Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    valid_tasks = get_valid_tasks(config.DEFAULT_GENERATION_MODE)

    if not valid_tasks:
        print("\n[Error] No valid tasks found.")
        sys.exit(0)

    tasks_to_process = valid_tasks[:config.NUM_POCKETS_TO_PROCESS] if config.NUM_POCKETS_TO_PROCESS != -1 else valid_tasks
    print(f"\nTasks to process: {len(tasks_to_process)}")

            
    output_dir = config.OUTPUT_DIR_GENERATION
    if not output_dir: # Fallback
        output_dir = "generated_molecules"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    for task in tqdm(tasks_to_process, desc="Processing Tasks"):
        pocket_id = task['id']

        print(f"\n\n{'='*30} > Task ID: {pocket_id} < {'='*30}")

        try:
            raw_pocket_pdb = task.get('pocket_pdb_path')
            if not raw_pocket_pdb or not os.path.exists(raw_pocket_pdb):
                print(f"  [Warn] Raw Pocket PDB not found for {pocket_id}. Refinement might skip.")
                raw_pocket_pdb = None

            processed_fixed_ligand = None
            fixed_ligand_edges = None
            num_fixed_nodes = 0

            if config.DEFAULT_GENERATION_MODE == 'SURVIVAL':
                complex_data = torch.load(task['complex_pt_path'], weights_only=False)
                pocket_data_raw, ligand_data_raw, fixed_ligand_edges = split_complex_data(complex_data)

                processed_pocket = preprocess_pocket_data(pocket_data_raw)

                processed_fixed_ligand = preprocess_fixed_ligand_data(ligand_data_raw)
                num_fixed_nodes = processed_fixed_ligand.num_nodes

                fixed_ligand_edges = {
                    'edge_index': fixed_ligand_edges['edge_index'].to(device),
                    'edge_attr': fixed_ligand_edges['edge_attr'].to(device)
                }

                print(f"  [Survival] Pocket: {processed_pocket.num_nodes} nodes | Fixed Ligand: {num_fixed_nodes} nodes | Fixed Edges: {fixed_ligand_edges['edge_index'].size(1)}")

            else:
                                            
                pocket_data_raw = torch.load(task['pocket_pt_path'], weights_only=False)
                processed_pocket = preprocess_pocket_data(pocket_data_raw)
                                                                    
                print(f"  [Creative] Pocket: {processed_pocket.num_nodes} nodes")
            if size_predictor is not None:
                if not hasattr(processed_pocket, 'batch') or processed_pocket.batch is None:
                    processed_pocket.batch = torch.zeros(processed_pocket.num_nodes, dtype=torch.long, device=device)
                    processed_pocket.num_graphs = 1 
                
                # =================================================================
                # 【核心修复】：为旧版 Size Predictor 降维并剔除 Dummy 节点
                # =================================================================
                predictor_pocket = processed_pocket.clone().to(device)
                
                # 1. 过滤掉 dummy 节点，不让其干扰大小预测
                if hasattr(predictor_pocket, 'is_dummy'):
                    valid_mask = ~predictor_pocket.is_dummy
                    predictor_pocket.x = predictor_pocket.x[valid_mask]
                    predictor_pocket.pos = predictor_pocket.pos[valid_mask]
                    predictor_pocket.batch = predictor_pocket.batch[valid_mask]
                    # 【核心补漏】：必须同步切除参考系张量，否则 size_predictor 拼接时会崩溃！
                    if hasattr(predictor_pocket, 'ref_coords'):
                        predictor_pocket.ref_coords = predictor_pocket.ref_coords[valid_mask]
                
                with torch.no_grad():
                    pred_total_float = size_predictor(predictor_pocket).item()
                
                predicted_total_nodes = int(round(pred_total_float))
                
                if num_fixed_nodes > 0:
                    dynamic_noise_nodes = predicted_total_nodes - num_fixed_nodes
                    dynamic_noise_nodes = max(3, dynamic_noise_nodes)
                else:
                    dynamic_noise_nodes = max(3, predicted_total_nodes)                    
                
                dynamic_noise_nodes = min(dynamic_noise_nodes, 30)
                print(f"  [Smart Size] Predicted Total: {pred_total_float:.1f} | Fixed: {num_fixed_nodes} -> Target Free Nodes: {dynamic_noise_nodes}")
            else:
                dynamic_noise_nodes = config.INPUT_NOISE_NODES
                print(f"  [Manual Size] Fallback to configured target Free Nodes: {dynamic_noise_nodes}")
            for run_idx in range(config.LIGANDS_PER_POCKET):
                print(f"\n  --->> Run {run_idx + 1}/{config.LIGANDS_PER_POCKET} Start ...")
                generation_successful = False
                for round_num in range(1, config.MAX_GENERATION_ROUNDS + 1):
                    print(f"\n      --- [Round {round_num}] Debug Trace ---")
                    
                    try:
                        print(f"      [Debug Gen] Running E3NN diffusion (t=T -> t=0) with {dynamic_noise_nodes} free nodes...")
                        
                        pred_tuple, edge_logits, filtered_edge_index = generate_ligand(
                            model=model,
                            diffusion_process=diffusion_generator,
                            pocket_data=processed_pocket,
                            num_noise_nodes=dynamic_noise_nodes,
                            ddim_steps=config.DDIM_STEPS,
                            device=device,
                            mode=config.DEFAULT_GENERATION_MODE,
                            fixed_ligand_data=processed_fixed_ligand,
                            fixed_ligand_edges=fixed_ligand_edges
                        )
                        pred_feat, pred_pos, pred_ref = pred_tuple
                    except Exception as e:
                        print(f"      [Error] Generation crashed: {e}"); traceback.print_exc(); break
                    
                    print("      [Funnel 1] Decoding fragments...")
                    aligned_frags, final_smiles, star_levels, success_mask, is_fixed_mask = decode_and_align_fragments(
                        pred_feat, pred_pos, pred_ref, 
                        anchor_vocab_embeds, anchor_vocab_smiles,         
                        free_vocab_embeds, free_vocab_smiles,             
                        num_fixed_nodes=num_fixed_nodes,
                        verbose=False
                    )
                    frags_raw_snapshot = [Chem.Mol(frag) if frag is not None else None for frag in aligned_frags]
                    
                    is_success = False
                    final_clean_mol = None
                    node_count = len(success_mask)
                    min_dist_val = float('inf')
                    ligand_center_dist = (
                        torch.norm(pred_pos.mean(dim=0)).item()
                        if pred_pos.numel() > 0 else float('inf')
                    )
                    status_msg = ""
                    center_threshold = getattr(config, 'GENERATION_LIGAND_CENTER_MAX_RADIUS', 0.0)
                    
                    if center_threshold > 0 and ligand_center_dist > center_threshold:
                        status_msg = f"Center Fail ({ligand_center_dist:.2f}A > {center_threshold:.2f}A)"
                        print(f"      [Funnel 0] Failed: center distance {ligand_center_dist:.2f}A > {center_threshold:.2f}A.")
                    elif not np.all(success_mask):
                        status_msg = f"Decode Fail ({np.sum(success_mask)}/{node_count})"
                        print(f"      [Funnel 1] Failed: decoded {np.sum(success_mask)}/{node_count} nodes.")
                    else:
                        print(f"      [Funnel 1] Passed: decoded {node_count}/{node_count} nodes.")

                                                        
                        N_protein = processed_pocket.num_nodes - processed_pocket.is_ligand.sum().item()
                        
                        # 1. 计算概率分布
                        pred_edge_probs = torch.softmax(edge_logits, dim=-1)
                        pred_edge_classes = torch.argmax(edge_logits, dim=-1)
                        
                        # 2. 【核心修复】：从 Config 动态读取置信度双重门槛
                        confidence_thresh = getattr(config, 'EDGE_CONFIDENCE_THRESHOLD', 0.20)
                        
                        is_class_0 = (pred_edge_classes == 0)
                        is_confident = (pred_edge_probs[:, 0] > confidence_thresh)
                        
                        llcov_mask = is_class_0 & is_confident
                        
                        global_edge_index_llcov = filtered_edge_index[:, llcov_mask]

                        u, v = global_edge_index_llcov
                        valid_mask = (u >= N_protein) & (v >= N_protein)
                        local_u = u[valid_mask] - N_protein
                        local_v = v[valid_mask] - N_protein
                        predicted_llcov = torch.stack([local_u, local_v], dim=0)

                        if fixed_ligand_edges is not None:
                            fe_idx = fixed_ligand_edges['edge_index']
                            fe_attr = fixed_ligand_edges['edge_attr']
                            is_llcov_fixed = (fe_attr[:, 0] == 1.0)
                            
                            # 💡 [修复核心]：剔除 Dummy 节点的影响，获取原始的口袋边界索引
                            original_n_p = processed_pocket.num_nodes - processed_pocket.is_dummy.sum().item()
                            
                            # 使用 original_n_p 过滤出纯粹的配体边
                            both_lig = (fe_idx[0] >= original_n_p) & (fe_idx[1] >= original_n_p)
                            fixed_llcov_mask = is_llcov_fixed & both_lig
                            
                            # 使用 original_n_p 将全局索引偏移回完美的 0-based 配体内部索引
                            fixed_llcov_edges = fe_idx[:, fixed_llcov_mask] - original_n_p           
                        else:
                            fixed_llcov_edges = torch.empty((2, 0), dtype=torch.long, device=device)

                                                    
                        if num_fixed_nodes > 0 and predicted_llcov.shape[1] > 0:
                            is_both_fixed = (predicted_llcov[0] < num_fixed_nodes) & (predicted_llcov[1] < num_fixed_nodes)
                            pred_keep = ~is_both_fixed
                            predicted_llcov_filtered = predicted_llcov[:, pred_keep]
                        else:
                            predicted_llcov_filtered = predicted_llcov

                        local_edge_index_llcov = torch.cat([fixed_llcov_edges, predicted_llcov_filtered], dim=1)

                        if num_fixed_nodes > 0:
                            required_anchor_indices = set(range(num_fixed_nodes))
                            lcc_nodes = get_anchor_connected_component(
                                node_count, local_edge_index_llcov,
                                anchor_indices=required_anchor_indices
                            )
                            has_all_anchors = len(lcc_nodes) > 0
                        else:
                            lcc_nodes = get_largest_connected_component(node_count, local_edge_index_llcov)
                            has_all_anchors = True                        

                        if num_fixed_nodes > 0:
                            total_free_nodes = node_count - num_fixed_nodes
                            connected_free_nodes = len(lcc_nodes) - num_fixed_nodes
                            
                            lcc_ratio = (connected_free_nodes / total_free_nodes) if total_free_nodes > 0 else 1.0 
                        else:
                            lcc_ratio = len(lcc_nodes) / node_count if node_count > 0 else 0
                        min_lcc_ratio = getattr(config, 'MIN_LCC_RATIO', 0.75)

                        if lcc_ratio < min_lcc_ratio or not has_all_anchors:
                            if not has_all_anchors:
                                status_msg = "Topology Fail (Anchors disconnected)"
                                print("      [Funnel 2] Failed: fixed anchors are disconnected.")
                            else:
                                status_msg = f"Topology Fail (LCC ratio {lcc_ratio:.2f} < {min_lcc_ratio})"
                                print(f"      [Funnel 2] Failed: component too small ({len(lcc_nodes)}/{node_count}).")

                                    
                            u_all, v_all = filtered_edge_index
                            ll_cand_mask = (u_all >= N_protein) & (v_all >= N_protein)
                            total_ll_cands = ll_cand_mask.sum().item()
                            ll_classes = pred_edge_classes[ll_cand_mask]
                            class_counts = torch.bincount(ll_classes, minlength=6).tolist()
                            print(f"        -> LL candidate edges: {total_ll_cands}; class counts: {class_counts}")
                        else:
                            print(
                                f"      [Funnel 2] Passed: component {len(lcc_nodes)}/{node_count}; "
                                f"LLCov {fixed_llcov_edges.shape[1]} fixed + "
                                f"{predicted_llcov_filtered.shape[1]} predicted."
                            )

                            sorted_lcc_nodes = sorted(list(lcc_nodes))
                            lcc_aligned_frags = [aligned_frags[i] for i in sorted_lcc_nodes]
                            lcc_smiles = [final_smiles[i] for i in sorted_lcc_nodes]

                                                
                            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_lcc_nodes)}

                                            
                            u_list = local_edge_index_llcov[0].tolist()
                            v_list = local_edge_index_llcov[1].tolist()
                            new_u, new_v = [], []

                            for ui, vi in zip(u_list, v_list):
                                                    
                                if ui in old_to_new_idx and vi in old_to_new_idx:
                                    new_u.append(old_to_new_idx[ui])
                                    new_v.append(old_to_new_idx[vi])

                            device_tensor = local_edge_index_llcov.device
                            lcc_edge_index = torch.tensor([new_u, new_v], dtype=torch.long, device=device_tensor)

                            remapped_fixed_set = set()
                            for old_idx in range(num_fixed_nodes):
                                if old_idx in old_to_new_idx:
                                    remapped_fixed_set.add(old_to_new_idx[old_idx])

                            if raw_pocket_pdb and os.path.exists(raw_pocket_pdb):
                                print("      [Refine] Executing rigid-body translation to resolve clashes before bonding...")
                                lcc_aligned_frags = refine_fragments_in_pocket(
                                    ligand_fragments=lcc_aligned_frags,
                                    pocket_pdb_path=raw_pocket_pdb,
                                    fixed_indices=list(remapped_fixed_set), # 锁定真实的锚点不让其发生滑移
                                    mode=config.DEFAULT_GENERATION_MODE,
                                    output_complex_pdb_path=None
                                )

                            dist_threshold = getattr(config, 'LLCOV_MAX_BOND_DIST', 4.0)

                            known_bonds = build_known_bonds_for_fixed_pairs(
                                lcc_aligned_frags, lcc_edge_index, remapped_fixed_set
                            )

                            # Keep known fixed-fixed bonds first, then plan new fixed-generated
                            # and generated-generated bonds with the remaining hydrogen budget.
                            sim_success, planned_bonds, sand_debug = simulate_hydrogen_consumption_and_plan_bonds(
                                lcc_aligned_frags, lcc_edge_index, dist_threshold,
                                fixed_node_set=remapped_fixed_set,
                                known_bonds=known_bonds,
                                return_debug=True
                            )

                            if not sim_success:
                                specific_reason = format_sand_table_reason(
                                    sand_debug, lcc_smiles, sorted_lcc_nodes
                                )
                                status_msg = f"Sand Table Fail ({specific_reason})"
                                print(f"      [Funnel 3] Failed: {specific_reason}.")
                            else:
                                num_fixed_generated_bonds = sum(
                                    1 for frag_A, _, frag_B, _ in planned_bonds
                                    if (frag_A in remapped_fixed_set) ^ (frag_B in remapped_fixed_set)
                                )
                                print(
                                    f"      [Funnel 3] Passed: kept {len(known_bonds)} fixed-fixed bonds, "
                                    f"planned {num_fixed_generated_bonds} fixed-generated + "
                                    f"{len(planned_bonds) - num_fixed_generated_bonds} generated-generated bonds; "
                                    f"skipped {sand_debug.get('skipped_fixed_generated_count', 0)} over-budget fixed-generated candidates."
                                )

                                actual_edge_index = bond_blueprints_to_edge_index(
                                    list(known_bonds) + list(planned_bonds),
                                    device_tensor
                                )
                                is_connected, connected_count, connectivity_reason = validate_actual_fragment_connectivity(
                                    len(lcc_aligned_frags), actual_edge_index,
                                    remapped_fixed_set
                                )

                                if not is_connected:
                                    status_msg = f"Final Connectivity Fail ({connectivity_reason})"
                                    print(f"      [Funnel 4] Failed: {connectivity_reason}.")
                                else:
                                    print(f"      [Funnel 4] Connectivity passed: {connected_count}/{len(lcc_aligned_frags)} fragments.")

                                    # Build the final RDKit molecule from the planned bond blueprint.
                                    final_clean_mol = build_molecule_from_blueprint(
                                        lcc_aligned_frags, planned_bonds,
                                        known_bonds=known_bonds
                                    )

                                    if final_clean_mol is None:
                                        status_msg = "Build Fail (RDKit Sanitize Rejected)"
                                        print("      [Funnel 5] Failed: RDKit sanitize rejected assembly.")
                                    else:
                                        print("      [Funnel 5] Passed: RDKit molecule is chemically valid.")
                                        is_success = True
                                        status_msg = "Perfect Assembly!"

                                        pos = final_clean_mol.GetConformer().GetPositions()
                                        min_dist_val = np.min(np.linalg.norm(pos, axis=1))

                    print(f"    [Round {round_num} Summary] Nodes: {node_count} | CenterDist: {ligand_center_dist:.2f} | MinDist: {min_dist_val:.2f} | Status: {status_msg}")
                    if is_success and final_clean_mol:
                        print("      -> [SUCCESS] Criteria Met. Executing File Saving & Refinement...")
                        generation_successful = True

                        current_run_dir = os.path.join(output_dir, f"{pocket_id}_run{run_idx+1}")
                        os.makedirs(current_run_dir, exist_ok=True)
                        

                        raw_save_path = os.path.join(current_run_dir, "1_original_all_fragments.sdf")
                                                                    
                        save_fragments_as_single_sdf_entry(
                            frags_raw_snapshot, raw_save_path, smiles_list=final_smiles
                        )
                        print(f"        -> Saved: {os.path.basename(raw_save_path)}")

                        # ==========================================
                        # [新增代码] 保存平移后的片段，用于前后对比
                        # ==========================================
                        translated_save_path = os.path.join(current_run_dir, "1.5_translated_fragments.sdf")
                        save_fragments_as_single_sdf_entry(
                            lcc_aligned_frags, translated_save_path, smiles_list=lcc_smiles
                        )
                        print(f"        -> Saved: {os.path.basename(translated_save_path)}")
                        # ==========================================


                        pocket_aware_done = False
                        if (
                            raw_pocket_pdb
                            and os.path.exists(raw_pocket_pdb)
                            and getattr(config, 'ENABLE_POCKET_AWARE_POST_FF', True)
                        ):
                            final_clean_mol, pocket_ff_debug = optimize_ligand_with_fixed_pocket(
                                final_clean_mol,
                                raw_pocket_pdb,
                                max_iters=getattr(config, 'POCKET_AWARE_UFF_STEPS', 100)
                            )
                            if pocket_ff_debug.get('reason') == 'ok':
                                pocket_aware_done = True
                                before_dist = pocket_ff_debug.get('min_lp_dist_before')
                                after_dist = pocket_ff_debug.get('min_lp_dist_after')
                                local_atoms = pocket_ff_debug.get('local_protein_atom_count', 0)
                                local_cutoff = pocket_ff_debug.get('local_cutoff', 0.0)
                                if before_dist is not None and after_dist is not None:
                                    print(
                                        f"        -> [Geometry] Pocket-aware {pocket_ff_debug.get('method')} done "
                                        f"(fixed local protein atoms={local_atoms}, cutoff={local_cutoff:.1f}A; "
                                        f"LP min {before_dist:.2f}A -> {after_dist:.2f}A)."
                                    )
                                else:
                                    print(
                                        f"        -> [Geometry] Pocket-aware {pocket_ff_debug.get('method')} done "
                                        f"(fixed local protein atoms={local_atoms}, cutoff={local_cutoff:.1f}A)."
                                    )
                                for stage_info in pocket_ff_debug.get('stages', []):
                                    print(
                                        "        -> [Geometry Stage] "
                                        f"{stage_info.get('stage', '?')}: "
                                        f"ok={stage_info.get('ok', '?')}, "
                                        f"reason={stage_info.get('reason', '?')}, "
                                        f"return_code={stage_info.get('return_code', '?')}"
                                    )
                                    if stage_info.get('atom_summary'):
                                        print(f"        -> [MMFF Atom Summary] {stage_info['atom_summary']}")
                            else:
                                print(f"        [Warn] Pocket-aware geometry skipped: {pocket_ff_debug.get('reason')}.")
                                for stage_info in pocket_ff_debug.get('stages', []):
                                    print(
                                        "        -> [Geometry Stage] "
                                        f"{stage_info.get('stage', '?')}: "
                                        f"ok={stage_info.get('ok', '?')}, "
                                        f"reason={stage_info.get('reason', '?')}, "
                                        f"return_code={stage_info.get('return_code', '?')}"
                                    )
                                    if stage_info.get('atom_summary'):
                                        print(f"        -> [MMFF Atom Summary] {stage_info['atom_summary']}")
                                if pocket_ff_debug.get('traceback'):
                                    print("        [Warn] Pocket-aware traceback:")
                                    print(pocket_ff_debug['traceback'].rstrip())

                        if not pocket_aware_done:
                            try:
                                AllChem.UFFOptimizeMolecule(
                                    final_clean_mol,
                                    maxIters=getattr(config, 'POCKET_AWARE_UFF_STEPS', 100)
                                )
                                AllChem.MMFFOptimizeMolecule(
                                    final_clean_mol,
                                    maxIters=getattr(config, 'POCKET_AWARE_LIGAND_ONLY_MMFF_STEPS', 30)
                                )
                                print("        -> [Geometry] UFF + MMFF94 ligand-only clean-up done.")
                            except Exception as e:
                                print(f"        [Warn] Geometry clean-up failed: {e}")
                        
                        optimized_ligand_path = os.path.join(current_run_dir, "2_mmff94_optimized_ligand.sdf")
                                
                        with Chem.SDWriter(optimized_ligand_path) as w: 
                            w.write(final_clean_mol)
                        print(f"        -> Saved: {os.path.basename(optimized_ligand_path)}")

                        mmff94_complex_path = os.path.join(current_run_dir, "3_mmff94_complex.pdb")
                        if raw_pocket_pdb and os.path.exists(raw_pocket_pdb):
                            combine_ligand_and_protein(
                                optimized_ligand_path, 
                                raw_pocket_pdb, 
                                mmff94_complex_path
                            )
                            print(f"        -> Saved: {os.path.basename(mmff94_complex_path)}")

                        break                   
                            
                    if not is_success:
                        print("      [Reset] Generation failed. Starting completely fresh next round...")
                        
                if not generation_successful:
                    print(f"  [Fail] Run {run_idx+1} max rounds reached.")

        except Exception as e:
            print(f"[Critical Task Error] {pocket_id}: {e}"); traceback.print_exc()

    print("\n" + "="*60 + "\n--- All Tasks Completed ---\n" + "="*60)
