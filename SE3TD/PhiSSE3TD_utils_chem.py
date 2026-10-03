import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import csv

# RDKit
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem
from rdkit import rdBase

# OpenBabel
from openbabel import openbabel
from openbabel import openbabel as ob

# Local Project Imports
import PhiSSE3TD_config as config



# Disable RDKit warnings
rdBase.DisableLog('rdApp.warning')
# Disable OpenBabel warnings
ob_log = openbabel.OBMessageHandler()
ob_log.SetOutputLevel(0)

def neutralize_all_charges(mol):
    """
    Neutralize all formal charges on atoms (e.g., [c-] -> c, [O-] -> O, [NH+] -> N).
    Adjusts explicit hydrogens accordingly to satisfy valence.
    """
    pattern_found = False
    
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge != 0:
            atom.SetFormalCharge(0)
            
            # 动态调整显式氢原子数量以平衡化合价
            num_explicit = atom.GetNumExplicitHs()
            if charge > 0 and num_explicit >= charge:
                # 消除正电荷，通常需要脱去多余的质子(H)
                atom.SetNumExplicitHs(num_explicit - charge)
            elif charge < 0:
                # 消除负电荷，通常需要加上质子(H)来闭合孤对电子
                atom.SetNumExplicitHs(num_explicit + abs(charge))
                
            pattern_found = True
    
    if pattern_found:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mol

def remove_coordinate_hydrogens(mol):
    """
    Remove real H atoms from a fragment while keeping RDKit's heavy-atom
    hydrogen valence bookkeeping. This avoids bad H coordinates from AddHs.
    """
    if mol is None:
        return None

    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        return mol

    try:
        params = Chem.RemoveHsParameters()
        params.removeDefiningBondStereo = True
        params.removeDegreeZero = True
        params.removeHigherDegrees = True
        params.removeHydrides = True
        params.removeNonimplicit = True
        params.removeOnlyHNeighbors = True
        params.removeWithWedgedBond = True
        params.updateExplicitCount = True
        stripped = Chem.RemoveHs(mol, params, sanitize=False)
    except Exception:
        rw_mol = Chem.RWMol(mol)
        h_indices = sorted(
            [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 1],
            reverse=True,
        )
        for h_idx in h_indices:
            rw_mol.RemoveAtom(h_idx)
        stripped = rw_mol.GetMol()

    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        try:
            stripped.UpdatePropertyCache(strict=False)
        except Exception:
            pass

    return stripped


def get_atom_h_budget(atom):
    """Return the current attachable H budget for one heavy atom."""
    try:
        return max(0, int(atom.GetTotalNumHs()))
    except Exception:
        return 0


def rdkit_mol_to_ob_mol(rdkit_mol):
    """Convert an RDKit Mol into an OpenBabel OBMol."""
    if rdkit_mol is None:
        return None
    smiles = Chem.MolToSmiles(rdkit_mol)
    ob_mol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInFormat("smi")
    conv.ReadString(ob_mol, smiles)
    return ob_mol

def ob_mol_to_rdkit_mol(ob_mol):
    """Convert an OpenBabel OBMol into an RDKit Mol while keeping coordinates."""
    if ob_mol is None:
        return None
    conv = openbabel.OBConversion()
    conv.SetOutFormat("sdf")
    sdf_string = conv.WriteString(ob_mol)
    rdkit_mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
    return rdkit_mol

def calculate_reference_coordinate_system(mol_with_conf):
    """
    Build a deterministic 3x3 reference coordinate frame for a fragment.

    The frame is defined from heavy-atom anchor points and is used as the
    node intrinsic orientation target.
    """
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0:
        return None

    conf = mol_with_conf.GetConformer(0)
    num_total_atoms = mol_with_conf.GetNumAtoms()
    heavy_atom_indices = [a.GetIdx() for a in mol_with_conf.GetAtoms() if a.GetAtomicNum() > 1]
    num_heavy_atoms = len(heavy_atom_indices)

    if num_heavy_atoms == 0:
        return None

    all_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_total_atoms)])

    try:
        full_ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(mol_with_conf, breakTies=True))
        canonical_success = True
    except Exception:
        full_ranks = list(range(num_total_atoms))
        canonical_success = False

    heavy_ranks_map = {idx: full_ranks[idx] for idx in heavy_atom_indices}

    if canonical_success:
        idx_a = min(heavy_atom_indices, key=lambda x: heavy_ranks_map[x])
    else:
        idx_a = -1
        max_mass = -1.0
        for idx in heavy_atom_indices:
            atom = mol_with_conf.GetAtomWithIdx(idx)
            current_mass = atom.GetMass()
            if current_mass > max_mass:
                max_mass = current_mass
                idx_a = idx
            elif abs(current_mass - max_mass) < 1e-8 and idx_a != -1 and idx < idx_a:
                idx_a = idx

    coord_a = all_coords[idx_a]

    if num_heavy_atoms == 1:
        coord_b = coord_a + np.array([1.0, 0.0, 0.0])
        coord_c = coord_a + np.array([0.0, 1.0, 0.0])
        return np.array([coord_a, coord_b, coord_c])

    if num_heavy_atoms == 2:
        idx_b = next(idx for idx in heavy_atom_indices if idx != idx_a)
        coord_b = all_coords[idx_b]
        vec_ab = coord_b - coord_a
        vec_ab_len = np.linalg.norm(vec_ab)

        if vec_ab_len < 1e-6:
            coord_b = coord_a + np.array([1.0, 0.0, 0.0])
            coord_c = coord_a + np.array([0.0, 1.0, 0.0])
            return np.array([coord_a, coord_b, coord_c])

        min_axis = np.argmin(np.abs(vec_ab))
        perturb = np.array([0.0, 0.0, 0.0])
        perturb[min_axis] = 1.0
        ortho_vec = np.cross(vec_ab, np.cross(vec_ab, perturb))
        ortho_norm = np.linalg.norm(ortho_vec)
        if ortho_norm > 1e-6:
            ortho_vec = ortho_vec / ortho_norm
        else:
            ortho_vec = np.array([1.0, 0.0, 0.0]) if min_axis != 0 else np.array([0.0, 1.0, 0.0])

        coord_c = coord_a + ortho_vec
        return np.array([coord_a, coord_b, coord_c])

    max_dist_sq = -1.0
    idx_b = -1
    for idx in heavy_atom_indices:
        if idx == idx_a:
            continue
        dist_sq = np.sum((all_coords[idx] - coord_a) ** 2)
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            idx_b = idx
        elif abs(dist_sq - max_dist_sq) < 1e-8 and heavy_ranks_map[idx] > heavy_ranks_map[idx_b]:
            idx_b = idx

    coord_b = all_coords[idx_b]
    vec_ab = coord_b - coord_a
    vec_ab_len_sq = np.sum(vec_ab ** 2)
    max_perp_dist_sq = -1.0
    idx_c = -1

    if vec_ab_len_sq < 1e-8:
        for idx in heavy_atom_indices:
            if idx != idx_a:
                idx_c = idx
                break
    else:
        for idx in heavy_atom_indices:
            if idx in [idx_a, idx_b]:
                continue

            vec_ap = all_coords[idx] - coord_a
            projection_len = np.dot(vec_ap, vec_ab) / vec_ab_len_sq
            closest_point = coord_a + projection_len * vec_ab
            perp_dist_sq = np.sum((all_coords[idx] - closest_point) ** 2)

            if perp_dist_sq > max_perp_dist_sq:
                max_perp_dist_sq = perp_dist_sq
                idx_c = idx
            elif abs(perp_dist_sq - max_perp_dist_sq) < 1e-8 and idx_c != -1:
                if heavy_ranks_map[idx] > heavy_ranks_map[idx_c]:
                    idx_c = idx

    if idx_c == -1 or max_perp_dist_sq < 1e-4:
        min_axis = np.argmin(np.abs(vec_ab))
        perturb = np.array([0.0, 0.0, 0.0])
        perturb[min_axis] = 1.0
        ortho_vec = np.cross(vec_ab, np.cross(vec_ab, perturb))
        ortho_norm = np.linalg.norm(ortho_vec)
        if ortho_norm > 1e-6:
            ortho_vec = ortho_vec / ortho_norm
        else:
            ortho_vec = np.array([1.0, 0.0, 0.0]) if min_axis != 0 else np.array([0.0, 1.0, 0.0])
        coord_c = coord_a + ortho_vec
    else:
        coord_c = all_coords[idx_c]

    return np.array([coord_a, coord_b, coord_c])


def get_largest_connected_component(num_nodes, edge_index_llcov):
    """Return the node index set of the largest connected component."""
    if num_nodes == 0:
        return set()
    if num_nodes == 1:
        return {0}

    adj_list = {i: set() for i in range(num_nodes)}
    if hasattr(edge_index_llcov, 'cpu'):
        edges = edge_index_llcov.cpu().numpy()
    else:
        edges = edge_index_llcov

    for i in range(edges.shape[1]):
        u, v = int(edges[0, i]), int(edges[1, i])
        adj_list[u].add(v)
        adj_list[v].add(u)

    visited_global = set()
    max_cc = set()
    for i in range(num_nodes):
        if i in visited_global:
            continue

        queue = [i]
        visited_local = {i}
        while queue:
            curr = queue.pop(0)
            for neighbor in adj_list[curr]:
                if neighbor not in visited_local:
                    visited_local.add(neighbor)
                    queue.append(neighbor)

        visited_global.update(visited_local)
        if len(visited_local) > len(max_cc):
            max_cc = visited_local

    return max_cc


def get_anchor_connected_component(num_nodes, edge_index_llcov, anchor_indices):
    """Return the connected component that contains all anchor nodes."""
    if num_nodes == 0:
        return set()
    if not anchor_indices:
        return get_largest_connected_component(num_nodes, edge_index_llcov)

    adj_list = {i: set() for i in range(num_nodes)}
    if hasattr(edge_index_llcov, 'cpu'):
        edges = edge_index_llcov.cpu().numpy()
    else:
        edges = edge_index_llcov

    for i in range(edges.shape[1]):
        u, v = int(edges[0, i]), int(edges[1, i])
        adj_list[u].add(v)
        adj_list[v].add(u)

    visited_global = set()
    all_components = []
    for i in range(num_nodes):
        if i in visited_global:
            continue

        queue = [i]
        visited_local = {i}
        while queue:
            curr = queue.pop(0)
            for neighbor in adj_list[curr]:
                if neighbor not in visited_local:
                    visited_local.add(neighbor)
                    queue.append(neighbor)

        visited_global.update(visited_local)
        all_components.append(visited_local)

    for cc in all_components:
        if anchor_indices.issubset(cc):
            return cc

    return set()


def decode_and_align_fragments(pred_feat, pred_pos, pred_ref_coords, 
                               anchor_vocab_embeds, anchor_vocab_smiles,
                               free_vocab_embeds, free_vocab_smiles,
                               num_fixed_nodes=0, top_k=config.DECODING_TOP_K,
                               verbose=True):
    """
    Decode fragment identities and align each fragment to the predicted
    center and reference frame.
    """
    aligned_fragments = []
    final_smiles_list = []
    star_levels = []
    success_mask = []

    pred_embeds = pred_feat[:, :config.EMBEDDING_DIM_IN]
    pred_pos_np = pred_pos.cpu().numpy()
    pred_ref_coords_np = pred_ref_coords.cpu().numpy()
    num_nodes = pred_feat.shape[0]
    
    is_fixed_mask = np.zeros(num_nodes, dtype=bool)
    is_fixed_mask[:num_fixed_nodes] = True

    top_k_idx_anchors = None
    if num_fixed_nodes > 0:
        dist_anchors = torch.cdist(pred_embeds[:num_fixed_nodes], anchor_vocab_embeds, p=2.0)
        
                                        
        _, top_k_idx_anchors = torch.topk(dist_anchors, k=top_k, dim=1, largest=False)
        top_k_idx_anchors = top_k_idx_anchors.cpu().numpy()

                                                
    num_free = num_nodes - num_fixed_nodes
    top_k_idx_free = None
    if num_free > 0:
        dist_free = torch.cdist(pred_embeds[num_fixed_nodes:], free_vocab_embeds, p=2.0)
        
                                        
        _, top_k_idx_free = torch.topk(dist_free, k=top_k, dim=1, largest=False)
        top_k_idx_free = top_k_idx_free.cpu().numpy()


    for i in range(num_nodes):
        is_successful = False
        current_pred_pos = pred_pos_np[i]
        current_pred_ref = pred_ref_coords_np[i]

                    
        if i < num_fixed_nodes:
            candidates = top_k_idx_anchors[i][:1]
            current_vocab_smiles = anchor_vocab_smiles
        else:
            candidates = top_k_idx_free[i - num_fixed_nodes]
            current_vocab_smiles = free_vocab_smiles

        debug_trace = []

        for rank, smiles_idx in enumerate(candidates):
            original_smiles = current_vocab_smiles[smiles_idx]
            
            if original_smiles == "<UNK>":
                debug_trace.append(f"[Rank {rank}] <UNK> -> Skipped.")
                continue

            try:
                cleaned_smiles = original_smiles.strip().replace('*', '')
                if not cleaned_smiles:
                    debug_trace.append(f"[Rank {rank}] {original_smiles} -> Failed: Empty after cleaning.")
                    continue

                mol = Chem.MolFromSmiles(cleaned_smiles)
                if mol is None:
                    debug_trace.append(f"[Rank {rank}] {cleaned_smiles} -> Failed: MolFromSmiles rejected (Valence/Syntax).")
                    continue
                
                if getattr(config, 'REMOVE_UNSAFE_EXPLICIT_H', True):
                    mol = remove_coordinate_hydrogens(mol)
            
                mol = neutralize_all_charges(mol)
                
                if getattr(config, 'REMOVE_UNSAFE_EXPLICIT_H', True):
                    mol = remove_coordinate_hydrogens(mol)

                num_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                if num_heavy == 0:
                    debug_trace.append(f"[Rank {rank}] {cleaned_smiles} -> Failed: no heavy atoms after H cleanup.")
                    continue

                if num_heavy <= 1:
                    level = original_smiles.count('*')
                    from rdkit.Geometry import Point3D
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    pt = Point3D(float(current_pred_pos[0]), float(current_pred_pos[1]), float(current_pred_pos[2]))
                    
                    # Place every atom at the predicted fragment center.
                    for j in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(j, pt)
                    mol.AddConformer(conf, assignId=True)

                    aligned_fragments.append(mol)
                    final_smiles_list.append(cleaned_smiles)
                    star_levels.append(level)
                    is_successful = True
                    # Emit detailed debug only when a fallback rank was needed.
                    if verbose and rank > 0:
                        for trace in debug_trace:
                            print(f"          -> {trace}")
                    elif verbose and rank == 0:
                        print(f"      [Debug Node {i}]  Succeeded at Top-1 ({cleaned_smiles}) (Single Atom)!")
                        
                    break                       


                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                embed_res = AllChem.EmbedMolecule(mol, params)
                
                if embed_res == -1:
                    embed_res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
                    if embed_res == -1:
                        debug_trace.append(f"[Rank {rank}] {cleaned_smiles} -> Failed: EmbedMolecule (-1) ETKDG & Random failed.")
                        continue

                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass

                if mol.GetNumConformers() == 0:
                    debug_trace.append(f"[Rank {rank}] {cleaned_smiles} -> Failed: 0 conformers generated.")
                    continue

                level = original_smiles.count('*')
                conf = mol.GetConformer()
                atom_pos = conf.GetPositions()

                heavy_mask = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
                mol_center = atom_pos[heavy_mask].mean(axis=0) if np.any(heavy_mask) else atom_pos.mean(axis=0)

                ref_coords_std = calculate_reference_coordinate_system(mol)
                if ref_coords_std is None:
                    debug_trace.append(f"[Rank {rank}] {cleaned_smiles} -> Failed: calculate_reference_coordinate_system returned None.")
                    continue

                                
                P = ref_coords_std - mol_center
                Q = current_pred_ref

                H = P.T @ Q
                U, S, Vt = np.linalg.svd(H)
                V = Vt.T
                R = V @ U.T
                if np.linalg.det(R) < 0:
                    V[:, -1] *= -1
                    R = V @ U.T

                pos_centered = atom_pos - mol_center
                aligned_coords = pos_centered @ R.T + current_pred_pos
                for j in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(j, tuple(aligned_coords[j]))

                aligned_fragments.append(mol)
                final_smiles_list.append(cleaned_smiles)
                star_levels.append(level)
                is_successful = True
                if verbose and rank > 0:
                    print(f"      [Debug Node {i}]  Succeeded at Rank {rank} ({cleaned_smiles}). Previous failures:")
                    for trace in debug_trace:
                        print(f"          -> {trace}")
                elif verbose and rank == 0:
                    print(f"      [Debug Node {i}] Succeeded at Top-1 ({cleaned_smiles})!")
                    
                break

            except Exception as e:
                error_msg = str(e).split("\n")[0]
                debug_trace.append(f"[Rank {rank}] {original_smiles} -> Exception: {type(e).__name__} ({error_msg})")
                continue

        success_mask.append(is_successful)
        if not is_successful:
            if not verbose:
                aligned_fragments.append(None)
                final_smiles_list.append(None)
                star_levels.append(-1)
                continue
            print(f"      [Debug Decode] 鉂?Node {i} COMPLETELY FAILED. Tested {len(candidates)} candidates.")
            for trace in debug_trace:
                print(f"          {trace}")

            aligned_fragments.append(None)
            final_smiles_list.append(None)
            star_levels.append(-1)

    return aligned_fragments, final_smiles_list, np.array(star_levels), np.array(success_mask, dtype=bool), is_fixed_mask


def save_fragments_as_single_sdf_entry(fragments, file_path, smiles_list=None, identities=None):
    """Save multiple fragments as one combined SDF entry."""
    if not fragments: return
    combined_mol = Chem.Mol()
    valid_fragments = [f for f in fragments if f is not None]
    if not valid_fragments: return 
    
    for frag in valid_fragments:
        combined_mol = Chem.CombineMols(combined_mol, frag)
        
    if combined_mol.GetNumAtoms() == 0: return

    combined_conformer = rdchem.Conformer(combined_mol.GetNumAtoms())
    atom_offset = 0
    for frag in valid_fragments:
        frag_conf = frag.GetConformer(0)
        for atom in frag.GetAtoms():
            atom_idx = atom.GetIdx()
            pos = frag_conf.GetAtomPosition(atom_idx)
            combined_conformer.SetAtomPosition(atom_offset + atom_idx, pos)
        atom_offset += frag.GetNumAtoms()
        
    combined_mol.AddConformer(combined_conformer, assignId=True)
    combined_mol.SetProp("_Name", os.path.basename(file_path).replace('.sdf', ''))
    if smiles_list:
        combined_mol.SetProp("SMILES", ".".join(filter(None, smiles_list)))
    
    if identities is not None and len(identities) == len(valid_fragments):
        atom_offset = 0
        for i, frag in enumerate(valid_fragments):
            identity_val = float(identities[i])
            for atom in frag.GetAtoms():
                global_idx = atom_offset + atom.GetIdx()
                combined_mol.GetAtomWithIdx(global_idx).SetDoubleProp("TempFactor", identity_val)
            atom_offset += frag.GetNumAtoms()

    try:
        with Chem.SDWriter(file_path) as writer:
            writer.write(combined_mol)
    except Exception as e:
        print(f"[閿欒] 鍐欏叆SDF澶辫触 '{file_path}': {e}")

def combine_ligand_and_protein(ligand_sdf_path, protein_pdb_path, output_complex_pdb_path):
    """Combine a ligand SDF and a protein PDB into one complex PDB."""
    try:
        protein_mol = Chem.MolFromPDBFile(protein_pdb_path, removeHs=False, sanitize=True)
        if protein_mol is None: raise IOError("Failed to read protein PDB file.")

        suppl = Chem.SDMolSupplier(ligand_sdf_path)
        if len(suppl) == 0: raise IOError("Ligand SDF contains no molecules.")
        ligand_mol = suppl[0]
        if ligand_mol is None: raise IOError("Failed to read ligand molecule from SDF.")
            
        complex_mol = Chem.CombineMols(protein_mol, ligand_mol)
        writer = Chem.PDBWriter(output_complex_pdb_path)
        writer.write(complex_mol)
        writer.close()
        print(f"-> [Complex Saved]: {os.path.basename(output_complex_pdb_path)}")
    except Exception as e:
        print(f"-> [Complex Error]: {e}")


def refine_fragments_in_pocket(ligand_fragments, pocket_pdb_path, fixed_indices, mode, output_complex_pdb_path, log_save_path=None):
    """
    Refine fragment placement inside the pocket with OpenBabel force-field
    relaxation while tracking rigid-body movement diagnostics.
    """

                        
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("pdb")
    complex_ob = ob.OBMol()
    
    if not os.path.exists(pocket_pdb_path):
        print(f"[Refine] Error: PDB not found: {pocket_pdb_path}")
        return ligand_fragments
        
            
    is_loaded = obConversion.ReadFile(complex_ob, pocket_pdb_path)
    if not is_loaded:
        print("[Refine] OpenBabel failed to load protein PDB.")
        return ligand_fragments

    ligand_ranges = []
    sdf_conversion = ob.OBConversion()
    sdf_conversion.SetInFormat("sdf")
    fixed_indices_set = set(fixed_indices) if fixed_indices else set()

    for i, frag in enumerate(ligand_fragments):
        if frag is None: continue
        
        # RDKit -> SDF Block -> OpenBabel
        sdf_block = Chem.MolToMolBlock(frag)
        frag_ob = ob.OBMol()
        sdf_conversion.ReadString(frag_ob, sdf_block)
        
        start_idx = complex_ob.NumAtoms() + 1
        complex_ob += frag_ob         
        end_idx = complex_ob.NumAtoms()
        
        ligand_ranges.append({
            'frag_idx': i, 
            'start': start_idx, 
            'end': end_idx,
            'is_anchor': (i in fixed_indices_set)
        })

                    
    ff = ob.OBForceField.FindForceField("UFF")
    if ff is None:
        print("[Refine] Could not find UFF force field.")
        return ligand_fragments

    constraints = ob.OBFFConstraints()
    constraints.AddIgnore(0)        
    max_steps = getattr(config, 'TGD_MAX_STEPS', 100)
    learning_rate = getattr(config, 'TGD_LEARNING_RATE', 0.05)
    max_step_dist = getattr(config, 'TGD_MAX_STEP_DIST', 0.2)
    force_tol = getattr(config, 'TGD_FORCE_TOL', 1.0)
    
    fragment_atoms_map = []
    for info in ligand_ranges:
        atoms = []
        for idx in range(info['start'], info['end'] + 1):
            atoms.append(complex_ob.GetAtom(idx))
        fragment_atoms_map.append({
            'atoms': atoms,
            'is_anchor': info['is_anchor'],
            'frag_idx': info['frag_idx']                
        })

    tgd_logs = []
    tgd_logs.append(["Step", "Frag_Idx", "Is_Anchor", "Net_Force_Mag", "Displacement_Mag", "Reversed"])

    # ==========================================
    # [新增] 快速计算片段质心的辅助函数
    # ==========================================
    def get_centroid(atoms):
        if not atoms: 
            return np.array([0.0, 0.0, 0.0])
        cx = sum(a.GetX() for a in atoms)
        cy = sum(a.GetY() for a in atoms)
        cz = sum(a.GetZ() for a in atoms)
        n = len(atoms)
        return np.array([cx/n, cy/n, cz/n])

    try:
        with tqdm(total=max_steps, desc="  [Refine] Rigid Translation", leave=False) as pbar:
            for step in range(max_steps):
                ff.Setup(complex_ob, constraints)
                ff.Energy() # 必须唤醒力场
                
                max_net_force_mag = 0.0
                has_movement = False
                
                # --- 1. 在此步开始前，记录所有片段的当前质心 ---
                current_centroids = {
                    fd['frag_idx']: get_centroid(fd['atoms'])
                    for fd in fragment_atoms_map
                }
                
                for frag_data in fragment_atoms_map:
                    frag_idx = frag_data['frag_idx']
                    is_anchor = frag_data['is_anchor']
                    
                    net_grad_x, net_grad_y, net_grad_z = 0.0, 0.0, 0.0
                    atoms = frag_data['atoms']
                    
                    for atom in atoms:
                        grad = ff.GetGradient(atom)
                        net_grad_x += grad.GetX()
                        net_grad_y += grad.GetY()
                        net_grad_z += grad.GetZ()
                    
                    grad_mag = np.sqrt(net_grad_x**2 + net_grad_y**2 + net_grad_z**2)
                    if grad_mag > max_net_force_mag:
                        max_net_force_mag = grad_mag

                    move_dist_record = 0.0
                    is_reversed = False # 用于记录是否触发了反转踢出

                    if grad_mag >= force_tol: 
                        if is_anchor:
                            move_dist_record = 0.0
                        else:
                            # 默认正向受力方向 (OpenBabel 提供的推开方向)
                            move_x = net_grad_x * learning_rate
                            move_y = net_grad_y * learning_rate
                            move_z = net_grad_z * learning_rate
                                                    
                            # 限制单步最大位移 (防止飞出屏幕)
                            move_dist = np.sqrt(move_x**2 + move_y**2 + move_z**2)
                            if move_dist > max_step_dist:
                                scale = max_step_dist / move_dist
                                move_x *= scale
                                move_y *= scale
                                move_z *= scale
                                move_dist = max_step_dist                                        
                            
                            # ==============================================================
                            # [核心新增] 质心反转仲裁逻辑
                            # ==============================================================
                            my_centroid = current_centroids[frag_idx]
                            
                            # 计算当前片段与其他所有片段质心的距离之和 (老距离)
                            old_dist_sum = sum(
                                np.linalg.norm(my_centroid - other_c)
                                for other_idx, other_c in current_centroids.items()
                                if other_idx != frag_idx
                            )
                            
                            if old_dist_sum > 0: # 只要系统里有超过1个片段就进行判定
                                # 预演质心位置
                                proposed_centroid = my_centroid + np.array([move_x, move_y, move_z])
                                
                                # 计算预演后与其他片段质心的距离之和 (新距离)
                                new_dist_sum = sum(
                                    np.linalg.norm(proposed_centroid - other_c)
                                    for other_idx, other_c in current_centroids.items()
                                    if other_idx != frag_idx
                                )
                                
                                # 💡 终极仲裁：如果合力导致系统收缩（越靠越近）
                                if new_dist_sum < old_dist_sum:
                                    # 判定为拓扑死锁陷入吸引，强制反转向量，暴力踢开！
                                    move_x *= -1.0
                                    move_y *= -1.0
                                    move_z *= -1.0
                                    is_reversed = True
                            # ==============================================================

                            move_dist_record = move_dist
                            
                            # 执行最终确定的位移
                            for atom in atoms:
                                atom.SetVector(atom.GetX() + move_x, 
                                               atom.GetY() + move_y, 
                                               atom.GetZ() + move_z)
                            
                            has_movement = True
                    
                    # 日志中增加一列，记录这一步是否触发了反转
                    tgd_logs.append([step, frag_idx, is_anchor, f"{grad_mag:.4f}", f"{move_dist_record:.4f}", is_reversed])
                
                pbar.set_postfix({"MaxForce": f"{max_net_force_mag:.1f}"})
                pbar.update(1)

                if not has_movement or max_net_force_mag < force_tol:
                    break
                    
    except Exception as e:
        print(f"  [Refine] Error during translation loop: {e}")

                    
    if log_save_path:
        try:
            with open(log_save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(tgd_logs)
        except Exception as e:
            print(f"  [Refine] Failed to write log: {e}")


    for i, info in enumerate(ligand_ranges):
        target_frag = ligand_fragments[info['frag_idx']]
        if target_frag is None: continue
        
        conf = target_frag.GetConformer()
        atom_count = 0
        
        for global_ob_idx in range(info['start'], info['end'] + 1):
            if atom_count >= target_frag.GetNumAtoms(): break
            ob_atom = complex_ob.GetAtom(global_ob_idx)
            conf.SetAtomPosition(atom_count, (ob_atom.GetX(), ob_atom.GetY(), ob_atom.GetZ()))
            atom_count += 1


    if output_complex_pdb_path:
        try:
            prot_mol_for_save = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False, sanitize=False)
            if prot_mol_for_save:
                combined_debug = prot_mol_for_save
                for frag in ligand_fragments:
                    if frag: combined_debug = Chem.CombineMols(combined_debug, frag)
                Chem.MolToPDBFile(combined_debug, output_complex_pdb_path)
        except: pass

    return ligand_fragments

def build_known_bonds_for_fixed_pairs(fragments, edge_index, fixed_node_set):
    """Build atom-level bond blueprints for fixed-fixed fragment edges."""
    if not fixed_node_set:
        return []

    frag_coords = {}
    frag_heavy_atoms = {}
    for f_idx, frag in enumerate(fragments):
        if frag is None:
            continue

        conf = frag.GetConformer()
        heavy_indices = []
        heavy_positions = []
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue

            atom_idx = atom.GetIdx()
            heavy_indices.append(atom_idx)
            heavy_positions.append(np.array(conf.GetAtomPosition(atom_idx)))

        if heavy_positions:
            frag_coords[f_idx] = np.array(heavy_positions)
            frag_heavy_atoms[f_idx] = heavy_indices

    if hasattr(edge_index, 'cpu'):
        edges = edge_index.t().cpu().numpy()
    else:
        edges = edge_index.T

    fixed_pairs = {
        tuple(sorted((int(u), int(v))))
        for u, v in edges
        if int(u) != int(v) and int(u) in fixed_node_set and int(v) in fixed_node_set
    }

    known_bonds = []
    for frag_A, frag_B in sorted(fixed_pairs):
        if frag_A not in frag_coords or frag_B not in frag_coords:
            continue

        dists = cdist(frag_coords[frag_A], frag_coords[frag_B])
        min_idx_flat = np.argmin(dists)
        idx_A, idx_B = np.unravel_index(min_idx_flat, dists.shape)
        known_bonds.append((
            frag_A,
            frag_heavy_atoms[frag_A][idx_A],
            frag_B,
            frag_heavy_atoms[frag_B][idx_B]
        ))

    return known_bonds


def simulate_hydrogen_consumption_and_plan_bonds(fragments, edge_index_llcov, dist_threshold,
                                                 fixed_node_set=None, known_bonds=None,
                                                 return_debug=False):
    """
    Plan inter-fragment bonds under a heavy-atom hydrogen budget with Strict Topo-Chemical Rules.
    """
    if fixed_node_set is None:
        fixed_node_set = set()
    if known_bonds is None:
        known_bonds = []

    def finish(success, planned_bonds, debug_info=None):
        if return_debug:
            return success, planned_bonds, debug_info or {}
        return success, planned_bonds

    from rdkit import Chem
    ledger = {}
    coords = {}
    is_frame = {}
    offsets = {}
    current_offset = 0
    combined_mol = Chem.Mol()

    # 1. 初始化片段信息与 RWMol 骨架
    for f_idx, frag in enumerate(fragments):
        offsets[f_idx] = current_offset
        if frag is None:
            is_frame[f_idx] = False
            continue

        is_frame[f_idx] = frag.GetRingInfo().NumRings() > 0
        combined_mol = Chem.CombineMols(combined_mol, frag)
        current_offset += frag.GetNumAtoms()

        ledger[f_idx] = {}
        coords[f_idx] = {}
        try:
            frag.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        conf = frag.GetConformer()

        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            atom_idx = atom.GetIdx()
            # 优先使用隐式氢账本
            try:
                ledger[f_idx][atom_idx] = max(0, int(atom.GetTotalNumHs()))
            except:
                ledger[f_idx][atom_idx] = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
            coords[f_idx][atom_idx] = np.array(conf.GetAtomPosition(atom_idx))

    initial_ledger = {f: dict(h) for f, h in ledger.items()}
    rw_mol = Chem.RWMol(combined_mol)
    frag_adj = {i: set() for i in range(len(fragments))}
    frame_atom_used = set() # Rule 7 账本：记录已经参与过片段间连接的 Frame 原子 (frag_idx, atom_idx)

    # 2. 注入 Known Bonds 到拓扑记录器中
    for frag_A, atom_A, frag_B, atom_B in known_bonds:
        if frag_A in ledger and atom_A in ledger[frag_A]:
            ledger[frag_A][atom_A] = max(0, ledger[frag_A][atom_A] - 1)
        if frag_B in ledger and atom_B in ledger[frag_B]:
            ledger[frag_B][atom_B] = max(0, ledger[frag_B][atom_B] - 1)
        
        frag_adj[frag_A].add(frag_B)
        frag_adj[frag_B].add(frag_A)
        
        # 将使用的 Frame 原子记入账本
        if is_frame[frag_A]: frame_atom_used.add((frag_A, atom_A))
        if is_frame[frag_B]: frame_atom_used.add((frag_B, atom_B))

        gA = offsets[frag_A] + atom_A
        gB = offsets[frag_B] + atom_B
        if rw_mol.GetBondBetweenAtoms(gA, gB) is None:
            rw_mol.AddBond(gA, gB, Chem.BondType.SINGLE)

    if hasattr(edge_index_llcov, 'cpu'):
        edges = edge_index_llcov.t().cpu().numpy()
    else:
        edges = edge_index_llcov.T

    unique_edges = {tuple(sorted((int(u), int(v)))) for u, v in edges if int(u) != int(v)}

    fixed_generated_edges, generated_generated_edges, fixed_fixed_edges = [], [], []
    for frag_A, frag_B in sorted(unique_edges):
        if frag_A in fixed_node_set and frag_B in fixed_node_set:
            fixed_fixed_edges.append((frag_A, frag_B))
        elif frag_A in fixed_node_set or frag_B in fixed_node_set:
            fixed_generated_edges.append((frag_A, frag_B))
        else:
            generated_generated_edges.append((frag_A, frag_B))

    degree_total = {idx: 0 for idx in ledger}
    degree_fixed_fixed = {idx: 0 for idx in ledger}
    degree_fixed_generated = {idx: 0 for idx in ledger}
    degree_generated_generated = {idx: 0 for idx in ledger}
    for fA, fB in unique_edges:
        degree_total[fA] = degree_total.get(fA, 0) + 1
        degree_total[fB] = degree_total.get(fB, 0) + 1
        if fA in fixed_node_set and fB in fixed_node_set:
            degree_fixed_fixed[fA] = degree_fixed_fixed.get(fA, 0) + 1
            degree_fixed_fixed[fB] = degree_fixed_fixed.get(fB, 0) + 1
        elif fA in fixed_node_set or fB in fixed_node_set:
            degree_fixed_generated[fA] = degree_fixed_generated.get(fA, 0) + 1
            degree_fixed_generated[fB] = degree_fixed_generated.get(fB, 0) + 1
        else:
            degree_generated_generated[fA] = degree_generated_generated.get(fA, 0) + 1
            degree_generated_generated[fB] = degree_generated_generated.get(fB, 0) + 1

    # 3. 核心通行证审批引擎 (7大规则)
    def is_bond_approved(f_A, a_A, f_B, a_B):
        g_A = offsets[f_A] + a_A
        g_B = offsets[f_B] + a_B
        atom_A = rw_mol.GetAtomWithIdx(g_A)
        atom_B = rw_mol.GetAtomWithIdx(g_B)

        sym_A, sym_B = atom_A.GetSymbol(), atom_B.GetSymbol()
        arom_A, arom_B = atom_A.GetIsAromatic(), atom_B.GetIsAromatic()
        hyb_A, hyb_B = atom_A.GetHybridization(), atom_B.GetHybridization()

        # Rule 5: 截断 <=4 元的新环形成
        path = Chem.rdmolops.GetShortestPath(rw_mol, g_A, g_B)
        if 0 < len(path) <= 4:
            return False, f"Rule 5 (<=4 membered global ring detected, path len {len(path)})"

        # Rule 3: 杂原子直接相连拦截
        if sym_A in ['N', 'O', 'S'] and sym_B in ['N', 'O', 'S']:
            return False, f"Rule 3 (Hetero-Hetero bond {sym_A}-{sym_B})"

        # Rule 6: 拦截缩醛/缩胺醛
        def check_acetal(target_atom, target_sym, target_hyb, other_sym):
            if other_sym in ['O', 'N'] and target_sym == 'C' and target_hyb == Chem.HybridizationType.SP3:
                for nbr in target_atom.GetNeighbors():
                    if nbr.GetSymbol() in ['O', 'N']:
                        return True
            return False
        if check_acetal(atom_A, sym_A, hyb_A, sym_B) or check_acetal(atom_B, sym_B, hyb_B, sym_A):
            return False, "Rule 6 (Unstable Acetal/Aminal formation)"

        # Rule 7: Frame 原子单次连接限制
        if is_frame[f_A] and (f_A, a_A) in frame_atom_used:
            return False, "Rule 7 (Frame atom A already used for a shorter bond)"
        if is_frame[f_B] and (f_B, a_B) in frame_atom_used:
            return False, "Rule 7 (Frame atom B already used for a shorter bond)"

        # 检查是否均为 Frame
        if is_frame[f_A] and is_frame[f_B]:
            # Rule 2: 共同 Linker 检查与多环限制
            shared = frag_adj[f_A].intersection(frag_adj[f_B])
            shared_linkers = [n for n in shared if not is_frame[n]]
            
            if len(shared_linkers) > 0:
                if len(shared_linkers) > 1:
                    return False, "Rule 2 (Denied: Multiple shared linkers would form complex polycyclic system)"
                return True, "Approved by Rule 2 (Shared Linker Frame-Frame)"
            
            # Rule 1: 无共同 Linker 时的直连审查
            is_aryl_aryl = (sym_A == 'C' and arom_A and sym_B == 'C' and arom_B)
            
            # 识别杂原子与碳的组合，严格限制杂原子必须是脂肪族（非芳香）
            is_valid_hetero_A = (sym_A in ['N', 'O', 'S'] and not arom_A) and (sym_B == 'C')
            is_valid_hetero_B = (sym_B in ['N', 'O', 'S'] and not arom_B) and (sym_A == 'C')
            has_valid_heteroatom = is_valid_hetero_A or is_valid_hetero_B

            # 拦截纯碳的脂环-脂环 / 芳香-脂肪连接
            if sym_A == 'C' and sym_B == 'C' and not is_aryl_aryl:
                return False, "Rule 1 (Denied Frame-Frame Aliphatic C-C, falling back to Heteroatom search)"
            
            if is_aryl_aryl or has_valid_heteroatom:
                return True, "Approved by Rule 1 (Aryl-Aryl or AliphaticHetero-C Frame-Frame)"
                
            return False, "Rule 1 (Denied direct Frame-Frame bond or Invalid Aromatic Heteroatom)"

        # Rule 4: Linker/Frame-Linker 仲碳/叔碳位阻检查 (限 Sp3 C)
        if sym_A == 'C' and hyb_A == Chem.HybridizationType.SP3 and \
           sym_B == 'C' and hyb_B == Chem.HybridizationType.SP3:
            deg_A = sum(1 for n in atom_A.GetNeighbors() if n.GetAtomicNum() > 1)
            deg_B = sum(1 for n in atom_B.GetNeighbors() if n.GetAtomicNum() > 1)
            if deg_A >= 2 and deg_B >= 2:
                return False, f"Rule 4 (SP3 Steric Clash deg_{deg_A} - deg_{deg_B})"

        return True, "Approved (General)"

    # 4. 融合审批的几何搜寻与杂原子优先机制 (Deferred Acceptance)
    def get_current_best_pair(frag_A, frag_B):
        avail_A = [a for a, h in ledger[frag_A].items() if h > 0]
        avail_B = [b for b, h in ledger[frag_B].items() if h > 0]

        if not avail_A or not avail_B:
            return {'ok': False, 'reason': 'no_available_h', 'avail_A': avail_A, 'avail_B': avail_B}

        pos_A = np.array([coords[frag_A][a] for a in avail_A])
        pos_B = np.array([coords[frag_B][b] for b in avail_B])
        dists = cdist(pos_A, pos_B)
        
        flat_indices = np.argsort(dists, axis=None)
        last_reason = 'distance_exceeded'
        min_dist_record = None
        
        # 备胎记录器：专门用于暂存脂肪碳-脂肪碳连接
        fallback_cc_pair = None 
        # 判断是否属于需要应用优先级的组合 (Frame-Linker 或 Linker-Linker)
        is_FL_or_LL = not (is_frame[frag_A] and is_frame[frag_B])

        for idx_flat in flat_indices:
            idx_A, idx_B = np.unravel_index(idx_flat, dists.shape)
            dist = dists[idx_A, idx_B]
            
            if min_dist_record is None:
                min_dist_record = dist
            
            if dist > dist_threshold:
                break
                
            b_atom_A, b_atom_B = int(avail_A[idx_A]), int(avail_B[idx_B])
            
            # 执行通行证审批 (若被拦截，不会退出循环，顺延查找下一个距离更远的原子对)
            is_appr, reason = is_bond_approved(frag_A, b_atom_A, frag_B, b_atom_B)
            
            if is_appr:
                if is_FL_or_LL:
                    g_A_temp = offsets[frag_A] + b_atom_A
                    g_B_temp = offsets[frag_B] + b_atom_B
                    a_A_obj = rw_mol.GetAtomWithIdx(g_A_temp)
                    a_B_obj = rw_mol.GetAtomWithIdx(g_B_temp)
                    
                    is_C_A = (a_A_obj.GetSymbol() == 'C')
                    is_C_B = (a_B_obj.GetSymbol() == 'C')
                    
                    if is_C_A and is_C_B:
                        # 判定是否为不饱和碳（芳香性，或杂化态为 SP2 / SP）
                        is_unsat_A = a_A_obj.GetIsAromatic() or a_A_obj.GetHybridization() in [Chem.HybridizationType.SP2, Chem.HybridizationType.SP]
                        is_unsat_B = a_B_obj.GetIsAromatic() or a_B_obj.GetHybridization() in [Chem.HybridizationType.SP2, Chem.HybridizationType.SP]
                        
                        if is_unsat_A and is_unsat_B:
                            # 💡 新增逻辑：两端均为不饱和碳 (如高效的交叉偶联)，具备极高合成可及性，立即确立！
                            return {
                                'ok': True, 'min_dist': float(dist),
                                'best_atom_A': b_atom_A, 'best_atom_B': b_atom_B,
                                'avail_A': avail_A, 'avail_B': avail_B
                            }
                        else:
                            # 只要有一端是 sp3 脂肪碳的纯碳键，存入备胎并继续检索杂原子
                            if fallback_cc_pair is None:
                                fallback_cc_pair = {
                                    'ok': True, 'min_dist': float(dist),
                                    'best_atom_A': b_atom_A, 'best_atom_B': b_atom_B,
                                    'avail_A': avail_A, 'avail_B': avail_B
                                }
                            continue # 继续向后检索，优先寻找杂原子
                    else:
                        # 存在包含杂原子的优质连接，立刻高优返回
                        return {
                            'ok': True, 'min_dist': float(dist),
                            'best_atom_A': b_atom_A, 'best_atom_B': b_atom_B,
                            'avail_A': avail_A, 'avail_B': avail_B
                        }
                else:
                    # 对于 Frame-Frame，遇到合规即返回 (Frame-Frame 的严苛审核已由 Rule 1 完成)
                    return {
                        'ok': True, 'min_dist': float(dist),
                        'best_atom_A': b_atom_A, 'best_atom_B': b_atom_B,
                        'avail_A': avail_A, 'avail_B': avail_B
                    }
                    
            last_reason = f"pass_denied: {reason}"

        # 循环结束。检查是否有暂存的脂肪碳-脂肪碳备胎
        if fallback_cc_pair is not None:
            return fallback_cc_pair

        return {
            'ok': False, 'reason': last_reason, 
            'min_dist': float(min_dist_record) if min_dist_record is not None else float('inf'),
            'avail_A': avail_A, 'avail_B': avail_B
        }

    planned_bonds = []
    skipped_fixed_generated_edges = []
    skipped_generated_edges = []

    debug_base = {
        'total_unique_edges': len(unique_edges),
        'fixed_fixed_edges': len(fixed_fixed_edges),
        'fixed_generated_edges': len(fixed_generated_edges),
        'generated_generated_edges': len(generated_generated_edges),
        'known_bonds_count': len(known_bonds),
        'dist_threshold': float(dist_threshold),
        'initial_total_h': {f: int(sum(h.values())) for f, h in initial_ledger.items()},
        'degree_total': degree_total,
        'degree_fixed_fixed': degree_fixed_fixed,
        'degree_fixed_generated': degree_fixed_generated,
        'degree_generated_generated': degree_generated_generated,
    }

    # 5. 执行规划逻辑 (引入优先级、距离排序与 Rule 5 软跳过)
    fixed_generated_candidates = []
    for frag_A, frag_B in fixed_generated_edges:
        best_pair = get_current_best_pair(frag_A, frag_B)
        # 拓扑优先级：Frame-Frame 优先级设为 1（靠后），其他为 0（靠前）
        prio = 1 if (is_frame[frag_A] and is_frame[frag_B]) else 0
        fixed_generated_candidates.append((prio, best_pair.get('min_dist', float('inf')), frag_A, frag_B))

    for _, _, frag_A, frag_B in sorted(fixed_generated_candidates):
        best_pair = get_current_best_pair(frag_A, frag_B)
        if not best_pair['ok']:
            skipped_fixed_generated_edges.append({
                'edge': (frag_A, frag_B), 'reason': best_pair.get('reason', 'unknown'),
                'min_dist': best_pair.get('min_dist'),
            })
            continue

        bA, bB = best_pair['best_atom_A'], best_pair['best_atom_B']
        ledger[frag_A][bA] -= 1; ledger[frag_B][bB] -= 1
        planned_bonds.append((frag_A, bA, frag_B, bB))
        
        # 记入拓扑账本与 Rule 7 账本
        frag_adj[frag_A].add(frag_B); frag_adj[frag_B].add(frag_A)
        if is_frame[frag_A]: frame_atom_used.add((frag_A, bA))
        if is_frame[frag_B]: frame_atom_used.add((frag_B, bB))
        
        rw_mol.AddBond(offsets[frag_A] + bA, offsets[frag_B] + bB, Chem.BondType.SINGLE)

    # 对游离边也执行拓扑排序
    generated_generated_candidates = []
    for frag_A, frag_B in generated_generated_edges:
        best_pair = get_current_best_pair(frag_A, frag_B)
        prio = 1 if (is_frame[frag_A] and is_frame[frag_B]) else 0
        generated_generated_candidates.append((prio, best_pair.get('min_dist', float('inf')), frag_A, frag_B))

    for edge_order, (_, _, frag_A, frag_B) in enumerate(sorted(generated_generated_candidates)):
        best_pair = get_current_best_pair(frag_A, frag_B)

        if not best_pair['ok']:
            skipped_generated_edges.append({
                'edge': (frag_A, frag_B), 
                'reason': best_pair.get('reason', 'unknown'),
                'min_dist': best_pair.get('min_dist')
            })
            continue 

        bA, bB = best_pair['best_atom_A'], best_pair['best_atom_B']
        ledger[frag_A][bA] -= 1; ledger[frag_B][bB] -= 1
        planned_bonds.append((frag_A, bA, frag_B, bB))
        
        frag_adj[frag_A].add(frag_B); frag_adj[frag_B].add(frag_A)
        if is_frame[frag_A]: frame_atom_used.add((frag_A, bA))
        if is_frame[frag_B]: frame_atom_used.add((frag_B, bB))
        
        rw_mol.AddBond(offsets[frag_A] + bA, offsets[frag_B] + bB, Chem.BondType.SINGLE)

    debug_info = dict(debug_base)
    debug_info.update({
        'reason': 'success',
        'planned_bonds_count': len(planned_bonds),
        'skipped_fixed_generated_edges': skipped_fixed_generated_edges,
        'skipped_generated_edges': skipped_generated_edges,
        'skipped_fixed_generated_count': len(skipped_fixed_generated_edges),
        'final_total_h': {f: int(sum(h.values())) for f, h in ledger.items()},
    })
    return finish(True, planned_bonds, debug_info)    

def build_molecule_from_blueprint(fragments, planned_bonds, known_bonds=None):
    """Assemble a ligand from fragments and planned inter-fragment bonds."""
    from rdkit import Chem

    if not fragments:
        return None

    combined_mol = Chem.Mol()
    offsets = []
    current_offset = 0

    for frag in fragments:
        offsets.append(current_offset)
        if frag is None:
            continue

        combined_mol = Chem.CombineMols(combined_mol, frag)
        current_offset += frag.GetNumAtoms()

    rw_mol = Chem.RWMol(combined_mol)
    h_indices_to_remove = set()

    all_bonds = list(known_bonds or []) + list(planned_bonds)

    def consume_attachable_h(global_idx):
        atom_obj = rw_mol.GetAtomWithIdx(global_idx)
        for neighbor in atom_obj.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIdx() not in h_indices_to_remove:
                h_indices_to_remove.add(neighbor.GetIdx())
                return

        explicit_h_count = atom_obj.GetNumExplicitHs()
        if explicit_h_count > 0:
            atom_obj.SetNumExplicitHs(explicit_h_count - 1)
            try:
                atom_obj.UpdatePropertyCache(strict=False)
            except Exception:
                pass

    for frag_A, local_A, frag_B, local_B in all_bonds:
        global_A = offsets[frag_A] + local_A
        global_B = offsets[frag_B] + local_B

        if rw_mol.GetBondBetweenAtoms(global_A, global_B) is not None:
            continue

        consume_attachable_h(global_A)
        consume_attachable_h(global_B)

        rw_mol.AddBond(global_A, global_B, Chem.BondType.SINGLE)

    for h_idx in sorted(h_indices_to_remove, reverse=True):
        rw_mol.RemoveAtom(h_idx)

    final_mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return final_mol
    except Exception:
        return None
