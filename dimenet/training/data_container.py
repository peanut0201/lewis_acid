import numpy as np
import scipy.sparse as sp
import pandas as pd
import json
import re

index_keys = ["batch_seg", "idnb_i", "idnb_j", "id_expand_kj",
              "id_reduce_ji", "id3dnb_i", "id3dnb_j", "id3dnb_k"]


class DataContainer:
    def __init__(self, json_file, xlsx_file, cutoff, target_keys):
        # Load data from json file
        with open(json_file, 'r') as f:
            self.json_data = json.load(f)
        
        # Load data from Excel file
        self.df = pd.read_excel(xlsx_file)
        
        self.cutoff = cutoff
        self.target_keys = target_keys
        
        # Extract data from JSON
        self._extract_data_from_json()
        
        if self.N is None:
            self.N = np.zeros(len(self.targets), dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

        assert self.R is not None

    def _parse_xyz_coordinates(self, xyz_string):
        """parse XYZ coordinates and extract atomic numbers"""

        lines = xyz_string.strip().split('\n')
        # the first line is the number of atoms
        n_atoms = int(lines[0])
        
        # skip the second line (comment line)
        coord_lines = lines[2:2+n_atoms]
        
        coordinates = []
        atomic_numbers = []
        
        for line in coord_lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                # convert element symbol to atomic number
                atomic_number = self._element_to_atomic_number(element)
                
                coordinates.append([x, y, z])
                atomic_numbers.append(atomic_number)
        
        return np.array(coordinates), np.array(atomic_numbers), n_atoms

    def _element_to_atomic_number(self, element):
        """convert element symbol to atomic number"""

        element_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Br': 35, 'I': 53
        }
        return element_map.get(element, 0)

    def _extract_data_from_json(self):
        """extract la-PBEh3c coordinates data from JSON"""

        compound_ids = []
        all_coordinates = []
        all_atomic_numbers = []
        all_n_atoms = []
        
        for compound_id, compound_data in self.json_data.items():
            if 'la-PBEh3c' in compound_data:
                xyz_string = compound_data['la-PBEh3c']
                coordinates, atomic_numbers, n_atoms = self._parse_xyz_coordinates(xyz_string)
                
                compound_ids.append(compound_id)
                all_coordinates.append(coordinates)
                all_atomic_numbers.append(atomic_numbers)
                all_n_atoms.append(n_atoms)
        
        # convert to numpy array
        self.id = np.array(compound_ids)
        self.N = np.array(all_n_atoms, dtype=np.int32)
        self.Z = np.concatenate(all_atomic_numbers).astype(np.int32)
        self.R = np.concatenate(all_coordinates, axis=0).astype(np.float32)
        
        # extract real FIA targets from Excel
        self._extract_targets_from_excel(compound_ids)

    def _extract_targets_from_excel(self, compound_ids):
        """extract FIA targets from Excel"""
        
        # FIA related column names
        fia_columns = ['fia_gas-DSDBLYP', 'fia_gas-PBEh3c', 'fia_solv-DSDBLYP', 'fia_solv-PBEh3c']
        
        # create target value array
        targets_list = []
        
        for compound_id in compound_ids:
            # find the corresponding row in Excel
            mask = self.df['Compound'] == compound_id
            if mask.any():
                # extract FIA values
                fia_values = []
                for col in fia_columns:
                    if col in self.df.columns:
                        value = self.df.loc[mask, col].iloc[0]
                        fia_values.append(float(value) if pd.notna(value) else 0.0)
                    else:
                        fia_values.append(0.0)
                targets_list.append(fia_values)
            else:
                # if the corresponding compound is not found, use zero value
                targets_list.append([0.0] * len(fia_columns))
                print(f"警告: 在Excel文件中未找到化合物 {compound_id}")
        
        # convert to numpy array
        self.targets = np.array(targets_list, dtype=np.float32)
        
        # 标准化目标值（FIA值范围很大，需要标准化）
        self.targets_mean = np.mean(self.targets, axis=0)
        self.targets_std = np.std(self.targets, axis=0)
        self.targets = (self.targets - self.targets_mean) / self.targets_std
        
        # update target_keys
        self.target_keys = fia_columns
        
        print(f"成功提取 {len(compound_ids)} 个化合物的FIA目标值")
        print(f"目标值形状: {self.targets.shape}")
        print(f"FIA列: {fia_columns}")
        print(f"目标值标准化完成 - 均值: {self.targets_mean}, 标准差: {self.targets_std}")

    def _bmat_fast(self, mats):
        new_data = np.concatenate([mat.data for mat in mats])

        ind_offset = np.zeros(1 + len(mats))
        ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
        new_indices = np.concatenate(
            [mats[i].indices + ind_offset[i] for i in range(len(mats))])

        indptr_offset = np.zeros(1 + len(mats))
        indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
        new_indptr = np.concatenate(
            [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
        return sp.csr_matrix((new_data, new_indices, new_indptr))

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {}
        data['targets'] = self.targets[idx]
        data['id'] = self.id[idx]
        data['N'] = self.N[idx]
        data['batch_seg'] = np.repeat(np.arange(len(idx), dtype=np.int32), data['N'])
        adj_matrices = []

        data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
        data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)

        nend = 0
        for k, i in enumerate(idx):
            n = data['N'][k]  # number of atoms
            nstart = nend
            nend = nstart + n

            if self.Z is not None:
                data['Z'][nstart:nend] = self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]]

            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            data['R'][nstart:nend] = R

            Dij = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(Dij <= self.cutoff))
            adj_matrices[-1] -= sp.eye(n, dtype=bool)

        # Entry x,y is edge x<-y (!)
        adj_matrix = self._bmat_fast(adj_matrices)
        # Entry x,y is edgeid x<-y (!)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target (i) and source (j) nodes of edges
        data['idnb_i'] = edgeid_to_target
        data['idnb_j'] = edgeid_to_source

        # Indices of triplets k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Indices of triplets that are not i->j->i
        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
        data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
        data['id3dnb_k'] = id3ynb_k[id3_y_to_d]

        # Edge indices for interactions
        # j->i => k->j
        data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # j->i => k->j => j->i
        data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        return data
