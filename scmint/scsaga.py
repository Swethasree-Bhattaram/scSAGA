import argparse
import pydantic
import os
import time
import yaml
import typing as t
import numpy as np
import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds, cg, LinearOperator
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import dijkstra, connected_components
import ot  # Python Optimal Transport (POT) library
from geosketch import gs
from joblib import Parallel, delayed
import torch
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gc

# ==============================================================================
# ==== 0. TORCH CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- PyTorch will use device: {DEVICE} ---\n")

def _maybe_gc(do_cuda: bool = True):
    gc.collect()
    if do_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# ==== 1. HELPERS (I/O, metrics, sketching, plotting)
# ==============================================================================

def load_txt_lines(path: str) -> np.ndarray:
    with open(path, "r") as f:
        return np.array([line.strip() for line in f])

def analyze_transport_plan(df: pd.DataFrame) -> float:
    correct_matches = 0
    total_matches = len(df)
    if total_matches == 0:
        print("Accuracy: 0.00% (no rows)")
        return 0.0
    for row_barcode, row in df.iterrows():
        max_col_barcode = row.idxmax()
        if row_barcode == max_col_barcode:
            correct_matches += 1
    acc = 100.0 * correct_matches / total_matches
    print(f"Correctly matched cells: {correct_matches}/{total_matches}")
    print(f"Accuracy (1:1): {acc:.2f}%")
    return acc

def test_alignment_score_pair(X_left_aligned: np.ndarray, X_anchor: np.ndarray, k: int = 30) -> float:
    n_left = X_left_aligned.shape[0]
    combined = np.vstack((X_left_aligned, X_anchor)).astype(np.float32)
    labels = np.array([0] * n_left + [1] * X_anchor.shape[0], dtype=np.int32)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(combined)
    indices = nbrs.kneighbors(combined, return_distance=False)[:, 1:]
    neighbor_labels = labels[indices]
    same_domain_counts = np.sum(neighbor_labels == labels[:, None], axis=1)
    bar_x = np.mean(same_domain_counts)
    expected_random_neighbors = k / 2
    score = 1 - ((bar_x - expected_random_neighbors) / (k - expected_random_neighbors))
    return float(max(0.0, min(1.0, score)))

def test_alignment_score_multi(embeddings_by_name: dict, k: int = 30) -> float:
    names = list(embeddings_by_name.keys())
    arrays = [embeddings_by_name[n] for n in names]
    sizes = [a.shape[0] for a in arrays]
    X = np.vstack(arrays).astype(np.float32)
    labels = np.concatenate([[i]*sizes[i] for i in range(len(sizes))]).astype(np.int32)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    neighbor_labels = labels[idx]
    same = (neighbor_labels == labels[:, None]).sum(axis=1).mean()
    N = len(labels)
    props = (np.array(sizes, dtype=np.float32) / N).astype(np.float32)
    expected = k * (props**2).sum()
    score = 1 - ((same - expected) / (k - expected))
    return float(max(0.0, min(1.0, score)))

def sample_anchor_pairs_from_plan(T: np.ndarray, M: int):
    N, K = T.shape
    p_rows = T.sum(axis=1)
    if p_rows.sum() < 1e-9:
        return [ (np.random.randint(0, N), np.random.randint(0, K)) for _ in range(M) ]
    p_rows = p_rows / p_rows.sum()
    sampled_row_indices = np.random.choice(N, size=M, p=p_rows)
    sampled_col_indices = np.zeros(M, dtype=np.int32)
    for i, r in enumerate(sampled_row_indices):
        p_cols = T[r, :]
        rs = p_cols.sum()
        if rs > 1e-10:
            p_cols = p_cols / rs
            sampled_col_indices[i] = int(np.random.choice(K, p=p_cols))
        else:
            sampled_col_indices[i] = int(np.random.choice(K))
    return list(zip(sampled_row_indices, sampled_col_indices))

def _dijkstra_worker(graph, source_node):
    return dijkstra(csgraph=graph, directed=False, indices=source_node)

def save_joint_embedding_plot(aligned_by_name: dict, outdir: str, method: str = "pca"):
    names = list(aligned_by_name.keys())
    Xs = [aligned_by_name[n] for n in names]
    X = np.vstack(Xs)
    labels = np.concatenate([[i]*Xs[i].shape[0] for i in range(len(Xs))])
    reducer = PCA(n_components=2, random_state=0)
    X2 = reducer.fit_transform(X)
    df = pd.DataFrame({
        "x": X2[:, 0], "y": X2[:, 1],
        "dataset": [names[i] for i in labels]
    })
    csv_path = os.path.join(outdir, "joint_embedding_2d.csv")
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(8, 6), dpi=150)
    for i, name in enumerate(names):
        mask = (labels == i)
        plt.scatter(X2[mask, 0], X2[mask, 1], s=3, alpha=0.5, label=name)
    plt.legend(markerscale=3)
    plt.title("Simultaneous Joint Embedding (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    png_path = os.path.join(outdir, "joint_embedding_2d.png")
    plt.tight_layout(); plt.savefig(png_path); plt.close()
    print(f"Saved joint 2D embedding: {png_path}\nSaved coordinates: {csv_path}")

# ==============================================================================
# ==== 2. CORE ALGORITHM
# ==============================================================================

class Saga:
    def __init__(self, device: torch.device):
        self.device = device
        self.knn_time, self.dijkstra_time, self.ot_time = 0.0, 0.0, 0.0
        self.gw_iterations, self.final_gw_error = 0, 0.0
        self.ot_sampling_time, self.ot_cost_matrix_time = 0.0, 0.0
        self.ot_augmentation_time, self.ot_solver_time, self.ot_update_time = 0.0, 0.0, 0.0

    def _build_knn_graph(self, X: np.ndarray, k: int = 30) -> csr_matrix:
        print(f"    - Building k-NN graph for data of shape {X.shape} with k={k}...")
        knn_start = time.time()
        knn_graph = kneighbors_graph(X, k, mode='distance', n_jobs=-1)
        self.knn_time += time.time() - knn_start
        n_components, labels = connected_components(knn_graph, directed=False)
        if n_components > 1:
            print(f"    - Graph is disconnected ({n_components} components). Stitching them together...")
            while n_components > 1:
                comp0_indices = np.where(labels == 0)[0]
                other_indices = np.where(labels != 0)[0]
                other_tree = KDTree(X[other_indices])
                dist, ind = other_tree.query(X[comp0_indices], k=1)
                best_idx = np.argmin(dist)
                pt_from, pt_to = comp0_indices[best_idx], other_indices[ind[best_idx]]
                knn_graph = knn_graph.tolil()
                knn_graph[pt_from, pt_to] = dist[best_idx]
                knn_graph[pt_to, pt_from] = dist[best_idx]
                knn_graph = knn_graph.tocsr()
                n_components, labels = connected_components(knn_graph, directed=False)
                print(f"    - Bridge added. Remaining components: {n_components}")
        return knn_graph.astype(np.float32)

    def _solve_pair(self, X_left: np.ndarray, X_anchor: np.ndarray,
                    graph_left: csr_matrix, graph_anchor: csr_matrix,
                    s_shared_cells: int, M_samples: int, alpha: float, 
                    S_iterations: int, gw_epsilon: float, gw_reg: float,
                    verbose_every: int = 20) -> np.ndarray:
        N, K = X_left.shape[0], X_anchor.shape[0]
        s_eff = s_shared_cells if s_shared_cells is not None and s_shared_cells <= min(N, K) else min(N, K)
        print(f"\n--- Preparing for Partial Alignment (s={s_eff}, N={N}, K={K}) ---")
        p_real = torch.from_numpy(ot.unif(N)).to(self.device, dtype=torch.float32)
        q_real = torch.from_numpy(ot.unif(K)).to(self.device, dtype=torch.float32)
        m_transport_fraction = s_eff / max(N, K)
        p_aug = torch.cat([p_real, torch.tensor([torch.sum(q_real) - m_transport_fraction], device=self.device, dtype=torch.float32)])
        q_aug = torch.cat([q_real, torch.tensor([torch.sum(p_real) - m_transport_fraction], device=self.device, dtype=torch.float32)])
        T_real = torch.outer(p_real, q_real)
        initial_reg = gw_reg if gw_reg > 0 else 1e-4
        final_reg = 5e-4
        decay_rate = (final_reg / initial_reg) ** (1 / max(1, S_iterations))
        ot_start_time = time.time()
        for i in range(S_iterations):
            current_reg = initial_reg * (decay_rate ** i)
            T_real_prev = T_real.clone()
            t0 = time.time()
            T_cpu = T_real.detach().cpu().numpy()
            anchor_pairs = sample_anchor_pairs_from_plan(T_cpu, M_samples)
            self.ot_sampling_time += time.time() - t0
            j_left, l_anchor = zip(*anchor_pairs)
            unique_j, j_inv = np.unique(j_left, return_inverse=True)
            unique_l, l_inv = np.unique(l_anchor, return_inverse=True)
            dijkstra_start = time.time()
            left_results = Parallel(n_jobs=-1)(delayed(_dijkstra_worker)(graph_left, idx) for idx in unique_j)
            anch_results = Parallel(n_jobs=-1)(delayed(_dijkstra_worker)(graph_anchor, idx) for idx in unique_l)
            self.dijkstra_time += time.time() - dijkstra_start
            D_left_unique = torch.from_numpy(np.vstack(left_results)).float().to(self.device).T
            D_anch_unique = torch.from_numpy(np.vstack(anch_results)).float().to(self.device).T
            for D_mat in [D_left_unique, D_anch_unique]:
                inf_mask = torch.isinf(D_mat)
                if torch.any(inf_mask):
                    max_dist = D_mat[~inf_mask].max()
                    D_mat[inf_mask] = max_dist * 1.5
                max_val = D_mat.max()
                if max_val > 0: D_mat /= max_val
            D_left, D_anch = D_left_unique[:, j_inv], D_anch_unique[:, l_inv]
            t0 = time.time()
            term_A = torch.mean(D_left**2, dim=1, keepdim=True)
            term_C = torch.mean(D_anch**2, dim=1, keepdim=True).T
            term_B = -2 * (D_left @ D_anch.T) / M_samples
            Lambda_real = term_A + term_B + term_C
            self.ot_cost_matrix_time += time.time() - t0
            t0 = time.time()
            Lambda_aug = torch.zeros(N + 1, K + 1, device=self.device, dtype=torch.float32)
            Lambda_aug[:N, :K] = Lambda_real
            max_val = torch.max(Lambda_real)
            penalty = 100 * max_val if max_val > 0 else torch.tensor(100.0, device=self.device, dtype=torch.float32)
            Lambda_aug[:-1, -1], Lambda_aug[-1, :-1] = penalty, penalty
            self.ot_augmentation_time += time.time() - t0
            t0 = time.time()
            T_prime_aug = ot.sinkhorn(p_aug, q_aug, Lambda_aug, current_reg, numItermax=100, stopThr=5e-4, verbose=False)
            self.ot_solver_time += time.time() - t0
            t0 = time.time()
            T_prime_real = T_prime_aug[:-1, :-1]
            T_real = (1 - alpha) * T_real_prev + alpha * T_prime_real
            self.ot_update_time += time.time() - t0
            err = torch.linalg.norm(T_real - T_real_prev)
            if (i + 1) % max(1, verbose_every) == 0 or i == S_iterations - 1:
                print(f"    Iter {i+1:>4}/{S_iterations} | Update Error: {err.item():.4e}")
            if err < gw_epsilon and i > 50:
                print(f"\n    Converged at iteration {i+1} with error {err.item():.4e}")
                break
            del D_left_unique, D_anch_unique, D_left, D_anch, Lambda_real, Lambda_aug, T_prime_aug, T_prime_real
            _maybe_gc(do_cuda=True)
        self.ot_time += time.time() - ot_start_time
        self.gw_iterations, self.final_gw_error = i + 1, float(err.item())
        return T_real.detach().cpu().numpy()

    def _define_linear_operator(self, laplacians, T_dict, query_names, anchor_name, lambda_reg):
        T_path_list = [T_dict[(name, anchor_name)] for name in query_names]
        T_shapes = [np.load(p, mmap_mode='r').shape for p in T_path_list]
        Sigma_x_list = [diags(np.load(p, mmap_mode='r').sum(axis=1)) for p in T_path_list]
        Sigma_y_list = [diags(np.load(p, mmap_mode='r').sum(axis=0)) for p in T_path_list]
        S_xx_blocks = [laplacians[name] + lambda_reg * Sigma_x_list[i] for i, name in enumerate(query_names)]
        S_yy = laplacians[anchor_name] + lambda_reg * sum(Sigma_y_list)
        N_q = sum(block.shape[0] for block in S_xx_blocks)
        N_a = S_yy.shape[0]

        diag_S_yy = S_yy.diagonal()
        diag_S_yy[diag_S_yy == 0] = 1.0
        precond_y = diags(1.0 / diag_S_yy)
        
        # --- NEW: Helper function for parallel CG calls ---
        def _cg_worker(block, v_chunk):
            diag_block = block.diagonal()
            diag_block[diag_block == 0] = 1.0
            precond_x = diags(1.0 / diag_block)
            x, _ = cg(block, v_chunk.astype(np.float32), rtol=1e-4, M=precond_x)
            return x.ravel()

        def H_x_matvec(v):
            v = v.ravel()
            
            # --- NEW: Parallelize the for loop over S_xx_blocks ---
            v_chunks = []
            current_pos = 0
            for block in S_xx_blocks:
                n_cells = block.shape[0]
                v_chunks.append(v[current_pos : current_pos + n_cells])
                current_pos += n_cells

            # Use joblib to run cg solver on each block in parallel
            # n_jobs=-1 uses all available cores
            results = Parallel(n_jobs=-1)(delayed(_cg_worker)(S_xx_blocks[i], v_chunks[i]) for i in range(len(S_xx_blocks)))
            
            # Concatenate the results from all parallel jobs
            return np.concatenate(results)
        
        def H_y_matvec(v):
            v = v.ravel()
            x, _ = cg(S_yy, v.astype(np.float32), rtol=1e-4, M=precond_y)
            return x.ravel()

        def H_matvec(v):
            y_mult = H_y_matvec(v)
            xy_mult = np.zeros(N_q, dtype=np.float32)
            current_pos = 0
            for i, T_path in enumerate(T_path_list):
                T = np.load(T_path, mmap_mode='r')
                n_cells = T_shapes[i][0]
                xy_mult[current_pos : current_pos + n_cells] = T @ y_mult
                current_pos += n_cells
            return H_x_matvec(xy_mult)

        def H_rmatvec(v):
            x_mult = H_x_matvec(v)
            xy_T_mult = np.zeros(N_a, dtype=np.float32)
            current_pos = 0
            for i, T_path in enumerate(T_path_list):
                T = np.load(T_path, mmap_mode='r')
                n_cells = T_shapes[i][0]
                xy_T_mult += T.T @ x_mult[current_pos : current_pos + n_cells]
                current_pos += n_cells
            return H_y_matvec(xy_T_mult)
            
        return LinearOperator((N_q, N_a), matvec=H_matvec, rmatvec=H_rmatvec), H_x_matvec, H_y_matvec

    def joint_embedding(self, anchor_name: str, data_by_name: dict,
                        graphs_by_name: dict, T_dict: dict, 
                        lambda_reg: float = 1.0, out_dim: int = 30) -> dict:
        print("\n--- Performing Memory-Safe Joint Manifold Alignment ---")
        query_names = [name for name in data_by_name.keys() if name != anchor_name]
        laplacians = {}
        for name, graph in graphs_by_name.items():
            W = graph.copy()
            W.data = 1.0 / (W.data + 1e-10)
            W = W.maximum(W.T)
            D = diags(W.sum(axis=1).A1)
            laplacians[name] = (D - W).asformat('csr')
        H_op, H_x_matvec, H_y_matvec = self._define_linear_operator(laplacians, T_dict, query_names, anchor_name, lambda_reg)
        print("    - Solving for singular vectors using iterative solver (svds)...")
        
        try:
            k_svds = min(out_dim + 2, H_op.shape[0] - 1, H_op.shape[1] - 1)
            U, _, V_t = svds(H_op, k=k_svds, tol=1e-4)
        
            U = U[:, ::-1][:, :out_dim]
            V = V_t.T[:, ::-1][:, :out_dim]

        except Exception as e:
            print(f"    - svds failed with error: {e}. Check matrix ranks and sizes.")
            return {}

        print("    - Constructing final embeddings from singular vectors...")
        fx = U.astype(np.float32)
        fy = V.astype(np.float32)
    
        aligned_embeddings = {anchor_name: fy.astype(np.float32)}
        current_pos = 0
        for name in query_names:
            n_cells = data_by_name[name].shape[0]
            aligned_embeddings[name] = fx[current_pos : current_pos + n_cells].astype(np.float32)
            current_pos += n_cells
        print("--- Joint Embedding Complete ---")
        return aligned_embeddings

    def run_multi(self, anchor_name: str, data_by_name: dict, outdir: str,
                  params: dict, k_for_graph: int = 30) -> t.Tuple[dict, dict]:
        print("--- Building k-NN graphs for all datasets ---")
        graphs = {name: self._build_knn_graph(X, k=k_for_graph) for name, X in data_by_name.items()}
        T_dict: dict = {}
        for name, X_left in data_by_name.items():
            if name == anchor_name: continue
            print(f"\n=== Solving OT: {name} -> {anchor_name} ===")
            T = self._solve_pair(
                X_left=X_left, X_anchor=data_by_name[anchor_name],
                graph_left=graphs[name], graph_anchor=graphs[anchor_name], **params
            )
            plan_path = os.path.join(outdir, f"T_{name}_to_{anchor_name}.npy")
            print(f"    - Saving transport plan to: {plan_path}")
            np.save(plan_path, T)
            T_dict[(name, anchor_name)] = plan_path
        min_dim = min(X.shape[0] for X in data_by_name.values())
        out_dim = min(30, min_dim - 2 if min_dim > 2 else 1)
        aligned_embeddings = self.joint_embedding(
            anchor_name=anchor_name, data_by_name=data_by_name,
            graphs_by_name=graphs, T_dict=T_dict,
            lambda_reg=1.0, out_dim=out_dim
        )
        return T_dict, aligned_embeddings

# ==============================================================================
# ==== 3. CONFIG & MAIN
# ==============================================================================
class DatasetSpec(pydantic.BaseModel):
    name: str; pca: str
    modality: t.Optional[str] = None; counts: t.Optional[str] = None
    barcodes: t.Optional[str] = None; features: t.Optional[str] = None
class InputArgsGeneric(pydantic.BaseModel):
    anchor: str; datasets: t.List[DatasetSpec]; output_dir: str
    s_shared_cells: int; sketch_size: t.Optional[int] = None
    M_samples: int = 50; alpha: float = 0.9; S_iterations: int = 500
    gw_epsilon: float = 1e-5; gw_reg: float = 0.001
@t.final
class SCData:
    def __init__(self, raw_cfg: dict) -> None:
        os.makedirs(raw_cfg["output_dir"], exist_ok=True)
        self.output_dir = raw_cfg["output_dir"]
        parsed = InputArgsGeneric.model_validate(raw_cfg)
        self.anchor = parsed.anchor
        self.data_by_name: dict = {}; self.barcodes_by_name: dict = {}
        for d in parsed.datasets:
            print(f"Loading PCA for {d.name} from: {d.pca}")
            X = np.loadtxt(d.pca, comments="#", dtype=np.float32)
            idx = None
            if parsed.sketch_size and X.shape[0] > parsed.sketch_size:
                 print(f"  - Geometric sketching: {X.shape[0]} -> {parsed.sketch_size} cells")
                 idx = gs(X, parsed.sketch_size, replace=False)
                 X = X[idx, :]
            self.data_by_name[d.name] = X
            if d.barcodes:
                try:
                    barcodes = load_txt_lines(d.barcodes)
                    if idx is not None:
                        self.barcodes_by_name[d.name] = barcodes[idx]
                    else:
                        self.barcodes_by_name[d.name] = barcodes
                except Exception as e:
                    print(f"Warning: failed to read barcodes for {d.name}: {e}")
        self.params = dict(
            s_shared_cells=parsed.s_shared_cells, M_samples=parsed.M_samples,
            alpha=parsed.alpha, S_iterations=parsed.S_iterations,
            gw_epsilon=parsed.gw_epsilon, gw_reg=parsed.gw_reg,
        )
        if parsed.sketch_size:
            new_s = min(self.params["s_shared_cells"], parsed.sketch_size)
            if new_s != self.params["s_shared_cells"]:
                print(f"Adjusting s_shared_cells: {self.params['s_shared_cells']} -> {new_s} (due to sketch_size)")
                self.params["s_shared_cells"] = new_s
def main(raw_cfg: dict):
    scdata = SCData(raw_cfg)
    saga = Saga(device=DEVICE)
    
    print("\nRunning SAGA (multi-dataset, anchor hub)...")
    
    T_dict, aligned = saga.run_multi(
        anchor_name=scdata.anchor, data_by_name=scdata.data_by_name,
        outdir=scdata.output_dir, 
        params=scdata.params
    )
    outdir = scdata.output_dir
    os.makedirs(outdir, exist_ok=True)
    
    # print("Saving Transport Plans as CSVs (this may be slow)")
    # for (name, anchor), plan_path in T_dict.items():
    #     print(f"Loading plan from {plan_path} to save as CSV...")
    #     T = np.load(plan_path)
    #     rows, cols = scdata.barcodes_by_name.get(name), scdata.barcodes_by_name.get(anchor)
    #     df = pd.DataFrame(T, index=rows, columns=cols)
    #     path = os.path.join(outdir, f"transport_{name}_to_{anchor}.csv")
    #     df.to_csv(path)
    #     print(f"Saved transport plan: {path}")

    global_score = test_alignment_score_multi(aligned, k=30)
    print(f"\nGlobal alignment score (all datasets): {global_score:.4f}")
    pair_lines = []
    anchor_embedding = aligned[scdata.anchor]
    for name, embedding in aligned.items():
        if name == scdata.anchor: continue
        score = test_alignment_score_pair(embedding, anchor_embedding, k=30)
        print(f"Pairwise alignment score in joint space ({name} -> {scdata.anchor}): {score:.4f}")
        pair_lines.append(f"{name}->{scdata.anchor} score={score:.4f}")
    save_joint_embedding_plot(aligned_by_name=aligned, outdir=outdir)
    timing_breakdown = f"""
--- SAGA-Torch Timing Breakdown ---
k-NN Graph Construction took: {saga.knn_time:.4f} seconds
Intra-domain Distances (Dijkstra) took: {saga.dijkstra_time:.4f} seconds
Total Optimal Transport Solver took: {saga.ot_time:.4f} seconds
--- SAGA-Torch Metrics ---
Global Alignment Score (all datasets): {global_score:.4f}
Pairwise: {", ".join(pair_lines)}
Total GW Iterations (last run): {saga.gw_iterations}
Final GW Convergence Error (last run): {saga.final_gw_error:.4e}
"""
    print(timing_breakdown)
    with open(os.path.join(outdir, "saga_runtimes.txt"), "w") as f:
        f.write(timing_breakdown)
    print(f"SAGA detailed runtimes saved to: {os.path.join(outdir, 'saga_runtimes.txt')}")

def parse_yaml(yaml_file: str):
    with open(yaml_file) as ymfx:
        return yaml.safe_load(ymfx)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="scSAGA")
    parser.add_argument("yaml_file", help="Yaml input file with configuration")
    run_args = parser.parse_args()
    yaml_config = parse_yaml(run_args.yaml_file)
    main(yaml_config)
