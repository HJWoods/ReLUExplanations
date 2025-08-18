from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.optimize import linprog
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

@dataclass
class Polytope:
    A: np.ndarray
    b: np.ndarray
    def add_constraints(self, A_add: np.ndarray, b_add: np.ndarray) -> "Polytope":
        if A_add.size == 0:
            return self
        if self.A.size == 0:
            return Polytope(A_add.copy(), b_add.copy())
        A_new = np.vstack([self.A, A_add])
        b_new = np.concatenate([self.b, b_add])
        return Polytope(A_new, b_new)
    @property
    def dim(self) -> int:
        return self.A.shape[1] if self.A.size else 0

@dataclass
class ActivationSignature:
    s_per_layer: List[np.ndarray]
    def to_flat(self) -> np.ndarray:
        return np.concatenate([s.astype(np.int8).ravel() for s in self.s_per_layer])
    @staticmethod
    def from_flat(flat: np.ndarray, layer_sizes: List[int]) -> "ActivationSignature":
        s_per_layer = []
        p = 0
        for n in layer_sizes:
            s_per_layer.append(flat[p:p+n].astype(bool))
            p += n
        return ActivationSignature(s_per_layer)

class ReLUNetworkWrapper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.linears: List[nn.Linear] = []
        self._extract_linears()
        self.hidden_sizes = [lin.out_features for lin in self.linears[:-1]]
        self.input_dim = self.linears[0].in_features
        self.num_classes = self.linears[-1].out_features
    def _extract_linears(self) -> None:
        layers = []
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                layers.append(m)
        self.linears = layers
        if len(self.linears) < 1:
            raise ValueError("Model must contain at least one nn.Linear layer.")
    def forward_collect(self, x: torch.Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self.model.eval()
        with torch.no_grad():
            a = x
            z_list = []
            s_list = []
            for lin in self.linears[:-1]:
                z = lin(a)
                s = (z > 0).cpu().numpy().astype(bool)
                z_list.append(z.cpu().numpy())
                s_list.append(s)
                a = torch.relu(z)
            _ = self.linears[-1](a)
        return z_list, s_list
    def effective_affines(self, signature: ActivationSignature) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(signature.s_per_layer) != len(self.linears) - 1:
            raise ValueError("Signature length mismatch.")
        W1 = self.linears[0].weight.detach().cpu().numpy()
        b1 = self.linears[0].bias.detach().cpu().numpy()
        A_prev, b_prev = W1, b1
        A_list, b_list = [A_prev], [b_prev]
        for k in range(1, len(self.linears) - 1):
            Wk = self.linears[k].weight.detach().cpu().numpy()
            bk = self.linears[k].bias.detach().cpu().numpy()
            D = np.diag(signature.s_per_layer[k-1].astype(float))
            A_k = Wk @ D @ A_prev
            b_k = Wk @ D @ b_prev + bk
            A_list.append(A_k)
            b_list.append(b_k)
            A_prev, b_prev = A_k, b_k
        W_out = self.linears[-1].weight.detach().cpu().numpy()
        b_out = self.linears[-1].bias.detach().cpu().numpy()
        D_last = np.diag(signature.s_per_layer[-1].astype(float)) if signature.s_per_layer else np.eye(W_out.shape[1])
        A_out = W_out @ D_last @ A_prev
        b_out_eff = W_out @ D_last @ b_prev + b_out
        return (A_list + [A_out]), (b_list + [b_out_eff])
    def polytope_from_signature(self, signature: ActivationSignature) -> Polytope:
        A_layers, b_layers = self.effective_affines(signature)
        A_hidden = A_layers[:-1]
        b_hidden = b_layers[:-1]
        A_rows, b_rows = [], []
        for k, (A_k, b_k) in enumerate(zip(A_hidden, b_hidden)):
            s_k = signature.s_per_layer[k]
            if np.any(s_k):
                A_rows.append(-A_k[s_k])
                b_rows.append(b_k[s_k])
            if np.any(~s_k):
                A_rows.append(A_k[~s_k])
                b_rows.append(-b_k[~s_k])
        if len(A_rows) == 0:
            return Polytope(np.zeros((0, self.input_dim)), np.zeros((0,)))
        return Polytope(np.vstack(A_rows), np.concatenate(b_rows))
    def output_subset_polytope(self, signature: ActivationSignature, cls: int) -> Polytope:
        A_layers, b_layers = self.effective_affines(signature)
        A_out, b_out = A_layers[-1], b_layers[-1]
        C = A_out.shape[0]
        A_rows, b_rows = [], []
        for j in range(C):
            if j == cls:
                continue
            A_rows.append(A_out[j] - A_out[cls])
            b_rows.append(b_out[cls] - b_out[j])
        if not A_rows:
            return Polytope(np.zeros((0, self.input_dim)), np.zeros((0,)))
        return Polytope(np.vstack(A_rows), np.array(b_rows))

def is_feasible(poly: Polytope) -> bool:
    if poly.A.size == 0:
        return True
    c = np.zeros(poly.A.shape[1])
    res = linprog(c, A_ub=poly.A, b_ub=poly.b, bounds=[(None, None)]*poly.A.shape[1], method="highs")
    return res.success

class ExplanationEngine:
    def __init__(self, model: nn.Module):
        self.net = ReLUNetworkWrapper(model)
        self.device = next(model.parameters()).device
    def why(self, x: np.ndarray) -> Dict[str, object]:
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, s_list = self.net.forward_collect(x_t)
        signature = ActivationSignature([s[0] for s in s_list])
        with torch.no_grad():
            a = x_t
            for lin in self.net.linears[:-1]:
                a = torch.relu(lin(a))
            logits = self.net.linears[-1](a)
            cls = int(torch.argmax(logits, dim=1).item())
        P = self.net.polytope_from_signature(signature)
        P_out = self.net.output_subset_polytope(signature, cls)
        P_total = P.add_constraints(P_out.A, P_out.b)
        return {"class": cls, "signature": signature.to_flat(), "polytope": P_total, "A": P_total.A, "b": P_total.b}
    

    def why_not(self, x: np.ndarray, counterfactual_cls: int,
                max_hamming: int | None = None,
                max_visited: int = 50000,
                show_progress: bool = True) -> Dict[str, object]:
        """
        March across activation regions via BFS until we find the nearest region
        where the counterfactual class is feasible. If max_hamming is None,
        there is no distance cap (only max_visited limits the search).
        Progress bar shows total regions explored (unique signatures visited).
        """
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, s_list = self.net.forward_collect(x_t)
        sig_start = ActivationSignature([s[0] for s in s_list])

        # Find the actual class
        with torch.no_grad():
            a = x_t
            for lin in self.net.linears[:-1]:
                a = torch.relu(lin(a))
            logits = self.net.linears[-1](a)
            actual_cls = int(torch.argmax(logits, dim=1).item())

        # Check whether the counterfactual class is within the same polytope (O(1) case)
        P0 = self.net.polytope_from_signature(sig_start)
        Pcf0 = self.net.output_subset_polytope(sig_start, counterfactual_cls)
        if is_feasible(P0.add_constraints(Pcf0.A, Pcf0.b)):
            A_layers, B_layers = self.net.effective_affines(sig_start)
            w = A_layers[-1][actual_cls] - A_layers[-1][counterfactual_cls]
            beta = B_layers[-1][actual_cls] - B_layers[-1][counterfactual_cls]
            return {
                "result": "same_polytope",
                "actual": actual_cls,
                "counterfactual": counterfactual_cls,
                "separator_hyperplane": (w, beta),
            }

        # Not found within the same polytope, so march across activation regions via BFS. Worst case O(2^n)
        # but in reality much faster than this in the average case.
        start_bits = sig_start.to_flat().astype(np.int8)
        visited = {tuple(start_bits)}
        queue = deque([(start_bits, 0)])  # (bits, dist)

        pbar = tqdm(disable=not show_progress, total=max_visited if max_visited else None,
                    desc="Regions explored", unit="region")
        pbar.update(1)

        n_bits = len(start_bits)

        while queue:
            bits, dist = queue.popleft()

            if max_hamming is not None and dist >= max_hamming:
                continue

            for i in range(n_bits):
                new_bits = bits.copy()
                new_bits[i] = 1 - new_bits[i]
                key = tuple(new_bits)
                if key in visited:
                    continue

                visited.add(key)
                if max_visited and len(visited) >= max_visited:
                    pbar.close()
                    return {
                        "result": "not_found_within_budget",
                        "actual": actual_cls,
                        "counterfactual": counterfactual_cls,
                        "visited": len(visited),
                        "message": "Stopped due to max_visited limit."
                    }

                pbar.update(1)

                # Check if the counterfactual class is feasible in this new region
                sig_new = ActivationSignature.from_flat(np.array(new_bits, dtype=np.int8), self.net.hidden_sizes)
                Pn = self.net.polytope_from_signature(sig_new)
                Pcf = self.net.output_subset_polytope(sig_new, counterfactual_cls)
                if is_feasible(Pn.add_constraints(Pcf.A, Pcf.b)):
                    pbar.close()
                    return {
                        "result": "adjacent_region",
                        "actual": actual_cls,
                        "counterfactual": counterfactual_cls,
                        "hamming_distance": dist + 1,
                        "target_signature": sig_new.to_flat(),
                        "visited": len(visited),
                    }

                # Increment hamming distance by 1 and enqueue those bit vectors
                queue.append((new_bits, dist + 1))

        pbar.close()
        return {
            "result": "not_found_within_budget",
            "actual": actual_cls,
            "counterfactual": counterfactual_cls,
            "visited": len(visited),
            "message": "BFS exhausted reachable signatures without finding a counterfactual region."
        }

    def format_why_explanation(self, why_result: Dict[str, object]) -> str:
        cls = why_result.get("class")
        A = why_result.get("A")
        b = why_result.get("b")
        constraints_str = [f"Constraint {i+1}: {np.array2string(ai, precision=4)}·x <= {bi:.4f}" for i, (ai, bi) in enumerate(zip(A, b))]
        return f"The input is classified as class {cls} because all of the following constraints are satisfied:\n" + "\n".join(constraints_str)
    def format_why_not_explanation(self, why_not_result: Dict[str, object]) -> str:
        actual = int(why_not_result.get("actual", -1))
        counter = int(why_not_result.get("counterfactual", -1))
        result_type = why_not_result.get("result")
        def format_constraints(A: np.ndarray, b: np.ndarray) -> str:
            return "\n".join([f"Constraint {i+1}: {np.array2string(ai, precision=4)}·x <= {bi:.4f}" for i, (ai, bi) in enumerate(zip(A, b))])
        if result_type == "same_polytope":
            pair = why_not_result.get("separator_hyperplane")
            if pair is not None:
                w, beta = pair
                return (f" (CASE: SAME POLYTOPE)The input is classified as class {actual} and not class {counter}.\nIt would be classified as class {counter} if the following inequality were true:\n" + format_constraints(np.array([w]), np.array([beta])))        
        elif result_type == "adjacent_region":
            dist = why_not_result.get("hamming_distance")
            return f" (CASE: ADJACENT REGION) The input is classified as class {actual} and not class {counter}.\nIt would be class {counter} if {dist} ReLU activations were flipped."
        elif result_type == "not_found_within_budget":
            return f" (CASE: NOT FOUND WITHIN BUDGET) The input is classified as class {actual} and not class {counter}.\nNo region within the search budget satisfies class {counter}."



def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """Train the model on the given dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # Flatten
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return model

def save_model(model, filepath='model.pth'):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath='model.pth'):
    """Load a trained model from a file."""
    if os.path.exists(filepath):
        # Load to CPU first, then move to the same device as the model
        device = next(model.parameters()).device
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath} to {device}")
        return True
    else:
        print(f"Model file {filepath} not found.")
        return False

