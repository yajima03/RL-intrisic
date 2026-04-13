from __future__ import annotations

import csv
import datetime
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces
import torch
from PIL import Image as _PIL_Image


class RewardSpec:
    def __init__(
        self,
        depth: int,
        coordinates: Optional[Sequence[int]],
        value: float,
        reward_type: str = "fix",
        variance: float = 0.0,
    ) -> None:
        self.depth = depth
        self.coordinates = coordinates
        self.value = value
        self.reward_type = reward_type
        self.variance = variance


class Node:
    """Single node in the image pyramid tree."""

    def __init__(
        self,
        hyperplane_dim: int,
        coordinates: Tuple[float, ...],
        depth: int,
        general_rng: np.random.Generator,
        reward_specs: Sequence[RewardSpec],
    ) -> None:
        self.hyperplane_dim = hyperplane_dim
        self.coordinates = coordinates
        self.depth = depth
        self.features: List[Tuple[float, float, float, float]] | None = None
        self.visit_count = 0

        self.reward_type: Optional[str] = None
        self.reward_mu = 0.0
        self.reward_var = 0.0
        self.general_rng = general_rng

        self._set_reward_condition(reward_specs)

    def _set_reward_condition(self, reward_specs: Sequence[RewardSpec]) -> None:
        layer_depth = self.depth + 1  # reward definitions are 1-indexed
        for spec in reward_specs:
            if spec.depth != layer_depth:
                continue

            if spec.coordinates is None:
                match = True
            else:
                indices = list(spec.coordinates)
                if len(indices) != self.hyperplane_dim:
                    match = False
                else:
                    arrays = [np.linspace(-layer_depth / 2, layer_depth / 2, layer_depth + 1) for _ in range(self.hyperplane_dim)]
                    target_vals = np.array([arrays[ax][indices[ax]] for ax in range(self.hyperplane_dim)], dtype=float)
                    node_vals = np.array(self.coordinates[: self.hyperplane_dim], dtype=float)
                    match = np.allclose(node_vals, target_vals, atol=1e-9)

            if match:
                self.reward_type = spec.reward_type
                self.reward_mu = spec.value
                self.reward_var = spec.variance
                return

    def sample_reward(self) -> float:
        if self.reward_type == "fix":
            return float(self.reward_mu)
        if self.reward_type == "normal":
            return float(self.general_rng.normal(self.reward_mu, np.sqrt(max(self.reward_var, 1e-12))))
        if self.reward_type == "binominal":  # typo preserved from original specification
            return 1.0 if self.general_rng.random() < self.reward_mu else 0.0
        return 0.0

    def count(self) -> None:
        self.visit_count += 1


class ImgPyramidEnv:
    """Deterministic tree-structured image environment."""

    def __init__(
        self,
        image_size: int = 64,
        sigma_range: Tuple[float, float] = (1.0, 5.0),
        coeff_range: Tuple[float, float] = (-1.0, 1.0),
        *,
        depth: int,
        hyperplane_dim: int,
        start_random: bool,
        reward_specs: Sequence[RewardSpec],
        general_seed: int,
        feature_seed: int,
        obs_noise_std: float = 0.0,
    ) -> None:
        self.image_size = int(image_size)
        self.sigma_range = sigma_range
        self.coeff_range = coeff_range
        self.depth = int(depth)
        self.hyperplane_dim = int(hyperplane_dim)
        self.start_random = bool(start_random)
        self.reward_specs = reward_specs

        dt_now = datetime.datetime.now()
        if general_seed < 0:
            general_seed = dt_now.hour * 3600 + dt_now.minute * 60 + dt_now.second
        if feature_seed < 0:
            feature_seed = dt_now.microsecond

        self.general_seed = int(general_seed)
        self.feature_seed = int(feature_seed)
        self.general_rng = np.random.default_rng(self.general_seed)
        self.feature_rng = np.random.default_rng(self.feature_seed)
        self.obs_noise_std = float(max(obs_noise_std, 0.0))

        self.action_space_size = 2 ** self.hyperplane_dim

        self.nodes, self.node_dict = self._generate_nodes()
        self.node_images, self.gaussian_params = self._generate_images(
            sigma_range, coeff_range
        )

        self._validate_reward_specs_reachability()

        self.current_node_idx: Optional[int] = None
        self.total_reward = 0.0
        self.done = False

    # ------------------------------------------------------------------
    # graph construction utilities
    # ------------------------------------------------------------------

    def _generate_nodes(self) -> Tuple[List[Node], Dict[Tuple[float, ...], int]]:
        nodes: List[Node] = []
        node_dict: Dict[Tuple[float, ...], int] = {}
        index = 0
        for depth in range(self.depth):
            layer_size = depth + 1
            arrays = [np.linspace(-layer_size / 2, layer_size / 2, layer_size + 1) for _ in range(self.hyperplane_dim)]
            grid = np.meshgrid(*arrays, indexing="ij")
            flat_coords = np.stack([g.flatten() for g in grid], axis=-1)
            depth_column = np.full((flat_coords.shape[0], 1), float(depth))
            layer_coords = np.hstack([flat_coords, depth_column])
            for coord in layer_coords:
                coord_tuple = tuple(coord.tolist())
                node = Node(
                    self.hyperplane_dim,
                    coord_tuple,
                    depth,
                    self.general_rng,
                    self.reward_specs,
                )
                nodes.append(node)
                node_dict[coord_tuple] = index
                index += 1
        return nodes, node_dict

    def _generate_images(
        self,
        sigma_range: Tuple[float, float],
        coeff_range: Tuple[float, float],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[float, float, float, float]]]]:
        gaussian_params: List[List[Tuple[float, float, float, float]] | None] = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            if node.depth == 0:
                params: List[Tuple[float, float, float, float]] = []
                for _ in range(self.action_space_size):
                    mu_x = self.feature_rng.uniform(0, self.image_size)
                    mu_y = self.feature_rng.uniform(0, self.image_size)
                    sigma = self.feature_rng.uniform(*sigma_range)
                    coeff = self.feature_rng.uniform(*coeff_range)
                    params.append((mu_x, mu_y, sigma, coeff))
                gaussian_params[i] = params
                node.features = params

        for depth in range(1, self.depth):
            parent_nodes = [n for n in self.nodes if n.depth == depth - 1]
            child_nodes = [n for n in self.nodes if n.depth == depth]

            for child in child_nodes:
                inherited_params: List[Tuple[float, float, float, float]] = []
                valid_parents = [
                    p
                    for p in parent_nodes
                    if np.sum(np.abs(np.array(p.coordinates[: self.hyperplane_dim]) - np.array(child.coordinates[: self.hyperplane_dim])))
                    == 0.5 * self.hyperplane_dim
                ]
                share_per_parent = 1
                for parent in valid_parents:
                    if parent.features is None:
                        continue
                    chosen = self.feature_rng.choice(len(parent.features), size=share_per_parent, replace=False)
                    for c in chosen:
                        inherited_params.append(parent.features[c])

                while len(inherited_params) < self.action_space_size:
                    mu_x = self.feature_rng.uniform(0, self.image_size)
                    mu_y = self.feature_rng.uniform(0, self.image_size)
                    sigma = self.feature_rng.uniform(*sigma_range)
                    coeff = self.feature_rng.uniform(*coeff_range)
                    inherited_params.append((mu_x, mu_y, sigma, coeff))

                idx = self.node_dict[tuple(child.coordinates)]
                gaussian_params[idx] = inherited_params
                child.features = inherited_params

        node_images: List[np.ndarray] = []
        for params in gaussian_params:
            if params is None:
                img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                img = self._generate_gaussian_image(self.image_size, params)
            node_images.append(img[np.newaxis, :, :])
        return node_images, gaussian_params

    def export_state_vectors_to_directory(self, base_dir: str) -> bool:
        try:
            self._export_state_vectors(base_dir)
            return True
        except Exception:
            return False

    def _export_state_vectors(self, base_dir: str) -> None:
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to create state_vec directory '{base_dir}': {exc}") from exc
        file_path = os.path.join(base_dir, "input_state_vec.csv")
        size = self.image_size * self.image_size
        header = ["node_id"] + [f"pixel_{i}" for i in range(size)]
        with open(file_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for idx, img in enumerate(self.node_images):
                flat = np.asarray(img, dtype=np.float32).reshape(-1)
                writer.writerow([idx] + [f"{float(v):.8f}" for v in flat])

    def _validate_reward_specs_reachability(self) -> None:
        if not self.reward_specs:
            return

        start_nodes = [idx for idx, node in enumerate(self.nodes) if node.depth == 0]
        if not start_nodes:
            return

        nodes_by_depth: Dict[int, List[int]] = defaultdict(list)
        for idx, node in enumerate(self.nodes):
            nodes_by_depth[node.depth].append(idx)

        def reachable_from(start_idx: int) -> set[int]:
            stack = [start_idx]
            seen: set[int] = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                node = self.nodes[current]
                next_depth = node.depth + 1
                if next_depth >= self.depth:
                    continue
                for child_idx in nodes_by_depth[next_depth]:
                    child = self.nodes[child_idx]
                    diff = np.abs(
                        np.array(child.coordinates[: self.hyperplane_dim])
                        - np.array(node.coordinates[: self.hyperplane_dim])
                    )
                    if np.sum(diff) == 0.5 * self.hyperplane_dim:
                        stack.append(child_idx)
            return seen

        reachable_sets = {start: reachable_from(start) for start in start_nodes}

        for spec in self.reward_specs:
            if spec.coordinates is None:
                continue

            layer_depth = int(spec.depth)
            if layer_depth <= 0 or layer_depth > self.depth:
                raise ValueError(
                    f"reward depth {layer_depth} is outside the valid range [1, {self.depth}]"
                )

            try:
                arrays = [
                    np.linspace(-layer_depth / 2, layer_depth / 2, layer_depth + 1)
                    for _ in range(self.hyperplane_dim)
                ]
                coord_values = [arrays[ax][int(spec.coordinates[ax])] for ax in range(self.hyperplane_dim)]
            except (IndexError, ValueError, TypeError) as exc:
                raise ValueError(
                    "reward coordinates must be integer indices within the layer grid"
                ) from exc

            target_coord = tuple(float(v) for v in coord_values) + (float(layer_depth - 1),)
            node_idx = self.node_dict.get(target_coord)
            if node_idx is None:
                raise ValueError(
                    f"no node matches reward coordinates {spec.coordinates} at depth {layer_depth}"
                )

            if not all(node_idx in reachable for reachable in reachable_sets.values()):
                raise ValueError(
                    "reward specification is not reachable from every initial depth-0 node: "
                    f"depth={layer_depth}, coordinates={spec.coordinates}"
                )

    def _generate_gaussian_image(
        self, image_size: int, params: Sequence[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        x = np.arange(image_size, dtype=np.float32)
        y = np.arange(image_size, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        feature_map = np.zeros((image_size, image_size), dtype=np.float32)
        for mu_x, mu_y, sigma, coeff in params:
            gauss = np.exp(-((X - mu_x) ** 2 + (Y - mu_y) ** 2) / (2 * sigma**2))
            feature_map += coeff * gauss
        return np.clip(feature_map, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # interaction helpers
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        start_candidates = [i for i, node in enumerate(self.nodes) if node.depth == 0]
        if self.start_random and start_candidates:
            self.current_node_idx = int(self.feature_rng.choice(start_candidates))
        else:
            self.current_node_idx = start_candidates[0] if start_candidates else 0
        self.nodes[self.current_node_idx].count()
        self.total_reward = 0.0
        self.done = False
        return self._observe(self.current_node_idx)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.current_node_idx is None:
            raise RuntimeError("Environment must be reset before stepping")
        current_node = self.nodes[self.current_node_idx]
        depth = current_node.depth

        if depth >= self.depth - 1:
            self.done = True
            return self._observe(self.current_node_idx), 0.0, True

        next_depth = depth + 1
        candidates = [
            idx
            for idx, node in enumerate(self.nodes)
            if node.depth == next_depth
            and np.sum(
                np.abs(
                    np.array(node.coordinates[: self.hyperplane_dim])
                    - np.array(current_node.coordinates[: self.hyperplane_dim])
                )
            )
            == 0.5 * self.hyperplane_dim
        ]

        if not candidates:
            self.done = True
            return self._observe(self.current_node_idx), 0.0, True

        chosen_idx = candidates[action % len(candidates)]
        self.current_node_idx = chosen_idx
        self.nodes[self.current_node_idx].count()

        node = self.nodes[self.current_node_idx]
        reward = float(node.sample_reward())
        self.total_reward += reward
        self.done = node.depth == self.depth - 1
        return self._observe(self.current_node_idx), reward, self.done

    def _observe(self, node_idx: int) -> np.ndarray:
        base = self.node_images[node_idx]
        if self.obs_noise_std <= 0.0:
            return base.copy()
        noise = self.general_rng.normal(0.0, self.obs_noise_std, size=base.shape)
        noisy = np.clip(base + noise, -1.0, 1.0)
        return noisy.astype(np.float32)


class ImgSPGymEnv(Env):
    """Gymnasium wrapper around :class:`ImgPyramidEnv`."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        args: Optional[Dict[str, Any]] = None,
        *,
        image_size: int = 64,
        sigma_range: Tuple[float, float] = (1.0, 5.0),
        coeff_range: Tuple[float, float] = (-1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__()

        merged_args: Dict[str, Any] = dict(args or {})
        merged_args.update(kwargs)

        param = dict(merged_args.get("param", {}))
        depth = int(param.get("depth", 6))
        hyperplane_dim = int(param.get("hyperplane_dim", 2))
        start_random = bool(param.get("start_random", True))
        general_seed = int(param.get("general_seed", -1))
        features_cfg = dict(param.get("features", {}))
        feature_seed = int(features_cfg.get("seed", -1))

        reward_specs = self._parse_reward_spec(param.get("rewards", {}), depth, hyperplane_dim)

        obs_noise_std = float(merged_args.get("obs_noise_std", param.get("obs_noise_std", 0.0)))

        self.env = ImgPyramidEnv(
            image_size=int(merged_args.get("image_size", image_size)),
            sigma_range=tuple(merged_args.get("sigma_range", sigma_range)),
            coeff_range=tuple(merged_args.get("coeff_range", coeff_range)),
            depth=depth,
            hyperplane_dim=hyperplane_dim,
            start_random=start_random,
            reward_specs=reward_specs,
            general_seed=general_seed,
            feature_seed=feature_seed,
            obs_noise_std=obs_noise_std,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, self.env.image_size, self.env.image_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.env.action_space_size)

        self._np_random = np.random.default_rng(general_seed if general_seed >= 0 else None)
        save_dir = merged_args.get("save_initial_observations", None)
        if save_dir:
            try:
                base_path = str(save_dir)
                self.save_all_observations(base_path)
            except Exception:
                pass
        self.state_action_counts = np.zeros(
            (len(self.env.nodes), self.env.action_space_size), dtype=np.int64
        )

    # Gymnasium API -------------------------------------------------

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        obs = self.env.reset()
        return obs.astype(np.float32), {"depth": 1}

    def step(self, action: int):
        current_idx = getattr(self.env, "current_node_idx", None)
        if current_idx is not None:
            try:
                node = self.env.nodes[current_idx]
                if int(node.depth) < int(self.env.depth) - 1:
                    a_idx = int(action) % self.env.action_space_size
                    self.state_action_counts[current_idx, a_idx] += 1
            except Exception:
                pass
        obs, reward, done = self.env.step(int(action))
        info = {"depth": self.env.nodes[self.env.current_node_idx].depth + 1 if self.env.current_node_idx is not None else 0}
        return obs.astype(np.float32), reward, done, False, info

    def render(self):
        if self.env.current_node_idx is None:
            return None
        img = self.env.node_images[self.env.current_node_idx][0]
        # Convert to RGB array by stacking channels
        return np.repeat(img[..., None], 3, axis=2)

    def close(self) -> None:
        pass

    # Helpers -------------------------------------------------------

    def _parse_reward_spec(
        self,
        rewards_cfg: Dict[str, Any],
        depth: int,
        hyperplane_dim: int,
    ) -> List[RewardSpec]:
        if not rewards_cfg:
            return []

        depths = list(rewards_cfg.get("depth", []))
        coordinates_list = rewards_cfg.get("coordinates", [])
        values = rewards_cfg.get("value", rewards_cfg.get("values", []))
        types = rewards_cfg.get("type", [])
        variances = rewards_cfg.get("var", [])

        specs: List[RewardSpec] = []
        for idx, reward_depth in enumerate(depths):
            coord = None
            if idx < len(coordinates_list):
                coord = tuple(int(c) for c in coordinates_list[idx])
                if len(coord) != hyperplane_dim:
                    raise ValueError("reward coordinates must match hyperplane_dim")
            value = float(values[idx]) if idx < len(values) else 1.0
            reward_type = types[idx] if idx < len(types) else "fix"
            variance = float(variances[idx]) if idx < len(variances) else 0.0
            specs.append(
                RewardSpec(
                    depth=int(reward_depth),
                    coordinates=coord,
                    value=value,
                    reward_type=reward_type,
                    variance=variance,
                )
            )
        return specs

    # Convenience utilities used by evaluation/logging ----------------

    def export_all_observations(self) -> Optional["torch.Tensor"]:
        if torch is None:
            return None
        all_obs = np.array(self.env.node_images, dtype=np.float32)
        return torch.from_numpy(all_obs)

    def export_state_action_counts(self, base_dir: str, step_label: int) -> bool:
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create env_status_actions_log directory '{base_dir}': {exc}"
            ) from exc
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(
            base_dir,
            f"sp_img_train_stateaction_step_{int(step_label)}_{timestamp}.csv",
        )
        counts = np.asarray(self.state_action_counts, dtype=np.int64)
        header = ["node_id", "depth", "action_id", "count"]
        try:
            with open(filename, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for node_id, node in enumerate(self.env.nodes):
                    if int(node.depth) >= int(self.env.depth) - 1:
                        continue
                    depth = int(node.depth) + 1
                    for action_id in range(self.env.action_space_size):
                        writer.writerow(
                            [
                                str(node_id),
                                str(depth),
                                str(action_id),
                                str(int(counts[node_id, action_id])),
                            ]
                        )
            return True
        except Exception as exc:
            raise RuntimeError(f"Failed to write state-action counts: {exc}") from exc

    def save_all_observations(self, base_path: str) -> bool:
        if _PIL_Image is None:
            return False

        try:
            base = Path(base_path)
            base.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(self.env.node_images):
                arr = np.clip((img[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
                _PIL_Image.fromarray(arr, mode="L").save(base / f"node_{idx:04d}.png")
            return True
        except Exception:
            return False

    # Logging utilities ------------------------------------------------

    def fprint_env_status(
        self,
        role: str,
        episode: Optional[int] = None,
        worker_id: Optional[int] = None,
        base_dir: Optional[str] = None,
        step_count: Optional[int] = None,
    ) -> bool:
        if base_dir is None:
            base_dir = os.getcwd()
        dirname = os.path.join(base_dir, "env_status_log")
        os.makedirs(dirname, exist_ok=True)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        role_map = {"t": "train_", "g": "generation_", "e": "evaluation_"}
        role_prefix = role_map.get(role, "")
        label = step_count if step_count is not None else (episode if episode is not None else 0)
        logname = f"sp_img_{role_prefix}step{label}_{now}.csv"
        filename = os.path.join(dirname, logname)

        layer_idx = defaultdict(int)
        action_num = self.env.action_space_size

        with open(filename, "w", encoding="utf-8") as file:
            file.write("general_seed, feature_seed\n")
            file.write(f"{self.env.general_seed}, {self.env.feature_seed}\n")

            coord_headers = [f"coordinate_{i}" for i in range(len(self.env.nodes[0].coordinates))]
            base_columns = ["index", "depth", "layer_index", *coord_headers, "reward", "visit_count"]
            feat_headers = []
            for i in range(action_num):
                feat_headers.extend([f"mu_x_{i}", f"mu_y_{i}", f"sigma_{i}", f"coeff_{i}"])
            file.write(",".join(base_columns + feat_headers) + "\n")

            for idx, node in enumerate(self.env.nodes):
                depth_out = node.depth + 1
                layer_idx[depth_out] += 1
                reward_out = "-" if node.reward_type is None else f"{node.reward_mu}"
                coord_values = [f"{float(v):.12g}" for v in node.coordinates]

                row = [
                    str(idx),
                    str(depth_out),
                    str(layer_idx[depth_out] - 1),
                    *coord_values,
                    reward_out,
                    str(node.visit_count),
                ]
                feature_info = node.features or []
                for j in range(action_num):
                    if j < len(feature_info):
                        mu_x, mu_y, sigma, coeff = feature_info[j]
                        row.extend([
                            f"{float(mu_x):.12g}",
                            f"{float(mu_y):.12g}",
                            f"{float(sigma):.12g}",
                            f"{float(coeff):.12g}",
                        ])
                    else:
                        row.extend(["", "", "", ""])
                file.write(", ".join(row) + "\n")

        return True

    def fprint_env_rnd(
        self,
        logname: str,
        step: int,
        all_intrinsic: Any,
        base_dir: Optional[str] = None,
    ) -> bool:
        """ Log IntrisicReward per node to CSV (one column per step).
        
        - If the file does not exist, create a new file with the following columns:
          idx, depth, layer_idx, step{step}_ir
        """
        if base_dir is None:
            base_dir = os.getcwd()
        dirname = os.path.join(base_dir, "env_status_log")
        filename = os.path.join(dirname, logname)

        col = f"step{int(step)}_ir"

        if isinstance(all_intrinsic, torch.Tensor):
            ir_np = all_intrinsic.detach().float().cpu().numpy().reshape(-1)
        else:
            ir_np = np.asarray(all_intrinsic, dtype=np.float32).reshape(-1)

        layer_idx = defaultdict(int)

        os.makedirs(dirname, exist_ok=True)

        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8", newline="") as f:
                f.write(f"idx,depth,layer_idx,{col}\n")

                lines: List[str] = []
                for idx, v in enumerate(ir_np):
                    node = self.env.nodes[idx]
                    depth_out = int(node.depth) + 1  # 1-indexed
                    layer_idx[depth_out] += 1
                    lines.append(f"{idx},{depth_out},{layer_idx[depth_out]-1},{float(v):.8f}")

                f.write("\n".join(lines) + "\n")
            return True

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        if not lines:
            with open(filename, "w", encoding="utf-8", newline="") as f:
                f.write(f"idx,depth,layer_idx,{col}\n")
                lines_out: List[str] = []
                for idx, v in enumerate(ir_np):
                    node = self.env.nodes[idx]
                    depth_out = int(node.depth) + 1
                    layer_idx[depth_out] += 1
                    lines_out.append(f"{idx},{depth_out},{layer_idx[depth_out]-1},{float(v):.8f}")
                f.write("\n".join(lines_out) + "\n")
            return True

        header = [h.strip() for h in lines[0].split(",")]
        header.append(col)

        new_lines = [",".join(header)]
        for i, line in enumerate(lines[1:]):
            val = f"{float(ir_np[i]):.8f}" if i < len(ir_np) else ""
            new_lines.append(line + "," + val)

        with open(filename, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(new_lines) + "\n")

        return True