# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, Optional, List, Tuple
from .motion_loader import MotionLoader


class MultiMotionLoader:
    r"""Load *multiple* motion clips that share同一拓扑(DOF/body 名称)。
    每一次 ``sample()`` 时随机挑选一个 clip 再走单文件的插值逻辑。
    
    Args:
        motion_files:   一组 joblib/npy Motion 文件路径；顺序不影响采样概率。
        device:         torch 设备。
        indices:        对每个文件想加载的 clip 索引；None=都取 0（与旧接口对齐）。
        clip_prob:      采样时各 clip 被选中的概率；None=均匀。
    """
    def __init__(
        self,
        motion_lists: str,
        device: torch.device,
        clip_prob: Optional[Sequence[float]] = None,
    ) -> None:

        # ---- 解析 motion_lists ----
        motion_files = []
        indices = []

        with open(motion_lists, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != 2:
                    continue
                motion_files.append(tokens[0])
                indices.append(tokens[1])
        self.device = device
        self.loaders: List[MotionLoader] = []
        self.clip_prob = (
            torch.tensor(clip_prob, dtype=torch.float32, device=device)
            if clip_prob is not None
            else None
        )

        # ---- 批量构造单文件 loader ----
        for fpath, idx in zip(motion_files, indices):
            self.loaders.append(MotionLoader(fpath, device, idx))

        # ---- 一致性检查 (dof/body names) ----
        ref_dofs = self.loaders[0].dof_names
        ref_bodies = self.loaders[0].body_names
        for ld in self.loaders[1:]:
            assert ld.dof_names == ref_dofs,  "DOF names mismatch across clips"
            assert ld.body_names == ref_bodies, "Body names mismatch across clips"

        self._dof_names = ref_dofs
        self._body_names = ref_bodies
        self.dt = self.loaders[0].dt
        # record meta for fast access
        self.num_dofs  = len(ref_dofs)
        self.num_bodies = len(ref_bodies)
        self.num_clips = len(self.loaders)

    @property
    def dof_names(self):
        return self._dof_names

    @property
    def body_names(self):
        return self._body_names

    def _rand_clip_ids(self, n: int) -> torch.Tensor:
        # 返回一个随机采样的 clip id 列表，长度为 n。
        if self.clip_prob is None:
            probs = torch.ones(self.num_clips, device=self.device)
        else:
            probs = self.clip_prob / self.clip_prob.sum()
        return torch.multinomial(probs, num_samples=n, replacement=True)

    def sample_times(self, num_samples: int, *, clip_ids: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """同时返回*(times, clip_ids)*，供外部可重用。"""
        clip_ids = clip_ids if clip_ids is not None else self._rand_clip_ids(num_samples)
        times = np.zeros(num_samples, dtype=np.float32)
        for k in range(self.num_clips):
            mask = (clip_ids == k).cpu().numpy()
            if mask.any():
                times[mask] = self.loaders[k].sample_times(mask.sum())
        return times, clip_ids

    def sample(self, num_samples: int, *, times: Optional[np.ndarray] = None, clip_ids: Optional[torch.Tensor] = None):
        if times is None or clip_ids is None:
            times, clip_ids = self.sample_times(num_samples)

        dof_p = torch.zeros(times.shape[0], self.num_dofs, device=self.device)
        dof_v = torch.zeros(times.shape[0], self.num_dofs, device=self.device)
        pos_b = torch.zeros(times.shape[0], self.num_bodies, 3, device=self.device)
        rot_b = torch.zeros(times.shape[0], self.num_bodies, 4, device=self.device)
        proj_g = torch.zeros(times.shape[0], 3, device=self.device)
        lin_b = torch.zeros(times.shape[0], self.num_bodies, 3, device=self.device)
        ang_b = torch.zeros(times.shape[0], self.num_bodies, 3, device=self.device)

        # 分 clip 插值填充
        for k, loader in enumerate(self.loaders):
            idx_mask = (clip_ids == k).cpu().numpy()
            if not idx_mask.any():
                continue
            local_times = times[idx_mask]
            res = loader.sample(num_samples=idx_mask.sum(), times=local_times)
            (dof_p[idx_mask], dof_v[idx_mask], pos_b[idx_mask], 
             rot_b[idx_mask], proj_g[idx_mask], lin_b[idx_mask], ang_b[idx_mask]) = res

        return (dof_p, dof_v, pos_b, rot_b, proj_g, lin_b, ang_b)

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes    

if __name__ == "__main__":
    loader = MultiMotionLoader(
        ["CMU02.pkl", "CMU13.pkl"],
        indices=[0, 0],
        device=torch.device("cpu")
    )
    dof_p, *_ = loader.sample(8)
    print(dof_p.shape)   # (8, num_dofs)
