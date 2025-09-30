# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, Optional, List, Tuple
from motion_loader import MotionLoader


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
        motion_files: Sequence[str],
        device: torch.device,
        indices: Optional[Sequence[int]],
        clip_prob: Optional[Sequence[float]] = None,
    ) -> None:

        assert len(motion_files) > 0, "motion_files cannot be empty"
        indices = indices or [0] * len(motion_files)
        assert len(indices) == len(motion_files), "`indices` length mismatch"

        self.device = device
        self.loaders: List[MotionLoader] = []
        self.clip_prob = (
            torch.tensor(clip_prob, dtype=torch.float32, device=device)
            if clip_prob is not None
            else None
        )

        # ---- 批量构造单文件 loader ----
        for fpath, idx in zip(motion_files, indices):
            assert os.path.isfile(fpath), f"{fpath} 不存在"
            self.loaders.append(MotionLoader(fpath, device, idx))

        # ---- 一致性检查 (dof/body names) ----
        ref_dofs = self.loaders[0].dof_names
        ref_bodies = self.loaders[0].body_names
        for ld in self.loaders[1:]:
            assert ld.dof_names == ref_dofs,  "DOF names mismatch across clips"
            assert ld.body_names == ref_bodies, "Body names mismatch across clips"

        self._dof_names = ref_dofs
        self._body_names = ref_bodies

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
        """
        回传与旧版 MotionLoader.sample 完全相同的 6 tuple，
        只是数据来自多 clip 混合。
        """
        if times is None or clip_ids is None:
            times, clip_ids = self.sample_times(num_samples)

        # 输出容器：拼接后同 dtype / device
        shape_dof  = (num_samples, self.num_dofs)
        shape_body = (num_samples, self.num_bodies)
        zeros = lambda *s: torch.zeros(*s, device=self.device)

        dof_p = zeros(*shape_dof)
        dof_v = zeros(*shape_dof)
        pos_b = zeros(*shape_body, 3)
        rot_b = zeros(*shape_body, 4)
        lin_b = zeros(*shape_body, 3)
        ang_b = zeros(*shape_body, 3)

        # 分 clip 插值填充
        for k, loader in enumerate(self.loaders):
            idx_mask = (clip_ids == k).cpu().numpy()
            if not idx_mask.any():
                continue
            local_times = times[idx_mask]
            res = loader.sample(num_samples=idx_mask.sum(), times=local_times)
            (dof_p[idx_mask], dof_v[idx_mask], pos_b[idx_mask], 
             rot_b[idx_mask], lin_b[idx_mask], ang_b[idx_mask]) = res

        return (dof_p, dof_v, pos_b, rot_b, lin_b, ang_b)
if __name__ == "__main__":
    loader = MultiMotionLoader(
        ["CMU02.pkl", "CMU13.pkl"],
        indices=[0, 0],
        device=torch.device("cpu")
    )
    dof_p, *_ = loader.sample(8)
    print(dof_p.shape)   # (8, num_dofs)
