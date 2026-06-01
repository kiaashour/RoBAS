from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar

import numpy as np
import torch
from tqdm.auto import tqdm

BASE_SEED = 42

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


def set_global_random_seed(seed: int = BASE_SEED) -> None:
    """
    Set NumPy and PyTorch random seeds.

    Parameters
    ----------
    seed : int, default=BASE_SEED
        Seed value used for all RNGs.

    Returns
    -------
    None
        This function updates global RNG states in place.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_image_trial_splits(
    *,
    pool_size: int,
    n_cal: int | str,
    n_test: int,
    num_trials: int,
    split_seed: int = BASE_SEED,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], int]:
    """
    Build test and calibration index splits for each image-dataset trial.

    Parameters
    ----------
    pool_size : int
        Number of candidate points in the evaluation pool.
    n_cal : int or str
        Calibration-set size or ``"standard"``.
    n_test : int
        Number of test points per trial.
    num_trials : int
        Number of trial splits to generate.
    split_seed : int, default=BASE_SEED
        Seed used to sample trial splits.

    Returns
    -------
    tuple[list[tuple[int, np.ndarray, np.ndarray]], int]
        Trial tasks ``(trial_idx, test_indices, cal_indices)`` and resolved calibration size.
    """
    rng = np.random.RandomState(split_seed)
    all_indices = np.arange(pool_size)

    if num_trials <= 0:
        raise ValueError("num_trials must be >= 1.")

    tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    effective_n_cal = None if n_cal == "standard" else int(n_cal)

    for trial_idx in range(num_trials):
        test_indices = rng.choice(all_indices, size=n_test, replace=False)
        remaining_indices = np.setdiff1d(all_indices, test_indices)

        if effective_n_cal is None:
            effective_n_cal = int(len(remaining_indices))
        elif effective_n_cal > len(remaining_indices):
            raise ValueError(
                "Requested n_cal is infeasible for this split: "
                f"n_cal={effective_n_cal}, available={len(remaining_indices)}."
            )

        cal_indices = rng.choice(remaining_indices, size=effective_n_cal, replace=False)
        tasks.append((trial_idx, test_indices, cal_indices))

    assert effective_n_cal is not None
    return tasks, effective_n_cal


def run_trial_tasks(
    task_fn: Callable[[TaskT], ResultT],
    tasks: Iterable[TaskT],
    *,
    num_workers: int,
    desc: str | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    mp_start_method: str = "spawn",
) -> list[ResultT]:
    """
    Execute trial tasks sequentially or with multiprocessing workers.

    Parameters
    ----------
    task_fn : Callable[[TaskT], ResultT]
        Function applied to each task.
    tasks : Iterable[TaskT]
        Iterable of task objects.
    num_workers : int
        Number of workers. Use ``1`` for sequential execution.
    desc : str or None, optional
        Progress-bar label.
    initializer : Callable[..., None] or None, optional
        Worker initializer function.
    initargs : tuple, default=()
        Positional arguments passed to ``initializer``.
    mp_start_method : str, default="spawn"
        Multiprocessing start method or ``"default"``.

    Returns
    -------
    list[ResultT]
        Task outputs in task order.
    """
    task_list = list(tasks)
    if num_workers <= 1:
        return [task_fn(task) for task in tqdm(task_list, desc=desc)]

    if mp_start_method == "default":
        mp_context = None
    else:
        mp_context = mp.get_context(mp_start_method)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=initializer,
        initargs=initargs,
        mp_context=mp_context,
    ) as executor:
        return list(tqdm(executor.map(task_fn, task_list), total=len(task_list), desc=desc))
