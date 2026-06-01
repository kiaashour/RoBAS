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


def build_trial_splits(
    remaining_indices_after_train: np.ndarray,
    n_cal: int,
    n_test: int,
    num_trials: int,
    split_seed: int = BASE_SEED,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Build calibration and test index splits for each trial.

    Parameters
    ----------
    remaining_indices_after_train : np.ndarray
        Candidate indices available after selecting training data.
    n_cal : int
        Number of calibration points per trial.
    n_test : int
        Number of test points per trial.
    num_trials : int
        Number of trial splits to generate.
    split_seed : int, default=BASE_SEED
        Seed used to sample trial splits.

    Returns
    -------
    list[tuple[int, np.ndarray, np.ndarray]]
        List of ``(trial_idx, cal_indices, test_indices)`` tuples.
    """
    rng = np.random.RandomState(split_seed)
    tasks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for trial_idx in range(num_trials):
        cal_indices = rng.choice(remaining_indices_after_train, size=n_cal, replace=False)
        remaining_indices_after_cal = np.setdiff1d(
            remaining_indices_after_train, cal_indices, assume_unique=True
        )
        test_indices = rng.choice(remaining_indices_after_cal, size=n_test, replace=False)
        tasks.append((trial_idx, cal_indices, test_indices))
    return tasks


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
