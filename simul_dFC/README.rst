============================================
PydFC: simul_dFC Module Documentation
============================================

The ``simul_dFC`` module generates **synthetic task-based fMRI data** for benchmarking dFC methods under controlled conditions.

It uses `The Virtual Brain (TVB) <https://www.thevirtualbrain.org>`_ simulator to produce BOLD signals driven by a known task design, allowing ground-truth evaluation of dFC methods.

Two task paradigms are supported:

*   **Real task-derived** (``tasks_info_ds003465.json``) — task timing extracted from an OpenNeuro dataset (ds003465) to drive the simulation.
*   **Synthetic pulse-train** (``tasks_info_pulseTrain.json``) — parametric block designs with configurable onset, duration, and frequency.

Running
-------

Set ``VENV_PATH`` and ``PYDFC_CODE_DIR`` in the cluster configuration block at the top of the job script, then submit::

    # SLURM
    sbatch --array=1-N run_scripts_slurm/run_simulator.sh

    # SGE
    qsub -t 1-N run_scripts_sge/run_simulator.sh

The script expects a ``subj_list.txt`` (one subject ID per line), a ``dataset_info.json``, and a ``tasks_info.json`` in the same directory as the run script.

Simulated outputs are consumed directly by the ``task_dFC`` pipeline starting at ``FCS_estimate.py``.
