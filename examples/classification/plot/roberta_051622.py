"""
Plot 1) spectral decay, 2) retrain curves.
"""

import fire

from swissknife import utils
import torch


# python -m classification.plot.roberta_051622 --task plot1
def plot1(
    ckpt_path=f"/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj/global_step_000002.pt"
):
    """Eigenvalues.

    Run on gvm.
    """
    ckpt = torch.load(ckpt_path)
    eigenvalues = ckpt["eigenvalues"]
    print(eigenvalues.size())


# python -m classification.plot.roberta_051622 --task plot2
def plot2(
    seeds=(42, 101, 20598, 90828, 9008),
    ranks=(10, 20, 100, None),
    base_dir="/Users/xuechenli/Desktop/dump_a100/privlm",
    dump_dir="./classification/plot",
    markers=('x', '^', '+', 'o'),
):
    """Retrain.

    Run locally.
    """
    errorbars = []
    for rank, marker in utils.zip_(ranks, markers):
        results = []
        for seed in seeds:
            output_dir = utils.join(
                f"{base_dir}/roberta_prompt_retrain_{rank}_{seed}/sst-2",
                'log_history.json'
            )
            record = utils.jload(output_dir)
            results.append([dumpi['dev']['eval_acc'] for dumpi in record])
            steps = [dumpi['step'] for dumpi in record]

        label = f"subspace rank={rank}" if rank is not None else "original"
        mu, si = utils.average_over_seed(results)
        errorbar = dict(x=steps, y=mu, yerr=si, label=label, marker=marker)
        errorbars.append(errorbar)

    img_path = utils.join(dump_dir, 'plot2')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="iteration", ylabel="SST-2 classification accuracy (dev)")
    )


# python -m classification.plot.roberta_051622 --task plot_all
def plot_all():
    plot1()
    plot2()


def main(task="plot2"):
    utils.runs_tasks(
        task=task,
        task_names=("plot_all", "plot1", "plot2"),
        task_callables=(plot_all, plot1, plot2)
    )


if __name__ == "__main__":
    fire.Fire(main)
