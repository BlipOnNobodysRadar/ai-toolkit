"""Render the exported EDM2 weighting curve without loading a model."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    with open(args.csv, encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('The EDM2 CSV has no rows')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sigma = [float(row['sigma']) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(sigma, [float(row['prediction_weight']) for row in rows])
    axes[0].set_ylabel('Prediction weight')
    axes[1].plot(sigma, [float(row['log_variance']) for row in rows])
    axes[1].set_ylabel('Learned log variance')
    axes[1].set_xlabel('Canonical noise fraction (sigma)')
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle('EDM2 weighting: ' + Path(args.csv).parent.name)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
