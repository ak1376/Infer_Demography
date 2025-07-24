#!/usr/bin/env python3
import argparse, json, os, sys, pickle
from pathlib import Path

# >>> make src importable <<<
ROOT = Path(__file__).resolve().parents[1]   # project root (adjust if needed)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting_helpers import create_color_scheme


def main(experiment_config, out_dir: Path):
    num_params = len(experiment_config['priors'])
    color_scheme, main_colors = create_color_scheme(num_params)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'color_shades.pkl', 'wb') as f:
        pickle.dump(color_scheme, f)
    with open(out_dir / 'main_colors.pkl', 'wb') as f:
        pickle.dump(main_colors, f)
    print(f"Colors saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up color scheme for demographic model parameters.")
    parser.add_argument('--config',   required=True, help='Path to the experiment configuration JSON file.')
    parser.add_argument('--out-dir',  required=False, default=None,
                        help='Output directory for pickles (default: experiments/<model>/modeling)')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} does not exist.")

    experiment_config = json.load(open(args.config))
    default_out = Path(f"experiments/{experiment_config['demographic_model']}/modeling")
    out_dir = Path(args.out_dir) if args.out_dir else default_out

    main(experiment_config, out_dir)
