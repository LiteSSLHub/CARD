#!/usr/bin/env python3
"""
collect_results.py - Collect multiple model evalscope evaluation results into an Excel file.

Usage:
  python collect_results.py model_path1 model_path2 ... -o output.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def find_latest_version_dir(model_path: str) -> Path:
    """Find the latest version_* directory under model_path/evalscope/."""
    evalscope_dir = Path(model_path) / 'evalscope'
    if not evalscope_dir.exists():
        return None
    version_dirs = sorted(evalscope_dir.glob('version_*'), reverse=True)
    return version_dirs[0] if version_dirs else None


def load_model_results(version_dir: Path) -> pd.DataFrame:
    """
    Load results from version directory, return DataFrame with columns benchmark, dataset, mean, std.
    Prefer multi_run_summary.json (multi-run), fallback to summary.xlsx (single-run).
    """
    multi_json = version_dir / 'multi_run_summary.json'
    if multi_json.exists():
        with open(multi_json, encoding='utf-8') as f:
            data = json.load(f)
        rows = data.get('results', [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        run_cols = sorted([c for c in df.columns if c.startswith('run_')])
        keep_cols = ['benchmark', 'dataset', 'mean', 'std'] + run_cols
        df = df[[c for c in keep_cols if c in df.columns]]
        df['key'] = df['benchmark'] + ' / ' + df['dataset']
        return df.set_index('key').drop(columns=['benchmark', 'dataset'])

    summary_xlsx = version_dir / 'summary.xlsx'
    if summary_xlsx.exists():
        df = pd.read_excel(summary_xlsx)
        df['key'] = df['benchmark'] + ' / ' + df['dataset']
        df = df.set_index('key')[['score']].rename(columns={'score': 'mean'})
        df['std'] = 0.0
        return df

    return None


def collect(model_paths: list, output: str, num_runs: int):
    model_results = {}
    for mp in model_paths:
        name = Path(mp).name
        vdir = find_latest_version_dir(mp)
        if vdir is None:
            print(f"[Skip] evalscope result directory not found: {mp}")
            continue
        df = load_model_results(vdir)
        if df is None:
            print(f"[Skip] Unable to read results: {vdir}")
            continue
        model_results[name] = df
        print(f"[Read] {name} <- {vdir.name}")

    if not model_results:
        print("No summarizable results found, exiting.")
        sys.exit(1)

    # ---- Build comparison table ----
    # Union of all model benchmark/dataset rows
    all_keys = []
    seen = set()
    for df in model_results.values():
        for k in df.index:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Split key back to benchmark / dataset
    split = pd.Series(all_keys, name='key').str.split(' / ', n=1, expand=True)
    summary_df = pd.DataFrame({
        'benchmark': split[0].values,
        'dataset':   split[1].values,
    })

    for name, df in model_results.items():
        summary_df[f'{name}_mean'] = df['mean'].reindex(all_keys).values
        if num_runs > 1:
            summary_df[f'{name}_std%'] = (
                df['std'].reindex(all_keys) /
                df['mean'].reindex(all_keys).replace(0, float('nan')) * 100
            ).round(2).values
            run_cols = sorted([c for c in df.columns if c.startswith('run_')])
            for rc in run_cols:
                summary_df[f'{name}_{rc}'] = df[rc].reindex(all_keys).values

    # ---- Write Excel ----
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Each model gets a separate sheet (with full fields)
        for name, df in model_results.items():
            df_full = df.reset_index().rename(columns={'key': 'benchmark/dataset'})
            safe_name = name[:31]  # Excel sheet name max 31 chars
            df_full.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"\n[Done] Summary saved to: {output_path}")
    print(f"       Contains {len(model_results)} models x {len(all_keys)} evaluation metrics")


def main():
    parser = argparse.ArgumentParser(description='Collect multi-model evalscope evaluation results')
    parser.add_argument('model_paths', nargs='+', help='Model directory path list')
    parser.add_argument('-o', '--output', default='vllm_logs/batch_summary.xlsx',
                        help='Output Excel path (default: vllm_logs/batch_summary.xlsx)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of runs per model, adds std columns when >1 (default: 1)')
    args = parser.parse_args()
    collect(args.model_paths, args.output, args.num_runs)


if __name__ == '__main__':
    main()
