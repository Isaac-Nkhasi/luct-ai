"""End-to-end Phase 8 + Phase 9 pipeline and CLI.

Usage
-----
    python -m luct_ai.pipeline clean --input data/raw/laptop_prices_raw.csv
    python -m luct_ai.pipeline train --input data/processed/laptop_prices_clean.csv
    python -m luct_ai.pipeline all   --input data/raw/laptop_prices_raw.csv

Outputs land under ``data/processed``, ``models`` and ``reports``.
Nothing is fabricated: if the input file has no rows the pipeline says so and
stops instead of producing fake metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import cleaning, evaluate, modeling

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# --------------------------------------------------------------------------- #
# Tiny markdown helpers (avoids a tabulate dependency)
# --------------------------------------------------------------------------- #


def _md_table(frame: pd.DataFrame, *, floatfmt: str = "{:.2f}") -> str:
    if frame is None or frame.empty:
        return "_(no rows)_"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else floatfmt.format(value))
            else:
                cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _source_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    if "source" not in frame.columns:
        return pd.DataFrame(columns=["source", "records"])
    return (
        frame.assign(source=frame["source"].fillna("(unspecified)"))
        .groupby("source").size().reset_index(name="records")
        .sort_values("records", ascending=False)
    )


# --------------------------------------------------------------------------- #
# Phase 8
# --------------------------------------------------------------------------- #


def run_cleaning(
    input_path: str | Path,
    *,
    rates_path: str | Path | None = None,
    output_root: Path = ROOT,
    drop_flagged: bool = False,
) -> cleaning.CleaningResult:
    input_path = Path(input_path)
    raw = cleaning.read_dataset(input_path)
    if raw.empty:
        raise SystemExit(
            f"{input_path} contains no data rows.\n"
            "Collect the dataset first (see docs/data_collection_plan.md), then re-run.\n"
            "No records have been invented for you."
        )

    rates = cleaning.load_fx_rates(rates_path) if rates_path else \
        cleaning.load_fx_rates(DATA_DIR / "fx_rates.csv")

    result = cleaning.clean_dataset(raw, rates=rates, drop_flagged=drop_flagged)

    processed_dir = output_root / "data" / "processed"
    reports_dir = output_root / "reports"
    cleaned_path = cleaning.write_dataset(result.frame, processed_dir / "laptop_prices_clean.csv")
    result.write_report(reports_dir / "cleaning_report.json")

    report_md = [
        "# Phase 8 — data cleaning report",
        "",
        f"- Input: `{input_path}`",
        f"- Cleaned output: `{cleaned_path}`",
        f"- Records after cleaning: **{result.usable_records}**",
        f"- Duplicates removed: **{len(result.duplicates_removed)}**",
        "",
        "## Notes",
        *[f"- {note}" for note in result.notes],
        "",
        "## Missing values (before cleaning)",
        _md_table(result.missingness_before),
        "",
        "## Missing values (after cleaning)",
        _md_table(result.missingness_after),
        "",
        "## Quality flags",
        _md_table(result.flag_counts),
        "",
        "## Currency conversion",
        _md_table(result.currency_summary, floatfmt="{:.0f}"),
        "",
        "## Split sizes",
        _md_table(result.split_summary, floatfmt="{:.0f}"),
        "",
        "## Source breakdown",
        _md_table(_source_breakdown(result.frame), floatfmt="{:.0f}"),
        "",
    ]
    (reports_dir / "cleaning_report.md").write_text("\n".join(report_md), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Phase 9
# --------------------------------------------------------------------------- #


def run_training(
    input_path: str | Path,
    *,
    output_root: Path = ROOT,
    include_model: bool = False,
    tune: bool = True,
) -> modeling.TrainResult:
    input_path = Path(input_path)
    cleaned = pd.read_csv(input_path, dtype="string")
    if cleaned.empty:
        raise SystemExit(f"{input_path} contains no data rows — nothing to train on.")

    result = modeling.train_models(cleaned, include_model=include_model, tune=tune)

    models_dir = output_root / "models"
    figures_dir = output_root / "reports" / "figures"
    reports_dir = output_root / "reports"

    paths = modeling.save_model_bundle(result, models_dir, cleaned=cleaned)

    # Figures.
    if result.test_truth.size:
        evaluate.save_figure(
            evaluate.actual_vs_predicted_plot(
                result.test_truth, result.test_predictions[result.best_model_name],
                title=f"Actual vs predicted — {result.best_model_name} (test set)"),
            figures_dir / "actual_vs_predicted.png")

        band_table = evaluate.error_by_band(
            result.test_truth, result.test_predictions[result.best_model_name], result.test_bands)
        evaluate.save_figure(evaluate.error_by_band_plot(band_table),
                             figures_dir / "error_by_price_band.png")
    else:
        band_table = pd.DataFrame(columns=["band", "n", "mae", "rmse"])

    train_only = cleaned[modeling.usable_mask(cleaned) & (cleaned["split"] != "test")]
    curve = evaluate.learning_curve_data(
        result.best_model,
        modeling.feature_frame(train_only, include_model=include_model),
        pd.to_numeric(train_only[modeling.TARGET_COLUMN], errors="coerce"),
        groups=modeling.group_keys(train_only),
    )
    evaluate.save_figure(evaluate.learning_curve_plot(curve), figures_dir / "learning_curve.png")

    if result.feature_importances is not None:
        evaluate.save_figure(
            evaluate.feature_importance_plot(result.feature_importances),
            figures_dir / "feature_importance.png")

    report = {
        **result.metadata,
        "model_file": str(paths["model"]),
        "version_file": str(paths["version"]),
        "test_error_by_band": band_table.to_dict(orient="records"),
        "learning_curve": {
            "train_sizes": curve.train_sizes.tolist(),
            "train_mae": curve.train_mae.tolist(),
            "val_mae": curve.val_mae.tolist(),
        },
        "source_breakdown": _source_breakdown(cleaned).to_dict(orient="records"),
    }
    (reports_dir / "model_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    report_md = [
        "# Phase 9 — model report",
        "",
        f"- Data: `{input_path}` ({len(cleaned)} records)",
        f"- Best model: **{result.best_model_name}**",
        f"- Trained at: {result.metadata['trained_at']}",
        f"- scikit-learn: {result.metadata['sklearn_version']} | Python: {result.metadata['python_version']}",
        f"- Split sizes — train {result.metadata['n_train']} / val {result.metadata['n_val']} / test {result.metadata['n_test']}",
        "",
        "MAE and RMSE are in maloti (LSL); lower is better. R² is closer to 1 for better fit.",
        "",
        "## Model comparison (validation)",
        _md_table(result.comparison_val),
        "",
        "## Model comparison (held-out test — untouched during tuning)",
        _md_table(result.comparison_test),
        "",
        "## Test error by price band",
        _md_table(band_table, floatfmt="{:.0f}"),
        "",
        "## Figures",
        "- `reports/figures/actual_vs_predicted.png`",
        "- `reports/figures/error_by_price_band.png`",
        "- `reports/figures/learning_curve.png`",
        "- `reports/figures/feature_importance.png`",
        "",
        "## Limitations",
        "- Metrics are only as trustworthy as the sample size; a few hundred records "
        "gives wide confidence intervals.",
        "- Prices are **asking** prices unless the dataset was filtered to completed "
        "sales; asking and completed prices are kept separate.",
        "- Accuracy is reported as MAE/RMSE/R², not as a single 'accuracy' percentage.",
        "",
    ]
    (reports_dir / "model_report.md").write_text("\n".join(report_md), encoding="utf-8")

    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luct_ai.pipeline",
                                     description="LUCT Market AI — Phase 8/9 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="Phase 8: clean and prepare the dataset")
    clean.add_argument("--input", required=True, help="raw dataset CSV")
    clean.add_argument("--rates", default=None, help="currency->LSL rate CSV (optional)")
    clean.add_argument("--drop-flagged", action="store_true",
                       help="remove flagged records from the output")

    train = sub.add_parser("train", help="Phase 9: train and compare the models")
    train.add_argument("--input", default=str(PROCESSED_DIR / "laptop_prices_clean.csv"))
    train.add_argument("--include-model", action="store_true",
                       help="include the high-cardinality 'model' column as a feature")
    train.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")

    both = sub.add_parser("all", help="run cleaning then training")
    both.add_argument("--input", required=True)
    both.add_argument("--rates", default=None)
    both.add_argument("--include-model", action="store_true")
    both.add_argument("--no-tune", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "clean":
        result = run_cleaning(args.input, rates_path=args.rates, drop_flagged=args.drop_flagged)
        print(f"cleaned: {result.usable_records} records -> data/processed/laptop_prices_clean.csv")
        for note in result.notes:
            print(f"  - {note}")
        return 0

    if args.command == "train":
        result = run_training(args.input, include_model=args.include_model, tune=not args.no_tune)
        print(f"best model: {result.best_model_name}")
        print(result.comparison_test.to_string(index=False))
        return 0

    if args.command == "all":
        result = run_cleaning(args.input, rates_path=args.rates)
        train_result = run_training(PROCESSED_DIR / "laptop_prices_clean.csv",
                                    include_model=args.include_model, tune=not args.no_tune)
        print(f"cleaned {result.usable_records} records; best model: {train_result.best_model_name}")
        print(train_result.comparison_test.to_string(index=False))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
