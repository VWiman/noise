import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    EVALUATION_SUMMARY_PATH,
    METRIC_COMPARISON_PATH,
    RESULTS_DIR,
    STAGE_ONE_HISTORY_DATA_PATH,
    STAGE_TWO_HISTORY_DATA_PATH,
    TILE_METRICS_PATH,
    WHOLE_IMAGE_METRICS_PATH,
)


VARIANT_NAMES = ("noisy", "stage_one", "stage_two")
VARIANT_LABELS = {
    "noisy": "Brusig baseline",
    "stage_one": "Efter steg 1",
    "stage_two": "Efter steg 2 (residual)",
}
METRIC_NAMES = ("mse", "mae", "psnr", "ssim")
METRIC_KEYS = tuple(
    f"{variant_name}_{metric_name}"
    for variant_name in VARIANT_NAMES
    for metric_name in METRIC_NAMES
)
COMPARISON_KEYS = (
    "stage_one_mse_vs_noisy_percent",
    "stage_two_mse_vs_noisy_percent",
    "stage_two_mse_vs_stage_one_percent",
    "stage_one_mae_vs_noisy_percent",
    "stage_two_mae_vs_noisy_percent",
    "stage_two_mae_vs_stage_one_percent",
    "stage_one_psnr_vs_noisy",
    "stage_two_psnr_vs_noisy",
    "stage_two_psnr_vs_stage_one",
    "stage_one_ssim_vs_noisy",
    "stage_two_ssim_vs_noisy",
    "stage_two_ssim_vs_stage_one",
)


# ============================================================
# Beräkna rekonstruktionsmått per bild
# ============================================================


def crop_to_content(image, content_mask):
    mask = content_mask[..., 0] > 0.5
    row_indices = np.flatnonzero(np.any(mask, axis=1))
    column_indices = np.flatnonzero(np.any(mask, axis=0))

    return image[
        row_indices[0] : row_indices[-1] + 1,
        column_indices[0] : column_indices[-1] + 1,
    ]


def calculate_ssim(clean_image, compared_image):
    min_dimension = min(clean_image.shape[0], clean_image.shape[1])
    filter_size = min(11, min_dimension)

    if filter_size % 2 == 0:
        filter_size -= 1

    ssim = tf.image.ssim(
        np.clip(clean_image, 0.0, 1.0)[np.newaxis, ...],
        np.clip(compared_image, 0.0, 1.0)[np.newaxis, ...],
        max_val=1.0,
        filter_size=filter_size,
    )

    return float(ssim.numpy()[0])


def calculate_batched_ssim(clean_images, compared_images, batch_size=16):
    ssim_values = []

    for batch_start in range(0, len(clean_images), batch_size):
        batch_end = batch_start + batch_size
        batch_values = tf.image.ssim(
            np.clip(clean_images[batch_start:batch_end], 0.0, 1.0),
            np.clip(compared_images[batch_start:batch_end], 0.0, 1.0),
            max_val=1.0,
        )
        ssim_values.append(batch_values.numpy())

    return np.concatenate(ssim_values).astype(np.float64)


def calculate_unmasked_metrics(clean_images, compared_images):
    difference = compared_images - clean_images
    axes = (1, 2, 3)
    mse_values = np.mean(np.square(difference), axis=axes, dtype=np.float64)
    mae_values = np.mean(np.abs(difference), axis=axes, dtype=np.float64)
    psnr_values = -10.0 * np.log10(np.maximum(mse_values, 1e-12))
    ssim_values = calculate_batched_ssim(clean_images, compared_images)

    return mse_values, mae_values, psnr_values, ssim_values


def calculate_masked_metrics(clean_images, compared_images, content_masks):
    mse_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []

    for clean_image, compared_image, content_mask in zip(
        clean_images,
        compared_images,
        content_masks,
    ):
        channel_mask = np.broadcast_to(content_mask, clean_image.shape)
        num_values = np.sum(channel_mask)
        difference = compared_image - clean_image
        mse = (
            np.sum(
                np.square(difference) * channel_mask,
                dtype=np.float64,
            )
            / num_values
        )
        mae = (
            np.sum(
                np.abs(difference) * channel_mask,
                dtype=np.float64,
            )
            / num_values
        )
        clean_content = crop_to_content(clean_image, content_mask)
        compared_content = crop_to_content(compared_image, content_mask)

        mse_values.append(mse)
        mae_values.append(mae)
        psnr_values.append(-10.0 * np.log10(max(mse, 1e-12)))
        ssim_values.append(calculate_ssim(clean_content, compared_content))

    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (mse_values, mae_values, psnr_values, ssim_values)
    )


def calculate_image_metrics(
    clean_images,
    noisy_images,
    stage_one_images,
    stage_two_images,
    content_masks=None,
):
    compared_variants = {
        "noisy": noisy_images,
        "stage_one": stage_one_images,
        "stage_two": stage_two_images,
    }
    metric_values = {}

    for variant_name, compared_images in compared_variants.items():
        if content_masks is None:
            values = calculate_unmasked_metrics(clean_images, compared_images)
        else:
            values = calculate_masked_metrics(
                clean_images,
                compared_images,
                content_masks,
            )

        for metric_name, metric_array in zip(METRIC_NAMES, values):
            metric_values[f"{variant_name}_{metric_name}"] = metric_array

    return metric_values


def percent_reduction(baseline, candidate):
    return (baseline - candidate) / max(abs(baseline), 1e-12) * 100.0


def calculate_comparisons(values):
    return {
        "stage_one_mse_vs_noisy_percent": percent_reduction(
            values["noisy_mse"], values["stage_one_mse"]
        ),
        "stage_two_mse_vs_noisy_percent": percent_reduction(
            values["noisy_mse"], values["stage_two_mse"]
        ),
        "stage_two_mse_vs_stage_one_percent": percent_reduction(
            values["stage_one_mse"], values["stage_two_mse"]
        ),
        "stage_one_mae_vs_noisy_percent": percent_reduction(
            values["noisy_mae"], values["stage_one_mae"]
        ),
        "stage_two_mae_vs_noisy_percent": percent_reduction(
            values["noisy_mae"], values["stage_two_mae"]
        ),
        "stage_two_mae_vs_stage_one_percent": percent_reduction(
            values["stage_one_mae"], values["stage_two_mae"]
        ),
        "stage_one_psnr_vs_noisy": (values["stage_one_psnr"] - values["noisy_psnr"]),
        "stage_two_psnr_vs_noisy": (values["stage_two_psnr"] - values["noisy_psnr"]),
        "stage_two_psnr_vs_stage_one": (
            values["stage_two_psnr"] - values["stage_one_psnr"]
        ),
        "stage_one_ssim_vs_noisy": (values["stage_one_ssim"] - values["noisy_ssim"]),
        "stage_two_ssim_vs_noisy": (values["stage_two_ssim"] - values["noisy_ssim"]),
        "stage_two_ssim_vs_stage_one": (
            values["stage_two_ssim"] - values["stage_one_ssim"]
        ),
    }


def summarize_metrics(metric_values):
    summary = {"count": len(metric_values["noisy_mse"])}

    for metric_name, values in metric_values.items():
        summary[metric_name] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))

    summary.update(calculate_comparisons(summary))

    return summary


# ============================================================
# Spara rapporter och visualiseringar
# ============================================================


def save_metrics_csv(output_path, sample_names, metric_values):
    fieldnames = ["sample", *METRIC_KEYS, *COMPARISON_KEYS]

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample_index, sample_name in enumerate(sample_names):
            row = {
                "sample": sample_name,
                **{key: float(metric_values[key][sample_index]) for key in METRIC_KEYS},
            }
            row.update(calculate_comparisons(row))
            writer.writerow(row)


def save_training_history(history, output_path):
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "loss", "val_loss"])

        for epoch_index, (loss, val_loss) in enumerate(
            zip(history["loss"], history["val_loss"]),
            start=1,
        ):
            writer.writerow([epoch_index, loss, val_loss])


def describe_change(value, unit=""):
    if value > 0:
        result = "förbättring"
    elif value < 0:
        result = "försämring"
    else:
        result = "oförändrat"

    return f"{value:+.3f}{unit} ({result})"


def format_summary_section(title, summary):
    lines = [
        title,
        "-" * len(title),
        f"Antal exempel: {summary['count']}",
    ]

    for metric_name, label, unit in (
        ("mse", "MSE", ""),
        ("mae", "MAE", ""),
        ("psnr", "PSNR", " dB"),
        ("ssim", "SSIM", ""),
    ):
        lines.extend(
            [
                f"{label}, brusig:      {summary[f'noisy_{metric_name}']:.6f}{unit}",
                f"{label}, efter steg 1: "
                f"{summary[f'stage_one_{metric_name}']:.6f}{unit}",
                f"{label}, efter steg 2: "
                f"{summary[f'stage_two_{metric_name}']:.6f}{unit}",
            ]
        )

    lines.extend(
        [
            "",
            "Steg två jämfört med steg ett",
            f"MSE:  {describe_change(summary['stage_two_mse_vs_stage_one_percent'], '%')}",
            f"MAE:  {describe_change(summary['stage_two_mae_vs_stage_one_percent'], '%')}",
            f"PSNR: {describe_change(summary['stage_two_psnr_vs_stage_one'], ' dB')}",
            f"SSIM: {describe_change(summary['stage_two_ssim_vs_stage_one'])}",
        ]
    )

    return lines


def save_evaluation_summary(tile_summary, whole_image_summary):
    summary_lines = [
        "Utvärderingsrapport för tvåstegs-U-Net med residualförfining",
        "=================================================================",
        "",
        "MSE och MAE ska minska. PSNR och SSIM ska öka.",
        "Positiva jämförelsevärden betyder förbättring.",
        "Värdena redovisas som medelvärden per exempel.",
        "",
        *format_summary_section("Testresultat: tiles", tile_summary),
        "",
        *format_summary_section(
            "Testresultat: hela bilder",
            whole_image_summary,
        ),
    ]

    EVALUATION_SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def save_metric_comparison(tile_summary, whole_image_summary):
    summaries = (tile_summary, whole_image_summary)
    labels = ("Tiles", "Hela bilder")
    metric_definitions = (
        ("MSE", "mse", "Lägre är bättre"),
        ("MAE", "mae", "Lägre är bättre"),
        ("PSNR (dB)", "psnr", "Högre är bättre"),
        ("SSIM", "ssim", "Högre är bättre"),
    )
    colors = ("#E45756", "#4C78A8", "#54A24B")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )
    x_positions = np.arange(len(labels))
    bar_width = 0.24

    for axis, (title, metric_name, direction) in zip(
        axes.reshape(-1), metric_definitions
    ):
        for variant_index, variant_name in enumerate(VARIANT_NAMES):
            values = [summary[f"{variant_name}_{metric_name}"] for summary in summaries]
            positions = x_positions + (variant_index - 1) * bar_width
            bars = axis.bar(
                positions,
                values,
                bar_width,
                label=VARIANT_LABELS[variant_name],
                color=colors[variant_index],
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)

        axis.set_xticks(x_positions, labels)
        axis.set_title(f"{title} – {direction}")
        axis.grid(axis="y", alpha=0.3)

    axes[0, 0].legend()
    figure.suptitle("Baseline, steg ett och steg två", fontsize=16)
    figure.savefig(METRIC_COMPARISON_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_results(
    stage_one_history,
    stage_two_history,
    tile_metric_values,
    whole_image_metric_values,
    tile_summary,
    whole_image_summary,
    whole_image_paths,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tile_names = [
        f"tile_{tile_index:04d}"
        for tile_index in range(1, len(tile_metric_values["noisy_mse"]) + 1)
    ]
    whole_image_names = [Path(path).name for path in whole_image_paths]

    save_training_history(stage_one_history, STAGE_ONE_HISTORY_DATA_PATH)
    save_training_history(stage_two_history, STAGE_TWO_HISTORY_DATA_PATH)
    save_metrics_csv(TILE_METRICS_PATH, tile_names, tile_metric_values)
    save_metrics_csv(
        WHOLE_IMAGE_METRICS_PATH,
        whole_image_names,
        whole_image_metric_values,
    )
    save_evaluation_summary(tile_summary, whole_image_summary)
    save_metric_comparison(tile_summary, whole_image_summary)

    print(
        f"\nUtvärdering sparad\n"
        f"-------------------\n"
        f"Sammanfattning:       {EVALUATION_SUMMARY_PATH}\n"
        f"Tile-mått:            {TILE_METRICS_PATH}\n"
        f"Helbildsmått:         {WHOLE_IMAGE_METRICS_PATH}\n"
        f"Historik, steg ett:   {STAGE_ONE_HISTORY_DATA_PATH}\n"
        f"Historik, steg två:   {STAGE_TWO_HISTORY_DATA_PATH}\n"
        f"Jämförelsefigur:      {METRIC_COMPARISON_PATH}"
    )
