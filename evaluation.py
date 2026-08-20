import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    EVALUATION_SUMMARY_PATH,
    METRIC_COMPARISON_PATH,
    RESULTS_DIR,
    TILE_METRICS_PATH,
    TRAINING_HISTORY_DATA_PATH,
    WHOLE_IMAGE_METRICS_PATH,
)


METRIC_KEYS = (
    "noisy_mse",
    "reconstruction_mse",
    "noisy_mae",
    "reconstruction_mae",
    "noisy_psnr",
    "reconstruction_psnr",
    "noisy_ssim",
    "reconstruction_ssim",
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
        clean_batch = np.clip(
            clean_images[batch_start:batch_end],
            0.0,
            1.0,
        )
        compared_batch = np.clip(
            compared_images[batch_start:batch_end],
            0.0,
            1.0,
        )
        batch_values = tf.image.ssim(
            clean_batch,
            compared_batch,
            max_val=1.0,
        )
        ssim_values.append(batch_values.numpy())

    return np.concatenate(ssim_values).astype(np.float64)


def calculate_image_metrics(
    clean_images,
    noisy_images,
    decoded_images,
    content_masks=None,
):
    metric_values = {key: [] for key in METRIC_KEYS}

    for image_index in range(len(clean_images)):
        clean_image = clean_images[image_index]
        noisy_image = noisy_images[image_index]
        decoded_image = decoded_images[image_index]

        if content_masks is None:
            channel_mask = None
            num_values = clean_image.size
            clean_for_ssim = clean_image
            noisy_for_ssim = noisy_image
            decoded_for_ssim = decoded_image
        else:
            content_mask = content_masks[image_index]
            channel_mask = np.broadcast_to(content_mask, clean_image.shape)
            num_values = np.sum(channel_mask)
            clean_for_ssim = crop_to_content(clean_image, content_mask)
            noisy_for_ssim = crop_to_content(noisy_image, content_mask)
            decoded_for_ssim = crop_to_content(decoded_image, content_mask)

        noisy_difference = noisy_image - clean_image
        reconstruction_difference = decoded_image - clean_image

        if channel_mask is None:
            noisy_mse = np.sum(
                np.square(noisy_difference),
                dtype=np.float64,
            ) / num_values
            reconstruction_mse = np.sum(
                np.square(reconstruction_difference),
                dtype=np.float64,
            ) / num_values
            noisy_mae = np.sum(
                np.abs(noisy_difference),
                dtype=np.float64,
            ) / num_values
            reconstruction_mae = np.sum(
                np.abs(reconstruction_difference),
                dtype=np.float64,
            ) / num_values
        else:
            noisy_mse = np.sum(
                np.square(noisy_difference) * channel_mask,
                dtype=np.float64,
            ) / num_values
            reconstruction_mse = np.sum(
                np.square(reconstruction_difference) * channel_mask,
                dtype=np.float64,
            ) / num_values
            noisy_mae = np.sum(
                np.abs(noisy_difference) * channel_mask,
                dtype=np.float64,
            ) / num_values
            reconstruction_mae = np.sum(
                np.abs(reconstruction_difference) * channel_mask,
                dtype=np.float64,
            ) / num_values

        metric_values["noisy_mse"].append(noisy_mse)
        metric_values["reconstruction_mse"].append(reconstruction_mse)
        metric_values["noisy_mae"].append(noisy_mae)
        metric_values["reconstruction_mae"].append(reconstruction_mae)
        metric_values["noisy_psnr"].append(
            -10.0 * np.log10(max(noisy_mse, 1e-12))
        )
        metric_values["reconstruction_psnr"].append(
            -10.0 * np.log10(max(reconstruction_mse, 1e-12))
        )
        if content_masks is not None:
            metric_values["noisy_ssim"].append(
                calculate_ssim(clean_for_ssim, noisy_for_ssim)
            )
            metric_values["reconstruction_ssim"].append(
                calculate_ssim(clean_for_ssim, decoded_for_ssim)
            )

    if content_masks is None:
        metric_values["noisy_ssim"] = calculate_batched_ssim(
            clean_images,
            noisy_images,
        )
        metric_values["reconstruction_ssim"] = calculate_batched_ssim(
            clean_images,
            decoded_images,
        )

    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in metric_values.items()
    }


def summarize_metrics(metric_values):
    summary = {"count": len(metric_values["noisy_mse"])}

    for metric_name, values in metric_values.items():
        summary[metric_name] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))

    summary["improvement"] = (
        summary["noisy_mse"] - summary["reconstruction_mse"]
    )
    summary["relative_improvement"] = (
        summary["improvement"] / summary["noisy_mse"] * 100.0
    )
    summary["mae_improvement"] = (
        summary["noisy_mae"] - summary["reconstruction_mae"]
    )
    summary["mae_relative_improvement"] = (
        summary["mae_improvement"] / summary["noisy_mae"] * 100.0
    )
    summary["psnr_improvement"] = (
        summary["reconstruction_psnr"] - summary["noisy_psnr"]
    )
    summary["ssim_improvement"] = (
        summary["reconstruction_ssim"] - summary["noisy_ssim"]
    )

    return summary


# ============================================================
# Spara rapporter och visualiseringar
# ============================================================


def save_metrics_csv(output_path, sample_names, metric_values):
    fieldnames = [
        "sample",
        *METRIC_KEYS,
        "mse_improvement_percent",
        "mae_improvement_percent",
        "psnr_improvement",
        "ssim_improvement",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample_index, sample_name in enumerate(sample_names):
            row = {
                "sample": sample_name,
                **{
                    key: float(metric_values[key][sample_index])
                    for key in METRIC_KEYS
                },
            }
            row["mse_improvement_percent"] = (
                (row["noisy_mse"] - row["reconstruction_mse"])
                / row["noisy_mse"]
                * 100.0
            )
            row["mae_improvement_percent"] = (
                (row["noisy_mae"] - row["reconstruction_mae"])
                / row["noisy_mae"]
                * 100.0
            )
            row["psnr_improvement"] = (
                row["reconstruction_psnr"] - row["noisy_psnr"]
            )
            row["ssim_improvement"] = (
                row["reconstruction_ssim"] - row["noisy_ssim"]
            )
            writer.writerow(row)


def save_training_history(history):
    with TRAINING_HISTORY_DATA_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "loss", "val_loss"])

        for epoch_index, (loss, val_loss) in enumerate(
            zip(history["loss"], history["val_loss"]),
            start=1,
        ):
            writer.writerow([epoch_index, loss, val_loss])


def format_summary_section(title, summary):
    return [
        title,
        "-" * len(title),
        f"Antal exempel: {summary['count']}",
        f"MSE, brusig:        {summary['noisy_mse']:.6f} "
        f"± {summary['noisy_mse_std']:.6f}",
        f"MSE, rekonstruerad: {summary['reconstruction_mse']:.6f} "
        f"± {summary['reconstruction_mse_std']:.6f}",
        f"Relativ MSE-förbättring: {summary['relative_improvement']:.2f}%",
        f"MAE, brusig:        {summary['noisy_mae']:.6f} "
        f"± {summary['noisy_mae_std']:.6f}",
        f"MAE, rekonstruerad: {summary['reconstruction_mae']:.6f} "
        f"± {summary['reconstruction_mae_std']:.6f}",
        f"Relativ MAE-förbättring: {summary['mae_relative_improvement']:.2f}%",
        f"PSNR, brusig:        {summary['noisy_psnr']:.3f} dB",
        f"PSNR, rekonstruerad: {summary['reconstruction_psnr']:.3f} dB",
        f"PSNR-förbättring:     {summary['psnr_improvement']:.3f} dB",
        f"SSIM, brusig:        {summary['noisy_ssim']:.4f}",
        f"SSIM, rekonstruerad: {summary['reconstruction_ssim']:.4f}",
        f"SSIM-förbättring:     {summary['ssim_improvement']:.4f}",
    ]


def save_evaluation_summary(tile_summary, whole_image_summary):
    summary_lines = [
        "Utvärderingsrapport för Denoising U-Net",
        "========================================",
        "",
        "MSE och MAE ska minska. PSNR och SSIM ska öka.",
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
        ("MSE", "noisy_mse", "reconstruction_mse", "Lägre är bättre"),
        ("MAE", "noisy_mae", "reconstruction_mae", "Lägre är bättre"),
        (
            "PSNR (dB)",
            "noisy_psnr",
            "reconstruction_psnr",
            "Högre är bättre",
        ),
        ("SSIM", "noisy_ssim", "reconstruction_ssim", "Högre är bättre"),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )
    x_positions = np.arange(len(labels))
    bar_width = 0.36

    for axis, metric_definition in zip(axes.reshape(-1), metric_definitions):
        title, noisy_key, reconstruction_key, direction = metric_definition
        noisy_values = [summary[noisy_key] for summary in summaries]
        reconstruction_values = [
            summary[reconstruction_key] for summary in summaries
        ]

        noisy_bars = axis.bar(
            x_positions - bar_width / 2,
            noisy_values,
            bar_width,
            label="Brusig baseline",
            color="#E45756",
        )
        reconstruction_bars = axis.bar(
            x_positions + bar_width / 2,
            reconstruction_values,
            bar_width,
            label="Rekonstruerad",
            color="#4C78A8",
        )
        axis.bar_label(noisy_bars, fmt="%.3f", fontsize=8, padding=3)
        axis.bar_label(
            reconstruction_bars,
            fmt="%.3f",
            fontsize=8,
            padding=3,
        )
        axis.set_xticks(x_positions, labels)
        axis.set_title(f"{title} – {direction}")
        axis.grid(axis="y", alpha=0.3)

    axes[0, 0].legend()
    figure.suptitle("Baseline jämfört med U-Net-rekonstruktion", fontsize=16)
    figure.savefig(METRIC_COMPARISON_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_results(
    history,
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

    save_training_history(history)
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
        f"Sammanfattning:    {EVALUATION_SUMMARY_PATH}\n"
        f"Tile-mått:         {TILE_METRICS_PATH}\n"
        f"Helbildsmått:      {WHOLE_IMAGE_METRICS_PATH}\n"
        f"Träningshistorik: {TRAINING_HISTORY_DATA_PATH}\n"
        f"Jämförelsefigur:  {METRIC_COMPARISON_PATH}"
    )
