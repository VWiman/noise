import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    EDA_DIR,
    EDA_OVERVIEW_PATH,
    EDA_SAMPLES_PATH,
    EDA_SUMMARY_PATH,
    TILE_SIZE,
)


SPLIT_NAMES = ("train", "val", "test")
SPLIT_LABELS = {
    "train": "Train",
    "val": "Val",
    "test": "Test",
}
SPLIT_COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test": "#54A24B",
}


# ============================================================
# Samla metadata och grundläggande statistik
# ============================================================


def collect_image_metadata(image_splits):
    metadata = []

    for split_name in SPLIT_NAMES:
        for image_path in image_splits[split_name]:
            image_data = tf.io.read_file(image_path)
            height, width, _ = tf.io.extract_jpeg_shape(image_data).numpy()
            metadata.append(
                {
                    "split": split_name,
                    "height": int(height),
                    "width": int(width),
                }
            )

    return metadata


def calculate_tile_statistics(tile_arrays):
    statistics = {}

    for split_name in SPLIT_NAMES:
        tiles = tile_arrays[split_name]
        statistics[split_name] = {
            "brightness": np.mean(
                tiles,
                axis=(1, 2, 3),
                dtype=np.float64,
            ),
            "channel_mean": np.mean(
                tiles,
                axis=(0, 1, 2),
                dtype=np.float64,
            ),
            "channel_std": np.std(
                tiles,
                axis=(0, 1, 2),
                dtype=np.float64,
            ),
        }

    return statistics


# ============================================================
# Visualiseringar
# ============================================================


def save_overview_plot(
    image_splits,
    tile_counts,
    image_metadata,
    tile_statistics,
):
    split_labels = [SPLIT_LABELS[name] for name in SPLIT_NAMES]
    split_colors = [SPLIT_COLORS[name] for name in SPLIT_NAMES]

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))

    image_counts = [len(image_splits[name]) for name in SPLIT_NAMES]
    axes[0, 0].bar(split_labels, image_counts, color=split_colors)
    axes[0, 0].set_title("Originalbilder per split")
    axes[0, 0].set_ylabel("Antal bilder")
    axes[0, 0].grid(axis="y", alpha=0.3)

    tile_count_values = [tile_counts[name] for name in SPLIT_NAMES]
    axes[0, 1].bar(split_labels, tile_count_values, color=split_colors)
    axes[0, 1].set_title("Tiles per split")
    axes[0, 1].set_ylabel("Antal tiles")
    axes[0, 1].grid(axis="y", alpha=0.3)

    for split_name in SPLIT_NAMES:
        split_metadata = [
            item for item in image_metadata if item["split"] == split_name
        ]
        axes[1, 0].scatter(
            [item["width"] for item in split_metadata],
            [item["height"] for item in split_metadata],
            color=SPLIT_COLORS[split_name],
            label=SPLIT_LABELS[split_name],
            alpha=0.7,
        )

    axes[1, 0].axvline(TILE_SIZE, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(TILE_SIZE, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Originalbildernas dimensioner")
    axes[1, 0].set_xlabel("Bredd i pixlar")
    axes[1, 0].set_ylabel("Höjd i pixlar")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    brightness_bins = np.linspace(0.0, 1.0, 31)
    for split_name in SPLIT_NAMES:
        axes[1, 1].hist(
            tile_statistics[split_name]["brightness"],
            bins=brightness_bins,
            color=SPLIT_COLORS[split_name],
            label=SPLIT_LABELS[split_name],
            alpha=0.5,
        )

    axes[1, 1].set_title("Genomsnittlig ljusstyrka per tile")
    axes[1, 1].set_xlabel("Ljusstyrka, normaliserad till [0, 1]")
    axes[1, 1].set_ylabel("Antal tiles")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    figure.suptitle("Explorativ dataanalys", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(EDA_OVERVIEW_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_sample_tiles(train_tiles):
    num_examples = min(6, len(train_tiles))
    sample_indices = np.linspace(
        0,
        len(train_tiles) - 1,
        num_examples,
        dtype=int,
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = np.asarray(axes).reshape(-1)

    for plot_index, axis in enumerate(axes):
        if plot_index < num_examples:
            tile_index = sample_indices[plot_index]
            axis.imshow(np.clip(train_tiles[tile_index], 0.0, 1.0))
            axis.set_title(f"Train-tile {tile_index + 1}")

        axis.axis("off")

    figure.suptitle("Exempel på rena tiles före brus", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(EDA_SAMPLES_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# Textsammanfattning
# ============================================================


def format_image_dimensions(image_metadata, split_name):
    split_metadata = [
        item for item in image_metadata if item["split"] == split_name
    ]
    heights = np.array([item["height"] for item in split_metadata])
    widths = np.array([item["width"] for item in split_metadata])

    return (
        f"höjd {heights.min()}/{np.median(heights):.0f}/{heights.max()}, "
        f"bredd {widths.min()}/{np.median(widths):.0f}/{widths.max()} px"
    )


def save_eda_summary(
    image_splits,
    tile_counts,
    image_metadata,
    tile_statistics,
):
    summary_lines = [
        "Explorativ dataanalys",
        "=====================",
        "",
        "Originalbilder och tiles",
        "-------------------------",
    ]

    for split_name in SPLIT_NAMES:
        num_images = len(image_splits[split_name])
        tiles_per_image = tile_counts[split_name] / num_images
        summary_lines.append(
            f"{SPLIT_LABELS[split_name]}: {num_images} originalbilder, "
            f"{tile_counts[split_name]} tiles, "
            f"{tiles_per_image:.1f} tiles per bild"
        )

    summary_lines.extend(
        [
            "",
            "Bilddimensioner, min/median/max",
            "--------------------------------",
        ]
    )

    for split_name in SPLIT_NAMES:
        summary_lines.append(
            f"{SPLIT_LABELS[split_name]}: "
            f"{format_image_dimensions(image_metadata, split_name)}"
        )

    summary_lines.extend(
        [
            "",
            "Pixelstatistik för tiles, RGB",
            "-------------------------------",
        ]
    )

    for split_name in SPLIT_NAMES:
        channel_mean = tile_statistics[split_name]["channel_mean"]
        channel_std = tile_statistics[split_name]["channel_std"]
        summary_lines.append(
            f"{SPLIT_LABELS[split_name]} mean: "
            f"R={channel_mean[0]:.3f}, G={channel_mean[1]:.3f}, "
            f"B={channel_mean[2]:.3f}"
        )
        summary_lines.append(
            f"{SPLIT_LABELS[split_name]} std:  "
            f"R={channel_std[0]:.3f}, G={channel_std[1]:.3f}, "
            f"B={channel_std[2]:.3f}"
        )

    summary_lines.extend(
        [
            "",
            "Modellens dataflöde",
            "--------------------",
            "Modellen tränas endast på 256x256 tiles från train-splitten.",
            "Validation loss beräknas endast på tiles från val-splitten.",
            "Efter träningen testas modellen på både tiles och nedskalade "
            "hela bilder från test-splitten.",
        ]
    )

    EDA_SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def run_eda(image_splits, tile_counts, tile_arrays):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    image_metadata = collect_image_metadata(image_splits)
    tile_statistics = calculate_tile_statistics(tile_arrays)

    save_overview_plot(
        image_splits,
        tile_counts,
        image_metadata,
        tile_statistics,
    )
    save_sample_tiles(tile_arrays["train"])
    save_eda_summary(
        image_splits,
        tile_counts,
        image_metadata,
        tile_statistics,
    )

    print(
        f"\nEDA sparad\n"
        f"----------\n"
        f"Översikt:       {EDA_OVERVIEW_PATH}\n"
        f"Exempel på tiles: {EDA_SAMPLES_PATH}\n"
        f"Sammanfattning:    {EDA_SUMMARY_PATH}"
    )
