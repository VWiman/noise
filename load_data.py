import math
import random

import tensorflow as tf
from config import (
    DATASET_DIR,
    RANDOM_SEED,
    TEST_PERCENT,
    TILE_SIZE,
    TRAIN_PERCENT,
    VALID_EXTENSIONS,
    VAL_PERCENT,
)


# ============================================================
# Dela upp originalbilderna
# ============================================================


def collect_image_paths():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Datasetmappen saknas: {DATASET_DIR}")

    image_paths = sorted(
        str(path)
        for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Datasetmappen innehåller inga bilder: {DATASET_DIR}"
        )

    return image_paths


def calculate_split_counts(num_images):
    split_percentages = {
        "train": TRAIN_PERCENT,
        "val": VAL_PERCENT,
        "test": TEST_PERCENT,
    }

    if sum(split_percentages.values()) != 100:
        raise ValueError(
            "TRAIN_PERCENT, VAL_PERCENT och TEST_PERCENT "
            "måste tillsammans bli 100."
        )

    if any(percent <= 0 for percent in split_percentages.values()):
        raise ValueError("Alla datasetandelar måste vara större än 0.")

    exact_counts = {
        name: num_images * percent / 100
        for name, percent in split_percentages.items()
    }
    split_counts = {
        name: math.floor(count) for name, count in exact_counts.items()
    }

    remaining_images = num_images - sum(split_counts.values())
    tie_priority = {"train": 0, "val": 1, "test": 2}
    remainder_order = sorted(
        split_percentages,
        key=lambda name: (
            exact_counts[name] - split_counts[name],
            tie_priority[name],
        ),
        reverse=True,
    )

    for split_name in remainder_order[:remaining_images]:
        split_counts[split_name] += 1

    if any(count == 0 for count in split_counts.values()):
        raise ValueError(
            "Datasetet innehåller för få bilder för train, val och test."
        )

    return split_counts


def split_image_paths(image_paths):
    shuffled_paths = list(image_paths)
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(shuffled_paths)

    split_counts = calculate_split_counts(len(shuffled_paths))
    train_end = split_counts["train"]
    val_end = train_end + split_counts["val"]

    image_splits = {
        "train": shuffled_paths[:train_end],
        "val": shuffled_paths[train_end:val_end],
        "test": shuffled_paths[val_end:],
    }

    return image_splits


# ============================================================
# Dela upp en bild i bildrutor
# ============================================================


def image_to_tiles(path):
    image_data = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_data, channels=3)
    image.set_shape([None, None, 3])

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    # Endast kompletta bildrutor används.
    row_count = height // TILE_SIZE
    column_count = width // TILE_SIZE
    crop_height = row_count * TILE_SIZE
    crop_width = column_count * TILE_SIZE

    # Centrera den del av bilden som kan delas jämnt.
    offset_y = (height - crop_height) // 2
    offset_x = (width - crop_width) // 2

    image = image[
        offset_y : offset_y + crop_height,
        offset_x : offset_x + crop_width,
    ]

    row_indices = tf.range(row_count)
    column_indices = tf.range(column_count)
    row_grid, column_grid = tf.meshgrid(
        row_indices,
        column_indices,
        indexing="ij",
    )
    coordinates = tf.stack(
        [
            tf.reshape(row_grid, [-1]),
            tf.reshape(column_grid, [-1]),
        ],
        axis=1,
    )

    def crop_tile(coord):
        y = coord[0] * TILE_SIZE
        x = coord[1] * TILE_SIZE
        tile = image[y : y + TILE_SIZE, x : x + TILE_SIZE, :]
        tile.set_shape([TILE_SIZE, TILE_SIZE, 3])

        return tile

    return tf.data.Dataset.from_tensor_slices(coordinates).map(
        crop_tile,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ============================================================
# Skapa dataset med bildrutor
# ============================================================


def create_tile_dataset(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.flat_map(image_to_tiles)

    dataset = dataset.map(
        lambda x: tf.cast(x, tf.float32) / 255.0,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Räkna tiles för sammanfattningen och kontrollera tomma splits.
    tile_count = dataset.reduce(
        tf.constant(0, dtype=tf.int64),
        lambda count, _: count + 1,
    )
    num_tiles = int(tile_count.numpy())

    if num_tiles == 0:
        raise ValueError("Datasetdelen skapade inga kompletta bildrutor.")

    return dataset, num_tiles


def load_and_preprocess_datasets():
    image_paths = collect_image_paths()
    image_splits = split_image_paths(image_paths)
    tile_datasets = {}
    tile_counts = {}

    for split_name, split_paths in image_splits.items():
        dataset, num_tiles = create_tile_dataset(split_paths)
        tile_datasets[split_name] = dataset
        tile_counts[split_name] = num_tiles

    return tile_datasets, image_splits, tile_counts


# ============================================================
# Skala hela bilder till modellens input
# ============================================================


def resize_full_image(path):
    image_data = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape([None, None, 3])

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    largest_dimension = tf.maximum(height, width)
    scale = tf.minimum(
        1.0,
        tf.cast(TILE_SIZE, tf.float32) / tf.cast(largest_dimension, tf.float32),
    )

    resized_height = tf.cast(
        tf.round(tf.cast(height, tf.float32) * scale),
        tf.int32,
    )
    resized_width = tf.cast(
        tf.round(tf.cast(width, tf.float32) * scale),
        tf.int32,
    )
    resized_height = tf.clip_by_value(resized_height, 1, TILE_SIZE)
    resized_width = tf.clip_by_value(resized_width, 1, TILE_SIZE)

    image = tf.image.resize(
        image,
        [resized_height, resized_width],
        antialias=True,
    )

    # Kantpadding ger rätt input-shape utan att ändra proportionerna.
    offset_y = (TILE_SIZE - resized_height) // 2
    offset_x = (TILE_SIZE - resized_width) // 2

    row_positions = tf.range(TILE_SIZE) - offset_y
    column_positions = tf.range(TILE_SIZE) - offset_x
    source_rows = tf.clip_by_value(
        row_positions,
        0,
        resized_height - 1,
    )
    source_columns = tf.clip_by_value(
        column_positions,
        0,
        resized_width - 1,
    )
    image = tf.gather(image, source_rows, axis=0)
    image = tf.gather(image, source_columns, axis=1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    content_rows = tf.logical_and(
        row_positions >= 0,
        row_positions < resized_height,
    )
    content_columns = tf.logical_and(
        column_positions >= 0,
        column_positions < resized_width,
    )
    content_mask = tf.logical_and(
        content_rows[:, tf.newaxis],
        content_columns[tf.newaxis, :],
    )
    content_mask = tf.cast(content_mask[..., tf.newaxis], tf.float32)

    image.set_shape([TILE_SIZE, TILE_SIZE, 3])
    content_mask.set_shape([TILE_SIZE, TILE_SIZE, 1])

    return image, content_mask


def load_and_preprocess_full_images(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    return dataset.map(
        resize_full_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
