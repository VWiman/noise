import csv
import shutil

import numpy as np
import tensorflow as tf

from config import (
    BATCH_SIZE,
    MAX_NOISE,
    MIN_NOISE,
    RANDOM_SEED,
    STAGE_TWO_DATASET_DIR,
    STAGE_TWO_MANIFEST_PATH,
    TILE_SIZE,
    VAL_NOISE,
)


STAGE_TWO_SPLITS = ("train", "val")
MANIFEST_FIELDS = (
    "split",
    "filename",
    "input_path",
    "target_path",
    "noise_factor",
)


# ============================================================
# Skapa ett parat dataset från steg ett
# ============================================================


def save_png(image, output_path):
    image = tf.convert_to_tensor(np.clip(image, 0.0, 1.0), dtype=tf.float32)
    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    encoded_image = tf.io.encode_png(image)
    tf.io.write_file(str(output_path), encoded_image)


def create_stage_two_directories():
    # Mappen är en genererad artefakt och byggs om för varje full körning.
    if STAGE_TWO_DATASET_DIR.exists():
        shutil.rmtree(STAGE_TWO_DATASET_DIR)

    directories = {}

    for split_name in STAGE_TWO_SPLITS:
        input_directory = STAGE_TWO_DATASET_DIR / split_name / "inputs"
        target_directory = STAGE_TWO_DATASET_DIR / split_name / "targets"
        input_directory.mkdir(parents=True, exist_ok=True)
        target_directory.mkdir(parents=True, exist_ok=True)
        directories[split_name] = {
            "inputs": input_directory,
            "targets": target_directory,
        }

    return directories


def create_stage_two_train_batch(clean_batch, random_generator):
    noise_factors = random_generator.uniform(
        MIN_NOISE,
        MAX_NOISE,
        size=len(clean_batch),
    ).astype(np.float32)
    noise = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=clean_batch.shape,
    ).astype(np.float32)
    noisy_batch = np.clip(
        clean_batch + noise_factors[:, None, None, None] * noise,
        0.0,
        1.0,
    ).astype(np.float32)

    return noisy_batch, noise_factors


def write_stage_two_split(
    manifest_writer,
    split_name,
    clean_images,
    stage_one_model,
    directories,
    random_generator=None,
    fixed_noisy_images=None,
):
    num_written = 0
    num_batches = (len(clean_images) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"Skapar {split_name}-par för steg två: "
        f"{len(clean_images)} tiles i {num_batches} batcher",
        flush=True,
    )

    for batch_start in range(0, len(clean_images), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(clean_images))
        batch_number = batch_start // BATCH_SIZE + 1
        clean_batch = clean_images[batch_start:batch_end]

        if split_name == "train":
            noisy_batch, noise_factors = create_stage_two_train_batch(
                clean_batch,
                random_generator,
            )
        else:
            noisy_batch = fixed_noisy_images[batch_start:batch_end]
            noise_factors = np.full(
                len(clean_batch),
                VAL_NOISE,
                dtype=np.float32,
            )

        reconstructed_batch = np.asarray(
            stage_one_model.predict_on_batch(noisy_batch),
            dtype=np.float32,
        )

        for batch_index, (reconstruction, target, noise_factor) in enumerate(
            zip(reconstructed_batch, clean_batch, noise_factors)
        ):
            tile_index = batch_start + batch_index + 1
            filename = f"tile_{tile_index:06d}.png"
            input_path = directories[split_name]["inputs"] / filename
            target_path = directories[split_name]["targets"] / filename

            save_png(reconstruction, input_path)
            save_png(target, target_path)
            manifest_writer.writerow(
                {
                    "split": split_name,
                    "filename": filename,
                    "input_path": str(input_path.relative_to(STAGE_TWO_DATASET_DIR)),
                    "target_path": str(target_path.relative_to(STAGE_TWO_DATASET_DIR)),
                    "noise_factor": f"{float(noise_factor):.8f}",
                }
            )
            num_written += 1

        if batch_number == 1 or batch_number % 10 == 0 or batch_number == num_batches:
            print(
                f"  {split_name}: {batch_end}/{len(clean_images)} tiles sparade",
                flush=True,
            )

    return num_written


def generate_stage_two_dataset(
    stage_one_model,
    train_clean_images,
    val_clean_images,
    val_noisy_images,
):
    directories = create_stage_two_directories()
    train_random_generator = np.random.default_rng(RANDOM_SEED + 10)

    with STAGE_TWO_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        manifest_writer = csv.DictWriter(
            manifest_file,
            fieldnames=MANIFEST_FIELDS,
        )
        manifest_writer.writeheader()

        train_count = write_stage_two_split(
            manifest_writer,
            "train",
            train_clean_images,
            stage_one_model,
            directories,
            random_generator=train_random_generator,
        )
        val_count = write_stage_two_split(
            manifest_writer,
            "val",
            val_clean_images,
            stage_one_model,
            directories,
            fixed_noisy_images=val_noisy_images,
        )

    print(
        f"\nDataset för steg två skapat\n"
        f"---------------------------\n"
        f"Train:    {train_count} par\n"
        f"Val:      {val_count} par\n"
        f"Manifest: {STAGE_TWO_MANIFEST_PATH}"
    )

    return {"train": train_count, "val": val_count}


# ============================================================
# Läs det parade datasetet från disk
# ============================================================


def load_png_pair(input_path, target_path):
    input_image = tf.io.decode_png(
        tf.io.read_file(input_path),
        channels=3,
    )
    target_image = tf.io.decode_png(
        tf.io.read_file(target_path),
        channels=3,
    )
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    target_image = tf.image.convert_image_dtype(target_image, tf.float32)
    input_image.set_shape([TILE_SIZE, TILE_SIZE, 3])
    target_image.set_shape([TILE_SIZE, TILE_SIZE, 3])

    return input_image, target_image


def read_stage_two_manifest():
    if not STAGE_TWO_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifestet för steg två saknas: {STAGE_TWO_MANIFEST_PATH}"
        )

    split_pairs = {split_name: [] for split_name in STAGE_TWO_SPLITS}

    with STAGE_TWO_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("Manifestet för steg två har fel kolumner.")

        for row in reader:
            split_name = row["split"]

            if split_name not in STAGE_TWO_SPLITS:
                raise ValueError(
                    "Datasetet för steg två får endast innehålla train och val."
                )

            input_path = STAGE_TWO_DATASET_DIR / row["input_path"]
            target_path = STAGE_TWO_DATASET_DIR / row["target_path"]

            if input_path.name != row["filename"]:
                raise ValueError("Inputfilen matchar inte manifestets filnamn.")

            if target_path.name != row["filename"]:
                raise ValueError("Targetfilen matchar inte manifestets filnamn.")

            if not input_path.is_file() or not target_path.is_file():
                raise FileNotFoundError(
                    f"Ett bildpar i manifestet saknas: {row['filename']}"
                )

            split_pairs[split_name].append((str(input_path), str(target_path)))

    if any(not pairs for pairs in split_pairs.values()):
        raise ValueError("Train och val för steg två måste innehålla bildpar.")

    return split_pairs


def load_stage_two_datasets():
    split_pairs = read_stage_two_manifest()
    datasets = {}
    counts = {}

    for split_name, pairs in split_pairs.items():
        input_paths, target_paths = zip(*pairs)
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(input_paths), list(target_paths))
        )
        dataset = dataset.map(
            load_png_pair,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if split_name == "train":
            dataset = dataset.shuffle(
                buffer_size=len(pairs),
                seed=RANDOM_SEED + 20,
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        datasets[split_name] = dataset
        counts[split_name] = len(pairs)

    return datasets, counts
