import gc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    ReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    EDA_DIR,
    EVALUATION_SUMMARY_PATH,
    LEARNING_RATE,
    MAX_NOISE,
    MIN_NOISE,
    RANDOM_SEED,
    RESULTS_DIR,
    STAGE_ONE_CHECKPOINT_PATH,
    STAGE_ONE_EPOCHS,
    STAGE_ONE_HISTORY_PATH,
    STAGE_TWO_CHECKPOINT_PATH,
    STAGE_TWO_DATASET_DIR,
    STAGE_TWO_EPOCHS,
    STAGE_TWO_HISTORY_PATH,
    TEST_PERCENT,
    TILE_RECONSTRUCTIONS_PATH,
    TILE_SIZE,
    TRAIN_PERCENT,
    VAL_NOISE,
    VAL_PERCENT,
    WHOLE_IMAGE_RECONSTRUCTIONS_PATH,
)
from eda import run_eda
from evaluation import (
    calculate_image_metrics,
    save_evaluation_results,
    summarize_metrics,
)
from load_data import (
    load_and_preprocess_datasets,
    load_and_preprocess_full_images,
)
from stage_two_data import (
    generate_stage_two_dataset,
    load_stage_two_datasets,
)


# Skapa output-mapparna om de inte redan finns.
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# För bättre reproducerbarhet.
tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# Hjälpfunktioner
# ============================================================


def materialize_dataset(dataset):
    return np.stack(list(dataset.as_numpy_iterator())).astype(np.float32)


def materialize_full_images(dataset):
    images_and_masks = list(dataset.as_numpy_iterator())
    images = np.stack([item[0] for item in images_and_masks]).astype(np.float32)
    content_masks = np.stack([item[1] for item in images_and_masks]).astype(np.float32)

    return images, content_masks


def create_kernel_initializer(seed_offset):
    # Explicita heltals-seeds fungerar med både vanlig Keras och NGC tf_keras.
    return GlorotUniform(seed=RANDOM_SEED + seed_offset)


def create_fixed_noisy_images(clean_images, seed, noise_factor):
    random_generator = np.random.default_rng(seed)
    noise = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=clean_images.shape,
    ).astype(np.float32)

    return np.clip(
        clean_images + noise_factor * noise,
        0.0,
        1.0,
    ).astype(np.float32)


def add_training_noise(clean_image):
    # Varje train-tile får en egen slumpad brusnivå.
    noise_factor = tf.random.uniform(
        shape=[],
        minval=MIN_NOISE,
        maxval=MAX_NOISE,
        dtype=clean_image.dtype,
    )
    noise = tf.random.normal(
        shape=tf.shape(clean_image),
        mean=0.0,
        stddev=noise_factor,
        dtype=clean_image.dtype,
    )
    noisy_image = tf.clip_by_value(clean_image + noise, 0.0, 1.0)

    # Input är brusig bild och target är motsvarande rena bild.
    return noisy_image, clean_image


def crop_to_content(image, content_mask):
    mask = content_mask[..., 0] > 0.5
    row_indices = np.flatnonzero(np.any(mask, axis=1))
    column_indices = np.flatnonzero(np.any(mask, axis=0))

    return image[
        row_indices[0] : row_indices[-1] + 1,
        column_indices[0] : column_indices[-1] + 1,
    ]


def save_reconstruction_plot(
    clean_images,
    noisy_images,
    stage_one_images,
    stage_two_images,
    output_path,
    title,
    content_masks=None,
):
    num_examples = min(6, len(clean_images))
    figure, axes = plt.subplots(
        4,
        num_examples,
        figsize=(max(10, num_examples * 2.4), 9.5),
        squeeze=False,
        constrained_layout=True,
    )
    row_titles = (
        "Brusig",
        "Efter steg 1",
        "Efter steg 2\n(residual)",
        "Original",
    )

    for image_index in range(num_examples):
        images = (
            noisy_images[image_index],
            stage_one_images[image_index],
            stage_two_images[image_index],
            clean_images[image_index],
        )

        if content_masks is not None:
            images = tuple(
                crop_to_content(image, content_masks[image_index]) for image in images
            )

        for row_index, image in enumerate(images):
            axis = axes[row_index, image_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_xticks([])
            axis.set_yticks([])

            if image_index == 0:
                axis.set_ylabel(row_titles[row_index], fontsize=11)

            if row_index == 0:
                axis.set_title(f"Exempel {image_index + 1}", fontsize=11)

    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_training_history_plot(history, output_path, title):
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history["loss"], label="Training loss")
    axis.plot(history["val_loss"], label="Validation loss")
    axis.axvline(
        best_epoch - 1,
        linestyle="--",
        label=f"Bästa epoch ({best_epoch})",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_training_summary(stage_name, history):
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    best_val_loss = float(np.min(history["val_loss"]))
    print(
        f"\nTräningsresultat: {stage_name}\n"
        f"-----------------------------\n"
        f"Training loss, start:   {history['loss'][0]:.6f}\n"
        f"Training loss, slut:    {history['loss'][-1]:.6f}\n"
        f"Validation loss, start: {history['val_loss'][0]:.6f}\n"
        f"Validation loss, slut:  {history['val_loss'][-1]:.6f}\n"
        f"Bästa epoch:            {best_epoch}\n"
        f"Bästa validation loss:  {best_val_loss:.6f}"
    )


def print_evaluation_summary(title, summary):
    stage_two_change = summary["stage_two_mse_vs_stage_one_percent"]

    if stage_two_change > 0:
        conclusion = "Steg två förbättrade MSE jämfört med steg ett."
    elif stage_two_change < 0:
        conclusion = "Steg två försämrade MSE jämfört med steg ett."
    else:
        conclusion = "Steg två gav oförändrad MSE jämfört med steg ett."

    print(
        f"\n{title}\n"
        f"{'-' * len(title)}\n"
        f"MSE, brusig:       {summary['noisy_mse']:.6f}\n"
        f"MSE, efter steg 1: {summary['stage_one_mse']:.6f}\n"
        f"MSE, efter steg 2: {summary['stage_two_mse']:.6f}\n"
        f"Steg 2 mot steg 1: {stage_two_change:+.2f}%\n"
        f"PSNR, steg 1:      {summary['stage_one_psnr']:.3f} dB\n"
        f"PSNR, steg 2:      {summary['stage_two_psnr']:.3f} dB\n"
        f"SSIM, steg 1:      {summary['stage_one_ssim']:.4f}\n"
        f"SSIM, steg 2:      {summary['stage_two_ssim']:.4f}\n"
        f"Resultat: {conclusion}"
    )


def create_callbacks(checkpoint_path):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ============================================================
# Återanvändbar U-Net-modell
# ============================================================


def double_convolution_block(
    input_tensor,
    filters,
    block_name,
    initializer_seed_offset,
):
    # Två 3x3-lager ger ett effektivt receptive field på ungefär 5x5.
    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=create_kernel_initializer(initializer_seed_offset),
        name=f"{block_name}_conv_1",
    )(input_tensor)
    x = Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=create_kernel_initializer(initializer_seed_offset + 1),
        name=f"{block_name}_conv_2",
    )(x)

    return x


def build_unet(
    model_name,
    initializer_seed_offset,
    residual_refinement=False,
):
    # Tre pooling-nivåer kräver att bildstorleken är delbar med åtta.
    if TILE_SIZE % 8 != 0:
        raise ValueError("TILE_SIZE måste vara delbart med 8.")

    input_image = Input(
        shape=(TILE_SIZE, TILE_SIZE, 3),
        name="input_image",
    )

    # Encoder.
    encoder_1 = double_convolution_block(
        input_image,
        filters=64,
        block_name="encoder_1",
        initializer_seed_offset=initializer_seed_offset,
    )
    x = MaxPooling2D(name="pool_1")(encoder_1)

    encoder_2 = double_convolution_block(
        x,
        filters=128,
        block_name="encoder_2",
        initializer_seed_offset=initializer_seed_offset + 2,
    )
    x = MaxPooling2D(name="pool_2")(encoder_2)

    encoder_3 = double_convolution_block(
        x,
        filters=256,
        block_name="encoder_3",
        initializer_seed_offset=initializer_seed_offset + 4,
    )
    x = MaxPooling2D(name="pool_3")(encoder_3)

    # Bottleneck.
    x = double_convolution_block(
        x,
        filters=512,
        block_name="bottleneck",
        initializer_seed_offset=initializer_seed_offset + 6,
    )

    # Decoder med skip connections.
    x = UpSampling2D(name="upsample_3")(x)
    x = Concatenate(name="skip_connection_3")([x, encoder_3])
    x = double_convolution_block(
        x,
        filters=256,
        block_name="decoder_3",
        initializer_seed_offset=initializer_seed_offset + 8,
    )

    x = UpSampling2D(name="upsample_2")(x)
    x = Concatenate(name="skip_connection_2")([x, encoder_2])
    x = double_convolution_block(
        x,
        filters=128,
        block_name="decoder_2",
        initializer_seed_offset=initializer_seed_offset + 10,
    )

    x = UpSampling2D(name="upsample_1")(x)
    x = Concatenate(name="skip_connection_1")([x, encoder_1])
    x = double_convolution_block(
        x,
        filters=64,
        block_name="decoder_1",
        initializer_seed_offset=initializer_seed_offset + 12,
    )

    if residual_refinement:
        # Steg två lär sig bara korrigeringen. Nollinitieringen gör att modellen
        # börjar som en identitetsfunktion och behåller bilden från steg ett.
        residual_correction = Conv2D(
            3,
            kernel_size=1,
            activation="tanh",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="residual_correction",
        )(x)
        corrected_image = Add(name="add_residual")([input_image, residual_correction])
        output_image = ReLU(
            max_value=1.0,
            name="output_image",
        )(corrected_image)
    else:
        output_image = Conv2D(
            3,
            kernel_size=1,
            activation="sigmoid",
            kernel_initializer=create_kernel_initializer(initializer_seed_offset + 14),
            name="output_image",
        )(x)

    model = Model(
        inputs=input_image,
        outputs=output_image,
        name=model_name,
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    return model


# ============================================================
# Läs in train, validation och test
# ============================================================


tile_datasets, image_splits, tile_counts = load_and_preprocess_datasets()

# Varje originalbild tillhör exakt en split innan tiles skapas.
x_train = materialize_dataset(tile_datasets["train"])
x_val = materialize_dataset(tile_datasets["val"])
x_test = materialize_dataset(tile_datasets["test"])

print(
    f"\nDataset\n"
    f"-------\n"
    f"Uppdelning: {TRAIN_PERCENT}% train, "
    f"{VAL_PERCENT}% val, {TEST_PERCENT}% test\n"
    f"Originalbilder: {sum(len(paths) for paths in image_splits.values())}\n"
    f"Train: {len(image_splits['train'])} bilder, "
    f"{tile_counts['train']} tiles\n"
    f"Val:   {len(image_splits['val'])} bilder, "
    f"{tile_counts['val']} tiles\n"
    f"Test:  {len(image_splits['test'])} bilder, "
    f"{tile_counts['test']} tiles\n"
    f"Tile-shape: {x_train.shape[1:]}, datatyp: {x_train.dtype}\n"
    f"Train-brus: slumpas per tile mellan {MIN_NOISE:.2f} och "
    f"{MAX_NOISE:.2f}\n"
    f"Val/test-brus: fast nivå {VAL_NOISE:.2f}\n"
    f"Helbilder: används endast för test efter båda träningsstegen"
)


# ============================================================
# Explorativ dataanalys före träning
# ============================================================


run_eda(
    image_splits,
    tile_counts,
    {
        "train": x_train,
        "val": x_val,
        "test": x_test,
    },
)


# ============================================================
# Träningsdata för steg ett
# ============================================================


stage_one_train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
stage_one_train_dataset = stage_one_train_dataset.shuffle(
    buffer_size=len(x_train),
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True,
)
stage_one_train_dataset = stage_one_train_dataset.map(
    add_training_noise,
    num_parallel_calls=tf.data.AUTOTUNE,
)
stage_one_train_dataset = stage_one_train_dataset.batch(BATCH_SIZE)
stage_one_train_dataset = stage_one_train_dataset.prefetch(tf.data.AUTOTUNE)

# Samma val-brus används varje epoch så att validation loss blir jämförbar.
x_val_noisy = create_fixed_noisy_images(
    x_val,
    RANDOM_SEED + 1,
    VAL_NOISE,
)


# ============================================================
# Steg ett: träna denoising U-Net
# ============================================================


print(
    "\n========================================\n"
    "STEG ETT: DENOISING\n"
    "========================================"
)

stage_one_model = build_unet(
    model_name="denoising_unet_stage_one",
    initializer_seed_offset=0,
)
stage_one_model.summary()
stage_one_history_object = stage_one_model.fit(
    stage_one_train_dataset,
    epochs=STAGE_ONE_EPOCHS,
    validation_data=(x_val_noisy, x_val),
    callbacks=create_callbacks(STAGE_ONE_CHECKPOINT_PATH),
)
stage_one_history = {
    name: list(values) for name, values in stage_one_history_object.history.items()
}
del stage_one_history_object
save_training_history_plot(
    stage_one_history,
    STAGE_ONE_HISTORY_PATH,
    "Steg ett: Denoising U-Net",
)
print_training_summary("steg ett", stage_one_history)


# ============================================================
# Skapa datasetet som steg två ska tränas på
# ============================================================


del stage_one_model
tf.keras.backend.clear_session()
gc.collect()

print(
    f"\nFörbereder steg två\n"
    f"-------------------\n"
    f"Laddar bästa checkpoint: {STAGE_ONE_CHECKPOINT_PATH}",
    flush=True,
)
stage_one_model = load_model(
    str(STAGE_ONE_CHECKPOINT_PATH),
    compile=False,
)
print("Checkpoint laddad. Skapar parade PNG-filer...", flush=True)
stage_two_pair_counts = generate_stage_two_dataset(
    stage_one_model,
    x_train,
    x_val,
    x_val_noisy,
)

if stage_two_pair_counts != {
    "train": tile_counts["train"],
    "val": tile_counts["val"],
}:
    raise ValueError("Antalet bildpar för steg två matchar inte tile-antalen.")

del stage_one_model
tf.keras.backend.clear_session()
gc.collect()


# ============================================================
# Steg två: träna residual U-Net från PNG-par
# ============================================================


stage_two_datasets, loaded_stage_two_counts = load_stage_two_datasets()

if loaded_stage_two_counts != stage_two_pair_counts:
    raise ValueError("Manifestet för steg två har oväntade antal bildpar.")

print(
    "\n========================================\n"
    "STEG TVÅ: RESIDUALFÖRFINING\n"
    "========================================"
)

stage_two_model = build_unet(
    model_name="residual_refinement_unet_stage_two",
    initializer_seed_offset=100,
    residual_refinement=True,
)
stage_two_model.summary()
stage_two_history_object = stage_two_model.fit(
    stage_two_datasets["train"],
    epochs=STAGE_TWO_EPOCHS,
    validation_data=stage_two_datasets["val"],
    callbacks=create_callbacks(STAGE_TWO_CHECKPOINT_PATH),
)
stage_two_history = {
    name: list(values) for name, values in stage_two_history_object.history.items()
}
del stage_two_history_object
save_training_history_plot(
    stage_two_history,
    STAGE_TWO_HISTORY_PATH,
    "Steg två: Residual refinement U-Net",
)
print_training_summary("steg två", stage_two_history)

del stage_two_model
tf.keras.backend.clear_session()
gc.collect()


# ============================================================
# Förbered det orörda testsetet
# ============================================================


x_test_noisy = create_fixed_noisy_images(
    x_test,
    RANDOM_SEED + 2,
    VAL_NOISE,
)
whole_image_dataset = load_and_preprocess_full_images(image_splits["test"])
x_test_whole, whole_content_masks = materialize_full_images(whole_image_dataset)
x_test_whole_noisy = create_fixed_noisy_images(
    x_test_whole,
    RANDOM_SEED + 3,
    VAL_NOISE,
)


# ============================================================
# Slutlig inferens: brusig bild -> steg ett -> steg två
# ============================================================


stage_one_model = load_model(
    str(STAGE_ONE_CHECKPOINT_PATH),
    compile=False,
)
stage_one_tiles = stage_one_model.predict(x_test_noisy, verbose=0)
stage_one_whole_images = stage_one_model.predict(
    x_test_whole_noisy,
    verbose=0,
)

del stage_one_model
tf.keras.backend.clear_session()
gc.collect()

stage_two_model = load_model(
    str(STAGE_TWO_CHECKPOINT_PATH),
    compile=False,
)
stage_two_tiles = stage_two_model.predict(stage_one_tiles, verbose=0)
stage_two_whole_images = stage_two_model.predict(
    stage_one_whole_images,
    verbose=0,
)

del stage_two_model
tf.keras.backend.clear_session()
gc.collect()


# ============================================================
# Utvärdera båda stegen
# ============================================================


tile_metric_values = calculate_image_metrics(
    x_test,
    x_test_noisy,
    stage_one_tiles,
    stage_two_tiles,
)
tile_metrics = summarize_metrics(tile_metric_values)

whole_image_metric_values = calculate_image_metrics(
    x_test_whole,
    x_test_whole_noisy,
    stage_one_whole_images,
    stage_two_whole_images,
    whole_content_masks,
)
whole_image_metrics = summarize_metrics(whole_image_metric_values)

save_evaluation_results(
    stage_one_history,
    stage_two_history,
    tile_metric_values,
    whole_image_metric_values,
    tile_metrics,
    whole_image_metrics,
    image_splits["test"],
)

print_evaluation_summary("Testresultat: tiles", tile_metrics)
print_evaluation_summary("Testresultat: hela bilder", whole_image_metrics)


# ============================================================
# Rekonstruktioner
# ============================================================


save_reconstruction_plot(
    x_test,
    x_test_noisy,
    stage_one_tiles,
    stage_two_tiles,
    TILE_RECONSTRUCTIONS_PATH,
    "Tvåstegsresultat för test-tiles",
)
save_reconstruction_plot(
    x_test_whole,
    x_test_whole_noisy,
    stage_one_whole_images,
    stage_two_whole_images,
    WHOLE_IMAGE_RECONSTRUCTIONS_PATH,
    "Tvåstegsresultat för hela testbilder",
    whole_content_masks,
)


# ============================================================
# Sammanfattning
# ============================================================


print(
    f"\nSparade filer\n"
    f"-------------\n"
    f"EDA:                       {EDA_DIR}\n"
    f"Dataset för steg två:      {STAGE_TWO_DATASET_DIR}\n"
    f"Checkpoint, steg ett:      {STAGE_ONE_CHECKPOINT_PATH}\n"
    f"Checkpoint, steg två:      {STAGE_TWO_CHECKPOINT_PATH}\n"
    f"Historik, steg ett:        {STAGE_ONE_HISTORY_PATH}\n"
    f"Historik, steg två:        {STAGE_TWO_HISTORY_PATH}\n"
    f"Utvärderingsrapport:       {EVALUATION_SUMMARY_PATH}\n"
    f"Tile-rekonstruktioner:     {TILE_RECONSTRUCTIONS_PATH}\n"
    f"Helbildsrekonstruktioner:  {WHOLE_IMAGE_RECONSTRUCTIONS_PATH}"
)
