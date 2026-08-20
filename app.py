import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    EDA_DIR,
    EPOCHS,
    EVALUATION_SUMMARY_PATH,
    LEARNING_RATE,
    NOISE_FACTOR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_PERCENT,
    TILE_RECONSTRUCTIONS_PATH,
    TILE_SIZE,
    TRAIN_PERCENT,
    TRAINING_HISTORY_PATH,
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
    content_masks = np.stack([item[1] for item in images_and_masks]).astype(
        np.float32
    )

    return images, content_masks


def create_kernel_initializer(seed_offset):
    # Explicita heltals-seeds fungerar med både vanlig Keras och NGC tf_keras.
    return GlorotUniform(seed=RANDOM_SEED + seed_offset)


def create_fixed_noisy_images(clean_images, seed):
    random_generator = np.random.default_rng(seed)
    noise = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=clean_images.shape,
    ).astype(np.float32)

    return np.clip(
        clean_images + NOISE_FACTOR * noise,
        0.0,
        1.0,
    ).astype(np.float32)


def describe_result(metrics):
    if metrics["improvement"] > 0:
        return "U-Net-modellen förbättrade de brusiga bilderna."

    return "U-Net-modellen slog inte brusets baseline."


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
    decoded_images,
    output_path,
    title,
    content_masks=None,
):
    num_examples = min(6, len(clean_images))
    figure, axes = plt.subplots(
        3,
        num_examples,
        figsize=(max(10, num_examples * 2.4), 7.5),
        squeeze=False,
        constrained_layout=True,
    )
    row_titles = ("Brusig", "Rekonstruerad", "Original")

    for image_index in range(num_examples):
        images = (
            noisy_images[image_index],
            decoded_images[image_index],
            clean_images[image_index],
        )

        if content_masks is not None:
            images = tuple(
                crop_to_content(image, content_masks[image_index])
                for image in images
            )

        for row_index, image in enumerate(images):
            axis = axes[row_index, image_index]
            axis.imshow(np.clip(image, 0, 1))
            axis.set_xticks([])
            axis.set_yticks([])

            if image_index == 0:
                axis.set_ylabel(row_titles[row_index], fontsize=11)

            if row_index == 0:
                axis.set_title(f"Exempel {image_index + 1}", fontsize=11)

    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)


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
    f"Träning: endast tiles från train-splitten\n"
    f"Helbilder: används endast för test efter träningen"
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
# Dynamiskt träningsbrus
# ============================================================


def add_training_noise(clean_image):
    noise = tf.random.normal(
        shape=tf.shape(clean_image),
        mean=0.0,
        stddev=NOISE_FACTOR,
        dtype=clean_image.dtype,
    )
    noisy_image = tf.clip_by_value(clean_image + noise, 0.0, 1.0)

    # Input är brusig bild och target är motsvarande rena bild.
    return noisy_image, clean_image


# ============================================================
# Tränings- och valideringsdata
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(
    buffer_size=len(x_train),
    seed=RANDOM_SEED,
    reshuffle_each_iteration=True,
)
train_dataset = train_dataset.map(
    add_training_noise,
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# Samma val-brus används varje epoch så att validation loss blir jämförbar.
x_val_noisy = create_fixed_noisy_images(x_val, RANDOM_SEED + 1)


# ============================================================
# Bygg U-Net-modellen
# ============================================================

# Tre pooling-nivåer kräver att bildstorleken är delbar med åtta.
if TILE_SIZE % 8 != 0:
    raise ValueError("TILE_SIZE måste vara delbart med 8.")

input_image = Input(
    shape=(TILE_SIZE, TILE_SIZE, 3),
    name="input_image",
)


# ============================================================
# Encoder
# ============================================================

encoder_1 = Conv2D(
    64,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(0),
    name="encoder_1",
)(input_image)
x = MaxPooling2D(name="pool_1")(encoder_1)

encoder_2 = Conv2D(
    128,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(1),
    name="encoder_2",
)(x)
x = MaxPooling2D(name="pool_2")(encoder_2)

encoder_3 = Conv2D(
    256,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(2),
    name="encoder_3",
)(x)
x = MaxPooling2D(name="pool_3")(encoder_3)


# ============================================================
# Bottleneck
# ============================================================

x = Conv2D(
    512,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(3),
    name="bottleneck",
)(x)


# ============================================================
# Decoder med skip connections
# ============================================================

x = UpSampling2D(name="upsample_3")(x)
x = Concatenate(name="skip_connection_3")([x, encoder_3])
x = Conv2D(
    256,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(4),
    name="decoder_3",
)(x)

x = UpSampling2D(name="upsample_2")(x)
x = Concatenate(name="skip_connection_2")([x, encoder_2])
x = Conv2D(
    128,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(5),
    name="decoder_2",
)(x)

x = UpSampling2D(name="upsample_1")(x)
x = Concatenate(name="skip_connection_1")([x, encoder_1])
x = Conv2D(
    64,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer=create_kernel_initializer(6),
    name="decoder_1",
)(x)

output_image = Conv2D(
    3,
    kernel_size=1,
    activation="sigmoid",
    kernel_initializer=create_kernel_initializer(7),
    name="output_image",
)(x)

autoencoder = Model(
    inputs=input_image,
    outputs=output_image,
    name="denoising_unet",
)
autoencoder.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
)

print("\nU-Net-modellens arkitektur:")
autoencoder.summary()


# ============================================================
# Callbacks
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)
model_checkpoint = ModelCheckpoint(
    filepath=str(CHECKPOINT_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)


# ============================================================
# Träna modellen
# ============================================================

print(
    "\n========================================\n"
    "TRÄNING\n"
    "========================================"
)

history = autoencoder.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=(x_val_noisy, x_val),
    callbacks=[early_stopping, model_checkpoint],
)


# ============================================================
# Test 1: tiles från testbilderna
# ============================================================

# Testdatan används först efter att träningen är avslutad.
x_test_noisy = create_fixed_noisy_images(x_test, RANDOM_SEED + 2)
decoded_tiles = autoencoder.predict(x_test_noisy, verbose=0)
tile_test_loss = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)
tile_metric_values = calculate_image_metrics(
    x_test,
    x_test_noisy,
    decoded_tiles,
)
tile_metrics = summarize_metrics(tile_metric_values)


# ============================================================
# Test 2: hela testbilder
# ============================================================

# Längsta sidan skalas ned till högst TILE_SIZE och proportionerna bevaras.
whole_image_dataset = load_and_preprocess_full_images(image_splits["test"])
x_test_whole, whole_content_masks = materialize_full_images(whole_image_dataset)
x_test_whole_noisy = create_fixed_noisy_images(
    x_test_whole,
    RANDOM_SEED + 3,
)
decoded_whole_images = autoencoder.predict(
    x_test_whole_noisy,
    verbose=0,
)
whole_image_metric_values = calculate_image_metrics(
    x_test_whole,
    x_test_whole_noisy,
    decoded_whole_images,
    whole_content_masks,
)
whole_image_metrics = summarize_metrics(whole_image_metric_values)


# ============================================================
# Resultat
# ============================================================

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = np.min(history.history["val_loss"])

save_evaluation_results(
    history.history,
    tile_metric_values,
    whole_image_metric_values,
    tile_metrics,
    whole_image_metrics,
    image_splits["test"],
)

print(
    f"\nTräningsresultat\n"
    f"-----------------\n"
    f"Training loss, start:   {history.history['loss'][0]:.6f}\n"
    f"Training loss, slut:    {history.history['loss'][-1]:.6f}\n"
    f"Validation loss, start: {history.history['val_loss'][0]:.6f}\n"
    f"Validation loss, slut:  {history.history['val_loss'][-1]:.6f}\n"
    f"Bästa epoch:            {best_epoch}\n"
    f"Bästa validation loss:  {best_val_loss:.6f}"
)

print(
    f"\nTestresultat: tiles\n"
    f"-------------------\n"
    f"Keras test loss:         {tile_test_loss:.6f}\n"
    f"Brusig baseline-MSE:     {tile_metrics['noisy_mse']:.6f}\n"
    f"Rekonstruktions-MSE:     "
    f"{tile_metrics['reconstruction_mse']:.6f}\n"
    f"MSE-förbättring:         {tile_metrics['improvement']:.6f}\n"
    f"Relativ förbättring:     "
    f"{tile_metrics['relative_improvement']:.2f}%\n"
    f"Brusig baseline-MAE:     {tile_metrics['noisy_mae']:.6f}\n"
    f"Rekonstruktions-MAE:     {tile_metrics['reconstruction_mae']:.6f}\n"
    f"Brusig baseline-PSNR:    {tile_metrics['noisy_psnr']:.3f} dB\n"
    f"Rekonstruktions-PSNR:    {tile_metrics['reconstruction_psnr']:.3f} dB\n"
    f"Brusig baseline-SSIM:    {tile_metrics['noisy_ssim']:.4f}\n"
    f"Rekonstruktions-SSIM:    {tile_metrics['reconstruction_ssim']:.4f}\n"
    f"Resultat: {describe_result(tile_metrics)}"
)

print(
    f"\nTestresultat: hela bilder\n"
    f"-------------------------\n"
    f"Antal originalbilder:     {len(x_test_whole)}\n"
    f"Brusig baseline-MSE:      "
    f"{whole_image_metrics['noisy_mse']:.6f}\n"
    f"Rekonstruktions-MSE:      "
    f"{whole_image_metrics['reconstruction_mse']:.6f}\n"
    f"MSE-förbättring:          "
    f"{whole_image_metrics['improvement']:.6f}\n"
    f"Relativ förbättring:      "
    f"{whole_image_metrics['relative_improvement']:.2f}%\n"
    f"Brusig baseline-MAE:      {whole_image_metrics['noisy_mae']:.6f}\n"
    f"Rekonstruktions-MAE:      "
    f"{whole_image_metrics['reconstruction_mae']:.6f}\n"
    f"Brusig baseline-PSNR:     "
    f"{whole_image_metrics['noisy_psnr']:.3f} dB\n"
    f"Rekonstruktions-PSNR:     "
    f"{whole_image_metrics['reconstruction_psnr']:.3f} dB\n"
    f"Brusig baseline-SSIM:     {whole_image_metrics['noisy_ssim']:.4f}\n"
    f"Rekonstruktions-SSIM:     "
    f"{whole_image_metrics['reconstruction_ssim']:.4f}\n"
    f"Resultat: {describe_result(whole_image_metrics)}"
)


# ============================================================
# Träningshistorik
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.axvline(
    best_epoch - 1,
    linestyle="--",
    label=f"Bästa epoch ({best_epoch})",
)
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Denoising U-Net med dynamiskt brus")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    TRAINING_HISTORY_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


# ============================================================
# Rekonstruktioner
# ============================================================

save_reconstruction_plot(
    x_test,
    x_test_noisy,
    decoded_tiles,
    TILE_RECONSTRUCTIONS_PATH,
    "Testresultat för tiles",
)
save_reconstruction_plot(
    x_test_whole,
    x_test_whole_noisy,
    decoded_whole_images,
    WHOLE_IMAGE_RECONSTRUCTIONS_PATH,
    "Testresultat för hela bilder",
    whole_content_masks,
)


# ============================================================
# Sammanfattning
# ============================================================

print(
    f"\nSparade filer\n"
    f"-------------\n"
    f"EDA:                     {EDA_DIR}\n"
    f"Checkpoint:              {CHECKPOINT_PATH}\n"
    f"Utvärderingsrapport:      {EVALUATION_SUMMARY_PATH}\n"
    f"Träningshistorik:        {TRAINING_HISTORY_PATH}\n"
    f"Tile-rekonstruktioner:   {TILE_RECONSTRUCTIONS_PATH}\n"
    f"Helbildsrekonstruktioner: {WHOLE_IMAGE_RECONSTRUCTIONS_PATH}"
)
