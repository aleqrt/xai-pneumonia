"""
XAI system for Pneumonia detection using chest X-ray images.

VGG16-based binary classifier with LIME and Grad-CAM explainability.
Mirrors the implementation in notebook/chest_xray_xai.ipynb.

Usage:
    # Evaluate a pre-trained model and explain predictions:
    python src/cxr_xai.py --dataset ./dataset --model ./models/model_functional

    # Train from scratch:
    python src/cxr_xai.py --dataset ./dataset --model ./models/model_functional --train

    # Augment NORMAL images first, then train:
    python src/cxr_xai.py --dataset ./dataset --model ./models/model_functional --augment --train
"""

import argparse
import itertools
import os
import random

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorflow import keras
from tqdm import tqdm

sns.set_theme()

IMG_SIZE = 96
BATCH_SIZE = 12
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 55


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def print_dataset_stats(train_path, test_path):
    print("\nTrain set:\n===================")
    try:
        print(f"PNEUMONIA = {len(os.listdir(os.path.join(train_path, 'PNEUMONIA')))}")
        print(f"NORMAL    = {len(os.listdir(os.path.join(train_path, 'NORMAL')))}")
        print("\nTest set:\n===================")
        print(f"PNEUMONIA = {len(os.listdir(os.path.join(test_path, 'PNEUMONIA')))}")
        print(f"NORMAL    = {len(os.listdir(os.path.join(test_path, 'NORMAL')))}")
    except FileNotFoundError:
        print("Dataset directory not found. Check the --dataset path.")


def plot_dataset_distribution(train_dir, test_dir):
    try:
        data = {
            "NORMAL": [
                len(os.listdir(os.path.join(train_dir, "NORMAL"))),
                len(os.listdir(os.path.join(test_dir, "NORMAL"))),
            ],
            "PNEUMONIA": [
                len(os.listdir(os.path.join(train_dir, "PNEUMONIA"))),
                len(os.listdir(os.path.join(test_dir, "PNEUMONIA"))),
            ],
        }
        df = pd.DataFrame(data, index=["Train", "Test"])
        df.plot.barh()
        plt.title("Train and Test dataset partitions")
        plt.ylabel("Dataset Split")
        plt.xlabel("Number of images (log scale)")
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig("dataset_distribution.png", dpi=100)
        plt.show()
    except Exception as e:
        print(f"Could not plot distribution: {e}")


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def augment_normal_images(normal_dir):
    """Offline augmentation on NORMAL training images to reduce class imbalance.

    Adds one `_aug` copy per original image. Safe to re-run — already-augmented
    files (ending in _aug.jpeg / _aug.jpg) are skipped.
    """
    import cv2
    import imgaug.augmenters as iaa

    augmentation = iaa.Sometimes(
        1,
        [
            iaa.GaussianBlur(sigma=(0.1, 2.0)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Rotate(rotate=(-8, 8)),
            iaa.Sometimes(1, [iaa.ScaleX(scale=1.1), iaa.ScaleY(scale=1.1)]),
        ],
    )

    all_files = os.listdir(normal_dir)
    originals = [
        f for f in all_files
        if not (f.endswith("_aug.jpeg") or f.endswith("_aug.jpg"))
    ]
    print(f"Augmenting {len(originals)} NORMAL images...")
    for fname in tqdm(originals):
        img_path = os.path.join(normal_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        aug_img = augmentation.augment_image(img)
        base, ext = os.path.splitext(img_path)
        cv2.imwrite(f"{base}_aug{ext}", aug_img)
    print("Augmentation complete.")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def build_datasets(train_dir, test_dir):
    from keras.preprocessing.image import ImageDataGenerator

    image_generator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)

    train_ds = image_generator.flow_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="binary",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="training",
    )
    val_ds = image_generator.flow_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="binary",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="validation",
    )
    test_ds = image_generator.flow_from_directory(
        test_dir,
        batch_size=1,
        shuffle=False,
        class_mode="binary",
        target_size=(IMG_SIZE, IMG_SIZE),
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model():
    from keras import layers
    from keras.applications import VGG16

    base_model = VGG16(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    last = base_model.get_layer("block3_pool").output
    x = layers.GlobalAveragePooling2D()(last)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    pred = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(base_model.input, pred)

    lr = 0.01
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, lr


def compute_class_weights(train_ds, num_pneumonia, num_normal):
    train_count = train_ds.__len__()
    weight_for_0 = (1 / num_normal) * train_count / 2.0
    weight_for_1 = (1 / num_pneumonia) * train_count / 2.0
    print(f"Weight for class 0 (NORMAL):    {weight_for_0:.2f}")
    print(f"Weight for class 1 (PNEUMONIA): {weight_for_1:.2f}")
    return {0: weight_for_0, 1: weight_for_1}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, train_ds, val_ds, class_weight, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    lr = 0.01
    decay = lr / INITIAL_EPOCHS

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "xray_model.h5"), save_best_only=True
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    def lr_phase1(epoch, current_lr):
        return current_lr if epoch < 10 else current_lr * 1 / (1 + decay * epoch)

    print("=== Phase 1: Training head only ===")
    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[
            checkpoint_cb,
            early_stopping_cb,
            tf.keras.callbacks.LearningRateScheduler(lr_phase1),
        ],
    )

    print("\n=== Phase 2: Fine-tuning ===")
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:4]:
        layer.trainable = False

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def lr_phase2(epoch, current_lr=0.01):
        return current_lr if epoch < 25 else current_lr * 1 / (1 + decay * epoch)

    history_fine = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[
            checkpoint_cb,
            early_stopping_cb,
            tf.keras.callbacks.LearningRateScheduler(lr_phase2),
        ],
    )
    return history, history_fine


def plot_training_history(history, history_fine=None):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if history_fine:
        acc += history_fine.history["accuracy"]
        val_acc += history_fine.history["val_accuracy"]
        loss += history_fine.history["loss"]
        val_loss += history_fine.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS - 1, linestyle="--", label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([0.8, 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    if history_fine:
        plt.axvline(x=INITIAL_EPOCHS - 1, linestyle="--", label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def extract_test_data(test_ds):
    X_test, y_test = [], []
    for i in tqdm(range(len(test_ds))):
        X, label = test_ds.__getitem__(i)
        y_test.append(label)
        X_test.append(np.squeeze(X, axis=0))
    return np.array(X_test), np.array(y_test)


def plot_confusion_matrix(
    cm_values,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    if normalize:
        cm_values = cm_values.astype("float") / cm_values.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm_values)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm_values, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm_values.max() / 2.0
    for i, j in itertools.product(range(cm_values.shape[0]), range(cm_values.shape[1])):
        plt.text(
            j,
            i,
            format(cm_values[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm_values[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"roc_auc = {roc_auc:.3f}")

    sns.set_theme()
    plt.figure(figsize=(5, 5))
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=15)
    plt.plot(fpr, tpr, color="red", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.legend(loc="lower right", prop={"size": 15})
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=100)
    plt.show()


def evaluate(model, test_ds):
    evaluation = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {evaluation[1] * 100:.2f}%")

    y_pred = model.predict(test_ds)
    y_pred_rounded = y_pred.round()

    X_test, y_test = extract_test_data(test_ds)

    print(classification_report(y_test, y_pred_rounded, target_names=["NORMAL", "PNEUMONIA"]))

    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred_rounded, average="macro"
    )
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F-score   : {fscore:.2f}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred_rounded):.2f}")

    cm_values = confusion_matrix(y_test, y_pred_rounded)
    plot_confusion_matrix(cm_values, ["NORMAL", "PNEUMONIA"], title="Confusion Matrix")
    plot_roc_curve(y_test, y_pred)

    return X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def explain_with_lime(model, X_test, y_test, y_pred, n=5):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    indices = random.sample(range(len(X_test)), min(n, len(X_test)))
    for index in indices:
        image = X_test[index]
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image.astype("double"),
            model.predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )
        pred_label = "Normal" if y_pred[index] == 0 else "Pneumonia"
        true_label = "Normal" if y_test[index] == 0 else "Pneumonia"
        plt.figure()
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(
            f"Predicted: {pred_label} | Ground Truth: {true_label}\n"
            "Green: supports prediction  |  Red: against prediction"
        )
        plt.tight_layout()
        out_path = f"lime_{index}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.show()
        print(f"LIME explanation saved: {out_path}")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


def _get_gradcam_layers(model):
    """Return (last_conv_layer_name, classifier_layer_names) for this model."""
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found in the model.")
    classifier_names = []
    found = False
    for layer in model.layers:
        if found:
            classifier_names.append(layer.name)
        if layer.name == last_conv_name:
            found = True
    return last_conv_name, classifier_names


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:  # avoid division by zero when all activations are negative
        heatmap /= max_val
    return heatmap


def superimpose_gradcam(img, heatmap, save_path="grad_cam_image.jpg", alpha=0.4):
    img_uint8 = np.uint8(255 * img)
    heatmap_uint8 = np.uint8(255 * heatmap)

    try:
        jet = mpl_cm.colormaps["jet"]  # matplotlib >= 3.7
    except AttributeError:
        jet = mpl_cm.get_cmap("jet")   # matplotlib < 3.7

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize((img_uint8.shape[1], img_uint8.shape[0]))
    jet_heatmap_arr = tf.keras.preprocessing.image.img_to_array(jet_heatmap_img)

    superimposed = jet_heatmap_arr * alpha + img_uint8
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed)
    superimposed_img.save(save_path)
    print(f"Grad-CAM saved: {save_path}")

    result = plt.imread(save_path)
    plt.figure(figsize=(4, 4))
    plt.imshow(result)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def explain_with_gradcam(model, X_test, output_dir="./gradcam_outputs", n=5):
    os.makedirs(output_dir, exist_ok=True)
    last_conv_name, classifier_names = _get_gradcam_layers(model)
    indices = random.sample(range(len(X_test)), min(n, len(X_test)))
    for index in indices:
        heatmap = make_gradcam_heatmap(
            np.expand_dims(X_test[index], axis=0),
            model,
            last_conv_name,
            classifier_names,
        )
        superimpose_gradcam(
            X_test[index],
            heatmap,
            save_path=os.path.join(output_dir, f"{index}.jpg"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI Pneumonia Detection — train, evaluate, and explain a VGG16 classifier."
    )
    parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Dataset root directory (must contain train/ and test/).",
    )
    parser.add_argument(
        "--model",
        default="./models/model_functional",
        help="Model path: saved here on --train, loaded from here otherwise.",
    )
    parser.add_argument(
        "--checkpoints",
        default="./checkpoints",
        help="Checkpoint directory (used during training).",
    )
    parser.add_argument(
        "--gradcam-output",
        default="./gradcam_outputs",
        help="Output directory for Grad-CAM images.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training. If omitted, loads an existing model.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Offline-augment NORMAL training images before training.",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=5,
        metavar="N",
        help="Number of LIME explanations to generate (default: 5).",
    )
    parser.add_argument(
        "--gradcam-samples",
        type=int,
        default=5,
        metavar="N",
        help="Number of Grad-CAM visualizations to generate (default: 5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dir = os.path.join(args.dataset, "train")
    test_dir = os.path.join(args.dataset, "test")

    print_dataset_stats(train_dir, test_dir)

    if args.augment:
        augment_normal_images(os.path.join(train_dir, "NORMAL"))

    train_ds, val_ds, test_ds = build_datasets(train_dir, test_dir)

    if args.train:
        num_pneumonia = len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))
        num_normal = len(os.listdir(os.path.join(train_dir, "NORMAL")))
        class_weight = compute_class_weights(train_ds, num_pneumonia, num_normal)
        model, _ = build_model()
        model.summary()
        history, history_fine = train_model(
            model, train_ds, val_ds, class_weight, checkpoint_dir=args.checkpoints
        )
        plot_training_history(history, history_fine)
        model_dir = os.path.dirname(args.model)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        model.save(args.model)
        print(f"Model saved to {args.model}")
    else:
        print(f"Loading model from {args.model}")
        model = keras.models.load_model(args.model)

    X_test, y_test, y_pred = evaluate(model, test_ds)

    print("\n=== LIME Explanations ===")
    explain_with_lime(model, X_test, y_test, y_pred, n=args.lime_samples)

    print("\n=== Grad-CAM Explanations ===")
    explain_with_gradcam(model, X_test, output_dir=args.gradcam_output, n=args.gradcam_samples)
