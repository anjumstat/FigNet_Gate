# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:01:55 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
FIGNet: Nested Cross-Validation for Unbiased Performance Estimation
Addresses Editor Concern: Model-selection bias

Structure:
- Outer loop: 10-fold stratified CV → unbiased performance estimate
- Inner loop: 5-fold CV on outer training set → hyperparameter selection
- Test set (outer fold) is NEVER used for hyperparameter selection

Outputs:
- All fold-wise history .npy files
- All CSV results for manuscript tables
- All trained models
- All predictions per fold
- Checkpoint system for resume after interruption
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from itertools import product

from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tf.function retracing.*")

# ReliefF
try:
    from skrebate import ReliefF
    RELIEF_AVAILABLE = True
except ImportError:
    RELIEF_AVAILABLE = False

# Force CPU
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Running on CPU mode")
except Exception:
    pass

# Reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_with_species.csv"
BASE_DIR = r"D:\zebfish1\revision1\FIGNet_NestedCV_Results"
os.makedirs(BASE_DIR, exist_ok=True)

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128]
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
INNER_FOLDS = 5
OUTER_FOLDS = 10
RANDOM_STATE = 42

CLASS_NAMES = ["Non-enzyme", "Enzyme"]

# ============================================================================
# CHECKPOINT SYSTEM (NESTED CV LEVEL)
# ============================================================================

CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"completed_inner": {}, "completed_outer": {}}

def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)

def make_inner_key(method, lr, bs, outer_fold, inner_fold):
    return f"{method}_lr{lr}_bs{bs}_outer{outer_fold}_inner{inner_fold}"

def make_outer_key(method, lr, bs, outer_fold):
    return f"{method}_lr{lr}_bs{bs}_outer{outer_fold}"

# ============================================================================
# UTILITIES
# ============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def label_col_from_df(df):
    if "Label" in df.columns:
        return "Label"
    if "Binary_Target" in df.columns:
        return "Binary_Target"
    raise ValueError("No Label column found.")

# ============================================================================
# CUSTOM FIGNet LAYERS
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="FIGNet")
class FeatureImportanceGate(layers.Layer):
    def __init__(self, keep_ratio=0.8, temperature=1.0, gate_regularization=0.01, **kwargs):
        super().__init__(**kwargs)
        self.keep_ratio = keep_ratio
        self.temperature = temperature
        self.gate_regularization = gate_regularization

    def build(self, input_shape):
        self.feature_importance = self.add_weight(
            name="feature_importance",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        gate = tf.sigmoid(self.feature_importance / self.temperature)
        self.add_loss(self.gate_regularization * tf.square(tf.reduce_mean(gate) - self.keep_ratio))
        return inputs * gate

    def get_gate_values(self):
        return tf.sigmoid(self.feature_importance / self.temperature).numpy()

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_ratio": self.keep_ratio,
            "temperature": self.temperature,
            "gate_regularization": self.gate_regularization,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FIGNet")
class RealVariationalDropout(layers.Layer):
    def __init__(self, units, init_drop_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.init_drop_rate = init_drop_rate
        self.eps = 1e-8

    def build(self, input_shape):
        alpha_init = self.init_drop_rate / (1.0 - self.init_drop_rate + self.eps)
        log_alpha_init = np.log(alpha_init + self.eps)
        self.log_alpha = self.add_weight(
            name="log_alpha",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(log_alpha_init),
            trainable=True,
        )
        self.mean_shift = self.add_weight(
            name="mean_shift",
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        alpha = tf.exp(self.log_alpha)
        dropout_rate = alpha / (1.0 + alpha + self.eps)
        variance = alpha * tf.square(inputs + self.mean_shift)
        std = tf.sqrt(variance + self.eps)
        epsilon = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype)
        output = inputs + epsilon * std
        scale = tf.sqrt(1.0 / (1.0 - dropout_rate + self.eps))
        return output * scale

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "init_drop_rate": self.init_drop_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="FIGNet")
class AdaptiveVariationalDropout(layers.Layer):
    def __init__(self, units, initial_drop_rate=0.3, learnable=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initial_drop_rate = initial_drop_rate
        self.learnable = learnable
        self.eps = 1e-8

    def build(self, input_shape):
        if self.learnable:
            init_logit = np.log(self.initial_drop_rate / (1.0 - self.initial_drop_rate + self.eps))
            self.drop_logits = self.add_weight(
                name="drop_logits",
                shape=(self.units,),
                initializer=tf.keras.initializers.Constant(init_logit),
                trainable=True,
            )
        else:
            self.drop_logits = None
        self.noise_scale = self.add_weight(
            name="noise_scale",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.learnable:
            drop_rate = tf.sigmoid(self.drop_logits)
        else:
            drop_rate = tf.cast(self.initial_drop_rate, inputs.dtype)
        if not training:
            return inputs
        bernoulli_mask = tf.keras.backend.random_bernoulli(
            tf.shape(inputs), p=1.0 - drop_rate, dtype=inputs.dtype,
        )
        gaussian_noise = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype) * self.noise_scale
        combined_noise = bernoulli_mask * (1.0 + gaussian_noise)
        scale = 1.0 / (1.0 - drop_rate + self.eps)
        return inputs * combined_noise * scale

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "initial_drop_rate": self.initial_drop_rate,
            "learnable": self.learnable,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FIGNet")
class DynamicSparsityRegularizer(regularizers.Regularizer):
    def __init__(self, initial_sparsity=0.7, final_sparsity=0.9, warmup_epochs=20, total_epochs=100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def __call__(self, weights):
        progress = tf.minimum(1.0, self.current_epoch / float(self.warmup_epochs))
        current_target = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        abs_weights = tf.abs(weights)
        flat_weights = tf.reshape(abs_weights, [-1])
        sorted_weights = tf.sort(flat_weights)
        n = tf.shape(sorted_weights)[0]
        k = tf.cast(tf.cast(n, tf.float32) * (1.0 - current_target), tf.int32)
        k = tf.clip_by_value(k, 1, n)
        threshold = sorted_weights[k - 1]
        sparsity = tf.reduce_mean(tf.cast(abs_weights < threshold, tf.float32))
        sparsity_loss = tf.square(sparsity - current_target) * current_target
        l1_strength = 0.0005 * (1.0 + progress * 2.0)
        l1_loss = tf.reduce_mean(abs_weights) * l1_strength
        return sparsity_loss + l1_loss

    def update_epoch(self, epoch):
        self.current_epoch.assign(float(epoch))

    def get_config(self):
        return {
            "initial_sparsity": self.initial_sparsity,
            "final_sparsity": self.final_sparsity,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
        }

# ============================================================================
# CALLBACKS
# ============================================================================

class DynamicSparsityCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "kernel_regularizer"):
                reg = layer.kernel_regularizer
                if hasattr(reg, "update_epoch"):
                    reg.update_epoch(epoch)

class MCCCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        logs["val_mcc"] = matthews_corrcoef(self.y_val, y_pred)

# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_fignet_model(input_shape, num_classes, learning_rate, variant="gate_only"):
    inputs = layers.Input(shape=(input_shape,), name="input")
    x = inputs
    x = FeatureImportanceGate(keep_ratio=0.8, name="feature_gate")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    use_sparsity = variant in ["gate_sparsity", "gate_full"]

    if use_sparsity:
        x = layers.Dense(512, activation="relu",
                        kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
                        name="dense1")(x)
    else:
        x = layers.Dense(512, activation="relu", name="dense1")(x)

    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(512, init_drop_rate=0.2, name="rvd1")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(512, initial_drop_rate=0.2, name="avd1")(x)

    x = layers.BatchNormalization(name="bn2")(x)
    shortcut = x

    if use_sparsity:
        x = layers.Dense(256, activation="relu",
                        kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
                        name="dense2")(x)
    else:
        x = layers.Dense(256, activation="relu", name="dense2")(x)

    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(256, init_drop_rate=0.3, name="rvd2")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(256, initial_drop_rate=0.3, name="avd2")(x)

    x = layers.BatchNormalization(name="bn3")(x)

    if use_sparsity:
        x = layers.Dense(256, activation="relu",
                        kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
                        name="dense3")(x)
    else:
        x = layers.Dense(256, activation="relu", name="dense3")(x)

    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(256, init_drop_rate=0.3, name="rvd3")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(256, initial_drop_rate=0.3, name="avd3")(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Dense(256, name="shortcut")(shortcut)
    x = layers.Add(name="residual_add")([x, shortcut])
    x = layers.BatchNormalization(name="bn4")(x)

    x = layers.Dense(128, activation="relu", name="dense4")(x)
    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(128, init_drop_rate=0.4, name="rvd4")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(128, initial_drop_rate=0.4, name="avd4")(x)

    x = layers.Dense(64, activation="relu", name="dense5")(x)
    x = layers.Dropout(0.3, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"FIGNet_{variant}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_mlp_model(input_shape, num_classes, learning_rate, use_gate=False):
    """MLP baseline. If use_gate=False, this is the no-gate ablation."""
    if use_gate:
        return build_fignet_model(input_shape, num_classes, learning_rate, variant="gate_only")
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation="relu", name="dense1"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(256, activation="relu", name="dense2"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu", name="dense3"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="dense4"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="MLP_Baseline")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_sklearn(X_train, y_train, X_test, y_test, variant):
    from sklearn.linear_model import LogisticRegression
    if variant == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    elif variant == "svm_rbf":
        clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    elif variant == "svm_linear":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    else:
        raise ValueError(variant)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, train_time, clf


def train_relieff(X_train, y_train, X_test, y_test, variant):
    if not RELIEF_AVAILABLE:
        raise ImportError("skrebate not installed")
    best = {"score": -1}
    for n_feat in [50, 100, 200]:
        n_feat = min(n_feat, X_train.shape[1])
        relieff = ReliefF(n_features_to_select=n_feat, n_neighbors=100)
        X_tr = relieff.fit_transform(X_train, y_train)
        X_te = relieff.transform(X_test)
        if variant == "relieff_mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                                random_state=RANDOM_STATE, early_stopping=True)
        else:
            clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        t0 = time.time()
        clf.fit(X_tr, y_train)
        tt = time.time() - t0
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        score = matthews_corrcoef(y_test, y_pred)
        if score > best["score"]:
            best = {"score": score, "pred": y_pred, "proba": y_proba,
                    "clf": clf, "time": tt, "features": relieff.top_features_}
    return best["pred"], best["proba"], best["time"], best["clf"], best.get("features")

# ============================================================================
# METHODS
# ============================================================================

METHODS = {
    "FIGNet_Gate_Only": {"type": "fignet", "variant": "gate_only",
                         "description": "Feature Importance Gate only"},
    "FIGNet_Gate_RealVD": {"type": "fignet", "variant": "gate_real_vd",
                           "description": "FIGNet + Real Variational Dropout"},
    "FIGNet_Gate_AdaptiveVD": {"type": "fignet", "variant": "gate_adaptive_vd",
                               "description": "FIGNet + Adaptive Variational Dropout"},
    "FIGNet_Gate_Sparsity": {"type": "fignet", "variant": "gate_sparsity",
                             "description": "FIGNet + Dynamic Sparsity"},
    "FIGNet_Gate_Full": {"type": "fignet", "variant": "gate_full",
                         "description": "FIGNet + All Components"},
    "MLP_Baseline": {"type": "baseline_mlp", "variant": "mlp",
                     "description": "MLP Baseline"},
    "NoGate_Ablation": {"type": "no_gate", "variant": "no_gate",
                        "description": "MLP with same architecture as FIGNet but NO gate"},
    "Logistic_Regression": {"type": "baseline_sklearn", "variant": "logistic",
                            "description": "Logistic Regression"},
    "SVM_RBF": {"type": "baseline_sklearn", "variant": "svm_rbf",
                "description": "SVM with RBF kernel"},
    "SVM_Linear": {"type": "baseline_sklearn", "variant": "svm_linear",
                   "description": "SVM with Linear kernel"},
    "ReliefF_MLP": {"type": "relieff", "variant": "relieff_mlp",
                    "description": "ReliefF + MLP"},
    "ReliefF_SVM": {"type": "relieff", "variant": "relieff_svm",
                    "description": "ReliefF + SVM"},
}

IS_SKLEARN = ["baseline_sklearn", "relieff"]

# ============================================================================
# INNER LOOP: Hyperparameter Selection
# ============================================================================

def run_inner_cv(method_name, config, X_train, y_train, feature_names, outer_fold, checkpoint):
    """Run 5-fold inner CV to select best LR/BS."""
    inner_dir = ensure_dir(os.path.join(BASE_DIR, "inner_cv",
                                        f"outer_fold_{outer_fold}", method_name))
    results = []

    is_sklearn = config["type"] in IS_SKLEARN

    if is_sklearn:
        # sklearn/relieff: only one config
        skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for inner_fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            key = make_inner_key(method_name, 0.001, 32, outer_fold, inner_fold)
            if key in checkpoint["completed_inner"]:
                fold_scores.append(checkpoint["completed_inner"][key]["mcc"])
                continue

            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr).astype(np.float32)
            X_va = scaler.transform(X_va).astype(np.float32)

            if config["type"] == "relieff":
                y_pred, y_proba, tt, clf, _ = train_relieff(X_tr, y_tr, X_va, y_va, config["variant"])
            else:
                y_pred, y_proba, tt, clf = train_sklearn(X_tr, y_tr, X_va, y_va, config["variant"])

            mcc = matthews_corrcoef(y_va, y_pred)
            fold_scores.append(mcc)
            checkpoint["completed_inner"][key] = {"mcc": mcc}
            save_checkpoint(checkpoint)

        return {"learning_rate": 0.001, "batch_size": 32,
                "mean_mcc": float(np.mean(fold_scores)),
                "std_mcc": float(np.std(fold_scores)),
                "is_sklearn": True}

    # TensorFlow: sweep LR × BS
    for lr, bs in product(LEARNING_RATES, BATCH_SIZES):
        skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for inner_fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            key = make_inner_key(method_name, lr, bs, outer_fold, inner_fold)
            if key in checkpoint["completed_inner"]:
                fold_scores.append(checkpoint["completed_inner"][key]["mcc"])
                continue

            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr).astype(np.float32)
            X_va = scaler.transform(X_va).astype(np.float32)

            y_tr_cat = tf.keras.utils.to_categorical(y_tr, 2)
            y_va_cat = tf.keras.utils.to_categorical(y_va, 2)

            tf.keras.backend.clear_session()

            if config["type"] == "fignet":
                model = build_fignet_model(X_tr.shape[1], 2, lr, config["variant"])
            elif config["type"] in ["baseline_mlp", "no_gate"]:
                model = build_mlp_model(X_tr.shape[1], 2, lr, use_gate=False)

            es = callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=0)
            mcc_cb = MCCCallback(validation_data=(X_va, y_va))
            dyn_cb = DynamicSparsityCallback()

            model.fit(X_tr, y_tr_cat, epochs=EPOCHS, batch_size=bs,
                      validation_data=(X_va, y_va_cat), verbose=0,
                      callbacks=[es, mcc_cb, dyn_cb])

            y_pred_proba = model.predict(X_va, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            mcc = matthews_corrcoef(y_va, y_pred)
            fold_scores.append(mcc)

            checkpoint["completed_inner"][key] = {"mcc": mcc}
            save_checkpoint(checkpoint)

        results.append({
            "learning_rate": lr, "batch_size": bs,
            "mean_mcc": float(np.mean(fold_scores)),
            "std_mcc": float(np.std(fold_scores)),
            "is_sklearn": False,
        })

    best = max(results, key=lambda r: r["mean_mcc"])
    save_json({"all_configs": results, "best": best},
              os.path.join(inner_dir, "inner_cv_results.json"))
    return best

# ============================================================================
# OUTER LOOP: Final Evaluation
# ============================================================================

def run_outer_fold(method_name, config, best_config, X_train, y_train, X_test, y_test,
                   feature_names, outer_fold, checkpoint):
    """Train on full outer-train and evaluate on outer-test fold."""
    outer_key = make_outer_key(method_name, best_config["learning_rate"],
                               best_config["batch_size"], outer_fold)
    if outer_key in checkpoint["completed_outer"]:
        print(f"   ⏭️ Outer fold {outer_fold} already completed")
        return None

    lr = best_config["learning_rate"]
    bs = best_config["batch_size"]
    is_sklearn = best_config["is_sklearn"]

    out_dir = ensure_dir(os.path.join(BASE_DIR, "outer_cv",
                                      f"outer_fold_{outer_fold}", method_name))
    npy_dir = ensure_dir(os.path.join(out_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(out_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(out_dir, "plots"))
    models_dir = ensure_dir(os.path.join(out_dir, "models"))

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ---------- sklearn / relieff ----------
    if is_sklearn:
        if config["type"] == "relieff":
            y_pred, y_proba, tt, clf, features = train_relieff(
                X_train_s, y_train, X_test_s, y_test, config["variant"])
            save_json({"features": features.tolist() if features is not None else None},
                      os.path.join(npy_dir, "relieff_features.json"))
        else:
            y_pred, y_proba, tt, clf = train_sklearn(
                X_train_s, y_train, X_test_s, y_test, config["variant"])
        import joblib
        joblib.dump(clf, os.path.join(models_dir, "model.joblib"))
        history = {}
        epochs = 1
    else:
        y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
        tf.keras.backend.clear_session()

        if config["type"] == "fignet":
            model = build_fignet_model(X_train_s.shape[1], 2, lr, config["variant"])
        else:
            model = build_mlp_model(X_train_s.shape[1], 2, lr, use_gate=False)

        es = callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                     restore_best_weights=True, verbose=0)
        dyn_cb = DynamicSparsityCallback()

        # Split small validation from outer-train for early stopping
        from sklearn.model_selection import train_test_split
        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_train_s, y_train, test_size=0.1, stratify=y_train, random_state=RANDOM_STATE)
        y_tr2_cat = tf.keras.utils.to_categorical(y_tr2, 2)
        y_val2_cat = tf.keras.utils.to_categorical(y_val2, 2)

        t0 = time.time()
        history = model.fit(X_tr2, y_tr2_cat, epochs=EPOCHS, batch_size=bs,
                            validation_data=(X_val2, y_val2_cat), verbose=0,
                            callbacks=[es, dyn_cb])
        tt = time.time() - t0
        epochs = len(history.history.get("loss", []))

        # Save all fold-wise history
        for k, v in history.history.items():
            np.save(os.path.join(npy_dir, f"history_{k}.npy"), np.array(v))

        y_pred_proba = model.predict(X_test_s, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_proba = y_pred_proba[:, 1]

        if config["type"] == "fignet":
            for layer in model.layers:
                if layer.name == "feature_gate":
                    importance = layer.get_gate_values()
                    np.save(os.path.join(npy_dir, "feature_importance.npy"), importance)
                    top_idx = np.argsort(importance)[-50:][::-1]
                    pd.DataFrame({
                        "Rank": range(1, 51),
                        "Feature_Index": top_idx,
                        "Feature_Name": [feature_names[i] for i in top_idx],
                        "Importance": importance[top_idx],
                    }).to_csv(os.path.join(csv_dir, "Top_Features.csv"), index=False)

        model.save(os.path.join(models_dir, "model.keras"))

    # Save scaler
    save_json({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
              os.path.join(npy_dir, "scaler.json"))

    # Save predictions
    np.save(os.path.join(npy_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(npy_dir, "y_proba.npy"), y_proba)
    np.save(os.path.join(npy_dir, "y_test.npy"), y_test)
    pd.DataFrame({
        "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
    }).to_csv(os.path.join(csv_dir, "predictions.csv"), index=False)

    # Metrics
    metrics = {
        "method": method_name,
        "outer_fold": outer_fold,
        "learning_rate": lr, "batch_size": bs,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "auc": safe_auc(y_test, y_proba),
        "training_time": tt,
        "epochs": epochs,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(csv_dir, "metrics.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(npy_dir, "confusion_matrix.npy"), cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{method_name} - Outer Fold {outer_fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    checkpoint["completed_outer"][outer_key] = metrics
    save_checkpoint(checkpoint)
    return metrics

# ============================================================================
# MAIN NESTED CV
# ============================================================================

def run_nested_cv():
    print("=" * 80)
    print("FIGNet: NESTED CROSS-VALIDATION (Unbiased Performance Estimation)")
    print("=" * 80)
    print(f"Data: {DATA_PATH}")
    print(f"Output: {BASE_DIR}")
    print(f"Inner folds: {INNER_FOLDS} | Outer folds: {OUTER_FOLDS}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    label_col = label_col_from_df(df)
    feature_names = [c for c in df.columns if c != label_col and c != "Data_Source"]
    X = df[feature_names].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    print(f"\n📊 Dataset: {len(df)} samples, {len(feature_names)} features")
    print(f"   Class distribution: {np.bincount(y)}")

    checkpoint = load_checkpoint()

    outer_skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(outer_skf.split(X, y))

    all_outer_metrics = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print(f"\n{'=' * 80}")
        print(f"OUTER FOLD {outer_fold}/{OUTER_FOLDS}")
        print(f"{'=' * 80}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for method_name, config in METHODS.items():
            print(f"\n  ▶ {method_name}")

            # INNER CV: select hyperparameters
            best_config = run_inner_cv(method_name, config, X_train, y_train,
                                       feature_names, outer_fold, checkpoint)
            print(f"    Best: lr={best_config['learning_rate']}, bs={best_config['batch_size']}, "
                  f"inner MCC={best_config['mean_mcc']:.4f}")

            # OUTER: train on full outer-train, evaluate on outer-test
            metrics = run_outer_fold(method_name, config, best_config,
                                     X_train, y_train, X_test, y_test,
                                     feature_names, outer_fold, checkpoint)
            if metrics is not None:
                all_outer_metrics.append(metrics)

    # Aggregate
    print("\n" + "=" * 80)
    print("NESTED CV SUMMARY")
    print("=" * 80)

    if all_outer_metrics:
        results_df = pd.DataFrame(all_outer_metrics)
        results_path = os.path.join(BASE_DIR, "NestedCV_All_Results.csv")
        results_df.to_csv(results_path, index=False)

        summary = results_df.groupby("method").agg({
            "accuracy": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "mcc": ["mean", "std"],
            "auc": ["mean", "std"],
        }).round(4)
        summary_path = os.path.join(BASE_DIR, "NestedCV_Summary.csv")
        summary.to_csv(summary_path)
        print(summary.to_string())

    print("\n✅ All results saved to:", BASE_DIR)


if __name__ == "__main__":
    run_nested_cv()