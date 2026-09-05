# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:36:41 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
FIGNet: Feature Importance Gate Network for Interpretable Enzyme Classification
COMPLETE REVISION - CD-HIT HOMOLOGY-AWARE SPLITS
WITH THREE-LEVEL CHECKPOINT SYSTEM

Models included:
1. FIGNet variants (5):
   - FIGNet_Gate_Only
   - FIGNet_Gate_RealVD
   - FIGNet_Gate_AdaptiveVD
   - FIGNet_Gate_Sparsity
   - FIGNet_Gate_Full

2. Baseline classifiers requested by reviewers:
   - Logistic_Regression (existing)
   - MLP_Baseline (existing)
   - SVM_RBF (new - Reviewer 1, Point 2)
   - SVM_Linear (new - Reviewer 1, Point 2)
   - ReliefF_MLP (new - Reviewer 1, Point 2)
   - ReliefF_SVM (new - Reviewer 1, Point 2)

3. Explanation methods (for interpretability comparison):
   - SHAP (applied to FIGNet) - REDUCED SAMPLES FOR SPEED
   - LIME (applied to FIGNet)

Evaluation: CD-HIT Homology-Aware Splits (Train/Val/Test)
ALL MODELS TRAINED ON CD-HIT TRAIN SET, VALIDATED ON VAL SET, TESTED ON TEST SET

Checkpoint Levels:
   Level 1: Individual model runs (method + lr + bs)
   Level 2: All models complete for a split
   Level 3: Individual model complete
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

from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================================
# SUPPRESS TENSORFLOW WARNINGS
# ============================================================================
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tf.function retracing.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

# For ReliefF
try:
    from skrebate import ReliefF
    RELIEF_AVAILABLE = True
except ImportError:
    RELIEF_AVAILABLE = False
    print("⚠️ skrebate not installed. ReliefF will be skipped.")

# For SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ shap not installed. SHAP explanations will be skipped.")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ lime not installed. LIME explanations will be skipped.")

# Force CPU-only execution
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Running on CPU mode")
except Exception:
    print("⚠️ Could not change GPU visibility; continuing with available devices.")

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATED: Use CD-HIT homology-aware splits
DATA_DIR = r"D:\zebfish\data\evaluation_splits\filtered_4568_cdhit"
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

BASE_DIR = r"D:\zebfish1\revision1\FIGNet_CDHIT_Results"
os.makedirs(BASE_DIR, exist_ok=True)

# Hyperparameters
LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128]

EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42

# Selection metric for best model
SELECTION_METRIC = "mean_mcc"

# ReliefF parameters
RELIEF_FEATURES = [50, 100, 200]

# SHAP parameters - REDUCED FOR SPEED
SHAP_SAMPLES = 20
SHAP_BACKGROUND = 50

CLASS_NAMES = ["Non-enzyme", "Enzyme"]


# ============================================================================
# THREE-LEVEL CHECKPOINT SYSTEM
# ============================================================================

CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except:
            return {
                "completed_runs": {},
                "completed_splits": [],
                "completed_models": []
            }
    return {
        "completed_runs": {},
        "completed_splits": [],
        "completed_models": []
    }

def save_checkpoint(checkpoint):
    ensure_dir(os.path.dirname(CHECKPOINT_FILE))
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)

# Level 1: Individual Model Run
def is_run_completed(checkpoint, method_name, lr, bs):
    key = f"{method_name}_lr{lr}_bs{bs}"
    return key in checkpoint.get("completed_runs", {})

def mark_run_completed(checkpoint, method_name, lr, bs):
    key = f"{method_name}_lr{lr}_bs{bs}"
    if "completed_runs" not in checkpoint:
        checkpoint["completed_runs"] = {}
    checkpoint["completed_runs"][key] = {
        "method": method_name,
        "learning_rate": lr,
        "batch_size": bs,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_checkpoint(checkpoint)

# Level 2: All Models Complete
def is_split_completed(checkpoint):
    return checkpoint.get("split_completed", False)

def mark_split_completed(checkpoint):
    checkpoint["split_completed"] = True
    save_checkpoint(checkpoint)

# Level 3: Individual Model Complete
def is_model_completed(checkpoint, method_name, lr, bs):
    key = f"model_{method_name}_lr{lr}_bs{bs}"
    return key in checkpoint.get("completed_models", [])

def mark_model_completed(checkpoint, method_name, lr, bs):
    key = f"model_{method_name}_lr{lr}_bs{bs}"
    if "completed_models" not in checkpoint:
        checkpoint["completed_models"] = []
    if key not in checkpoint["completed_models"]:
        checkpoint["completed_models"].append(key)
        save_checkpoint(checkpoint)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

METHODS = {
    # === FIGNet Feature Gate Variants ===
    "FIGNet_Gate_Only": {
        "type": "fignet",
        "variant": "gate_only",
        "description": "Feature Importance Gate only (differentiable feature selection)",
    },
    "FIGNet_Gate_RealVD": {
        "type": "fignet",
        "variant": "gate_real_vd",
        "description": "FIGNet + Real Variational Dropout",
    },
    "FIGNet_Gate_AdaptiveVD": {
        "type": "fignet",
        "variant": "gate_adaptive_vd",
        "description": "FIGNet + Adaptive Variational Dropout",
    },
    "FIGNet_Gate_Sparsity": {
        "type": "fignet",
        "variant": "gate_sparsity",
        "description": "FIGNet + Dynamic Sparsity",
    },
    "FIGNet_Gate_Full": {
        "type": "fignet",
        "variant": "gate_full",
        "description": "FIGNet + All Components",
    },

    # === Existing Baselines ===
    "Logistic_Regression": {
        "type": "baseline_sklearn",
        "variant": "logistic",
        "description": "Logistic Regression (sklearn)",
    },
    "MLP_Baseline": {
        "type": "baseline_mlp",
        "variant": "mlp",
        "description": "Multi-Layer Perceptron Baseline",
    },

    # === Reviewer 1, Point 2: SVM Baselines ===
    "SVM_RBF": {
        "type": "baseline_sklearn",
        "variant": "svm_rbf",
        "description": "SVM with RBF kernel",
    },
    "SVM_Linear": {
        "type": "baseline_sklearn",
        "variant": "svm_linear",
        "description": "SVM with Linear kernel",
    },

    # === Reviewer 1, Point 2: ReliefF + Classifier ===
    "ReliefF_MLP": {
        "type": "relieff",
        "variant": "relieff_mlp",
        "description": "ReliefF feature selection + MLP classifier",
    },
    "ReliefF_SVM": {
        "type": "relieff",
        "variant": "relieff_svm",
        "description": "ReliefF feature selection + SVM classifier",
    },
}


# ============================================================================
# CUSTOM FIGNet LAYERS
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="FIGNet")
class FeatureImportanceGate(layers.Layer):
    """Differentiable soft feature gate for FIGNet."""

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

    def get_top_features(self, feature_names=None, k=20):
        gate_values = self.get_gate_values()
        k = min(k, len(gate_values))
        top_idx = np.argsort(gate_values)[-k:][::-1]
        top_values = gate_values[top_idx]
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_idx]
            return top_names, top_values
        return top_idx, top_values

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
    """Real Variational Dropout-style layer."""

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

    def get_dropout_rates(self):
        alpha = tf.exp(self.log_alpha).numpy()
        return alpha / (1.0 + alpha + self.eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "init_drop_rate": self.init_drop_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FIGNet")
class AdaptiveVariationalDropout(layers.Layer):
    """Adaptive dropout layer with learnable per-neuron dropout rates."""

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
            tf.shape(inputs),
            p=1.0 - drop_rate,
            dtype=inputs.dtype,
        )
        gaussian_noise = tf.random.normal(tf.shape(inputs), dtype=inputs.dtype) * self.noise_scale
        combined_noise = bernoulli_mask * (1.0 + gaussian_noise)
        scale = 1.0 / (1.0 - drop_rate + self.eps)
        return inputs * combined_noise * scale

    def get_drop_rates(self):
        if self.learnable:
            return tf.sigmoid(self.drop_logits).numpy()
        return np.ones(self.units) * self.initial_drop_rate

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
    """Dynamic sparsity regularizer with progressive target sparsity."""

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
# MODEL BUILDERS
# ============================================================================

def build_fignet_model(input_shape, num_classes, learning_rate, variant="gate_only"):
    """Build FIGNet feature gate models."""
    inputs = layers.Input(shape=(input_shape,), name="input")
    x = inputs

    x = FeatureImportanceGate(keep_ratio=0.8, name="feature_gate")(x)
    x = layers.BatchNormalization(name="bn1")(x)

    use_sparsity = variant in ["gate_sparsity", "gate_full"]

    # Stage 1
    if use_sparsity:
        x = layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense1",
        )(x)
    else:
        x = layers.Dense(512, activation="relu", name="dense1")(x)

    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(512, init_drop_rate=0.2, name="rvd1")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(512, initial_drop_rate=0.2, name="avd1")(x)

    x = layers.BatchNormalization(name="bn2")(x)

    # Stage 2 with residual connection
    shortcut = x

    if use_sparsity:
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense2",
        )(x)
    else:
        x = layers.Dense(256, activation="relu", name="dense2")(x)

    if variant in ["gate_real_vd", "gate_full"]:
        x = RealVariationalDropout(256, init_drop_rate=0.3, name="rvd2")(x)
    if variant in ["gate_adaptive_vd", "gate_full"]:
        x = AdaptiveVariationalDropout(256, initial_drop_rate=0.3, name="avd2")(x)

    x = layers.BatchNormalization(name="bn3")(x)

    if use_sparsity:
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=DynamicSparsityRegularizer(total_epochs=EPOCHS),
            name="dense3",
        )(x)
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

    # Stage 3
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
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_mlp_model(input_shape, num_classes, learning_rate):
    """Multi-Layer Perceptron baseline."""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation="relu", name="dense1"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu", name="dense2"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", name="dense3"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="dense4"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="MLP_Baseline")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_sklearn_classifier(X_train, y_train, X_test, y_test, method_config):
    """Train sklearn-based classifiers (Logistic Regression, SVM)."""
    variant = method_config["variant"]
    
    if variant == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    elif variant == "svm_rbf":
        clf = SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    elif variant == "svm_linear":
        clf = SVC(
            kernel="linear",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unknown sklearn variant: {variant}")
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    return y_pred, y_proba, train_time, clf


def train_relieff_classifier(X_train, y_train, X_test, y_test, method_config, feature_names):
    """Train ReliefF + classifier."""
    if not RELIEF_AVAILABLE:
        raise ImportError("skrebate not installed. Please install: pip install skrebate")
    
    variant = method_config["variant"]
    
    best_score = -1
    best_clf = None
    best_pred = None
    best_proba = None
    best_train_time = 0
    best_features = None
    best_n_features = 100
    
    for n_feat in RELIEF_FEATURES:
        n_feat = min(n_feat, X_train.shape[1])
        
        relieff = ReliefF(n_features_to_select=n_feat, n_neighbors=100)
        X_train_selected = relieff.fit_transform(X_train, y_train)
        X_test_selected = relieff.transform(X_test)
        
        if variant == "relieff_mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=200,
                random_state=RANDOM_STATE,
                early_stopping=True
            )
        else:  # relieff_svm
            clf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        
        start_time = time.time()
        clf.fit(X_train_selected, y_train)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test_selected)
        y_proba = clf.predict_proba(X_test_selected)[:, 1]
        
        score = matthews_corrcoef(y_test, y_pred)
        
        if score > best_score:
            best_score = score
            best_clf = clf
            best_pred = y_pred
            best_proba = y_proba
            best_train_time = train_time
            best_features = relieff.top_features_
            best_n_features = n_feat
    
    return best_pred, best_proba, best_train_time, best_clf, best_features, best_n_features


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_auc(y_true, y_score):
    """Safely compute ROC-AUC. Returns NaN if only one class is present."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


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
    """Calculate MCC on validation data after each epoch."""
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        logs["val_mcc"] = matthews_corrcoef(self.y_val, y_pred)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_training_history(history_dict, output_dir, method_name, fold_num):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history_dict.get("accuracy", []), label="Train Accuracy", linewidth=2)
    axes[0, 0].plot(history_dict.get("val_accuracy", []), label="Val Accuracy", linewidth=2)
    axes[0, 0].set_title("Accuracy over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_dict.get("loss", []), label="Train Loss", linewidth=2)
    axes[0, 1].plot(history_dict.get("val_loss", []), label="Val Loss", linewidth=2)
    axes[0, 1].set_title("Loss over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history_dict.get("precision", []), label="Train Precision", linewidth=2)
    axes[0, 2].plot(history_dict.get("val_precision", []), label="Val Precision", linewidth=2)
    axes[0, 2].set_title("Precision over Epochs")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history_dict.get("recall", []), label="Train Recall", linewidth=2)
    axes[1, 0].plot(history_dict.get("val_recall", []), label="Val Recall", linewidth=2)
    axes[1, 0].set_title("Recall over Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history_dict.get("auc", []), label="Train AUC", linewidth=2)
    axes[1, 1].plot(history_dict.get("val_auc", []), label="Val AUC", linewidth=2)
    axes[1, 1].set_title("AUC over Epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history_dict.get("val_mcc", []), label="Val MCC", linewidth=2)
    axes[1, 2].set_title("MCC over Epochs")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("MCC")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f"{method_name} - Fold {fold_num} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_fold{fold_num}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, output_path, title, cmap="Blues", average=False):
    plt.figure(figsize=(8, 6))
    fmt = ".1f" if average else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_stability(feature_importances_list, output_dir, method_name, feature_names=None):
    if len(feature_importances_list) < 2:
        return

    importance_array = np.array(feature_importances_list)
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)

    top_20_idx = np.argsort(mean_importance)[-20:]
    top_20_importance = mean_importance[top_20_idx]
    top_20_std = std_importance[top_20_idx]

    if feature_names is None:
        labels = [f"F{idx}" for idx in top_20_idx]
    else:
        labels = [str(feature_names[idx]) for idx in top_20_idx]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_20_importance)), top_20_importance, xerr=top_20_std, capsize=3, alpha=0.7)
    plt.yticks(range(len(top_20_importance)), labels)
    plt.xlabel("Mean Feature Importance")
    plt.title(f"{method_name} - Top 20 Features (Mean ± Std across CV folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_top_features(feature_importances_list, csv_dir, feature_names=None, top_k=50):
    if not feature_importances_list:
        return

    importance_array = np.array(feature_importances_list)
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)
    top_k = min(top_k, len(mean_importance))
    top_idx = np.argsort(mean_importance)[-top_k:][::-1]

    if feature_names is None:
        names = [f"F{idx}" for idx in top_idx]
    else:
        names = [feature_names[idx] for idx in top_idx]

    top_df = pd.DataFrame({
        "Rank": range(1, top_k + 1),
        "Feature_Index": top_idx,
        "Feature_Name": names,
        "Mean_Importance": mean_importance[top_idx],
        "Std_Importance": std_importance[top_idx],
    })
    top_df.to_csv(os.path.join(csv_dir, "Top_Features_From_CV.csv"), index=False)


def extract_feature_importance(model, layer_name="feature_gate"):
    """Extract gate values from the Feature Importance Gate layer."""
    for layer in model.layers:
        if layer.name == layer_name and hasattr(layer, "get_gate_values"):
            return layer.get_gate_values()
    return None


def calculate_jaccard_stability(feature_importances_list, top_k=50):
    """Calculate Jaccard similarity of top-k features across folds."""
    if len(feature_importances_list) < 2:
        return 0.0

    top_feature_sets = []
    for importance in feature_importances_list:
        if importance is not None and len(importance) >= top_k:
            top_indices = np.argsort(importance)[-top_k:].tolist()
            top_feature_sets.append(set(top_indices))

    if len(top_feature_sets) < 2:
        return 0.0

    scores = []
    for i in range(len(top_feature_sets)):
        for j in range(i + 1, len(top_feature_sets)):
            union = len(top_feature_sets[i] | top_feature_sets[j])
            if union > 0:
                intersection = len(top_feature_sets[i] & top_feature_sets[j])
                scores.append(intersection / union)

    return float(np.mean(scores)) if scores else 0.0


# ============================================================================
# SHAP AND LIME EXPLANATIONS
# ============================================================================

def run_shap_explanations(model, X_sample, feature_names, output_dir):
    """Run SHAP explanations."""
    if not SHAP_AVAILABLE:
        print("  ⚠️ SHAP not available, skipping")
        return None
    
    try:
        X_sample = np.array(X_sample, dtype=np.float32)
        
        def predict_fn(x):
            x = np.array(x, dtype=np.float32)
            return model.predict(x, verbose=0)
        
        n_background = min(SHAP_BACKGROUND, len(X_sample))
        n_explain = min(SHAP_SAMPLES, len(X_sample))
        
        background = X_sample[:n_background]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample[:n_explain])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        shap_dir = ensure_dir(os.path.join(output_dir, "shap_explanations"))
        
        n_features = shap_values.shape[1]
        safe_feature_names = [f"Embedding_{i}" for i in range(n_features)]
        
        plot_features = min(50, n_features)
        plot_names = safe_feature_names[:plot_features]
        plot_values = shap_values[:, :plot_features]
        plot_X = X_sample[:n_explain, :plot_features]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            plot_values, 
            plot_X, 
            feature_names=plot_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        np.save(os.path.join(shap_dir, "shap_values.npy"), shap_values)
        
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
        
        shap_summary = {
            "method": "SHAP",
            "top_features_indices": top_indices.tolist(),
            "top_features_names": [f"Embedding_{i}" for i in top_indices],
            "mean_abs_shap_values": mean_abs_shap[top_indices].tolist()
        }
        save_json(shap_summary, os.path.join(shap_dir, "shap_summary.json"))
        
        print(f"  ✅ SHAP explanations saved to: {shap_dir}")
        return shap_summary
        
    except Exception as e:
        print(f"  ❌ SHAP failed: {e}")
        return None


def run_lime_explanations(model, X_sample, y_sample, feature_names, output_dir):
    """Run LIME explanations with safe feature names."""
    if not LIME_AVAILABLE:
        print("  ⚠️ LIME not available, skipping")
        return None
    
    try:
        def predict_fn(x):
            return model.predict(x, verbose=0)
        
        lime_dir = ensure_dir(os.path.join(output_dir, "lime_explanations"))
        
        n_features = X_sample.shape[1]
        if feature_names is None or len(feature_names) < n_features:
            safe_feature_names = [f"Embedding_{i}" for i in range(n_features)]
        else:
            safe_feature_names = feature_names[:n_features]
        
        explainer = LimeTabularExplainer(
            X_sample,
            feature_names=safe_feature_names,
            class_names=CLASS_NAMES,
            mode="classification"
        )
        
        lime_explanations = []
        n_samples = min(10, len(X_sample))
        
        for i in range(n_samples):
            exp = explainer.explain_instance(
                X_sample[i], 
                predict_fn, 
                num_features=20
            )
            
            top_features = exp.as_list()
            lime_explanations.append({
                "sample_index": i,
                "true_label": int(y_sample[i]),
                "top_features": top_features
            })
            
            exp.save_to_file(os.path.join(lime_dir, f"lime_explanation_sample_{i}.html"))
        
        save_json(lime_explanations, os.path.join(lime_dir, "lime_explanations_summary.json"))
        
        print(f"  ✅ LIME explanations saved to: {lime_dir}")
        return lime_explanations
        
    except Exception as e:
        print(f"  ❌ LIME failed: {e}")
        return None


# ============================================================================
# TRAIN AND EVALUATE ON CD-HIT SPLITS
# ============================================================================
def run_cdhit_experiment(
    method_name,
    config,
    learning_rate,
    batch_size,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_names=None,
    is_sklearn=False,
):
    """
    Train on CD-HIT train set, validate on val set, test on test set.
    """
    # Check if already completed (Level 1)
    checkpoint = load_checkpoint()
    
    if is_run_completed(checkpoint, method_name, learning_rate, batch_size):
        print(f"\n⏭️ Skipping {method_name} (lr={learning_rate}, bs={batch_size}) - Already completed")
        return {"status": "already_completed", "method": method_name}
    
    lr_str = f"{learning_rate:.5f}".replace(".", "_")
    output_dir = ensure_dir(os.path.join(BASE_DIR, "runs", f"lr_{lr_str}_bs_{batch_size}", method_name))
    npy_dir = ensure_dir(os.path.join(output_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))

    num_classes = 2
    input_shape = X_train.shape[1]

    print(f"\n{'=' * 80}")
    print(f"RUN: {method_name} | lr={learning_rate} | batch_size={batch_size}")
    print(f"Description: {config['description']}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"{'=' * 80}")

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    save_json({
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, os.path.join(output_dir, "scaler_params.json"))

    # For sklearn models
    if is_sklearn or config["type"] in ["baseline_sklearn", "relieff"]:
        if config["type"] == "relieff":
            if not RELIEF_AVAILABLE:
                raise ImportError("skrebate not installed. Please install: pip install skrebate")
            y_pred, y_proba, train_time, clf, best_features, n_feat = train_relieff_classifier(
                X_train_scaled, y_train, X_test_scaled, y_test, config, feature_names
            )
            save_json({
                "n_features_selected": n_feat,
                "best_features_indices": best_features.tolist() if best_features is not None else None
            }, os.path.join(npy_dir, "relieff_features.json"))
        else:
            y_pred, y_proba, train_time, clf = train_sklearn_classifier(
                X_train_scaled, y_train, X_test_scaled, y_test, config
            )
        
        import joblib
        joblib.dump(clf, os.path.join(models_dir, "model.joblib"))
        
        # ✅ FIXED: Use scalar values, not lists
        val_history = {
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
            "val_mcc": 0.0,
            "val_auc": 0.0
        }
        epochs_trained = 1

    else:
        # TensorFlow models
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

        tf.keras.backend.clear_session()
        
        if config["type"] == "fignet":
            model = build_fignet_model(input_shape, num_classes, learning_rate, config["variant"])
        elif config["type"] == "baseline_mlp":
            model = build_mlp_model(input_shape, num_classes, learning_rate)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
        dyn_callback = DynamicSparsityCallback()
        mcc_callback = MCCCallback(validation_data=(X_val_scaled, y_val))

        start_time = time.time()
        history = model.fit(
            X_train_scaled,
            y_train_cat,
            epochs=EPOCHS,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val_cat),
            verbose=0,
            callbacks=[early_stop, dyn_callback, mcc_callback],
        )
        train_time = time.time() - start_time
        epochs_trained = len(history.history.get("loss", []))

        # Save all history .npy files
        for metric, values in history.history.items():
            np.save(os.path.join(npy_dir, f"{metric}.npy"), np.array(values))

        plot_training_history(history.history, plots_dir, method_name, "final")

        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_proba = y_pred_proba[:, 1]

        # Get val predictions for metrics
        val_pred_proba = model.predict(X_val_scaled, verbose=0)
        val_pred = np.argmax(val_pred_proba, axis=1)

        # Calculate validation metrics
        val_acc = accuracy_score(y_val, val_pred)
        val_precision = precision_score(y_val, val_pred, zero_division=0)
        val_recall = recall_score(y_val, val_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        val_mcc = matthews_corrcoef(y_val, val_pred)
        val_auc = safe_auc(y_val, val_pred_proba[:, 1])
        
        # ✅ FIXED: Use scalar values
        val_history = {
            "val_accuracy": float(val_acc),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "val_mcc": float(val_mcc),
            "val_auc": float(val_auc) if not np.isnan(val_auc) else 0.0
        }

        if config["type"] == "fignet":
            importance = extract_feature_importance(model, layer_name="feature_gate")
            if importance is not None:
                np.save(os.path.join(npy_dir, "feature_importance.npy"), importance)
                top_idx = np.argsort(importance)[-50:][::-1]
                top_names = [feature_names[i] if feature_names is not None else f"F{i}" for i in top_idx]
                pd.DataFrame({
                    "Rank": range(1, len(top_idx) + 1),
                    "Feature_Index": top_idx,
                    "Feature_Name": top_names,
                    "Importance": importance[top_idx],
                }).to_csv(os.path.join(csv_dir, "Top_Features.csv"), index=False)

            # SHAP explanations (Reviewer 2, Point 5)
            if SHAP_AVAILABLE and len(X_test_scaled) > 0:
                print("  Running SHAP explanations...")
                shap_results = run_shap_explanations(
                    model, X_test_scaled[:min(100, len(X_test_scaled))], 
                    feature_names, output_dir
                )
            
            # LIME explanations (Reviewer 2, Point 5)
            if LIME_AVAILABLE and len(X_test_scaled) > 0:
                print("  Running LIME explanations...")
                lime_results = run_lime_explanations(
                    model, X_test_scaled[:min(50, len(X_test_scaled))], 
                    y_test[:min(50, len(y_test))], feature_names, output_dir
                )

        model.save(os.path.join(models_dir, "model.keras"))

    # Calculate test metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_pred)
    test_auc = safe_auc(y_test, y_proba)

    test_cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(npy_dir, "test_predictions.npy"), y_proba)
    np.save(os.path.join(npy_dir, "test_predicted_classes.npy"), y_pred)
    np.save(os.path.join(npy_dir, "test_true_labels.npy"), y_test)
    np.save(os.path.join(npy_dir, "test_confusion_matrix.npy"), test_cm)

    plot_confusion_matrix(
        test_cm,
        os.path.join(plots_dir, "test_confusion_matrix.png"),
        f"{method_name} - Test Confusion Matrix",
        cmap="Greens",
    )

    test_predictions_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
        "predicted_proba_enzyme": y_proba,
        "correct": y_test == y_pred,
    })
    test_predictions_df.to_csv(os.path.join(csv_dir, "Test_Predictions.csv"), index=False)

    print(f"\nTest Results:")
    print(f"  Acc: {test_acc:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}")

    summary = {
        "status": "success",
        "method": method_name,
        "description": config["description"],
        "variant": config["variant"],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "val_accuracy": float(val_history.get("val_accuracy", 0.0)),
        "val_precision": float(val_history.get("val_precision", 0.0)),
        "val_recall": float(val_history.get("val_recall", 0.0)),
        "val_f1": float(val_history.get("val_f1", 0.0)),
        "val_mcc": float(val_history.get("val_mcc", 0.0)),
        "val_auc": float(val_history.get("val_auc", 0.0)),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_mcc": float(test_mcc),
        "test_auc": float(test_auc) if not np.isnan(test_auc) else 0.0,
        "training_time_seconds": float(train_time),
        "epochs_trained": int(epochs_trained),
        "is_sklearn": is_sklearn or config["type"] in ["baseline_sklearn", "relieff"],
    }

    # Save summary
    summary_df = pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, list)}])
    summary_df.to_csv(os.path.join(csv_dir, "Experiment_Summary.csv"), index=False)

    # Mark Level 1: Model Run Completed
    mark_run_completed(checkpoint, method_name, learning_rate, batch_size)

    return summary



# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_cdhit_experiments():
    """
    Main experiment loop with three-level checkpointing.
    """
    print("=" * 80)
    print("FIGNet REVISION: CD-HIT HOMOLOGY-AWARE SPLITS")
    print("ALL MODELS TRAINED ON CD-HIT SPLITS")
    print("THREE-LEVEL CHECKPOINT SYSTEM ENABLED")
    print("=" * 80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {BASE_DIR}")
    print(f"Models: {len(METHODS)} total")
    print("=" * 80)

    # Load CD-HIT splits
    print("\n📊 Loading CD-HIT splits:")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"   Train: {len(train_df)} samples ({train_df['is_enzyme'].sum()} enzymes)")
    print(f"   Val:   {len(val_df)} samples ({val_df['is_enzyme'].sum()} enzymes)")
    print(f"   Test:  {len(test_df)} samples ({test_df['is_enzyme'].sum()} enzymes)")

    # Extract features and labels
    # Get embedding columns (all columns except metadata)
    exclude_cols = ['protein_id', 'is_enzyme', 'species', 'ec_numbers']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['is_enzyme'].values.astype(int)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['is_enzyme'].values.astype(int)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['is_enzyme'].values.astype(int)

    feature_names = feature_cols

    print(f"\n📊 Feature dimensions:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")

    print("\n" + "=" * 80)

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_runs = checkpoint.get("completed_runs", {})
    
    print(f"\n📌 Checkpoint loaded:")
    print(f"   Completed runs: {len(completed_runs)} / {len(METHODS) * len(LEARNING_RATES) * len(BATCH_SIZES)}")

    # Run experiments
    all_results = []
    is_sklearn_methods = ["baseline_sklearn", "relieff"]

    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for method_name, config in METHODS.items():
                is_sklearn = config["type"] in is_sklearn_methods
                
                # Use default lr/bs for sklearn (they don't use these params)
                actual_lr = learning_rate if not is_sklearn else 0.001
                actual_bs = batch_size if not is_sklearn else 32
                
                # Skip if already completed
                if is_run_completed(checkpoint, method_name, actual_lr, actual_bs):
                    print(f"⏭️ Skipping {method_name} (lr={actual_lr}, bs={actual_bs}) - Already done")
                    continue
                
                try:
                    result = run_cdhit_experiment(
                        method_name=method_name,
                        config=config,
                        learning_rate=actual_lr,
                        batch_size=actual_bs,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        feature_names=feature_names,
                        is_sklearn=is_sklearn,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error in {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

    # =============================================
    # AGGREGATE RESULTS
    # =============================================

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Load all completed results from checkpoint
    all_final_results = []
    for key, info in completed_runs.items():
        # Try to load the summary from the output directory
        method = info["method"]
        lr = info["learning_rate"]
        bs = info["batch_size"]
        
        lr_str = f"{lr:.5f}".replace(".", "_")
        summary_path = os.path.join(BASE_DIR, "runs", f"lr_{lr_str}_bs_{bs}", method, "csv_files", "Experiment_Summary.csv")
        
        if os.path.exists(summary_path):
            try:
                df = pd.read_csv(summary_path)
                all_final_results.append(df.iloc[0].to_dict())
            except:
                pass

    if all_final_results:
        results_df = pd.DataFrame(all_final_results)
        
        # Sort by test MCC
        if 'test_mcc' in results_df.columns:
            results_df = results_df.sort_values('test_mcc', ascending=False)
        
        results_path = os.path.join(BASE_DIR, "CDHIT_Results_All_Models.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        print(f"   Total completed: {len(results_df)}")

        # Summary by method
        if 'method' in results_df.columns and 'test_mcc' in results_df.columns:
            print("\n📊 Performance by Method:")
            print("-" * 80)
            
            method_summary = results_df.groupby('method').agg({
                'test_accuracy': ['mean', 'std'],
                'test_f1': ['mean', 'std'],
                'test_mcc': ['mean', 'std'],
                'test_auc': ['mean', 'std']
            }).round(4)
            
            method_summary.columns = ['Acc_mean', 'Acc_std', 'F1_mean', 'F1_std', 
                                       'MCC_mean', 'MCC_std', 'AUC_mean', 'AUC_std']
            method_summary = method_summary.sort_values('MCC_mean', ascending=False)
            
            print(method_summary.to_string())
            
            method_summary_path = os.path.join(BASE_DIR, "CDHIT_Method_Summary.csv")
            method_summary.to_csv(method_summary_path)
            print(f"\n✅ Method summary saved to: {method_summary_path}")
    else:
        print("\n⚠️ No results found. Check if experiments completed successfully.")

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print(f"All results saved to: {BASE_DIR}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_cdhit_experiments()