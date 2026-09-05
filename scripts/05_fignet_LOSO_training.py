# -*- coding: utf-8 -*-
"""
FIGNet: Feature Importance Gate Network for Interpretable Enzyme Classification
COMPLETE REVISION - Leave-One-Species-Out with All Requested Baselines
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

Evaluation: Leave-One-Species-Out (LOSO) cross-validation
ALL MODELS TESTED ON EACH HELD-OUT SPECIES

Checkpoint Levels:
   Level 1: Individual model runs (method + lr + bs)
   Level 2: Species CV complete (all models done)
   Level 3: Individual LOSO model complete (method + lr + bs)
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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
# Suppress tf.function retracing warnings
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

# Optional: force CPU-only execution.
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

DATA_PATH = r"D:\zebfish\revision\zebfish_processed_results\combined_data\binary_classification_with_species.csv"
BASE_DIR = r"D:\zebfish1\revision1\FIGNet_LOSO_Results\FIGNET_0.01_64"
os.makedirs(BASE_DIR, exist_ok=True)

# Hyperparameters
LEARNING_RATES = [0.01,0.001,0.0001]
BATCH_SIZES = [32,64,128]

EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
N_FOLDS = 10
RANDOM_STATE = 42

# Selection metric for best model (used for ranking, but ALL models are tested)
SELECTION_METRIC = "mean_mcc"

# ReliefF parameters
RELIEF_FEATURES = [50, 100, 200]

# SHAP parameters - REDUCED FOR SPEED
SHAP_SAMPLES = 20  # Reduced from 50 to speed up
SHAP_BACKGROUND = 50  # Reduced from 100 to speed up

CLASS_NAMES = ["Non-enzyme", "Enzyme"]


# ============================================================================
# THREE-LEVEL CHECKPOINT SYSTEM
# ============================================================================

CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")

def load_checkpoint():
    """Load checkpoint with three levels."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except:
            return {
                "completed_runs": {},
                "completed_cv_species": [],
                "completed_loso_models": []
            }
    return {
        "completed_runs": {},
        "completed_cv_species": [],
        "completed_loso_models": []
    }

def save_checkpoint(checkpoint):
    """Save current progress."""
    ensure_dir(os.path.dirname(CHECKPOINT_FILE))
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)

# Level 1: Individual Model Run
def is_run_completed(checkpoint, species, method_name, lr, bs):
    key = f"{species}_{method_name}_lr{lr}_bs{bs}"
    return key in checkpoint.get("completed_runs", {})

def mark_run_completed(checkpoint, species, method_name, lr, bs):
    key = f"{species}_{method_name}_lr{lr}_bs{bs}"
    if "completed_runs" not in checkpoint:
        checkpoint["completed_runs"] = {}
    checkpoint["completed_runs"][key] = {
        "species": species,
        "method": method_name,
        "learning_rate": lr,
        "batch_size": bs,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_checkpoint(checkpoint)

# Level 2: Species CV Complete
def is_cv_species_completed(checkpoint, species):
    return species in checkpoint.get("completed_cv_species", [])

def mark_cv_species_completed(checkpoint, species):
    if species not in checkpoint.get("completed_cv_species", []):
        if "completed_cv_species" not in checkpoint:
            checkpoint["completed_cv_species"] = []
        checkpoint["completed_cv_species"].append(species)
        save_checkpoint(checkpoint)

# Level 3: Individual LOSO Model Complete
def is_loso_model_completed(checkpoint, species, method_name, lr, bs):
    key = f"loso_{species}_{method_name}_lr{lr}_bs{bs}"
    return key in checkpoint.get("completed_loso_models", [])

def mark_loso_model_completed(checkpoint, species, method_name, lr, bs):
    key = f"loso_{species}_{method_name}_lr{lr}_bs{bs}"
    if "completed_loso_models" not in checkpoint:
        checkpoint["completed_loso_models"] = []
    if key not in checkpoint["completed_loso_models"]:
        checkpoint["completed_loso_models"].append(key)
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


def label_column_from_dataframe(df):
    """Support both versions of your dataset label column."""
    if "Label" in df.columns:
        return "Label"
    if "Binary_Target" in df.columns:
        return "Binary_Target"
    raise ValueError("Dataset must contain either a 'Label' or 'Binary_Target' column.")


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


# ============================================================================
# FEATURE STABILITY FUNCTIONS
# ============================================================================

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
# SHAP AND LIME EXPLANATIONS (Reviewer 2, Point 5) - REDUCED SAMPLES
# ============================================================================
def run_shap_explanations(model, X_sample, feature_names, output_dir):
    """Run SHAP explanations - AGGRESSIVE FIX (no external dependencies)."""
    if not SHAP_AVAILABLE:
        print("  ⚠️ SHAP not available, skipping")
        return None
    
    try:
        # Force float32 type
        X_sample = np.array(X_sample, dtype=np.float32)
        
        def predict_fn(x):
            x = np.array(x, dtype=np.float32)
            return model.predict(x, verbose=0)
        
        n_background = min(SHAP_BACKGROUND, len(X_sample))
        n_explain = min(SHAP_SAMPLES, len(X_sample))
        
        background = X_sample[:n_background]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample[:n_explain])
        
        # Handle list output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (Enzyme)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        shap_dir = ensure_dir(os.path.join(output_dir, "shap_explanations"))
        
        # ================================================================
        # CRITICAL FIX: Create our OWN feature names, completely ignore input
        # ================================================================
        n_features = shap_values.shape[1]
        safe_feature_names = [f"Embedding_{i}" for i in range(n_features)]
        
        # Limit to first 50 features for plotting (avoid overcrowding)
        plot_features = min(50, n_features)
        plot_names = safe_feature_names[:plot_features]
        plot_values = shap_values[:, :plot_features]
        plot_X = X_sample[:n_explain, :plot_features]
        
        # ================================================================
        # PLOT: Use our own safe feature names
        # ================================================================
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            plot_values, 
            plot_X, 
            feature_names=plot_names,  # SAFE: always valid list
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save SHAP values (full)
        np.save(os.path.join(shap_dir, "shap_values.npy"), shap_values)
        
        # Get top features from FULL shap_values
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
        import traceback
        traceback.print_exc()
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
        
        # Safe feature names
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
# CROSS-VALIDATION EXPERIMENT
# ============================================================================

def run_cross_validation_experiment(
    method_name,
    config,
    learning_rate,
    batch_size,
    X_train_val_raw,
    y_train_val,
    train_val_indices,
    feature_names=None,
    is_sklearn=False,
    species_name="unknown",
):
    """
    Run stratified 10-fold CV on the training+validation set.
    """
    # Check if already completed (Level 1)
    checkpoint = load_checkpoint()
    
    if is_run_completed(checkpoint, species_name, method_name, learning_rate, batch_size):
        print(f"\n⏭️ Skipping {method_name} (lr={learning_rate}, bs={batch_size}) - Already completed")
        return {"status": "already_completed", "method": method_name}
    
    lr_str = f"{learning_rate:.5f}".replace(".", "_")
    output_dir = ensure_dir(os.path.join(BASE_DIR, "cv_runs", species_name, f"lr_{lr_str}_bs_{batch_size}", method_name))
    npy_dir = ensure_dir(os.path.join(output_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))

    num_classes = 2
    input_shape = X_train_val_raw.shape[1]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_acc = []
    fold_precision = []
    fold_recall = []
    fold_f1 = []
    fold_mcc = []
    fold_auc = []
    fold_times = []
    fold_epochs = []
    all_predictions = []
    confusion_matrices = []
    feature_importances = []

    print(f"\n{'=' * 80}")
    print(f"CV RUN: {method_name} | lr={learning_rate} | batch_size={batch_size}")
    print(f"Description: {config['description']}")
    print(f"Species: {species_name}")
    print(f"{'=' * 80}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_raw, y_train_val), start=1):
        print(f"\nFold {fold}/{N_FOLDS} - {method_name}")

        X_tr_raw = X_train_val_raw[train_idx]
        X_val_raw = X_train_val_raw[val_idx]
        y_tr = y_train_val[train_idx]
        y_val = y_train_val[val_idx]

        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_val = fold_scaler.transform(X_val_raw).astype(np.float32)

        save_json({
            "train_indices_original_dataset": train_val_indices[train_idx].tolist(),
            "val_indices_original_dataset": train_val_indices[val_idx].tolist(),
        }, os.path.join(npy_dir, f"fold{fold}_indices.json"))

        save_json({
            "mean": fold_scaler.mean_.tolist(),
            "scale": fold_scaler.scale_.tolist(),
        }, os.path.join(npy_dir, f"fold{fold}_scaler_params.json"))

        # For sklearn models
        if is_sklearn or config["type"] in ["baseline_sklearn", "relieff"]:
            if config["type"] == "relieff":
                if not RELIEF_AVAILABLE:
                    raise ImportError("skrebate not installed. Please install: pip install skrebate")
                y_pred, y_proba, train_time, clf, best_features, n_feat = train_relieff_classifier(
                    X_tr, y_tr, X_val, y_val, config, feature_names
                )
                save_json({
                    "n_features_selected": n_feat,
                    "best_features_indices": best_features.tolist() if best_features is not None else None
                }, os.path.join(npy_dir, f"fold{fold}_relieff_features.json"))
            else:
                y_pred, y_proba, train_time, clf = train_sklearn_classifier(
                    X_tr, y_tr, X_val, y_val, config
                )
            
            fold_epochs.append(1)
            import joblib
            joblib.dump(clf, os.path.join(models_dir, f"fold{fold}_model.joblib"))

        else:
            # TensorFlow models
            y_tr_cat = tf.keras.utils.to_categorical(y_tr, num_classes)
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
            mcc_callback = MCCCallback(validation_data=(X_val, y_val))

            start_time = time.time()
            history = model.fit(
                X_tr,
                y_tr_cat,
                epochs=EPOCHS,
                batch_size=batch_size,
                validation_data=(X_val, y_val_cat),
                verbose=0,
                callbacks=[early_stop, dyn_callback, mcc_callback],
            )
            train_time = time.time() - start_time
            fold_epochs.append(len(history.history.get("loss", [])))

            # Save all history .npy files
            for metric, values in history.history.items():
                np.save(os.path.join(npy_dir, f"fold{fold}_{metric}.npy"), np.array(values))

            plot_training_history(history.history, plots_dir, method_name, fold)

            y_pred_proba = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_proba = y_pred_proba[:, 1]

            if config["type"] == "fignet":
                importance = extract_feature_importance(model, layer_name="feature_gate")
                if importance is not None:
                    feature_importances.append(importance)
                    np.save(os.path.join(npy_dir, f"fold{fold}_feature_importance.npy"), importance)

                    top_idx = np.argsort(importance)[-50:][::-1]
                    top_names = [feature_names[i] if feature_names is not None else f"F{i}" for i in top_idx]
                    pd.DataFrame({
                        "Rank": range(1, len(top_idx) + 1),
                        "Feature_Index": top_idx,
                        "Feature_Name": top_names,
                        "Importance": importance[top_idx],
                    }).to_csv(os.path.join(csv_dir, f"fold{fold}_Top_Features.csv"), index=False)

            model.save(os.path.join(models_dir, f"fold{fold}_model.keras"))

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_val, y_pred)
        auc = safe_auc(y_val, y_proba)

        cm = confusion_matrix(y_val, y_pred)
        confusion_matrices.append(cm)
        np.save(os.path.join(npy_dir, f"fold{fold}_confusion_matrix.npy"), cm)
        plot_confusion_matrix(
            cm,
            os.path.join(plots_dir, f"confusion_matrix_fold{fold}.png"),
            f"{method_name} - Fold {fold} Confusion Matrix",
            cmap="Blues",
        )

        for local_i, (true_label, pred_label, proba) in enumerate(zip(y_val, y_pred, y_proba)):
            original_sample_idx = int(train_val_indices[val_idx[local_i]])
            all_predictions.append({
                "fold": fold,
                "sample_idx_original_dataset": original_sample_idx,
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "predicted_proba_enzyme": float(proba),
                "correct": bool(true_label == pred_label),
            })

        print(
            f"  Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, "
            f"F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, Epochs: {fold_epochs[-1] if fold_epochs else 1}"
        )

        fold_acc.append(acc)
        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)
        fold_mcc.append(mcc)
        fold_auc.append(auc)
        fold_times.append(train_time)

    # Feature stability
    jaccard_stability = calculate_jaccard_stability(feature_importances, top_k=50)
    if len(feature_importances) >= 2 and config["type"] == "fignet":
        plot_feature_stability(feature_importances, plots_dir, method_name, feature_names=feature_names)
        save_top_features(feature_importances, csv_dir, feature_names=feature_names, top_k=50)

    # Average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    np.save(os.path.join(npy_dir, "average_confusion_matrix.npy"), avg_cm)
    plot_confusion_matrix(
        avg_cm,
        os.path.join(plots_dir, "average_confusion_matrix.png"),
        f"{method_name} - Average Confusion Matrix ({N_FOLDS}-Fold CV)",
        cmap="Blues",
        average=True,
    )

    # Save fold metrics
    fold_metrics_df = pd.DataFrame({
        "Fold": range(1, N_FOLDS + 1),
        "Accuracy": fold_acc,
        "Precision": fold_precision,
        "Recall": fold_recall,
        "F1": fold_f1,
        "MCC": fold_mcc,
        "AUC": fold_auc,
        "Training_Time_Seconds": fold_times,
        "Epochs_Trained": fold_epochs if fold_epochs else [1] * N_FOLDS,
    })
    fold_metrics_df.to_csv(os.path.join(csv_dir, "Fold_Metrics.csv"), index=False)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(os.path.join(csv_dir, "All_CV_Predictions.csv"), index=False)

    recommended_epochs = int(max(1, round(np.median(fold_epochs)))) if fold_epochs else 1

    summary = {
        "status": "success",
        "method": method_name,
        "description": config["description"],
        "variant": config["variant"],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_precision": float(np.mean(fold_precision)),
        "std_precision": float(np.std(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "std_recall": float(np.std(fold_recall)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "mean_mcc": float(np.mean(fold_mcc)),
        "std_mcc": float(np.std(fold_mcc)),
        "mean_auc": float(np.nanmean(fold_auc)),
        "std_auc": float(np.nanstd(fold_auc)),
        "feature_stability_jaccard": float(jaccard_stability),
        "mean_training_time_seconds": float(np.mean(fold_times)),
        "recommended_epochs_for_final_training": recommended_epochs,
        "output_dir": output_dir,
        "fold_accuracy_values": [float(x) for x in fold_acc],
        "fold_mcc_values": [float(x) for x in fold_mcc],
        "fold_f1_values": [float(x) for x in fold_f1],
        "fold_auc_values": [float(x) for x in fold_auc],
        "is_sklearn": is_sklearn or config["type"] in ["baseline_sklearn", "relieff"],
    }

    # FIXED: Save summary with safe column access
    summary_df = pd.DataFrame([
        {k: v for k, v in summary.items() if not isinstance(v, list)}
    ])
    summary_df.to_csv(os.path.join(csv_dir, "Experiment_Summary.csv"), index=False)

    print(f"\n{method_name} CV Summary")
    print("-" * 60)
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Mean F1:       {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Mean MCC:      {summary['mean_mcc']:.4f} ± {summary['std_mcc']:.4f}")
    print(f"Mean AUC:      {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"Jaccard Stability: {summary['feature_stability_jaccard']:.4f}")

    # Mark Level 1: Model Run Completed
    mark_run_completed(checkpoint, species_name, method_name, learning_rate, batch_size)

    return summary


# ============================================================================
# FINAL INDEPENDENT TEST EVALUATION (LOSO) - UPDATED
# ============================================================================

def run_final_test_evaluation_loso(
    best_result,
    X_train_val_raw,
    y_train_val,
    X_test_raw,
    y_test,
    test_indices,
    species_name,
    feature_names=None,
):
    """
    Train the CV-selected model on the full training data and evaluate on held-out species.
    """
    method_name = best_result["method"]
    config = METHODS[method_name]
    learning_rate = best_result["learning_rate"]
    batch_size = best_result["batch_size"]
    final_epochs = best_result["recommended_epochs_for_final_training"]
    is_sklearn = best_result.get("is_sklearn", False)

    # Create output directory with method and hyperparameters
    output_dir = ensure_dir(os.path.join(BASE_DIR, "loso_results", species_name, 
                           f"{method_name}_lr{learning_rate}_bs{batch_size}"))
    npy_dir = ensure_dir(os.path.join(output_dir, "npy_files"))
    csv_dir = ensure_dir(os.path.join(output_dir, "csv_files"))
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))

    print("\n" + "=" * 80)
    print(f"FINAL INDEPENDENT TEST EVALUATION - LOSO: {species_name}")
    print("=" * 80)
    print(f"Selected model: {method_name}")
    print(f"Description: {best_result['description']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Final training epochs from CV median: {final_epochs}")
    print(f"Test species: {species_name} ({len(y_test)} samples)")
    print("=" * 80)

    # Scale data
    final_scaler = StandardScaler()
    X_train_val = final_scaler.fit_transform(X_train_val_raw).astype(np.float32)
    X_test = final_scaler.transform(X_test_raw).astype(np.float32)

    save_json({
        "mean": final_scaler.mean_.tolist(),
        "scale": final_scaler.scale_.tolist(),
    }, os.path.join(output_dir, "final_scaler_params.json"))

    # For sklearn models
    if is_sklearn or config["type"] in ["baseline_sklearn", "relieff"]:
        start_time = time.time()
        
        if config["type"] == "relieff":
            if not RELIEF_AVAILABLE:
                raise ImportError("skrebate not installed. Please install: pip install skrebate")
            y_pred, y_proba, train_time, clf, best_features, n_feat = train_relieff_classifier(
                X_train_val, y_train_val, X_test, y_test, config, feature_names
            )
            save_json({
                "n_features_selected": n_feat,
                "best_features_indices": best_features.tolist() if best_features is not None else None
            }, os.path.join(npy_dir, "final_relieff_features.json"))
        else:
            y_pred, y_proba, train_time, clf = train_sklearn_classifier(
                X_train_val, y_train_val, X_test, y_test, config
            )
        
        import joblib
        joblib.dump(clf, os.path.join(models_dir, "final_selected_model.joblib"))
        final_training_time = train_time
        
    else:
        # TensorFlow models
        num_classes = 2
        input_shape = X_train_val.shape[1]
        y_train_val_cat = tf.keras.utils.to_categorical(y_train_val, num_classes)

        tf.keras.backend.clear_session()
        
        if config["type"] == "fignet":
            final_model = build_fignet_model(input_shape, num_classes, learning_rate, config["variant"])
        elif config["type"] == "baseline_mlp":
            final_model = build_mlp_model(input_shape, num_classes, learning_rate)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        start_time = time.time()
        final_history = final_model.fit(
            X_train_val,
            y_train_val_cat,
            epochs=final_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[DynamicSparsityCallback()],
        )
        final_training_time = time.time() - start_time

        # Save all history .npy files
        for metric, values in final_history.history.items():
            np.save(os.path.join(npy_dir, f"final_train_{metric}.npy"), np.array(values))

        y_pred_proba = final_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_proba = y_pred_proba[:, 1]

        if config["type"] == "fignet":
            importance = extract_feature_importance(final_model, layer_name="feature_gate")
            if importance is not None:
                np.save(os.path.join(npy_dir, "final_feature_importance.npy"), importance)
                top_idx = np.argsort(importance)[-50:][::-1]
                top_names = [feature_names[i] if feature_names is not None else f"F{i}" for i in top_idx]
                pd.DataFrame({
                    "Rank": range(1, len(top_idx) + 1),
                    "Feature_Index": top_idx,
                    "Feature_Name": top_names,
                    "Importance": importance[top_idx],
                }).to_csv(os.path.join(csv_dir, "Final_Top_Features.csv"), index=False)

            # SHAP explanations (Reviewer 2, Point 5)
            if SHAP_AVAILABLE and len(X_test) > 0:
                print("  Running SHAP explanations...")
                shap_results = run_shap_explanations(
                    final_model, X_test[:min(100, len(X_test))], 
                    feature_names, output_dir
                )
            
            # LIME explanations (Reviewer 2, Point 5)
            if LIME_AVAILABLE and len(X_test) > 0:
                print("  Running LIME explanations...")
                lime_results = run_lime_explanations(
                    final_model, X_test[:min(50, len(X_test))], 
                    y_test[:min(50, len(y_test))], feature_names, output_dir
                )

        final_model.save(os.path.join(models_dir, "final_selected_model.keras"))

    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_mcc = matthews_corrcoef(y_test, y_pred)
    test_auc = safe_auc(y_test, y_proba)

    test_cm = confusion_matrix(y_test, y_pred)

    np.save(os.path.join(npy_dir, "test_predictions.npy"), y_pred_proba if not is_sklearn else y_proba)
    np.save(os.path.join(npy_dir, "test_predicted_classes.npy"), y_pred)
    np.save(os.path.join(npy_dir, "test_predicted_proba.npy"), y_proba)
    np.save(os.path.join(npy_dir, "test_true_labels.npy"), y_test)
    np.save(os.path.join(npy_dir, "test_confusion_matrix.npy"), test_cm)

    plot_confusion_matrix(
        test_cm,
        os.path.join(plots_dir, "test_confusion_matrix.png"),
        f"{method_name} - {species_name} Test Confusion Matrix",
        cmap="Greens",
    )

    test_predictions_df = pd.DataFrame({
        "sample_idx_original_dataset": test_indices,
        "true_label": y_test,
        "predicted_label": y_pred,
        "predicted_proba_enzyme": y_proba,
        "correct": y_test == y_pred,
    })
    test_predictions_df.to_csv(os.path.join(csv_dir, "Independent_Test_Predictions.csv"), index=False)

    final_result = {
        "Species": species_name,
        "Method": method_name,
        "Learning_Rate": learning_rate,
        "Batch_Size": batch_size,
        "Description": best_result["description"],
        "CV_Mean_MCC": best_result.get("mean_mcc", 0),
        "Test_Accuracy": test_acc,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Test_MCC": test_mcc,
        "Test_AUC": test_auc,
        "Test_Size": len(y_test),
        "Test_Enzymes": int(y_test.sum()),
        "Test_Non_Enzymes": len(y_test) - int(y_test.sum()),
        "Final_Training_Time_Seconds": final_training_time,
    }

    pd.DataFrame([final_result]).to_csv(
        os.path.join(csv_dir, "Final_Independent_Test_Result.csv"),
        index=False,
    )

    print("\nIndependent Test Results")
    print("-" * 60)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test MCC:       {test_mcc:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")

    return final_result


# ============================================================================
# RUN LOSO EXPERIMENTS - UPDATED (TESTS ALL MODELS)
# ============================================================================

def run_loso_experiments():
    """
    Main LOSO experiment loop with three-level checkpointing.
    NOW TESTS ALL MODELS, NOT JUST THE BEST ONE.
    """
    print("=" * 80)
    print("FIGNet REVISION: LEAVE-ONE-SPECIES-OUT EVALUATION")
    print("ALL MODELS TESTED ON EACH HELD-OUT SPECIES")
    print("THREE-LEVEL CHECKPOINT SYSTEM ENABLED")
    print("=" * 80)
    print(f"Data path: {DATA_PATH}")
    print(f"Results directory: {BASE_DIR}")
    print(f"Models: {len(METHODS)} total")
    print(f"CV folds: {N_FOLDS}")
    print("=" * 80)

    # Load data with species
    df = pd.read_csv(DATA_PATH)
    label_col = label_column_from_dataframe(df)
    feature_names = [c for c in df.columns if c != label_col and c != 'Data_Source']
    species_list = df['Data_Source'].unique()

    print(f"\n📊 Dataset loaded:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Species: {len(species_list)}")
    print(f"   Species distribution:")
    for species in species_list:
        count = len(df[df['Data_Source'] == species])
        enzymes = (df[df['Data_Source'] == species][label_col] == 1).sum()
        print(f"     {species:20s}: {count:>5} samples (Enzymes: {enzymes:>4})")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Level 3: Check completed LOSO models
    completed_loso_models = checkpoint.get("completed_loso_models", [])
    completed_cv = checkpoint.get("completed_cv_species", [])
    
    print(f"\n📌 Checkpoint loaded:")
    print(f"   Level 1 - Completed model runs: {len(checkpoint.get('completed_runs', {}))}")
    print(f"   Level 2 - Completed CV species: {len(completed_cv)} / {len(species_list)}")
    if completed_cv:
        print(f"      CV done: {completed_cv}")
    print(f"   Level 3 - Completed LOSO models: {len(completed_loso_models)}")
    if completed_loso_models:
        print(f"      LOSO models done: {completed_loso_models[:5]}{'...' if len(completed_loso_models) > 5 else ''}")

    # Store all LOSO results
    all_loso_results = []

    for species in species_list:
        print(f"\n{'='*80}")
        print(f"🐟 SPECIES: {species}")
        print(f"{'='*80}")

        # Split data
        species_indices = df[df['Data_Source'] == species].index.values
        other_indices = df[df['Data_Source'] != species].index.values

        X = df[feature_names].values.astype(np.float32)
        y = df[label_col].values.astype(int)

        X_train_val_raw = X[other_indices]
        y_train_val = y[other_indices]
        X_test_raw = X[species_indices]
        y_test = y[species_indices]

        print(f"   Training: {len(X_train_val_raw)} samples (all other species)")
        print(f"   Testing:  {len(X_test_raw)} samples ({species})")

        # ============================================================
        # Level 2: Check if CV already complete for this species
        # ============================================================
        if species not in completed_cv:
            print(f"\n🔄 Running CV for {species}...")
            species_results = []
            is_sklearn_methods = ["baseline_sklearn", "relieff"]

            for learning_rate in LEARNING_RATES:
                for batch_size in BATCH_SIZES:
                    for method_name, config in METHODS.items():
                        is_sklearn = config["type"] in is_sklearn_methods
                        
                        # Skip if already completed
                        if is_run_completed(checkpoint, species, method_name,
                                           learning_rate if not is_sklearn else 0.001,
                                           batch_size if not is_sklearn else 32):
                            print(f"⏭️ Skipping {method_name} (lr={learning_rate}, bs={batch_size}) - Already done")
                            continue
                        
                        try:
                            result = run_cross_validation_experiment(
                                method_name=method_name,
                                config=config,
                                learning_rate=learning_rate if not is_sklearn else 0.001,
                                batch_size=batch_size if not is_sklearn else 32,
                                X_train_val_raw=X_train_val_raw,
                                y_train_val=y_train_val,
                                train_val_indices=other_indices,
                                feature_names=feature_names,
                                is_sklearn=is_sklearn,
                                species_name=species,
                            )
                            species_results.append(result)
                        except Exception as e:
                            print(f"❌ Error in {method_name} for {species}: {e}")
                            species_results.append({"status": "error", "method": method_name})

            successful = [r for r in species_results if r.get("status") == "success"]

            if not successful:
                print(f"⚠️ No successful CV experiments for {species}, skipping")
                continue

            # Level 2: Mark CV complete for this species
            mark_cv_species_completed(checkpoint, species)
            print(f"✅ CV complete for {species}")

            # Save CV summary
            cv_summary_rows = []
            for r in successful:
                row = {k: v for k, v in r.items() if not isinstance(v, list)}
                cv_summary_rows.append(row)

            cv_summary_df = pd.DataFrame(cv_summary_rows)
            cv_summary_df = cv_summary_df.sort_values(SELECTION_METRIC, ascending=False)
            species_cv_path = os.path.join(BASE_DIR, "loso_results", species, f"{species}_CV_Summary.csv")
            ensure_dir(os.path.dirname(species_cv_path))
            cv_summary_df.to_csv(species_cv_path, index=False)

            print(f"\n📊 CV Summary for {species} saved.")
            
            # Store successful results for LOSO testing
            all_successful_models = successful
        else:
            # CV already complete, load summary
            print(f"\n⏭️ CV already complete for {species}. Loading CV summary...")
            species_cv_path = os.path.join(BASE_DIR, "loso_results", species, f"{species}_CV_Summary.csv")
            if os.path.exists(species_cv_path):
                cv_summary_df = pd.read_csv(species_cv_path)
                all_successful_models = []
                for _, row in cv_summary_df.iterrows():
                    all_successful_models.append({
                        "status": "success",
                        "method": row['method'],
                        "description": METHODS[row['method']]['description'],
                        "learning_rate": row['learning_rate'],
                        "batch_size": row['batch_size'],
                        "mean_accuracy": row['mean_accuracy'],
                        "mean_f1": row['mean_f1'],
                        "mean_mcc": row['mean_mcc'],
                        "mean_auc": row['mean_auc'],
                        "feature_stability_jaccard": row.get('feature_stability_jaccard', 0.0),
                        "recommended_epochs_for_final_training": row.get('recommended_epochs_for_final_training', 50),
                        "is_sklearn": row.get('is_sklearn', False),
                    })
                print(f"   Loaded {len(all_successful_models)} models for LOSO testing")
            else:
                print(f"⚠️ CV summary not found for {species}. Skipping.")
                continue

        # ============================================================
        # Level 3: Run LOSO test for ALL models
        # ============================================================
        print(f"\n🔄 Running LOSO tests for ALL {len(all_successful_models)} models on {species}...")
        
        tested_count = 0
        skipped_count = 0
        
        for model_result in all_successful_models:
            method_name = model_result["method"]
            learning_rate = model_result["learning_rate"]
            batch_size = model_result["batch_size"]
            
            # Check if this LOSO test is already completed (Level 3)
            if is_loso_model_completed(checkpoint, species, method_name, learning_rate, batch_size):
                print(f"⏭️ Skipping LOSO for {method_name} (lr={learning_rate}, bs={batch_size}) - Already done")
                skipped_count += 1
                continue
            
            print(f"\n🔄 Testing: {method_name} (lr={learning_rate}, bs={batch_size})")
            
            try:
                # Run LOSO test
                final_result = run_final_test_evaluation_loso(
                    best_result=model_result,
                    X_train_val_raw=X_train_val_raw,
                    y_train_val=y_train_val,
                    X_test_raw=X_test_raw,
                    y_test=y_test,
                    test_indices=species_indices,
                    species_name=species,
                    feature_names=feature_names,
                )
                
                all_loso_results.append(final_result)
                tested_count += 1
                
                # Mark this LOSO test as completed (Level 3)
                mark_loso_model_completed(checkpoint, species, method_name, learning_rate, batch_size)
                
                print(f"   ✅ LOSO test completed for {method_name}")
                
            except Exception as e:
                print(f"   ❌ LOSO test failed for {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 LOSO Summary for {species}:")
        print(f"   Tested: {tested_count} models")
        print(f"   Skipped: {skipped_count} models (already done)")

    # =============================================
    # AGGREGATE RESULTS ACROSS ALL SPECIES
    # =============================================

    print("\n" + "=" * 80)
    print("LOSO RESULTS SUMMARY (ALL SPECIES, ALL MODELS)")
    print("=" * 80)

    if all_loso_results:
        loso_df = pd.DataFrame(all_loso_results)
        
        # Reorder columns for clarity
        cols = ['Species', 'Method', 'Learning_Rate', 'Batch_Size', 
                'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 
                'Test_MCC', 'Test_AUC', 'Test_Size']
        cols = [c for c in cols if c in loso_df.columns]
        loso_df = loso_df[cols]
        
        loso_path = os.path.join(BASE_DIR, "LOSO_Results_All_Models.csv")
        loso_df.to_csv(loso_path, index=False)
        print(f"\n✅ LOSO results for ALL models saved to: {loso_path}")
        print(f"   Total results: {len(loso_df)}")

        # Summary by method
        print("\n📊 Performance by Method (averaged across species):")
        print("-" * 80)
        
        method_summary = loso_df.groupby('Method').agg({
            'Test_Accuracy': ['mean', 'std'],
            'Test_F1': ['mean', 'std'],
            'Test_MCC': ['mean', 'std'],
            'Test_AUC': ['mean', 'std']
        }).round(4)
        
        method_summary.columns = ['Acc_mean', 'Acc_std', 'F1_mean', 'F1_std', 
                                   'MCC_mean', 'MCC_std', 'AUC_mean', 'AUC_std']
        method_summary = method_summary.sort_values('MCC_mean', ascending=False)
        
        print(method_summary.to_string())
        
        # Save method summary
        method_summary_path = os.path.join(BASE_DIR, "LOSO_Method_Summary.csv")
        method_summary.to_csv(method_summary_path)
        print(f"\n✅ Method summary saved to: {method_summary_path}")

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print(f"All results saved to: {BASE_DIR}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_loso_experiments()