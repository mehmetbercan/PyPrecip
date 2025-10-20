import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from .metrics import calc_metrics
from .classes import make_class_bins, to_class_indices
from .datasets import load_station_inputs
import pickle

def train_cnn(cfg):
    wanted_cols = cfg.feature_cols
    base_dir = cfg.inputs_dir

    df = load_station_inputs(cfg.stations, base_dir, cfg.train_col, [c for c in wanted_cols if c != cfg.train_col])

    input_cols = [f'{col}_{st}' for st in cfg.stations for col in wanted_cols]
    X = df[input_cols].values.astype(float)

    # target: t+1h of train_col in target station
    y = df[f"{cfg.train_col}_{cfg.target_station}"].shift(-1)
    valid = ~y.isna()
    X = X[valid.values]
    y = y[valid]

    # classes
    intervals = cfg.class_intervals
    bin_edges, class_means = make_class_bins(intervals)
    y_cls, mask = to_class_indices(y, bin_edges)
    X = X[mask.values]
    n_classes = len(intervals)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cls, test_size=0.2, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    model = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(cfg.hidden_units, activation="tanh"),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    os.makedirs(cfg.model_dir, exist_ok=True)
    hist_dir = os.path.join(cfg.model_dir, "histories")
    os.makedirs(hist_dir, exist_ok=True)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True, verbose=cfg.verbose)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[es],
        verbose=cfg.verbose
    )

    # Save model and requirements
    model_path = os.path.join(cfg.model_dir, f'NowcastMdl_st{cfg.target_station}_1h.keras')
    model.save(model_path, save_format='keras')
    model_history_path = os.path.join(hist_dir, f'NowcastMdl_st{cfg.target_station}_1h.pckl')
    with open(model_history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    try:
        from pip._internal.operations import freeze
        with open(os.path.join(cfg.model_dir, 'requirements.txt'), 'w') as f:
            for pkg in freeze.freeze():
                f.write(pkg + '\n')
    except Exception:
        pass

    # Evaluate
    y_prob = model.predict(X_test, batch_size=1024, verbose=0)
    y_pred_cls = y_prob.argmax(axis=1)
    metrics = calc_metrics(y_test, y_pred_cls, class_means)
    return model_path, metrics