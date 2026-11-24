#!/usr/bin/env bash
# Recrée les fichiers du Lab 1 manquants (extraits de tp1.md). À lancer à la racine du repo.
set -euo pipefail
mkdir -p src
cat > .dvcignore <<'LAB1_EOF'
# Fichier : .dvcignore
# Créé par `dvc init`. Dossiers que DVC n'a pas besoin de parcourir quand il cherche
# les dvc.yaml et les fichiers .dvc du repo (dvc status reste rapide).
.venv/
features/data/
mlruns/
LAB1_EOF
cat > params.yaml <<'LAB1_EOF'
# Fichier : params.yaml
# Paramètres du pipeline, suivis par DVC (dvc params diff).
# Le contrat des données (colonnes, cible, split, métriques, seuil) est dans dataset.yaml.
prepare:
  id_column: request_id  # identifiant de demande, créé s'il est absent (entité Feast)
  event_timestamp: "2026-01-01T00:00:00+00:00"  # date de valeur des features (Feast)

train:
  model: logistic_regression  # logistic_regression | random_forest | gradient_boosting | dummy
  C: 0.1  # régularisation (logistic_regression)
  class_weight: balanced  # balanced | none (logistic_regression, random_forest)
  n_estimators: 200  # random_forest, gradient_boosting
  max_depth: 8  # entier ou none (random_forest, gradient_boosting)
  learning_rate: 0.1  # gradient_boosting
  random_state: 42  # graine du modèle
LAB1_EOF
cat > src/prepare.py <<'LAB1_EOF'
# Fichier : src/prepare.py
"""Prépare les données : identifiant de demande, split stratifié, source du feature store.

Entrées : le fichier brut (raw_path de dataset.yaml) et params.yaml (section prepare).
Sorties : data/processed/train.csv, data/processed/test.csv (colonnes brutes + identifiant)
          et data/processed/features.parquet (source « offline » de Feast).
Aucun nom de colonne en dur : tout vient de dataset.yaml (load_config) et de params.yaml.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config

PROCESSED = Path("data/processed")


def main() -> None:
    cfg = load_config()
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["prepare"]
    id_col = params["id_column"]

    df = pd.read_csv(cfg.raw_path)
    cfg.validate_columns(df)

    # Identifiant de demande (clé de l'entité Feast). S'il est absent du fichier brut,
    # on numérote les lignes : l'ordre du fichier est figé par DVC, donc l'identifiant
    # est reproductible.
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))
    df = df[[id_col, *cfg.columns]]

    # Split stratifié sur la cible : même proportion de la classe positive des deux côtés
    y = cfg.encode_target(df[cfg.target])
    train, test = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    # Source offline de Feast : les features seules (jamais la cible), une ligne par
    # demande, avec un horodatage fixe (params.yaml) pour que le fichier soit reproductible.
    features = df[[id_col, *cfg.features]].copy()
    features[cfg.numeric] = features[cfg.numeric].astype("float64")
    features["event_timestamp"] = pd.Timestamp(params["event_timestamp"])
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    rate = cfg.encode_target(train[cfg.target]).mean(), cfg.encode_target(test[cfg.target]).mean()
    print(
        f"Train : {len(train)} / Test : {len(test)} · "
        f"classe '{cfg.positive_class}' : {rate[0]:.1%} / {rate[1]:.1%}"
    )


if __name__ == "__main__":
    main()
LAB1_EOF
cat > src/train.py <<'LAB1_EOF'
# Fichier : src/train.py
"""Entraîne un modèle de scoring et trace le run dans MLflow (expérience « scoring »).

Hyper-paramètres par défaut : params.yaml (section train) ; les options de la ligne de
commande les surchargent (utilisé par src/grid_search.py). Le modèle enregistré est un
Pipeline scikit-learn complet (prétraitement décrit par dataset.yaml + estimateur) : il
prend en entrée les colonnes brutes, exactement comme l'API de prédiction des labs 4-5.

Les métriques de décision (F1, précision, rappel, coût métier) sont calculées au seuil
qui minimise le coût métier (cfg.cost_threshold), pas à 0,5 : ce seuil est loggé en
paramètre `decision_threshold` et rangé dans les métadonnées du modèle.

Sorties : un run MLflow (params, métriques, tags, modèle) et une copie locale du modèle
au format MLflow dans models/model/ (sortie DVC, relue par src/evaluate.py).
"""

import argparse
import hashlib
import os
import shutil
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.dataset_config import DatasetConfig, load_config

EXPERIMENT = "scoring"
TRAIN_CSV = Path("data/processed/train.csv")
TEST_CSV = Path("data/processed/test.csv")
MODEL_DIR = Path("models/model")

# Type de modèle -> (classe scikit-learn, hyper-paramètres pris dans params.yaml / la CLI)
MODELS = {
    "logistic_regression": (LogisticRegression, ["C", "class_weight"]),
    "random_forest": (RandomForestClassifier, ["n_estimators", "max_depth", "class_weight"]),
    "gradient_boosting": (
        GradientBoostingClassifier,
        ["n_estimators", "max_depth", "learning_rate"],
    ),
    "dummy": (DummyClassifier, []),  # référence naïve : probabilité = taux de la classe
}
# MLflow 3 sérialise en skops (pas en pickle) : seuls les types déclarés « de confiance »
# peuvent être rechargés. Ceux-ci sont utilisés par OneHotEncoder et par les arbres.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]
# Dépendances nécessaires pour recharger le modèle (images de serving des labs 4-5).
# Sans cette liste, MLflow 3 recopie tout uv.lock (près de 300 paquets du cours).
PIP_REQUIREMENTS = [f"{p}=={version(p)}" for p in ("scikit-learn", "skops", "pandas", "numpy")]


def none_or(cast):
    """Convertisseur argparse acceptant « none » (ex. --max_depth none)."""
    return lambda value: None if str(value).lower() == "none" else cast(value)


def parse_args() -> argparse.Namespace:
    defaults = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["train"]
    parser = argparse.ArgumentParser(description="Entraîne un modèle et le trace dans MLflow")
    parser.add_argument("--model", choices=list(MODELS), default=defaults["model"])
    parser.add_argument("--C", type=float, default=defaults["C"])
    parser.add_argument("--class_weight", type=none_or(str), default=defaults["class_weight"])
    parser.add_argument("--n_estimators", type=int, default=defaults["n_estimators"])
    parser.add_argument("--max_depth", type=none_or(int), default=defaults["max_depth"])
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--random_state", type=int, default=defaults["random_state"])
    # NB : argparse applique aussi `type` aux valeurs par défaut écrites en texte
    # (« none » dans params.yaml devient None).
    return parser.parse_args()


def build_model(cfg: DatasetConfig, model_type: str, hyperparams: dict, seed: int) -> Pipeline:
    """Pipeline complet : prétraitement de dataset.yaml + estimateur."""
    cls, _ = MODELS[model_type]
    extra = {"random_state": seed} if model_type != "dummy" else {"strategy": "prior"}
    if model_type == "logistic_regression":
        extra["max_iter"] = 1000
    return Pipeline(
        [("preprocess", cfg.build_preprocessor()), ("model", cls(**hyperparams, **extra))]
    )


def decision_threshold(cfg: DatasetConfig, class_weight, y_train: pd.Series) -> float:
    """Seuil qui minimise le coût métier de dataset.yaml (et non 0,5).

    class_weight="balanced" donne aux positifs un poids n_neg/n_pos fois plus grand :
    les probabilités du modèle sont gonflées d'autant, le seuil remonte en conséquence.
    """
    k = (y_train == 0).sum() / (y_train == 1).sum() if class_weight == "balanced" else 1.0
    return cfg.cost_threshold(float(k))


def compute_metrics(cfg: DatasetConfig, y_true, proba, threshold: float) -> dict:
    """Métriques de classement (proba) et de décision au seuil donné, dont le coût métier."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "business_cost": cfg.business_cost(y_true, y_pred),
    }


def file_md5(path: str | Path) -> str:
    """Empreinte MD5 du fichier brut : la même que celle que DVC écrit dans dvc.lock."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    # Numériques en float64 : la signature MLflow les déclare `double` (cfg.read_xy)
    X_train, y_train = cfg.read_xy(TRAIN_CSV)
    X_test, y_test = cfg.read_xy(TEST_CSV)

    hyperparams = {name: getattr(args, name) for name in MODELS[args.model][1]}
    threshold = decision_threshold(cfg, hyperparams.get("class_weight"), y_train)
    run_name = "-".join([args.model, *(f"{k}={v}" for k, v in hyperparams.items())])
    # Qui a lancé le run ? DVC définit DVC_STAGE ; grid_search.py définit LAUNCHED_BY.
    launched_by = "dvc" if os.getenv("DVC_STAGE") else os.getenv("LAUNCHED_BY", "manual")

    # Serveur MLflow du cours (MLFLOW_TRACKING_URI dans .env) : aucun identifiant S3 ici,
    # le serveur fait proxy des artefacts vers s3://mlflow-artifacts.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_type": args.model,
                "dataset": cfg.name,
                "data_md5": file_md5(cfg.raw_path),
                "launched_by": launched_by,
            }
        )
        mlflow.log_params({"model": args.model, "random_state": args.random_state, **hyperparams})
        mlflow.log_param("decision_threshold", round(threshold, 4))

        model = build_model(cfg, args.model, hyperparams, args.random_state)
        model.fit(X_train, y_train)

        metrics = compute_metrics(cfg, y_test, model.predict_proba(X_test)[:, 1], threshold)
        mlflow.log_metrics(metrics)

        # Signature = contrat d'entrée/sortie du modèle, vérifié au chargement / au serving
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            pip_requirements=PIP_REQUIREMENTS,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            metadata={"decision_threshold": threshold},  # le seuil voyage avec le modèle
        )

        # Copie locale (sortie du stage DVC train, relue par src/evaluate.py)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        mlflow.sklearn.save_model(
            model,
            MODEL_DIR,
            signature=signature,
            pip_requirements=PIP_REQUIREMENTS,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            metadata={"mlflow_run_id": run.info.run_id, "decision_threshold": threshold},
        )

        summary = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "business_cost")
        cost = metrics["business_cost"]
        print(f"{run_name}: seuil={threshold:.3f}, {summary}, business_cost={cost:.0f}")
        print(f"run_id : {run.info.run_id}")


if __name__ == "__main__":
    main()
LAB1_EOF
cat > src/evaluate.py <<'LAB1_EOF'
# Fichier : src/evaluate.py
"""Évalue le modèle local (models/model) sur le jeu de test -> reports/metrics.json.

Stage DVC « evaluate » : ne dépend que de fichiers versionnés (pas du serveur MLflow),
pour que `dvc metrics show` / `dvc metrics diff` comparent les versions du pipeline.
Le seuil de décision est celui choisi par src/train.py (métadonnées du modèle).
"""

import json
from pathlib import Path

import mlflow
from mlflow.models import Model
from sklearn.metrics import confusion_matrix

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, compute_metrics

METRICS_JSON = Path("reports/metrics.json")


def main() -> None:
    cfg = load_config()
    X_test, y_test = cfg.read_xy(TEST_CSV)
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    threshold = Model.load(str(MODEL_DIR)).metadata["decision_threshold"]

    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(cfg, y_test, proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= threshold).astype(int)).ravel()
    report = {
        **{k: round(float(v), 4) for k, v in metrics.items()},
        "decision_threshold": round(threshold, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_test)),
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{cfg.primary_metric} = {report[cfg.primary_metric]} -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
LAB1_EOF
cat > dvc.yaml <<'LAB1_EOF'
# Fichier : dvc.yaml
# Pipeline du projet : download -> prepare -> train -> evaluate.
# `dvc repro` ne relance que les stages dont une dépendance (deps, params) a changé.
stages:
  download:
    cmd: python -m src.download
    deps:
      - src/dataset_config.py
      - src/download.py
    params:
      - dataset.yaml:
          - features
          - raw_path
          - target
    outs:
      - data/raw/dataset.csv
  prepare:
    cmd: python -m src.prepare
    deps:
      - src/dataset_config.py
      - src/prepare.py
      - data/raw/dataset.csv
    params:
      - dataset.yaml:
          - features
          - split
          - target
      - prepare
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
      - data/processed/features.parquet
  train:
    cmd: python -m src.train
    deps:
      - src/dataset_config.py
      - src/train.py
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - dataset.yaml:
          - metrics
      - train
    outs:
      - models/model
  evaluate:
    cmd: python -m src.evaluate
    deps:
      - src/evaluate.py
      - src/train.py
      - models/model
      - data/processed/test.csv
    params:
      - dataset.yaml:
          - metrics
    metrics:
      - reports/metrics.json:
          cache: false
LAB1_EOF
echo "Fichiers créés :"; ls -l params.yaml dvc.yaml .dvcignore src/prepare.py src/train.py src/evaluate.py
