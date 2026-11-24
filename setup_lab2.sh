#!/usr/bin/env bash
# =============================================================================
# setup_lab2.sh — automatise tout le Lab 2 (tp3.md) : Pipeline Kubeflow + Airflow
# À lancer à la RACINE du fork (~/mlops-cours-2026), Lab 1 réussi, cluster kind actif.
#
# Usage :
#   ./setup_lab2.sh                 # tout (branch files kfp image compile local run airflow dag check)
#   ./setup_lab2.sh files           # seulement recréer les fichiers du lab
#   ./setup_lab2.sh kfp image compile   # seulement certaines phases
#   ./setup_lab2.sh git             # commit + push (jamais inclus dans « tout »)
#
# Phases : branch files kfp image compile local run airflow dag report check git
# Variables : IMAGE_TAG (1.0.0)  SKIP_LOCAL=1 (saute le test kfp.local)
#
# Ce que le script NE fait PAS à ta place : le rapport comparatif (étape 9) doit être
# rédigé par toi (analyse honnête, notée sur 5 points) ; il copie seulement le modèle.
# =============================================================================
set -euo pipefail
 
IMAGE_TAG="${IMAGE_TAG:-1.0.0}"
IMAGE="localhost:5001/scoring-pipeline:${IMAGE_TAG}"
BRANCH="lab/m2-pipelines"
 
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✔ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m  ✘ %s\033[0m\n' "$*" >&2; exit 1; }
w()    { mkdir -p "$(dirname "$1")"; cat > "$1"; ok "écrit $1"; }   # w chemin <<'EOF'
has_make_target() { [ -f Makefile ] && grep -qE "^$1:" Makefile; }
 
# --- Préconditions ------------------------------------------------------------
[ -f dataset.yaml ] && [ -d src ] || die "Lance ce script depuis la racine du fork (dataset.yaml + src/ introuvables)."
[ -f src/dataset_config.py ] || die "Lab 1 incomplet (src/dataset_config.py manquant). Lance d'abord setup_lab1.sh."
 
[ -f .venv/bin/activate ] && source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
 
PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(branch files kfp image compile local run airflow dag check)
want() { local p; for p in "${PHASES[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }
 
ensure_pf() {  # port-forwards S3 :8333 et MLflow :5000
  if ! curl -fs http://localhost:5000/health >/dev/null 2>&1; then
    log "Port-forwards absents : make pf"
    make pf
    sleep 5
  fi
  curl -fs http://localhost:5000/health >/dev/null 2>&1 || warn "MLflow injoignable sur :5000 (make pf ?)"
}
 
ensure_kfp_ui() {
  if ! curl -fs http://localhost:8080/apis/v2beta1/healthz >/dev/null 2>&1; then
    log "Port-forward UI KFP"
    if has_make_target kfp-ui; then make kfp-ui
    else
      mkdir -p /tmp/mlops-port-forward
      nohup kubectl -n kubeflow port-forward svc/ml-pipeline-ui 8080:80 \
        >/tmp/mlops-port-forward/kfp-ui.log 2>&1 &
    fi
    for _ in $(seq 1 30); do
      curl -fs http://localhost:8080/apis/v2beta1/healthz >/dev/null 2>&1 && break
      sleep 2
    done
  fi
  curl -fs http://localhost:8080/apis/v2beta1/healthz >/dev/null || die "API KFP injoignable sur :8080"
}
 
# =============================================================================
# Phase : branch  (Branche de travail)
# =============================================================================
if want branch; then
  log "Branche de travail $BRANCH"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    current="$(git branch --show-current)"
    if [ "$current" != "$BRANCH" ]; then
      if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git switch "$BRANCH"
      else
        (git switch main && git pull --ff-only) || warn "main non mis à jour (on continue)"
        git switch -c "$BRANCH"
      fi
    fi
    ok "branche : $(git branch --show-current)"
  else
    warn "pas un repo git : phase ignorée"
  fi
fi
 
# =============================================================================
# Phase : files  (étapes 2.2 à 2.5, 3.1/6.1, 7.2, .gitignore)
# =============================================================================
if want files; then
  log "Création des fichiers du Lab 2"
  mkdir -p pipelines dags reports
 
  # ---------------------------------------------------------------- 2.2
  w src/validate_data.py <<'LAB2_EOF'
# Fichier : src/validate_data.py
"""Valide un fichier de données brutes contre le contrat dataset.yaml.
 
Gate du pipeline (Lab 2) : si un contrôle échoue, le script sort avec le code 1 et
l'orchestrateur (DVC, Kubeflow, Airflow, CI) s'arrête AVANT l'entraînement. Les contrôles
sont des contrôles de schéma simples, tous déduits de dataset.yaml (aucun nom de colonne
en dur). L'interface :
 
    python -m src.validate_data [fichier.csv]      (défaut : raw_path de dataset.yaml)
 
Sortie : un résumé JSON sur la sortie standard et dans reports/validation.json.
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import pandas as pd
 
from src.dataset_config import DatasetConfig, load_config
 
REPORT_JSON = Path("reports/validation.json")
 
 
def run_checks(cfg: DatasetConfig, df: pd.DataFrame) -> dict[str, bool]:
    """Un booléen par contrôle (True = conforme au contrat)."""
    try:
        cfg.validate_columns(df)  # toutes les colonnes déclarées sont présentes
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return {"colonnes_du_contrat": False}  # inutile d'aller plus loin
    target = df[cfg.target]
    classes = set(target.dropna().astype(str))
    return {
        "colonnes_du_contrat": True,
        "table_non_vide": len(df) > 0,
        "cible_sans_valeur_manquante": bool(target.notna().all()),
        "cible_avec_classe_positive_et_autre": str(cfg.positive_class) in classes
        and len(classes) >= 2,
        "numeriques_de_type_nombre": all(
            pd.api.types.is_numeric_dtype(df[col]) for col in cfg.numeric
        ),
        "aucune_variable_entierement_vide": not any(df[col].isna().all() for col in cfg.features),
    }
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", help="fichier à valider (défaut : raw_path)")
    args = parser.parse_args()
 
    cfg = load_config()
    df = pd.read_csv(args.csv or cfg.raw_path)
    checks = run_checks(cfg, df)
    failed = [name for name, ok in checks.items() if not ok]
    has_target = cfg.target in df.columns and len(df) > 0
    summary = {
        "success": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "rows": len(df),
        "positive_rate": round(float(cfg.encode_target(df[cfg.target]).mean()), 4)
        if has_target
        else None,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] else 1
 
 
if __name__ == "__main__":
    raise SystemExit(main())
LAB2_EOF
 
  # ---------------------------------------------------------------- 2.3
  touch pipelines/__init__.py
  ok "écrit pipelines/__init__.py"
 
  w pipelines/stage.py <<'LAB2_EOF'
# Fichier : pipelines/stage.py
"""Exécute une étape du projet (python -m src.<étape>) pour un orchestrateur.
 
Une étape = une stage de dvc.yaml : des entrées (deps), une commande, des sorties (outs).
Elle tourne dans un dossier de travail temporaire qui reproduit l'arborescence du projet
(params.yaml, data/, models/, reports/) : le code du Lab 1, écrit avec des chemins
relatifs, s'exécute SANS modification, que l'orchestrateur soit DVC, Kubeflow Pipelines
ou Airflow. Ce module est copié dans l'image du pipeline (pipelines/Dockerfile).
 
- Kubeflow (pipelines/components.py) : run_stage() avec les chemins des artefacts typés.
- Airflow (dags/scoring_dag.py) : python -m pipelines.stage src.<étape> --workspace s3://…
  recopie l'arborescence du run depuis/vers S3 ; --xcom publie les petites valeurs utiles
  aux tasks suivantes (run MLflow, métriques).
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]  # racine du projet (/app dans l'image)
OUTPUT_DIRS = ("data", "models", "reports")  # là où les étapes écrivent leurs sorties
XCOM_FILE = Path("/airflow/xcom/return.json")  # lu par le KubernetesPodOperator
 
 
def copy(src: str | Path, dst: str | Path) -> None:
    """Copie un fichier ou un dossier, en créant les dossiers parents."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copyfile(src, dst)
 
 
def execute(module: str, args, workdir: Path, env: dict | None = None) -> None:
    """Lance python -m <module> <args> dans workdir, avec le code et le contrat du projet."""
    shutil.copyfile(ROOT / "params.yaml", workdir / "params.yaml")
    run_env = {**os.environ, **(env or {})}
    run_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.getenv("PYTHONPATH")]))
    config = Path(os.getenv("DATASET_CONFIG", ROOT / "dataset.yaml")).resolve()
    run_env["DATASET_CONFIG"] = str(config)
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=workdir, env=run_env, check=True)
 
 
def run_stage(module: str, args=(), inputs=None, outputs=None, env=None) -> None:
    """Copie les entrées, lance l'étape, copie les sorties (composants KFP).
 
    inputs  : {chemin dans le projet : fichier ou dossier à y copier avant l'étape}
    outputs : {chemin dans le projet : destination où le recopier après l'étape}
    env     : variables d'environnement en plus (ex. LAUNCHED_BY=kfp)
    """
    with tempfile.TemporaryDirectory(prefix="stage-") as tmp:
        work = Path(tmp)
        for rel, src in (inputs or {}).items():
            copy(src, work / rel)
        execute(module, args, work, env)
        for rel, dst in (outputs or {}).items():
            copy(work / rel, dst)
 
 
# --- Airflow : dossier de travail du run, sur S3 (ou en local pour tester) ---------------
def _s3(workspace: str):
    """Client S3 (endpoint et identifiants lus dans l'environnement), bucket, préfixe."""
    import boto3
 
    bucket, _, prefix = workspace.removeprefix("s3://").partition("/")
    prefix = re.sub(r"[^A-Za-z0-9._/-]", "_", prefix).strip("/")  # run_id Airflow : « : + »
    return boto3.client("s3"), bucket, prefix
 
 
def pull(workspace: str, work: Path) -> None:
    """Récupère les sorties des étapes précédentes du run."""
    if not workspace.startswith("s3://"):
        if Path(workspace).is_dir():
            copy(workspace, work)
        return
    s3, bucket, prefix = _s3(workspace)
    if bucket not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
        s3.create_bucket(Bucket=bucket)  # premier run : le bucket n'existe pas encore
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            target = work / obj["Key"].removeprefix(f"{prefix}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
 
 
def snapshot(work: Path) -> dict[Path, int]:
    """Fichiers de sortie présents et leur date de modification."""
    return {
        p: p.stat().st_mtime_ns for d in OUTPUT_DIRS for p in (work / d).rglob("*") if p.is_file()
    }
 
 
def push(work: Path, workspace: str, before: dict[Path, int]) -> None:
    """Publie les fichiers créés ou modifiés par l'étape."""
    changed = [p for p, mtime in snapshot(work).items() if before.get(p) != mtime]
    if workspace.startswith("s3://"):
        s3, bucket, prefix = _s3(workspace)
        for p in changed:
            s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(work).as_posix()}")
    else:
        for p in changed:
            copy(p, Path(workspace) / p.relative_to(work))
    print(f"{len(changed)} fichier(s) publié(s) dans {workspace}", flush=True)
 
 
def xcom_summary(work: Path) -> dict:
    """Petites valeurs pour les tasks suivantes : run MLflow du modèle, métriques."""
    from mlflow.models import Model
 
    from src.evaluate import METRICS_JSON
    from src.train import MODEL_DIR
 
    summary = {}
    if (work / MODEL_DIR / "MLmodel").exists():
        summary["mlflow_run_id"] = Model.load(work / MODEL_DIR).metadata["mlflow_run_id"]
    if (work / METRICS_JSON).exists():
        summary["metrics"] = json.loads((work / METRICS_JSON).read_text())
    return summary
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exécute une étape du projet (task Airflow)", allow_abbrev=False
    )
    parser.add_argument("module", help="étape à lancer, ex. src.prepare")
    parser.add_argument("--workspace", required=True, help="s3://bucket/préfixe ou dossier")
    parser.add_argument("--xcom", action="store_true", help=f"résumé écrit dans {XCOM_FILE}")
    opts, args = parser.parse_known_args()
    args = [a for a in args if a != "--"]  # arguments de l'étape, après « -- »
 
    with tempfile.TemporaryDirectory(prefix="stage-") as tmp:
        work = Path(tmp)
        pull(opts.workspace, work)
        before = snapshot(work)
        execute(opts.module, args, work)
        push(work, opts.workspace, before)
        if opts.xcom:
            summary = xcom_summary(work)
            XCOM_FILE.parent.mkdir(parents=True, exist_ok=True)
            XCOM_FILE.write_text(json.dumps(summary), encoding="utf-8")
            print("XCom :", summary, flush=True)
 
 
if __name__ == "__main__":
    main()
LAB2_EOF
 
  # ---------------------------------------------------------------- 2.4
  w pipelines/requirements.txt <<'LAB2_EOF'
# Dépendances de l'image du pipeline (pipelines/Dockerfile) : de quoi exécuter src/
# (download, validate_data, prepare, train, evaluate, promote) et parler au serveur
# MLflow du cours. Versions alignées sur uv.lock. kfp est installé à part (Dockerfile).
mlflow-skinny==3.16.1  # client MLflow seul : le serveur tourne dans le cluster
scikit-learn==1.9.1
skops==0.16.0  # format de sérialisation des modèles scikit-learn par MLflow 3
pandas==2.3.3
numpy==2.5.3
pyarrow==25.0.1  # data/processed/features.parquet (src/prepare.py)
pyyaml==6.0.3
boto3==1.43.106  # dossier de travail S3 des tasks Airflow (pipelines/stage.py)
LAB2_EOF
 
  w pipelines/Dockerfile <<'LAB2_EOF'
# Fichier : pipelines/Dockerfile
# Image « projet » du pipeline : le code du Lab 1 (src/), le contrat dataset.yaml,
# params.yaml et leurs dépendances figées. TOUS les composants KFP (et les tasks Airflow)
# tournent dans cette image : le code est réutilisé tel quel, jamais recopié.
# Contexte de build = RACINE du repo :
#   docker build -f pipelines/Dockerfile -t localhost:5001/scoring-pipeline:1.0.0 .
FROM python:3.12-slim
 
WORKDIR /app
 
# 1. Les dépendances d'abord : cette couche, la plus lourde, reste en cache tant que
#    requirements.txt ne change pas (reconstruire après une modification de src/ = 2 s).
#    kfp sans ses dépendances suffit à exécuter un composant : c'est ce que KFP
#    installerait au démarrage de CHAQUE pod (install_kfp_package=False dans components.py).
COPY pipelines/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt \
    && pip install --no-cache-dir --root-user-action=ignore --no-deps kfp==2.17.0
 
# 2. Le code et le contrat, avec la même arborescence que le repo
COPY dataset.yaml params.yaml ./
COPY src/ src/
COPY pipelines/__init__.py pipelines/stage.py pipelines/
 
# src et pipelines importables depuis n'importe quel dossier, dataset.yaml trouvé partout.
# Pas d'utilisateur non-root ici : le lanceur KFP écrit les artefacts sous /minio, /s3…
# à la racine du conteneur.
ENV PYTHONPATH=/app \
    DATASET_CONFIG=/app/dataset.yaml \
    PYTHONUNBUFFERED=1
 
ARG VERSION=dev
LABEL org.opencontainers.image.title="scoring-pipeline" \
      org.opencontainers.image.version="${VERSION}"
LAB2_EOF
 
  w pipelines/Dockerfile.dockerignore <<'LAB2_EOF'
# Fichier : pipelines/Dockerfile.dockerignore
# .dockerignore propre à pipelines/Dockerfile (BuildKit le lit à la place de celui de la
# racine). Liste blanche : on exclut TOUT, puis on ré-inclut ce que l'image copie.
# Ni .venv/, ni data/, ni .git/, ni .env (secrets) ne partent dans le contexte de build.
*
!dataset.yaml
!params.yaml
!src/
!pipelines/requirements.txt
!pipelines/__init__.py
!pipelines/stage.py
**/__pycache__
LAB2_EOF
 
  # ---------------------------------------------------------------- 2.5
  w pipelines/components.py <<'LAB2_EOF'
# Fichier : pipelines/components.py
"""Composants KFP du pipeline de scoring (Lab 2).
 
Chaque composant tourne dans l'image « projet » PIPELINE_IMAGE (pipelines/Dockerfile), qui
contient le code du Lab 1. Un composant ne recopie AUCUNE logique métier : il relie ses
artefacts typés (Dataset, Model, Metrics) aux fichiers d'une stage de dvc.yaml, puis lance
la même commande que DVC (python -m src.<étape>) grâce à pipelines/stage.py.
 
Rappel KFP : seul le CODE de chaque fonction est envoyé au pod. Les imports se font donc
DANS la fonction (ils sont résolus dans l'image) et aucune valeur par défaut d'un
paramètre ne doit dépendre d'une constante de ce module.
"""
 
import os
 
from kfp import dsl
from kfp.dsl import Artifact, ClassificationMetrics, Dataset, Input, Metrics, Model, Output
 
# Image construite et poussée à l'étape 2.4. Nouveau code dans l'image = nouveau tag.
# Surcharge possible à la compilation : PIPELINE_IMAGE=… python -m pipelines.scoring_pipeline
PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", "localhost:5001/scoring-pipeline:1.0.0")
# Serveur MLflow du cours vu depuis un pod (DNS interne <service>.<namespace>)
MLFLOW_URI = "http://mlflow.mlflow:5000"
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def ingest_data(raw_data: Output[Dataset]) -> None:
    """Stage « download » : python -m src.download produit le fichier brut."""
    import pandas as pd
 
    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import file_md5
 
    cfg = load_config()
    run_stage("src.download", outputs={cfg.raw_path: raw_data.path})
    raw_data.metadata["dataset"] = cfg.name
    raw_data.metadata["rows"] = len(pd.read_csv(raw_data.path))
    raw_data.metadata["md5"] = file_md5(raw_data.path)  # le même hash que dans dvc.lock
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def validate_data(
    raw_data: Input[Dataset],
    report: Output[Artifact],
    checks: Output[Metrics],
) -> None:
    """Gate : python -m src.validate_data ; un contrôle en échec fait échouer le step."""
    import json
    from pathlib import Path
 
    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.validate_data import REPORT_JSON
 
    cfg = load_config()
    run_stage(
        "src.validate_data",
        inputs={cfg.raw_path: raw_data.path},
        outputs={str(REPORT_JSON): report.path},  # reports/validation.json
    )
    summary = json.loads(Path(report.path).read_text())
    for name in ("n_checks", "n_failed", "rows", "positive_rate"):
        checks.log_metric(name, summary[name])
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def prepare_features(
    raw_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
) -> None:
    """Stage « prepare » : python -m src.prepare (split stratifié décrit par dataset.yaml)."""
    import pandas as pd
 
    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import TEST_CSV, TRAIN_CSV
 
    cfg = load_config()
    run_stage(
        "src.prepare",
        inputs={cfg.raw_path: raw_data.path},
        outputs={str(TRAIN_CSV): train_data.path, str(TEST_CSV): test_data.path},
    )
    for split in (train_data, test_data):
        y = cfg.encode_target(pd.read_csv(split.path)[cfg.target])
        split.metadata["rows"] = len(y)
        split.metadata["positive_rate"] = round(float(y.mean()), 4)
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def train_model(
    raw_data: Input[Dataset],
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    model_type: str,
    C: float,
    class_weight: str,
    n_estimators: int,
    max_depth: str,
    learning_rate: float,
    random_state: int,
    mlflow_uri: str,
    kfp_run_id: str,
) -> str:
    """Stage « train » : python -m src.train ; renvoie l'identifiant du run MLflow."""
    from mlflow import MlflowClient
    from mlflow.models import Model as MlflowModel
 
    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.train import MODEL_DIR, TEST_CSV, TRAIN_CSV
 
    cfg = load_config()
    run_stage(
        "src.train",
        args=["--model", model_type, "--C", C, "--class_weight", class_weight,
              "--n_estimators", n_estimators, "--max_depth", max_depth,
              "--learning_rate", learning_rate, "--random_state", random_state],
        # le fichier brut sert au tag data_md5 (le même hash que dans dvc.lock)
        inputs={cfg.raw_path: raw_data.path, str(TRAIN_CSV): train_data.path,
                str(TEST_CSV): test_data.path},
        outputs={str(MODEL_DIR): model.path},
        env={"MLFLOW_TRACKING_URI": mlflow_uri, "LAUNCHED_BY": "kfp"},
    )  # fmt: skip
    metadata = MlflowModel.load(model.path).metadata  # écrites par src/train.py
    run_id = metadata["mlflow_run_id"]
    # Lineage : depuis le run MLflow, on retrouve le run KFP qui l'a produit
    MlflowClient(mlflow_uri).set_tag(run_id, "kfp_run_id", kfp_run_id)
    model.metadata["mlflow_run_id"] = run_id
    model.metadata["model_type"] = model_type
    model.metadata["decision_threshold"] = round(metadata["decision_threshold"], 4)
    return run_id
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    report: Output[Artifact],
    metrics: Output[Metrics],
    confusion: Output[ClassificationMetrics],
    run_id: str,
    mlflow_uri: str,
) -> float:
    """Stage « evaluate » : python -m src.evaluate ; métriques dans l'UI KFP et MLflow."""
    import json
    from pathlib import Path
 
    import pandas as pd
    from mlflow import MlflowClient
 
    from pipelines.stage import run_stage
    from src.dataset_config import load_config
    from src.evaluate import METRICS_JSON
    from src.train import MODEL_DIR, TEST_CSV
 
    cfg = load_config()
    run_stage(
        "src.evaluate",
        inputs={str(MODEL_DIR): model.path, str(TEST_CSV): test_data.path},
        outputs={str(METRICS_JSON): report.path},  # reports/metrics.json
    )
    results = json.loads(Path(report.path).read_text())
    for name, value in results.items():
        if isinstance(value, int | float):
            metrics.log_metric(name, value)
 
    # Matrice de confusion au seuil de décision, avec les libellés réels de la cible
    positive = str(cfg.positive_class)
    others = sorted(set(pd.read_csv(test_data.path)[cfg.target].astype(str)) - {positive})
    negative = others[0] if len(others) == 1 else f"non {positive}"
    c = results["confusion"]
    confusion.log_confusion_matrix([negative, positive], [[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    # Le même rapport que reports/metrics.json de DVC, rangé avec le run MLflow
    MlflowClient(mlflow_uri).log_dict(run_id, results, str(METRICS_JSON))
    return float(results[cfg.primary_metric])
 
 
@dsl.component(base_image=PIPELINE_IMAGE, install_kfp_package=False)
def register_model(run_id: str, mlflow_uri: str) -> str:
    """python -m src.promote --run-id : @challenger toujours, @champion si métrique ≥ seuil."""
    from mlflow import MlflowClient
 
    from pipelines.stage import run_stage
    from src.promote import MODEL_NAME
 
    run_stage("src.promote", args=["--run-id", run_id], env={"MLFLOW_TRACKING_URI": mlflow_uri})
    versions = MlflowClient(mlflow_uri).search_model_versions(
        f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
    )
    return str(versions[0].version)
LAB2_EOF
 
  # ---------------------------------------------------------------- 3.1 + 5.4 + 6.1 (version finale)
  w pipelines/scoring_pipeline.py <<'LAB2_EOF'
# Fichier : pipelines/scoring_pipeline.py
"""Pipeline KFP « scoring-pipeline » du Lab 2.
 
ingest -> validate -> prepare -> train -> evaluate -> register
Compilation : python -m pipelines.scoring_pipeline   (écrit scoring_pipeline.yaml)
Les hyper-paramètres par défaut sont lus dans params.yaml AU MOMENT DE LA COMPILATION :
params.yaml reste la seule source des valeurs par défaut, comme pour dvc repro.
"""
 
from pathlib import Path
 
import yaml
from kfp import compiler, dsl, kubernetes
 
from pipelines.components import (
    MLFLOW_URI,
    evaluate_model,
    ingest_data,
    prepare_features,
    register_model,
    train_model,
    validate_data,
)
 
TRAIN = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["train"]
 
 
@dsl.pipeline(
    name="scoring-pipeline",
    description="Credit scoring : ingest -> validate -> prepare -> train -> evaluate -> register",
)
def scoring_pipeline(
    model_type: str = TRAIN["model"],
    C: float = TRAIN["C"],
    class_weight: str = str(TRAIN["class_weight"]),  # « none » accepté, comme au Lab 1
    n_estimators: int = TRAIN["n_estimators"],
    max_depth: str = str(TRAIN["max_depth"]),  # « none » accepté, comme au Lab 1
    learning_rate: float = TRAIN["learning_rate"],
    random_state: int = TRAIN["random_state"],
    mlflow_uri: str = MLFLOW_URI,
):
    # Source externe (réseau) : 3 nouvelles tentatives, et jamais plus de 10 min
    ingest_task = ingest_data()
    ingest_task.set_retry(num_retries=3, backoff_duration="30s")
    ingest_task.set_cpu_limit("500m").set_memory_limit("512Mi")
    kubernetes.set_timeout(ingest_task, 600)
 
    validate_task = validate_data(raw_data=ingest_task.outputs["raw_data"])
    validate_task.set_memory_limit("512Mi")
 
    prep_task = prepare_features(raw_data=ingest_task.outputs["raw_data"])
    prep_task.after(validate_task)  # gate : pas de préparation sans validation réussie
    prep_task.set_memory_limit("1Gi")
 
    train_task = train_model(
        raw_data=ingest_task.outputs["raw_data"],
        train_data=prep_task.outputs["train_data"],
        test_data=prep_task.outputs["test_data"],
        model_type=model_type,
        C=C,
        class_weight=class_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        mlflow_uri=mlflow_uri,
        kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,  # remplacé par l'id du run KFP
    )
    # Une limit seule vaut aussi request : on demande peu, on autorise plus
    train_task.set_cpu_request("500m").set_cpu_limit("2")
    train_task.set_memory_request("512Mi").set_memory_limit("2Gi")
    kubernetes.set_timeout(train_task, 1800)
    # Toujours ré-entraîner : chaque run KFP doit produire un nouveau run MLflow
    train_task.set_caching_options(enable_caching=False)
 
    eval_task = evaluate_model(
        model=train_task.outputs["model"],
        test_data=prep_task.outputs["test_data"],
        run_id=train_task.outputs["Output"],
        mlflow_uri=mlflow_uri,
    )
    eval_task.set_memory_limit("1Gi")
 
    register_task = register_model(run_id=train_task.outputs["Output"], mlflow_uri=mlflow_uri)
    register_task.after(eval_task)  # on n'enregistre qu'un modèle évalué
    register_task.set_memory_limit("1Gi")
 
 
if __name__ == "__main__":
    compiler.Compiler().compile(scoring_pipeline, package_path="scoring_pipeline.yaml")
LAB2_EOF
 
  # ---------------------------------------------------------------- 7.2
  w dags/scoring_dag.py <<'LAB2_EOF'
# Fichier : dags/scoring_dag.py
"""DAG Airflow « scoring_dag » : la variante Airflow du pipeline KFP du Lab 2.
 
Même image (localhost:5001/scoring-pipeline), mêmes commandes que dvc.yaml et que les
composants KFP : chaque task est un pod lancé dans le cluster kind par le
KubernetesPodOperator. Différence clé avec KFP : Airflow ne transporte AUCUN fichier.
Chaque task recopie l'arborescence du projet depuis/vers un préfixe S3 propre au run
(pipelines/stage.py --workspace) ; seules de petites valeurs passent par XCom
(identifiant du run MLflow, métriques).
"""
 
from datetime import UTC, datetime, timedelta
 
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG
from kubernetes.client import models as k8s
 
IMAGE = "localhost:5001/scoring-pipeline:1.0.0"  # même tag que pipelines/components.py
WORKSPACE = "s3://airflow-data/scoring/{{ run_id }}"  # dossier de travail du run (S3)
ENV = {
    "MLFLOW_TRACKING_URI": "http://mlflow.mlflow:5000",  # aucun identifiant S3 pour MLflow
    "LAUNCHED_BY": "airflow",  # tag launched_by des runs MLflow
    "AWS_ENDPOINT_URL": "http://s3.storage:8333",  # S3 du cours, pour le dossier de travail
    "AWS_DEFAULT_REGION": "us-east-1",
}
S3_CREDENTIALS = k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="s3-credentials"))
 
 
def stage(task_id: str, module: str, *args: str, xcom=False, memory="1Gi", retries=0):
    """Une task = un pod qui exécute « python -m pipelines.stage <module> » (une stage DVC)."""
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"scoring-{task_id}",
        namespace="airflow",  # créé par make airflow, avec le secret s3-credentials
        image=IMAGE,
        cmds=["python", "-m", "pipelines.stage", module, "--workspace", WORKSPACE],
        arguments=[*(["--xcom"] if xcom else []), "--", *args],
        env_vars=ENV,
        env_from=[S3_CREDENTIALS],
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"}, limits={"memory": memory}
        ),
        do_xcom_push=xcom,  # lit /airflow/xcom/return.json écrit par --xcom
        retries=retries,
        get_logs=True,
        on_finish_action="delete_succeeded_pod",
    )
 
 
with DAG(
    dag_id="scoring_dag",
    description="Credit scoring : ingest -> validate -> prepare -> train -> evaluate -> register",
    schedule=None,  # déclenchement manuel (cron possible : "0 3 * * *")
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    default_args={"owner": "mlops-student", "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "lab2"],
) as dag:
    ingest = stage("ingest", "src.download", retries=3)  # source externe : retries
    validate = stage("validate", "src.validate_data", memory="512Mi")
    prepare = stage("prepare", "src.prepare")
    train = stage("train", "src.train", xcom=True, memory="2Gi")
    evaluate = stage("evaluate", "src.evaluate", xcom=True)
    register = stage(
        "register",
        "src.promote",
        "--run-id",
        "{{ ti.xcom_pull(task_ids='train')['mlflow_run_id'] }}",
    )
 
    ingest >> validate >> prepare >> train >> evaluate >> register
LAB2_EOF
 
  # ---------------------------------------------------------------- 7.1 (override : fourni par le repo)
  if [ ! -f infra/airflow/docker-compose.override.yaml ]; then
    w infra/airflow/docker-compose.override.yaml <<'LAB2_EOF'
# Fichier : infra/airflow/docker-compose.override.yaml
# Adaptations du docker-compose.yaml OFFICIEL d'Airflow (téléchargé par
# install.sh, non modifié) pour le cours. Docker Compose fusionne
# automatiquement ce fichier avec docker-compose.yaml.
#   - UI/API sur http://localhost:8081 (le 8080 est pris par l'UI KFP)
#   - pas des ~80 DAGs d'exemple
#   - DAGs lus dans le dossier dags/ à la racine du repo
#   - conteneurs branchés sur le réseau Docker « kind » + kubeconfig interne :
#     le KubernetesPodOperator lance ses pods dans le cluster mlops-lab
x-cours-mlops: &cours-mlops
  environment:
    AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    AIRFLOW_CONN_KUBERNETES_DEFAULT: >-
      {"conn_type": "kubernetes",
      "extra": {"kube_config_path": "/opt/airflow/kube/config"}}
  volumes:
    - ../../dags:/opt/airflow/dags
    - ./kubeconfig:/opt/airflow/kube/config:ro
 
services:
  airflow-apiserver:
    <<: *cours-mlops
    ports: !override
      - "8081:8080"
  airflow-scheduler: *cours-mlops
  airflow-dag-processor: *cours-mlops
  airflow-worker: *cours-mlops
  airflow-triggerer: *cours-mlops
  airflow-cli: *cours-mlops
 
networks:
  default:
    name: kind
    external: true
LAB2_EOF
  else
    ok "infra/airflow/docker-compose.override.yaml déjà fourni par le repo (conservé)"
  fi
 
  # ---------------------------------------------------------------- .gitignore (bloc Lab 2)
  if [ -f .gitignore ] && ! grep -q "Lab 2 :" .gitignore; then
    {
      echo ""
      echo "# Lab 2 : fichiers générés"
      echo "scoring_pipeline.yaml"
      echo "local_outputs/"
      echo "reports/validation.json"
    } >> .gitignore
    ok "bloc « Lab 2 » ajouté au .gitignore"
  fi
fi
 
# =============================================================================
# Phase : kfp  (étape 1)
# =============================================================================
if want kfp; then
  log "Étape 1 — Installer Kubeflow Pipelines"
  kubectl get nodes >/dev/null 2>&1 || die "kubectl ne répond pas. Essaie : docker start mlops-lab-control-plane mlops-lab-worker mlops-lab-worker2 kind-registry"
 
  # Pas de Airflow en même temps que KFP (VM de 12 Go) — encadré de l'étape 4
  if docker ps --format '{{.Names}}' | grep -q airflow; then
    warn "Airflow tourne : arrêt pendant les étapes Kubeflow"
    (cd infra/airflow && docker compose down) || true
  fi
 
  if has_make_target kfp; then
    make kfp
  else
    # shellcheck disable=SC1091
    source infra/versions.env
    KFP_MANIFESTS="github.com/kubeflow/pipelines/manifests/kustomize"
    kubectl apply -k "$KFP_MANIFESTS/cluster-scoped-resources?ref=$KFP_VERSION"
    kubectl wait --for condition=established --timeout=120s \
      crd/applications.app.k8s.io crd/workflows.argoproj.io
    kubectl apply -k "$KFP_MANIFESTS/env/platform-agnostic?ref=$KFP_VERSION"
    kubectl -n kubeflow wait --for=condition=available deployment --all --timeout=900s
  fi
  kubectl -n kubeflow get pods
  ensure_kfp_ui
  python -c "import kfp; print('kfp', kfp.__version__)"
  curl -s http://localhost:8080/apis/v2beta1/healthz; echo
  ok "KFP opérationnel"
fi
 
# =============================================================================
# Phase : image  (étape 2.2 test + 2.4)
# =============================================================================
if want image; then
  log "Étape 2 — Test de validate_data, build et push de l'image « projet »"
  ensure_pf
 
  python -m src.validate_data; echo "Code retour (données valides) : $?"
  python - <<'EOF'
import pandas as pd
 
from src.dataset_config import load_config
 
cfg = load_config()
df = pd.read_csv(cfg.raw_path)
df.drop(columns=cfg.numeric[:1]).to_csv("/tmp/dataset_bad.csv", index=False)
EOF
  if python -m src.validate_data /tmp/dataset_bad.csv; then
    warn "la copie corrompue aurait dû échouer !"
  else
    ok "gate OK : la copie corrompue est refusée (code retour $?)"
  fi
  # validate_data a écrit reports/validation.json pour la copie corrompue : on le régénère propre
  python -m src.validate_data >/dev/null
 
  docker build -f pipelines/Dockerfile --build-arg VERSION="$IMAGE_TAG" -t "$IMAGE" .
  docker push "$IMAGE"
  curl -s http://localhost:5001/v2/scoring-pipeline/tags/list; echo
  docker run --rm "$IMAGE" python -c \
    "from src.dataset_config import load_config; c = load_config(); print(c.name, len(c.features))"
  ok "image $IMAGE construite et poussée"
fi
 
# =============================================================================
# Phase : compile  (étapes 3.2, 6.1)
# =============================================================================
if want compile; then
  log "Étape 3/6 — Compilation du pipeline"
  PIPELINE_IMAGE="$IMAGE" python -m pipelines.scoring_pipeline
  ls -l scoring_pipeline.yaml
  echo "Exécuteurs sur l'image du projet : $(grep -c "image: ${IMAGE}" scoring_pipeline.yaml)"
  grep -nE "maxRetryCount|resourceMemoryLimit|activeDeadlineSeconds" scoring_pipeline.yaml || \
    warn "policies retries/ressources/timeouts introuvables dans le YAML"
  ok "scoring_pipeline.yaml généré"
fi
 
# =============================================================================
# Phase : local  (étape 3.3)
# =============================================================================
if want local && [ "${SKIP_LOCAL:-0}" != "1" ]; then
  log "Étape 3.3 — Test local (kfp.local, sans cluster)"
  ensure_pf
  python - <<'EOF'
from kfp import local
 
from pipelines.scoring_pipeline import scoring_pipeline
 
# Chaque composant tourne dans un sous-processus du venv courant (pas dans l'image)
local.init(runner=local.SubprocessRunner(use_venv=False))
scoring_pipeline(mlflow_uri="http://localhost:5000")
EOF
  ok "test local réussi (artefacts dans local_outputs/)"
fi
 
# =============================================================================
# Phase : run  (étapes 4, 5.2, 5.4, 6.2 — via le SDK au lieu de l'UI)
# =============================================================================
if want run; then
  log "Étape 4/5/6 — Upload du pipeline et deux runs (test du cache)"
  ensure_pf
  ensure_kfp_ui
  [ -f scoring_pipeline.yaml ] || PIPELINE_IMAGE="$IMAGE" python -m pipelines.scoring_pipeline
 
  # Airflow ne doit pas tourner pendant les runs Kubeflow
  if docker ps --format '{{.Names}}' | grep -q airflow; then
    warn "Airflow tourne : arrêt pendant les runs Kubeflow"
    (cd infra/airflow && docker compose down) || true
  fi
 
  # 5.2 — MLflow joignable depuis un pod du cluster
  kubectl -n mlflow get pods,svc
  kubectl -n kubeflow run mlflow-check --rm -i --restart=Never --image="$IMAGE" -- python -c \
    "import urllib.request as u; print(u.urlopen('http://mlflow.mlflow:5000/health').read())" || \
    warn "test MLflow depuis un pod échoué (make mlflow ?)"
 
  python - <<'EOF'
import sys
import time
 
import kfp
 
client = kfp.Client(host="http://localhost:8080")
PACKAGE = "scoring_pipeline.yaml"
NAME = "scoring-pipeline"
 
try:
    client.upload_pipeline(PACKAGE, pipeline_name=NAME)
    print("pipeline uploadé :", NAME)
except Exception as exc:  # déjà existant -> nouvelle version
    print("pipeline déjà présent, upload d'une nouvelle version :", type(exc).__name__)
    client.upload_pipeline_version(
        PACKAGE, pipeline_version_name=f"v{int(time.time())}", pipeline_name=NAME
    )
 
failed = False
for i in (1, 2):  # le 2e run sert les premières étapes depuis le cache (étape 6.2)
    run = client.create_run_from_pipeline_package(
        PACKAGE, experiment_name="scoring", run_name=f"scoring-run-{i}-{int(time.time())}"
    )
    print(f"run {i} lancé : {run.run_id}", flush=True)
    state = client.wait_for_run_completion(run.run_id, timeout=3600).state
    print(f"run {i} -> {state}", flush=True)
    failed |= state != "SUCCEEDED"
sys.exit(1 if failed else 0)
EOF
  ok "2 runs KFP réussis — regarde http://localhost:8080 (2e run : étapes data « Cached »)"
 
  log "Vérification MLflow (alias challenger / champion)"
  python - <<'EOF'
from mlflow import MlflowClient
 
client = MlflowClient()
for alias in ("challenger", "champion"):
    try:
        mv = client.get_model_version_by_alias("scoring-model", alias)
        tags = client.get_run(mv.run_id).data.tags
        print(f"@{alias}: version {mv.version}, launched_by={tags.get('launched_by')}, "
              f"kfp_run_id={tags.get('kfp_run_id')}")
    except Exception as exc:
        print(f"@{alias}: introuvable ({exc})")
EOF
fi
 
# =============================================================================
# Phase : airflow  (étape 7.1)
# =============================================================================
if want airflow; then
  log "Étape 7.1 — Installer Airflow (Docker Compose)"
  if has_make_target airflow; then
    make airflow
  else
    # shellcheck disable=SC1091
    source infra/versions.env
    : "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID absent (.env du cours)}"
    : "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY absent (.env du cours)}"
    (
      cd infra/airflow
      curl -LfO "https://airflow.apache.org/docs/apache-airflow/${AIRFLOW_VERSION}/docker-compose.yaml"
      mkdir -p ./logs ./plugins ./config ../../dags
      echo "AIRFLOW_UID=$(id -u)" > .env
      echo "AIRFLOW_IMAGE_NAME=apache/airflow:${AIRFLOW_VERSION}-python3.12" >> .env
      kind get kubeconfig --internal --name mlops-lab > kubeconfig
      chmod 600 kubeconfig
      kubectl create namespace airflow --dry-run=client -o yaml | kubectl apply -f -
      kubectl -n airflow create secret generic s3-credentials \
        --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
        --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -
      docker compose up airflow-init
      docker compose up -d
      docker compose ps
    )
  fi
  ok "Airflow : http://localhost:8081 (airflow / airflow)"
fi
 
# =============================================================================
# Phase : dag  (étape 7.3 — trigger via la CLI)
# =============================================================================
if want dag; then
  log "Étape 7.3 — Déclencher scoring_dag"
  [ -d infra/airflow ] || die "infra/airflow introuvable"
  ensure_pf
  AF() { (cd infra/airflow && docker compose exec -T airflow-scheduler airflow "$@"); }
 
  echo "Attente du chargement du DAG (jusqu'à 3 min)…"
  for _ in $(seq 1 36); do
    AF dags list 2>/dev/null | grep -q scoring_dag && break
    sleep 5
  done
  AF dags list-import-errors || true
  AF dags unpause scoring_dag
  AF dags trigger scoring_dag
 
  echo "Suivi du run (jusqu'à 30 min) — pods : kubectl -n airflow get pods -w"
  state=""
  for _ in $(seq 1 180); do
    state="$(AF dags list-runs scoring_dag -o json 2>/dev/null \
      | python -c 'import sys,json
try:
    runs=json.load(sys.stdin)
    print(runs[0]["state"] if runs else "")
except Exception:
    print("")' || true)"
    case "$state" in success|failed) break ;; esac
    sleep 10
  done
  [ "$state" = "success" ] && ok "scoring_dag : success" || warn "scoring_dag état = '${state:-inconnu}' (voir http://localhost:8081)"
fi
 
# =============================================================================
# Phase : report  (étape 9 — modèle seulement)
# =============================================================================
if want report; then
  log "Étape 9 — Modèle du rapport"
  if [ -f labs/m2-pipelines/starter/rapport-m2.md ] && [ ! -f reports/rapport-m2.md ]; then
    mkdir -p reports
    cp labs/m2-pipelines/starter/rapport-m2.md reports/rapport-m2.md
    ok "reports/rapport-m2.md créé : À RÉDIGER toi-même (~500 mots, 5 sections, aucun TODO)"
  else
    warn "modèle déjà copié ou starter introuvable"
  fi
fi
 
# =============================================================================
# Phase : check  (vérification automatique + mesures de l'étape 8)
# =============================================================================
if want check; then
  log "Étape 8 — Mesures utiles"
  docker stats --no-stream || true
  wc -l pipelines/components.py pipelines/scoring_pipeline.py dags/scoring_dag.py || true
  rm -rf /tmp/ws
  python -m pipelines.stage src.download --workspace /tmp/ws
  python -m pipelines.stage src.validate_data --workspace /tmp/ws
  ls -R /tmp/ws | head -20
 
  log "Vérification automatique : make check-m2"
  if has_make_target check-m2; then
    make check-m2 || warn "check-m2 a des ❌ (rapport pas encore rédigé ? voir la sortie)"
  else
    warn "cible check-m2 absente"
  fi
fi
 
# =============================================================================
# Phase : git  (étape 9 — jamais incluse dans « tout »)
# =============================================================================
if want git; then
  log "Commit + push de la branche $BRANCH"
  git add src/validate_data.py pipelines/ dags/
  [ -f reports/rapport-m2.md ] && git add reports/rapport-m2.md
  git status --short
  git commit -m "feat: pipeline KFP scoring-pipeline + DAG Airflow scoring_dag"
  git push -u origin "$BRANCH"
  echo "Ouvre ensuite la PR vers TON repo (base repository = VOTRE_USER/mlops-cours-2026)."
fi
 
log "Terminé ✅"
