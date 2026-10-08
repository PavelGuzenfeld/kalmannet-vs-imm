import re
from pathlib import Path

from scripts.eval import BKN_DROPOUT_P

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
MODELS = ["knet", "rkn", "bkn", "ukn"]
CHECKPOINT_FILES = ["best.pt", "best.npz", "training_log.json"]


def job_block(name):
    text = WORKFLOW.read_text()
    match = re.search(rf"^  {name}:\n(.*?)(?=^  \w[\w-]*:\n|\Z)", text, re.S | re.M)
    return match.group(1)


def test_release_attaches_every_model_checkpoint_and_exported_weight_as_an_asset():
    release = job_block("release")
    for model in MODELS:
        assert f"name: {model}-checkpoint" in release
        assert f"path: {model}/" in release
    assert "for file in " + " ".join(CHECKPOINT_FILES) in release
    assert 'cp "$model/$file" "assets/$model-$file"' in release
    assert re.search(r"gh release create .*--generate-notes assets/\*", release)


def test_training_jobs_export_the_weights_the_release_attaches():
    for job, model_dir in [("train-knet", "knet_vanilla"), ("train-rkn", "rkn"), ("train-bkn", "bkn"), ("train-ukn", "ukn")]:
        block = job_block(job)
        assert f"scripts/export_weights.py results/{model_dir}/best.pt" in block
        assert f"path: results/{model_dir}/" in block


def test_bayesian_kalmannet_trains_at_the_dropout_p_its_ensemble_runs_at():
    """MS-39: the dropout p chosen on validation is the one the release trains, not a second copy that can drift."""
    assert f"scripts/train_knet.py --dropout-p {BKN_DROPOUT_P} --out results/bkn" in job_block("train-bkn")
