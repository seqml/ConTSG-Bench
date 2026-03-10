"""
CLI entry point for ConTSG.

This module provides the main command-line interface for:
- Training models
- Evaluating experiments
- Listing available models and datasets
- Managing experiments

Usage:
    contsg train --dataset synth-m --model verbalts
    contsg evaluate experiments/20250101_synth-m_verbalts/
    contsg list-models
    contsg list-datasets
    contsg list-experiments
"""

from __future__ import annotations

import warnings
# Suppress FutureWarnings from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="local_attention")
warnings.filterwarnings("ignore", category=FutureWarning, module="colt5_attention")

from pathlib import Path
import os
from typing import Any, Dict, List, Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Create CLI app
app = typer.Typer(
    name="contsg",
    help="ConTSG - Conditional Time Series Generation Benchmark",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()



def _ensure_imports():
    """Lazy import to avoid slow startup for help commands."""
    import contsg.models
    import contsg.data.datasets
    from contsg.registry import Registry
    Registry.auto_discover("contsg")


# =============================================================================
# Train Command
# =============================================================================

@app.command()
def train(
    # Required arguments (but optional when using --config)
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d",
        help="Dataset name (e.g., synth-m, ettm1)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name (e.g., verbalts, bridge)",
    ),
    # Data paths
    data_folder: Optional[Path] = typer.Option(
        None, "--data-folder",
        help="Path to data folder (overrides default)",
    ),
    clip_folder: Optional[Path] = typer.Option(
        None, "--clip-folder",
        help="Path to CLIP model folder for CTTP metric",
    ),
    # Training parameters
    epochs: int = typer.Option(
        700, "--epochs", "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        256, "--batch-size", "-b",
        help="Training batch size",
    ),
    lr: float = typer.Option(
        1e-3, "--lr",
        help="Learning rate",
    ),
    # Multi-stage training
    stages: Optional[str] = typer.Option(
        None, "--stages",
        help="Stage preset: 'single', 'two_stage', 'pretrain_freeze', or custom YAML",
    ),
    # Experiment management
    name: Optional[str] = typer.Option(
        None, "--name", "-n",
        help="Custom experiment name",
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", "-r",
        help="Resume from experiment directory or checkpoint",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config file (overrides CLI arguments)",
    ),
    output_dir: Path = typer.Option(
        Path("experiments"), "--output-dir", "-o",
        help="Output directory for experiments",
    ),
    # Other options
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("cuda:0", "--device", help="Device to use"),
    debug: bool = typer.Option(
        False, "--debug",
        help="Debug mode (10 epochs, verbose logging)",
    ),
    smoke: bool = typer.Option(
        False, "--smoke",
        help="CPU smoke test (tiny batches, minimal eval)",
    ),
    smoke_gpu: bool = typer.Option(
        False, "--smoke-gpu",
        help="GPU smoke test (small epochs, full metrics, batch_size=128)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show configuration without running",
    ),
    no_eval: bool = typer.Option(
        False, "--no-eval",
        help="Skip automatic evaluation after training",
    ),
    strict_schema: Optional[bool] = typer.Option(
        None, "--strict-schema/--no-strict-schema",
        help=(
            "Enable strict model schema validation. If unspecified, "
            "falls back to CONTSG_STRICT_SCHEMA env var (default: relaxed)."
        ),
    ),
    progress: str = typer.Option(
        "auto", "--progress", "-p",
        help="Progress display mode: 'auto' (detect TTY), 'tqdm', 'log', or 'off'",
    ),
):
    """
    Train a conditional time series generation model.

    Supports multi-stage training (e.g., pretrain → finetune) with a single command.

    [bold]Examples:[/bold]

        contsg train -d synth-m -m verbalts

        contsg train -d ettm1 -m bridge --epochs 1000 --lr 5e-4

        contsg train -d synth-m -m verbalts --stages two_stage

        contsg train --config my_experiment.yaml

        contsg train -d synth-m -m verbalts --resume experiments/exp1/
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _ensure_imports()

    # Set progress mode
    from contsg.utils.progress import set_progress_mode, ProgressMode
    try:
        set_progress_mode(ProgressMode(progress))
    except ValueError:
        console.print(f"[red]Invalid progress mode:[/red] {progress}")
        console.print("Valid options: auto, tqdm, log, off")
        raise typer.Exit(1)

    import torch
    import pytorch_lightning as pl
    from contsg.config.loader import ConfigLoader
    from contsg.config.model_validation import validate_model_config
    from contsg.config.schema import STAGE_PRESETS
    from contsg.registry import Registry
    from contsg.tracker import ExperimentTracker
    from contsg.train.multi_stage import MultiStageTrainer

    # Validate: either config or (dataset + model) must be provided
    if config is None and (dataset is None or model is None):
        console.print("[red]Error: Either --config or both --dataset and --model must be provided.[/red]")
        console.print("\nExamples:")
        console.print("  contsg train --config experiments.yaml")
        console.print("  contsg train -d synth-m -m verbalts")
        raise typer.Exit(1)

    # Load configuration
    loader = ConfigLoader()

    if config:
        console.print(f"[blue]Loading config from:[/blue] {config}")
        cfg = loader.from_yaml(config)

        # Apply CLI overrides to config
        overrides = {}
        if seed != 42:  # Only override if explicitly set (non-default)
            overrides["seed"] = seed
        if output_dir != Path("experiments"):  # Only override if explicitly set
            overrides["output_dir"] = Path(output_dir)  # Keep as Path, not str
        if device != "cuda:0":
            overrides["device"] = device
        if overrides:
            cfg = cfg.model_copy(update=overrides)
            console.print(f"[blue]CLI overrides applied:[/blue] {overrides}")
    else:
        # Apply debug mode
        actual_epochs = 10 if debug else epochs
        try:
            cfg = loader.from_args(
                dataset=dataset,
                model=model,
                data_folder=str(data_folder) if data_folder else None,
                clip_folder=str(clip_folder) if clip_folder else None,
                epochs=actual_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                device=device,
                output_dir=str(output_dir),
            )
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    # Apply stages preset if specified
    if stages:
        if stages in STAGE_PRESETS:
            cfg.train.stages_preset = stages
            # Re-validate to apply preset
            cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"stages_preset": stages})})
            console.print(f"[blue]Using stages preset:[/blue] {stages}")
        else:
            console.print(f"[red]Unknown stages preset:[/red] {stages}")
            console.print(f"Available: {list(STAGE_PRESETS.keys())}")
            raise typer.Exit(1)

    # Handle resume
    if resume:
        if resume.is_dir():
            # Resume from experiment directory
            cfg.resume_from = resume / "checkpoints" / "last.ckpt"
        else:
            cfg.resume_from = resume

    # Apply smoke-test overrides for fast CPU debugging
    if smoke_gpu:
        cfg = _apply_smoke_gpu_overrides(cfg)
    elif smoke:
        cfg = _apply_smoke_overrides(cfg)

    # Validate model config with hybrid schema strategy
    try:
        model_validation = validate_model_config(cfg, strict_schema=strict_schema)
        cfg = model_validation.config
    except ValueError as e:
        console.print(f"[red]Model configuration validation failed:[/red] {e}")
        raise typer.Exit(1)

    mode = "strict" if model_validation.strict_schema else "relaxed"
    if model_validation.schema_class is None:
        console.print(
            f"[blue]Model schema mode:[/blue] {mode} "
            f"(no model-specific schema for '{model_validation.resolved_model_name}')"
        )
    else:
        console.print(
            f"[blue]Model schema mode:[/blue] {mode} "
            f"({model_validation.schema_source}: {model_validation.schema_class.__name__})"
        )

    # Dry run - just show config
    if dry_run:
        console.print(Panel.fit(
            cfg.model_dump_yaml(),
            title="[bold]Experiment Configuration[/bold]",
            border_style="blue",
        ))
        return

    # Print configuration summary
    _print_config_summary(cfg)

    # Initialize experiment tracker
    tracker = ExperimentTracker(cfg, experiment_name=name)
    tracker.start()

    console.print(f"\n[green]Experiment directory:[/green] {tracker.experiment_dir}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create model class (not instance yet - MultiStageTrainer will create it)
    console.print(f"\n[blue]Model:[/blue] {cfg.model.name}")
    model_cls = Registry.get_model(cfg.model.name)

    # Validate stage support
    if hasattr(model_cls, "SUPPORTED_STAGES") and model_cls.SUPPORTED_STAGES:
        for stage in cfg.train.stages:
            if stage.name not in model_cls.SUPPORTED_STAGES:
                console.print(
                    f"[red]Model {cfg.model.name} does not support stage '{stage.name}'[/red]"
                )
                console.print(f"Supported stages: {model_cls.SUPPORTED_STAGES}")
                raise typer.Exit(1)

    # Create data module
    console.print(f"[blue]Creating data module:[/blue] {cfg.data.name}")
    dataset_cls = Registry.get_dataset(cfg.data.name)
    train_config = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "model_name": cfg.model.name,  # Pass model name for model-specific data handling
        "condition": cfg.condition,
        "seed": cfg.seed,
    }
    if smoke:
        train_config["debug_samples"] = {"train": 16, "valid": 4, "test": 4}
    elif smoke_gpu:
        train_config["debug_samples"] = {"train": 512, "valid": 128, "test": 128}

    datamodule = dataset_cls(cfg.data, train_config=train_config)

    # Use multi-stage trainer
    console.print(f"\n[bold green]Starting multi-stage training ({len(cfg.train.stages)} stage(s))...[/bold green]")
    for i, stage in enumerate(cfg.train.stages):
        console.print(f"  Stage {i+1}: {stage.name} ({stage.epochs} epochs, lr={stage.lr})")

    multi_trainer = MultiStageTrainer(
        cfg,
        model_cls,
        datamodule,
        checkpoint_root=tracker.experiment_dir,
    )
    final_checkpoint = multi_trainer.train()

    # Finish experiment
    tracker.finish(str(final_checkpoint))

    console.print(f"\n[bold green]✓ Training completed![/bold green]")
    console.print(f"  Final checkpoint: {final_checkpoint}")
    console.print(f"  Experiment: {tracker.experiment_dir}")

    # Automatic evaluation (default: enabled)
    if not no_eval:
        if cfg.model.name == "cttp":
            console.print(
                "\n[dim]Evaluation skipped for CTTP (representation model; no generate()).[/dim]"
            )
        else:
            console.print(f"\n[bold blue]Running automatic evaluation...[/bold blue]")
            results = _run_post_training_evaluation(tracker.experiment_dir, cfg)
            if results:
                output_path = _save_eval_results(tracker.experiment_dir, results)
                console.print(f"[green]Evaluation results saved to:[/green] {output_path}")
                _print_results_table(results)
    else:
        console.print(f"\n[dim]Evaluation skipped (--no-eval)[/dim]")


def _apply_smoke_overrides(cfg):
    """Apply fast CPU-friendly overrides for smoke testing."""
    train_updates = {
        "epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "early_stopping_patience": 0,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "num_sanity_val_steps": 0,
        "scheduler": "none",
    }
    train_cfg = cfg.train.model_copy(update=train_updates)

    updated_stages = [
        stage.model_copy(update={
            "epochs": train_cfg.epochs,
            "early_stopping_patience": 0,
        })
        for stage in train_cfg.stages
    ]
    train_cfg = train_cfg.model_copy(update={"stages": updated_stages})

    eval_cfg = cfg.eval.model_copy(update={
        "n_samples": 1,
        "metrics": ["ed"],
        "batch_size": 4,
        "save_samples": False,
    })

    return cfg.model_copy(update={
        "device": "cuda",
        "train": train_cfg,
        "eval": eval_cfg,
    })


def _apply_smoke_gpu_overrides(cfg):
    """Apply GPU-friendly overrides for smoke testing with full metrics."""
    train_updates = {
        "epochs": 1,
        "batch_size": 128,
        "num_workers": 4,
        "pin_memory": True,
        "early_stopping_patience": 0,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "num_sanity_val_steps": 0,
        "scheduler": "none",
    }
    train_cfg = cfg.train.model_copy(update=train_updates)

    updated_stages = [
        stage.model_copy(update={
            "epochs": train_cfg.epochs,
            "early_stopping_patience": 0,
        })
        for stage in train_cfg.stages
    ]
    train_cfg = train_cfg.model_copy(update={"stages": updated_stages})

    # Keep all metrics, just reduce n_samples and increase batch_size
    eval_cfg = cfg.eval.model_copy(update={
        "n_samples": 1,
        "batch_size": 128,
        "save_samples": False,
    })

    return cfg.model_copy(update={
        "device": "cuda",
        "train": train_cfg,
        "eval": eval_cfg,
    })


def _print_config_summary(cfg) -> None:
    """Print a summary of the configuration."""
    table = Table(title="Configuration Summary", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    table.add_row("Dataset", cfg.data.name)
    table.add_row("Model", cfg.model.name)
    table.add_row("Epochs", str(cfg.train.epochs))
    table.add_row("Batch Size", str(cfg.train.batch_size))
    table.add_row("Learning Rate", str(cfg.train.lr))
    table.add_row("Device", cfg.device)
    table.add_row("Seed", str(cfg.seed))

    console.print(table)


# =============================================================================
# Evaluate Command
# =============================================================================

@app.command()
def evaluate(
    experiment: Path = typer.Argument(
        ...,
        help="Path to experiment directory",
    ),
    checkpoint: str = typer.Option(
        "best", "--checkpoint", "-c",
        help="Checkpoint to use: 'best', 'last', or specific filename",
    ),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", "-m",
        help="Metrics to compute (comma-separated, e.g., 'dtw,fid,cttp')",
    ),
    n_samples: Optional[int] = typer.Option(
        None, "--n-samples", "-n",
        help="Number of samples to generate per condition (default: from config)",
    ),
    sampler: Optional[str] = typer.Option(
        None, "--sampler", "-s",
        help="Sampling method (ddpm, ddim); defaults to config",
    ),
    use_cache: bool = typer.Option(
        False, "--use-cache",
        help="Use and (if missing) save a prediction cache (predictions_cache.pkl)",
    ),
    refresh_cache: bool = typer.Option(
        False, "--refresh-cache",
        help="Regenerate predictions and overwrite the prediction cache (implies --use-cache)",
    ),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Use cache only, skip model loading (requires valid cache, implies --use-cache)",
    ),
    viz: bool = typer.Option(
        False, "--viz",
        help="Enable case study visualization (overrides config.eval.viz.enable)",
    ),
    viz_n_cases: Optional[int] = typer.Option(
        None, "--viz-n-cases",
        help="Number of cases to visualize (overrides config.eval.viz.k_cases)",
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output filename for results (default: eval_results.json)",
    ),
):
    """
    Evaluate a trained model.

    [bold]Examples:[/bold]

        contsg evaluate experiments/20250101_synth-m_verbalts/

        contsg evaluate experiments/exp1/ --checkpoint last --metrics dtw,fid

        contsg evaluate experiments/exp1/ --use-cache  # Reuse or create prediction cache

        contsg evaluate experiments/exp1/ --use-cache --refresh-cache  # Force regenerate cache

        contsg evaluate experiments/exp1/ --cache-only  # Skip model loading, use cache only

        contsg evaluate experiments/exp1/ --use-cache --viz --viz-n-cases 10  # Save figures
    """
    _ensure_imports()

    from contsg.tracker import ExperimentTracker
    from contsg.eval import Evaluator

    console.print(f"[blue]Loading experiment:[/blue] {experiment}")

    # Load experiment
    tracker = ExperimentTracker.from_experiment_dir(experiment)
    config = tracker.config

    # Handle cache_only mode
    if cache_only:
        use_cache = True  # Implicit
        # Check cache existence early
        cache_file = config.eval.cache_file if hasattr(config.eval, 'cache_file') else "predictions_cache.pkl"
        cache_path_check = experiment / "results" / cache_file
        if not cache_path_check.exists():
            console.print(f"[red]Cache not found:[/red] {cache_path_check}")
            console.print("[yellow]Hint: Run without --cache-only first to generate cache[/yellow]")
            raise typer.Exit(1)
        console.print(f"[blue]Cache-only mode:[/blue] skipping model loading")

    # Get checkpoint path (required if not cache_only)
    ckpt_path = None
    if not cache_only:
        if checkpoint == "best":
            ckpt_path = tracker.get_best_checkpoint()
        elif checkpoint == "last":
            ckpt_path = experiment / "checkpoints" / "last.ckpt"
        else:
            ckpt_path = experiment / "checkpoints" / checkpoint

        # Check existence only if we need the model
        if not ckpt_path or not ckpt_path.exists():
            console.print(f"[red]Checkpoint not found:[/red] {ckpt_path}")
            raise typer.Exit(1)
        console.print(f"[blue]Using checkpoint:[/blue] {ckpt_path}")
    else:
        console.print(f"[dim]Checkpoint loading skipped (cache-only mode)[/dim]")

    # Parse metrics
    metric_list = metrics.split(",") if metrics else config.eval.metrics

    # Use config values if not specified
    n_samples_actual = n_samples or config.eval.n_samples

    console.print(f"[blue]Computing metrics:[/blue] {', '.join(metric_list)}")
    console.print(f"[blue]Samples per condition:[/blue] {n_samples_actual}")
    sampler_actual = sampler or config.eval.sampler
    console.print(f"[blue]Sampler:[/blue] {sampler_actual}")

    # Initialize evaluator
    try:
        evaluator = Evaluator.from_experiment(experiment, checkpoint, cache_only=cache_only)
    except Exception as e:
        console.print(f"[red]Failed to load evaluator:[/red] {e}")
        raise typer.Exit(1)

    # Override visualization settings if requested (without mutating config.yaml)
    if viz:
        evaluator.config.eval.viz.enable = True
    if viz_n_cases is not None:
        evaluator.config.eval.viz.k_cases = viz_n_cases
    if viz or viz_n_cases is not None:
        console.print(
            f"[blue]Case study viz:[/blue] enable={evaluator.config.eval.viz.enable}, "
            f"k_cases={evaluator.config.eval.viz.k_cases}"
        )

    # Initialize CLIP if configured
    if config.eval.clip_config_path and config.eval.clip_model_path:
        console.print("[blue]Initializing CLIP embedder...[/blue]")
        clip_success = evaluator.init_clip(
            clip_config_path=config.eval.clip_config_path,
            clip_model_path=config.eval.clip_model_path,
            cache_folder=config.eval.cache_folder,
            use_longalign=config.eval.use_longalign,
        )
        if clip_success:
            console.print("[green]CLIP initialized successfully[/green]")
        else:
            console.print("[yellow]CLIP initialization failed, CLIP-dependent metrics will be skipped[/yellow]")
    else:
        console.print("[dim]No CLIP config provided, running CLIP-independent metrics only[/dim]")

    # Determine cache path
    cache_path = experiment / "results" / config.eval.cache_file

    if refresh_cache and not use_cache:
        use_cache = True

    if refresh_cache and cache_path.exists():
        try:
            cache_path.unlink()
            console.print(f"[yellow]Deleted existing cache:[/yellow] {cache_path}")
        except OSError as e:
            console.print(f"[red]Failed to delete cache:[/red] {cache_path} ({e})")
            raise typer.Exit(1)

    # Build metric configurations with experiment-specific paths
    metric_configs = _build_metric_configs(experiment, metric_list)

    # Run evaluation
    console.print("\n[bold]Running evaluation...[/bold]")
    try:
        results = evaluator.evaluate(
            metrics=metric_list,
            n_samples=n_samples_actual,
            sampler=sampler_actual,
            metric_configs=metric_configs,
            use_cache=use_cache,
            cache_path=cache_path,
        )
    except Exception as e:
        import traceback
        console.print(f"[red]Evaluation failed:[/red] {e}")
        console.print(f"[red]Exception type:[/red] {type(e).__name__}")
        console.print(f"[red]Traceback:[/red]")
        traceback.print_exc()
        raise typer.Exit(1)

    # Save results
    output_path = _save_eval_results(experiment, results, output_file)
    console.print(f"\n[green]Results saved to:[/green] {output_path}")

    # Print results table
    _print_results_table(results)


# =============================================================================
# List Commands
# =============================================================================

@app.command("list-models")
def list_models():
    """List all available models."""
    _ensure_imports()

    from contsg.registry import Registry

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Aliases", style="dim")
    table.add_column("Description")

    for name in Registry.list_models():
        info = Registry.get_model_info(name)
        aliases = ", ".join(info["aliases"]) if info["aliases"] else "-"
        doc_lines = [
            line.strip()
            for line in (info["docstring"] or "").splitlines()
            if line.strip()
        ]
        doc = doc_lines[0] if doc_lines else "-"
        table.add_row(name, aliases, doc)

    console.print(table)


@app.command("list-datasets")
def list_datasets():
    """List all available datasets."""
    _ensure_imports()

    from contsg.registry import Registry

    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for name in Registry.list_datasets():
        cls = Registry.get_dataset(name)
        doc = cls.__doc__.split("\n")[0] if cls.__doc__ else "-"
        table.add_row(name, doc)

    console.print(table)


@app.command("list-experiments")
def list_experiments_cmd(
    output_dir: Path = typer.Option(
        Path("experiments"), "--output-dir", "-o",
        help="Experiments directory",
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status (completed, running, etc.)",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Maximum number of experiments to show",
    ),
):
    """List experiments."""
    from contsg.tracker import list_experiments

    experiments = list_experiments(output_dir, status=status)[:limit]

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title=f"Experiments ({len(experiments)} shown)")
    table.add_column("ID", style="cyan")
    table.add_column("Dataset")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Updated")

    for exp in experiments:
        table.add_row(
            exp.get("experiment_id", "?"),
            exp.get("dataset", "?"),
            exp.get("model", "?"),
            exp.get("status", "?"),
            exp.get("updated_at", "?")[:16] if exp.get("updated_at") else "?",
        )

    console.print(table)


# =============================================================================
# Info Command
# =============================================================================

@app.command()
def info(
    experiment: Path = typer.Argument(
        ...,
        help="Path to experiment directory",
    ),
):
    """Show detailed information about an experiment."""
    import json

    if not experiment.exists():
        console.print(f"[red]Experiment not found:[/red] {experiment}")
        raise typer.Exit(1)

    # Load metadata
    metadata_path = experiment / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        console.print(Panel.fit(
            json.dumps(metadata, indent=2),
            title="[bold]Experiment Metadata[/bold]",
        ))

    # Load git info
    git_path = experiment / "git_info.json"
    if git_path.exists():
        with open(git_path) as f:
            git_info = json.load(f)

        console.print(Panel.fit(
            json.dumps(git_info, indent=2),
            title="[bold]Git Information[/bold]",
        ))

    # List checkpoints
    ckpt_dir = experiment / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            console.print("\n[bold]Checkpoints:[/bold]")
            for ckpt in sorted(ckpts):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                console.print(f"  - {ckpt.name} ({size_mb:.1f} MB)")


# =============================================================================
# Version Command
# =============================================================================

@app.command()
def version():
    """Show version information."""
    from contsg import __version__

    console.print(f"ConTSG version: [bold]{__version__}[/bold]")


# =============================================================================
# Precompute Embeddings Command
# =============================================================================

@app.command("precompute-embeddings")
def precompute_embeddings(
    data_folder: Path = typer.Argument(
        ...,
        help="Path to dataset folder",
    ),
    model_path: str = typer.Option(
        ..., "--model-path", "-m",
        help="Path to embedding model (e.g., Qwen3-Embedding)",
    ),
    embed_dim: int = typer.Option(
        1024, "--embed-dim", "-d",
        help="Target embedding dimension",
    ),
    splits: str = typer.Option(
        "train,valid,test", "--splits", "-s",
        help="Comma-separated list of splits to process",
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b",
        help="Batch size for encoding",
    ),
    device: str = typer.Option(
        "cuda:0", "--device",
        help="Device to use",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Overwrite existing embedding files",
    ),
):
    """
    Precompute text embeddings for a dataset.

    This command processes caption files and creates embedding files
    for efficient training without loading large language models.

    [bold]Examples:[/bold]

        contsg precompute-embeddings ./datasets/synth-m -m /path/to/Qwen3-Embedding

        contsg precompute-embeddings ./datasets/ettm1 -m /path/to/Qwen3-Embedding -d 512

        contsg precompute-embeddings ./datasets/synth-m -m /path/to/Qwen3-Embedding --splits train,valid
    """
    from contsg.data.precompute.sentence_transformer import SentenceTransformerPrecomputer
    from contsg.data.precompute.dataset import precompute_dataset_embeddings

    console.print(f"[blue]Dataset folder:[/blue] {data_folder}")
    console.print(f"[blue]Model path:[/blue] {model_path}")
    console.print(f"[blue]Embedding dimension:[/blue] {embed_dim}")
    console.print(f"[blue]Device:[/blue] {device}")

    if not data_folder.exists():
        console.print(f"[red]Dataset folder not found:[/red] {data_folder}")
        raise typer.Exit(1)

    # Parse splits
    split_list = [s.strip() for s in splits.split(",")]
    console.print(f"[blue]Splits:[/blue] {split_list}")

    # Create precomputer
    console.print("\n[blue]Loading embedding model...[/blue]")
    precomputer = SentenceTransformerPrecomputer(
        model_path=model_path,
        embed_dim=embed_dim,
        device=device,
    )

    # Precompute embeddings
    console.print(f"\n[bold green]Starting precomputation...[/bold green]")
    created_files = precompute_dataset_embeddings(
        dataset_dir=data_folder,
        precomputer=precomputer,
        splits=split_list,
        batch_size=batch_size,
        overwrite=overwrite,
    )

    console.print(f"\n[bold green]✓ Precomputation completed![/bold green]")
    console.print(f"Created {len(created_files)} files:")
    for f in created_files:
        console.print(f"  - {f}")


# =============================================================================
# Evaluation Helper Functions
# =============================================================================


def _build_metric_configs(
    experiment_dir: Path,
    metrics: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Build metric-specific configurations with experiment-relative paths.

    Args:
        experiment_dir: Path to experiment directory
        metrics: List of metric names

    Returns:
        Dictionary mapping metric names to their configurations
    """
    metric_configs: Dict[str, Dict[str, Any]] = {}

    # Configure metrics with save directories
    if "disc_auc" in metrics:
        metric_configs["disc_auc"] = {
            "save_dir": str(experiment_dir / "results" / "disc_auc")
        }

    if "tsne_viz" in metrics:
        metric_configs["tsne_viz"] = {
            "save_dir": str(experiment_dir / "results" / "visualization")
        }

    return metric_configs


def _save_eval_results(
    experiment_dir: Path,
    results: dict,
    output_file: Optional[str] = None,
) -> Path:
    """
    Save evaluation results to JSON file.

    Args:
        experiment_dir: Path to experiment directory
        results: Dictionary of metric results
        output_file: Optional custom output filename

    Returns:
        Path to saved results file
    """
    import json
    from datetime import datetime

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = output_file or "eval_results.json"
    output_path = results_dir / filename

    # Add metadata
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": str(experiment_dir),
        "metrics": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_path


def _print_results_table(results: dict) -> None:
    """
    Print evaluation results as a formatted table.

    Args:
        results: Dictionary of metric results
    """
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Group metrics by category
    categories = {
        "Basic": ["dtw", "wape", "ed", "crps"],
        "Statistical": ["acd", "sd", "kd", "mdd"],
        "Frechet": ["fid", "jftsd"],
        "PRDC": ["prdc_f1", "joint_prdc_f1", "precision", "recall", "density", "coverage"],
        "Specialized": ["cttp", "disc_auc", "segment_accuracy"],
    }

    # Find uncategorized metrics
    all_categorized = set()
    for metrics in categories.values():
        all_categorized.update(metrics)

    uncategorized = [k for k in results.keys() if k.lower() not in all_categorized]

    # Print by category
    for category, metric_names in categories.items():
        category_results = {k: v for k, v in results.items() if k.lower() in metric_names}
        if category_results:
            for name, value in category_results.items():
                if isinstance(value, float):
                    if value != value:  # NaN check
                        table.add_row(name, "[dim]N/A[/dim]")
                    else:
                        table.add_row(name, f"{value:.6f}")
                else:
                    table.add_row(name, str(value))

    # Print uncategorized
    for name in uncategorized:
        value = results[name]
        if isinstance(value, float):
            if value != value:  # NaN check
                table.add_row(name, "[dim]N/A[/dim]")
            else:
                table.add_row(name, f"{value:.6f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def _run_post_training_evaluation(
    experiment_dir: Path,
    config,
) -> Optional[dict]:
    """
    Run evaluation after training completes.

    Args:
        experiment_dir: Path to experiment directory
        config: ExperimentConfig

    Returns:
        Evaluation results or None if evaluation fails
    """
    from contsg.eval import Evaluator

    try:
        evaluator = Evaluator.from_experiment(experiment_dir, checkpoint="best")

        # Initialize CLIP if configured
        if config.eval.clip_config_path and config.eval.clip_model_path:
            clip_success = evaluator.init_clip(
                clip_config_path=config.eval.clip_config_path,
                clip_model_path=config.eval.clip_model_path,
                cache_folder=config.eval.cache_folder,
                use_longalign=config.eval.use_longalign,
            )
            if not clip_success:
                console.print("[yellow]CLIP initialization failed, running CLIP-independent metrics only[/yellow]")

        # Determine cache path for visualization output
        cache_path = experiment_dir / "results" / config.eval.cache_file

        # Build metric configurations with experiment-specific paths
        metric_configs = _build_metric_configs(experiment_dir, config.eval.metrics)

        results = evaluator.evaluate(
            metrics=config.eval.metrics,
            n_samples=config.eval.n_samples,
            sampler=config.eval.sampler,
            metric_configs=metric_configs,
            use_cache=config.eval.use_cache,
            cache_path=cache_path,
        )

        return results

    except Exception as e:
        import traceback
        console.print(f"[yellow]Post-training evaluation failed: {e}[/yellow]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


# =============================================================================
# Main Entry Point
# =============================================================================

# Global state for verbose error output
_verbose_errors = False


@app.callback()
def main_callback(
    verbose_errors: bool = typer.Option(
        False, "--verbose-errors", "-v",
        help="Show full traceback on errors",
    ),
):
    """ConTSG - Conditional Time Series Generation Benchmark."""
    global _verbose_errors
    _verbose_errors = verbose_errors


def main():
    """Main CLI entry point."""
    import sys

    try:
        app()
    except Exception as e:
        if _verbose_errors:
            # Show full traceback
            console.print_exception()
        else:
            # Show concise error message
            error_type = type(e).__name__
            console.print(f"\n[bold red]Error:[/bold red] {error_type}: {e}")
            console.print("[dim]Use --verbose-errors for full traceback[/dim]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
