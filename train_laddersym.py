"""Train LadderSym models with Hydra configuration files."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


def _validate_dataset_paths(cfg) -> None:
    """Validate required dataset paths before starting training."""
    help_msg = (
        "Set dataset paths via env vars:\n"
        "  LADDERSYM_COCO_ROOT, LADDERSYM_COCO_SPLIT_JSON\n"
        "  LADDERSYM_MAESTRO_ROOT, LADDERSYM_MAESTRO_SPLIT_JSON\n"
        "or pass Hydra overrides like:\n"
        "  dataset.train.root_dir=/path dataset.train.split_json_path=/path/to/split.json"
    )
    checks = [
        ("dataset.train.root_dir", cfg.dataset.train.root_dir, "dir"),
        ("dataset.train.split_json_path", cfg.dataset.train.split_json_path, "file"),
        ("dataset.val.root_dir", cfg.dataset.val.root_dir, "dir"),
        ("dataset.val.split_json_path", cfg.dataset.val.split_json_path, "file"),
    ]
    for key, value, expected_type in checks:
        if value is None or str(value).strip() == "":
            raise ValueError(f"{key} is empty.\n{help_msg}")
        path = Path(value)
        if expected_type == "dir" and not path.is_dir():
            raise FileNotFoundError(f"{key} does not exist or is not a directory: {value}")
        if expected_type == "file" and not path.is_file():
            raise FileNotFoundError(f"{key} does not exist or is not a file: {value}")


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Instantiate training and validation dataloaders from Hydra config."""
    train_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.train),
        **cfg.dataloader.train,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn),
    )
    val_loader = DataLoader(
        hydra.utils.instantiate(cfg.dataset.val),
        **cfg.dataloader.val,
        collate_fn=hydra.utils.get_method(cfg.dataset.collate_fn),
    )
    return train_loader, val_loader


def _fit_with_checkpoint_strategy(cfg, trainer, model, train_loader, val_loader) -> None:
    """Run training according to checkpoint initialization strategy."""
    checkpoint_path = str(cfg.get("path") or "").strip()
    if not checkpoint_path:
        trainer.fit(model, train_loader, val_loader)
        return

    if cfg.get("use_lightweight_checkpoint", False):
        print("Creating and using a lightweight checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        lightweight_ckpt_path = checkpoint_path.replace(".ckpt", "_lightweight.ckpt")
        torch.save({"state_dict": state_dict}, lightweight_ckpt_path)
        print(f"Saved lightweight checkpoint to {lightweight_ckpt_path}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys, flush=True)
        print("Unexpected keys:", unexpected_keys, flush=True)
        trainer.fit(model, train_loader, val_loader, ckpt_path=None)
        return

    if checkpoint_path.endswith(".ckpt"):
        print(f"Validating and resuming training from {checkpoint_path}...")
        trainer.validate(model, val_loader, ckpt_path=checkpoint_path)
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
        return

    if checkpoint_path.endswith(".pth"):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        load_target = model.model if hasattr(model, "model") else model
        missing_keys, unexpected_keys = load_target.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys, flush=True)
        print("Unexpected keys:", unexpected_keys, flush=True)
        trainer.fit(model, train_loader, val_loader)
        return

    raise ValueError(f"Invalid extension for path: {checkpoint_path}")


def _export_model_pt(
    model: torch.nn.Module,
    checkpoint_callback: ModelCheckpoint,
    output_dir: str,
    model_type: str,
    dataset_type: str,
) -> Path:
    """Export the current model state dict to a standalone `.pt` file."""
    export_ckpt_path = None
    for candidate in (checkpoint_callback.last_model_path, checkpoint_callback.best_model_path):
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                export_ckpt_path = candidate_path
                break

    if export_ckpt_path is None:
        export_path = Path(output_dir) / f"{model_type}_{dataset_type}.pt"
    else:
        export_path = export_ckpt_path.with_suffix(".pt")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith("model."):
            state_dict[key.replace("model.", "", 1)] = value
        else:
            state_dict[key] = value
    torch.save(state_dict, export_path)
    return export_path


@hydra.main(config_path="config", config_name="config_maestro", version_base="1.1")
def main(cfg) -> None:
    """Entry point for LadderSym training."""
    pl.seed_everything(cfg.seed)
    cfg.model.config.use_prompt = cfg.use_prompt
    _validate_dataset_paths(cfg)

    model = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    num_params = _count_trainable_parameters(model)
    print(f"Trainable parameters: {num_params:,}", flush=True)

    logger = WandbLogger(project=f"{cfg.model_type}_{cfg.dataset_type}")
    assert cfg.model_type == cfg.model._target_.split(".")[-1]

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=1),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    train_loader, val_loader = _build_dataloaders(cfg)

    dataset_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    num_steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    print(f"Calculated num_steps_per_epoch: {num_steps_per_epoch}")
    model.num_steps_per_epoch = num_steps_per_epoch

    _fit_with_checkpoint_strategy(cfg, trainer, model, train_loader, val_loader)

    if bool(cfg.get("compile_model", False)):
        print("Compiling the model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled.")

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    export_path = _export_model_pt(
        model=model,
        checkpoint_callback=checkpoint_callback,
        output_dir=output_dir,
        model_type=cfg.model_type,
        dataset_type=cfg.dataset_type,
    )
    print(f"Saved model in {export_path}.")


if __name__ == "__main__":
    main()
