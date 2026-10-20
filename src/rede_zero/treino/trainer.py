"""Loop de treinamento, checkpointing e logging para paradigmas de rastreamento (SCR-11)."""

import csv
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from rede_zero.paradigmas.base import ParadigmaBase, get_paradigma


def _merge_dicts(base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


def carregar_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.get("_base_")
    if base_name:
        base_path = p.parent / base_name
        base_cfg = carregar_config(base_path)
        cfg = _merge_dicts(base_cfg, cfg)

    exp = cfg.get("experimento", "default")
    cfg_dump = yaml.dump(cfg).replace("${experimento}", exp)
    return yaml.safe_load(cfg_dump)


class Treinador:
    """Treinador genérico com suporte a AdamW, AMP, checkpointing e logs CSV."""

    def __init__(
        self,
        config: Dict[str, Any] | str | Path,
        modelo: Optional[ParadigmaBase] = None,
        loader_treino: Optional[DataLoader] = None,
        loader_val: Optional[DataLoader] = None,
        resume: Optional[str | Path] = None,
    ):
        if isinstance(config, (str, Path)):
            self.config = carregar_config(config)
        else:
            self.config = dict(config)

        # Semente
        semente = int(self.config.get("semente", 42))
        torch.manual_seed(semente)
        np.random.seed(semente)

        # Device
        device_req = self.config.get("device", "cuda")
        self.device = torch.device("cuda" if device_req == "cuda" and torch.cuda.is_available() else "cpu")

        # Modelo
        if modelo is not None:
            self.modelo = modelo
        else:
            paradigma_nome = self.config.get("paradigma", "siamese")
            self.modelo = get_paradigma(paradigma_nome)
        self.modelo.to(self.device)

        self.loader_treino = loader_treino
        self.loader_val = loader_val

        treino_cfg = self.config.get("treino", {})
        self.lr = float(treino_cfg.get("lr", 1e-3))
        self.weight_decay = float(treino_cfg.get("weight_decay", 1e-4))
        self.total_epochs = int(treino_cfg.get("epochs", 50))

        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.modelo.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.total_epochs)
        )

        # AMP
        use_amp = bool(treino_cfg.get("amp", False)) and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # Pastas de saída
        exp = self.config.get("experimento", "default")
        self.ckpt_dir = Path(treino_cfg.get("ckpt_dir", f"runs/{exp}/checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = Path(f"runs/{exp}/train_log.csv")
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)

        self.inicio_epoch = 1
        self.best_loss = float("inf")

        # Resume
        caminho_resume = resume or treino_cfg.get("resume")
        if caminho_resume:
            self._carregar_checkpoint(Path(caminho_resume))

    def _carregar_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            candidato = self.ckpt_dir / "last.pt"
            if candidato.is_file():
                path = candidato
            else:
                return
        ckpt = torch.load(path, map_location=self.device)
        self.modelo.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.inicio_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_loss = float(ckpt.get("loss", float("inf")))

    def salvar_checkpoint(self, epoch: int, loss: float, nome: str = "last.pt") -> Path:
        destino = self.ckpt_dir / nome
        dados = {
            "epoch": epoch,
            "loss": loss,
            "paradigma": getattr(self.modelo, "nome", "desconhecido"),
            "versao": getattr(self.modelo, "versao", "0.0.0"),
            "model_state_dict": self.modelo.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(dados, destino)
        return destino

    def treinar_uma_epoca(self) -> float:
        if self.loader_treino is None:
            raise ValueError("Loader de treino não fornecido.")
        self.modelo.train()
        perdas = []

        for batch in self.loader_treino:
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            self.optimizer.zero_grad()
            autocast_ctx = (
                torch.amp.autocast("cuda", enabled=self.use_amp)
                if hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.use_amp)
            )
            with autocast_ctx:
                saidas = self.modelo(batch_dev)
                loss = self.modelo.perda_fn(saidas, batch_dev)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            perdas.append(loss.item())

        return float(np.mean(perdas)) if perdas else 0.0

    def treinar(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        num_epochs = epochs or self.total_epochs
        existe_csv = self.log_csv.is_file()

        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not existe_csv:
                writer.writerow(["epoch", "loss", "lr", "tempo_s"])

            for ep in range(self.inicio_epoch, num_epochs + 1):
                t0 = time.time()
                loss = self.treinar_uma_epoca()
                self.scheduler.step()
                dt = time.time() - t0
                lr_atual = self.optimizer.param_groups[0]["lr"]

                writer.writerow([ep, f"{loss:.6f}", f"{lr_atual:.6e}", f"{dt:.2f}"])
                f.flush()

                self.salvar_checkpoint(ep, loss, "last.pt")
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.salvar_checkpoint(ep, loss, "best.pt")

        return {"ult_epoch": num_epochs, "melhor_loss": self.best_loss, "ult_loss": loss}
