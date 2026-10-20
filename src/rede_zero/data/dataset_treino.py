"""Dataset PyTorch para treinamento dos paradigmas Siamese e Regressão (SCR-03)."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rede_zero.data.readers.base import BBox, ReaderBase


def recortar_com_padding(
    imagem: np.ndarray, cx: float, cy: float, w: float, h: float, out_size: int
) -> np.ndarray:
    """Recorta região centrada em (cx, cy) com tamanho (w, h) e redimensiona para (out_size, out_size).

    Preenche com zeros se a região ultrapassar os limites da imagem.
    """
    img_h, img_w = imagem.shape[:2]
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    x1_val = max(0, x1)
    y1_val = max(0, y1)
    x2_val = min(img_w, x2)
    y2_val = min(img_h, y2)

    crop = imagem[y1_val:y2_val, x1_val:x2_val]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop


class TrackingDataset(Dataset):
    """Dataset amostrador de pares (template, busca, patch, alvos) para rastreamento."""

    def __init__(
        self,
        readers: List[ReaderBase],
        template_size: int = 128,
        busca_size: int = 256,
        patch_size: int = 128,
        max_intervalo: int = 10,
        augment: bool = True,
        num_amostras: int = 1000,
    ):
        self.readers = readers
        self.template_size = template_size
        self.busca_size = busca_size
        self.patch_size = patch_size
        self.max_intervalo = max_intervalo
        self.augment = augment
        self.num_amostras = num_amostras

        self.trajetorias: List[Tuple[int, int, List[Tuple[int, BBox]]]] = []
        self._indexar_trajetorias()

    def _indexar_trajetorias(self) -> None:
        """Agrupa anotações de cada leitor por ID de objeto ao longo dos frames."""
        for r_idx, reader in enumerate(self.readers):
            por_id: Dict[int, List[Tuple[int, BBox]]] = {}
            for f_idx in range(reader.n_frames):
                bboxes = reader.get_gt(f_idx)
                for b_idx, b in enumerate(bboxes):
                    obj_id = b.id if b.id != -1 else (b_idx + 1)
                    if b.w > 0 and b.h > 0:
                        por_id.setdefault(obj_id, []).append((f_idx, b))

            for obj_id, tracks in por_id.items():
                if len(tracks) >= 2:
                    self.trajetorias.append((r_idx, obj_id, tracks))

    def __len__(self) -> int:
        if not self.trajetorias:
            return 0
        return self.num_amostras

    def _amostrar_par(self) -> Tuple[np.ndarray, BBox, np.ndarray, BBox]:
        traj_idx = np.random.randint(0, len(self.trajetorias))
        r_idx, _, tracks = self.trajetorias[traj_idx]
        reader = self.readers[r_idx]

        idx1 = np.random.randint(0, len(tracks) - 1)
        max_idx2 = min(len(tracks) - 1, idx1 + self.max_intervalo)
        idx2 = np.random.randint(idx1 + 1, max_idx2 + 1)

        f1_idx, b1 = tracks[idx1]
        f2_idx, b2 = tracks[idx2]

        f1 = reader.get_frame(f1_idx)
        f2 = reader.get_frame(f2_idx)
        return f1, b1, f2, b2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f1, b1, f2, b2 = self._amostrar_par()

        cx1, cy1 = b1.center
        w1, h1 = b1.w, b1.h

        cx2, cy2 = b2.center
        w2, h2 = b2.w, b2.h

        # 1. Template: 128x128 em f1 (ROI com margem de contexto)
        # Margem de contexto típica: 2x tamanho da bbox para dar bordas
        fator_ctx = 2.0
        template_crop = recortar_com_padding(f1, cx1, cy1, w1 * fator_ctx, h1 * fator_ctx, self.template_size)

        # 2. Busca (Siamese): 256x256 centrada na posição de f1 em f2 (área onde objeto se desloca)
        busca_w = w1 * 4.0
        busca_h = h1 * 4.0
        busca_crop = recortar_com_padding(f2, cx1, cy1, busca_w, busca_h, self.busca_size)

        # Alvo Siamese: deslocamento relativo do centro de b2 em relação ao centro da busca (cx1, cy1)
        # Normalizado pelo tamanho da busca em [-0.5, 0.5]
        siamese_dx = (cx2 - cx1) / max(1e-4, busca_w)
        siamese_dy = (cy2 - cy1) / max(1e-4, busca_h)

        # 3. Patch (Regressão): 128x128 = ROI atual (b1) expandida 2x sobre o frame f2
        patch_w = w1 * 2.0
        patch_h = h1 * 2.0
        patch_crop = recortar_com_padding(f2, cx1, cy1, patch_w, patch_h, self.patch_size)

        # Alvos da Regressão: dx, dy normalizados pelo patch_size, scale factor log(w2/w1)
        dx_reg = (cx2 - cx1) / max(1e-4, patch_w)
        dy_reg = (cy2 - cy1) / max(1e-4, patch_h)
        sw_reg = float(np.log(max(1e-4, w2) / max(1e-4, w1)))
        sh_reg = float(np.log(max(1e-4, h2) / max(1e-4, h1)))

        # Augmentações leves
        if self.augment:
            if np.random.rand() > 0.5:
                # Flip horizontal
                template_crop = np.ascontiguousarray(template_crop[:, ::-1, :])
                busca_crop = np.ascontiguousarray(busca_crop[:, ::-1, :])
                patch_crop = np.ascontiguousarray(patch_crop[:, ::-1, :])
                siamese_dx = -siamese_dx
                dx_reg = -dx_reg

            if np.random.rand() > 0.5:
                # Leve ruído gaussiano
                ruido = np.random.normal(0, 5, patch_crop.shape).astype(np.float32)
                patch_crop = np.clip(patch_crop.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

        # Converte para Tensores Float32 em [0, 1], canais primeiro (C, H, W)
        t_template = torch.from_numpy(template_crop).permute(2, 0, 1).float() / 255.0
        t_busca = torch.from_numpy(busca_crop).permute(2, 0, 1).float() / 255.0
        t_patch = torch.from_numpy(patch_crop).permute(2, 0, 1).float() / 255.0

        return {
            "template": t_template,
            "busca": t_busca,
            "patch": t_patch,
            "alvo_siamese": torch.tensor([siamese_dx, siamese_dy], dtype=torch.float32),
            "alvo_regressao": torch.tensor([dx_reg, dy_reg, sw_reg, sh_reg], dtype=torch.float32),
            "confianca": torch.tensor([1.0], dtype=torch.float32),
        }
