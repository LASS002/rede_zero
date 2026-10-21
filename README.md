# rede_zero — Rastreador de Objetos Neural do Zero

Esse é um repositório experimental para criar uma rede neural do zero. Exatamente, do zero!

A rede consiste em um rastreador de objetos, simples, testando algumas arquiteturas e rastreando até 8 objetos, sem sobreposição.

A ideia é que para a inferência seja aberto um frame do vídeo, a pessoa selecione um ROI que enquadra o objeto e esse ROI se desloque, acompanhando o objeto, até o fim do vídeo.

## Estrutura e execução (resumo)

Plano mestre em [PLAN.md](PLAN.md) · log de execução em [HISTORY.md](HISTORY.md).

```
src/rede_zero/   # pacote: paradigmas/ (a rede), data/, treino/, inferencia/, utils/
scripts/         # CLIs (treinar, avaliar, inferir, inventário)
configs/         # YAML por paradigma · data/ (brutos, gitignore) · runs/ (ckpts gitignore) · tests/ · docs/
```

- Gerenciador de pacotes: **uv** → `uv sync` e `uv run python ...` / `uv run pytest`.
- Separação de execução: a rede neural fica em `paradigmas/`; a inferência é script
  separado (`scripts/inferir.py`) que importa a rede, nunca o código de treino.
- Pipeline: `rede_zero.inferencia.executar_pipeline()` rastreia até oito ROIs de forma
  independente, sem GUI, e registra objetos removidos por baixa confiança.
- Smoke de dados: `uv run python scripts/gerar_sinteticos.py --output data/sinteticos/exp1`
  gera até 90 frames anotados com objetos coloridos sobre fundo verde variável.
- Paradigmas: o registry `rede_zero.paradigmas.get_paradigma()` já define os contratos
  `siamese`, `regressao` e `detector`; Siamese já possui backbone, correlação e offset,
  Regressão já possui backbone e MLP, e o detector permanece pendente. Perdas e métricas
  de avaliação vivem em `treino/`. Consulte `docs/paradigmas.md` e `docs/arquitetura.md`.

---

## Como treinar

### 1. Preparar o ambiente
```bash
uv sync
```

### 2. Gerar dados sintéticos (opcional, para smoke test)
```bash
uv run python scripts/gerar_sinteticos.py --output data/sinteticos/exp1
```

### 3. Smoke test (overfit em 10 amostras, ~1 min)
```bash
# Siamese
uv run python scripts/treinar.py --config configs/smoke.yaml --epochs 10 --num-amostras 10

# Regressão
uv run python scripts/treinar.py --config configs/smoke_regressao.yaml --epochs 10 --num-amostras 10
```

### 4. Treino completo no dataset real (requer GPU, ~2-4h por paradigma)
```bash
# Siamese (50 epochs)
uv run python scripts/treinar.py --config configs/siamese.yaml --epochs 50

# Regressão (50 epochs)
uv run python scripts/treinar.py --config configs/regressao.yaml --epochs 50
```

**Opções úteis do `scripts/treinar.py`:**
- `--config` — Caminho do YAML de configuração (obrigatório)
- `--paradigma` — Sobrepõe o paradigma da config (`siamese` | `regressao`)
- `--epochs` — Sobrepõe o número de epochs
- `--num-amostras` — Sobrepõe o número de amostras do dataset
- `--resume` — Caminho do checkpoint ou `last` para retomar
- `--dry-run` — Monta tudo e imprime resumo sem treinar

Checkpoints são salvos em `runs/<experimento>/checkpoints/{last,best}.pt` (ignorados pelo git).
Logs de treino em `runs/<experimento>/train_log.csv`.

---

## Como avaliar

```bash
# Avaliar Siamese
uv run python scripts/avaliar.py --config configs/siamese.yaml --ckpt runs/base/checkpoints/best.pt

# Avaliar Regressão
uv run python scripts/avaliar.py --config configs/regressao.yaml --ckpt runs/regressao/checkpoints/best.pt
```

Métricas (IoU médio, AUC success, precisão@20px) exportadas em JSON/CSV em `runs/<experimento>/`.

---

## Como inferir (rastrear em vídeo novo)

### Modo interativo (GUI para selecionar ROIs)
```bash
uv run python scripts/inferir.py \
  --video caminho/do/video.mp4 \
  --paradigma siamese \
  --ckpt runs/base/checkpoints/best.pt \
  --out runs/inferencia/resultado.mp4
```
Abre o primeiro frame — clique e arraste para selecionar até 8 ROIs, pressione **Enter** após cada uma, **ESC** para finalizar.

### Modo automático (ROIs pré-definidas, sem GUI)
```bash
uv run python scripts/inferir.py \
  --video caminho/do/video.mp4 \
  --paradigma regressao \
  --ckpt runs/regressao/checkpoints/best.pt \
  --out runs/inferencia/resultado.mp4 \
  --roi "100,100,50,50" "200,150,40,60"
```
Formato `--roi`: `x,y,w,h` (canto superior esquerdo + largura + altura). Um por objeto.

### Listar paradigmas disponíveis
```bash
uv run python scripts/inferir.py --listar-paradigmas
```

---

## Estado atual

| Paradigma | Status | Checkpoint | Métricas (val) |
|-----------|--------|------------|----------------|
| Siamese   | Treinado (50 epochs) | `runs/base/checkpoints/best.pt` | Loss ~0.96 |
| Regressão | Treinado (50 epochs) | `runs/regressao/checkpoints/best.pt` | — |
| Detector  | Pendente (SCR-18) | — | — |

Dataset real: 14 sequências (3 OTB + 11 vídeos Formato-B), ~10.8k frames, 133k bboxes.
Dados sintéticos: `data/sinteticos/exp1/` (90 frames, 3 objetos, fundo verde variável).

---

## Próximos passos

1. **RUN-06**: Avaliação comparativa Siamese vs Regressão no split de validação → escolher baseline.
2. **RUN-07**: Demo de inferência em vídeo real com ROI interativo.
3. **RUN-08**: Teste multi-objeto (3–8 objetos) + relatório por objeto.
4. **SCR-18 / RUN-09**: Implementar 3º paradigma (Detector + associação).
5. **DIR-07**: Revisão da estrutura pós-inferência.
6. **DOC-01**: README base completo (esta seção será expandida).