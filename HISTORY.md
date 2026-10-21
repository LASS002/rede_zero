# HISTORY.md — Log de execução do plano

Formato: [docs/historico_formato.md](docs/historico_formato.md). Entradas em ordem cronológica.

## [DIR-01] 2026-10-19T20:10Z
- Agente: opencode/big-pickle | Humano: lennin — moveu PLAN.md para a raiz e decretou uso de `uv`
- Ação: criar `HISTORY.md`, `docs/historico_formato.md` e `.gitignore` base
- Resultado: 3 arquivos criados; `.gitignore` cobre dados brutos, checkpoints, vídeos, cache
- Próxima: DIR-02

## [DIR-02] 2026-10-19T20:12Z
- Agente: opencode/big-pickle | Humano: none
- Ação: esqueleto de pastas do PLAN.md §4 com `__init__.py` vazios e READMEs de 1 linha (docs/, data/, runs/)
- Resultado: `src/rede_zero/{paradigmas/{siamese,regressao,detector},data/readers,treino,inferencia,utils}`, `scripts/`, `configs/`, `data/`, `runs/`, `tests/`, `docs/`
- Próxima: DIR-03

## [DIR-03] 2026-10-19T20:25Z
- Agente: opencode/big-pickle | Humano: none
- Ação: dependências via uv (ajuste decidido nesta task: sem requirements.txt; pyproject.toml PEP 621 + dependency-group dev + uv.lock)
- Resultado: `uv sync` OK — torch 2.14.1+cu130, CUDA disponível (NVIDIA GeForce RTX 2060); PLAN.md atualizado (premissa 5, DIR-03, §9 status)
- Nota: push SSH falhou (`Permission denied (publickey)` — chave local não autorizada no GitHub); push feito via HTTPS com credencial do `gh`
- Próxima: DIR-04

## [DIR-04] 2026-10-19T21:10Z
- Agente: opencode/big-pickle | Humano: lennin — forneceu path do dataset e descritivo dos formatos
- Ação: layout `data/{real,inventario,sinteticos}` + inspeção do dataset real em `/home/labiocom01/Documents/yolo_cfc_retrain/data/raw/` + schemas documentados em `data/README.md`
- Resultado: Formato-A = frames+labels YOLO/txt+JSON (sem id de objeto, ~20 obj/frame, fonte part1.mp4/1511 frames); Formato-B = vídeo+CSV de tracking com Tracker ID (`object_id_<N>.csv`); PLAN.md §1.3 e §9 atualizados
- Nota: remote do git migrado para HTTPS + `gh auth setup-git` (push normal voltou a funcionar)
- Próxima: DIR-05 (ou DOC-01 / SCR conforme preferência do humano)

## [DIR-05] 2026-10-19T21:25Z
- Agente: opencode/big-pickle | Humano: none
- Ação: criar `configs/base.yaml` (seed, device, imagens 128/256, treino, inferência) + `siamese.yaml` e `regressao.yaml` herdados via `_base_: base.yaml` + `detector.yaml` com `status: pendente`
- Resultado: 4 YAMLs parseados com `yaml.safe_load` OK; loader de herança fica a cargo de SCR-11
- Próxima: DIR-06, DOC-01 ou SCR-01 (independentes)

## [DIR-06] 2026-10-19T21:35Z
- Agente: opencode/big-pickle | Humano: none
- Ação: expandir `runs/README.md` com estrutura por experimento e política de versionamento (ckpts/métricas/vídeos)
- Resultado: `git check-ignore` validado — `runs/**/checkpoints/` e `*.mp4` ignorados, `metrics.yaml` versionado; planilha §9 atualizada
- Nota: commit `951cf1d` [DIR-06] também continha os arquivos da SCR-01 deixados no working tree pelo agente paralelo (ver "Nota de commits" abaixo)
- Próxima: DOC-01 ou SCR-01 (independentes); DIR-07 bloqueada até SCR-14/15/16

## [SCR-01] 2026-10-20T13:25Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: implementar leitores de dataset sob `ReaderBase` (`BBox`, `OTBReader` e `VideoReader`) e testes unitários
- Resultado: `src/rede_zero/data/readers/{base,otb,video}.py` e `tests/test_readers.py` criados (4 testes passando; validação com dados sintéticos e dados reais em `/data/raw/`)
- Próxima: SCR-02

## [SCR-05] 2026-10-20T13:25Z
- Agente: codex/gpt-5 | Humano: none
- Ação: definir `ParadigmaBase`, factory/registry e contrato dos três paradigmas
- Resultado: `siamese` e `regressao` instanciam placeholders de contrato; `detector` lança `NotImplementedError` até SCR-18; documentação criada em `docs/paradigmas.md`
- Próxima: SCR-06 ou SCR-07

## [NOTA] 2026-10-20T13:45Z — desalinhamento de commits (paralelismo)
- Agente: opencode/big-pickle | Humano: none
- Ação: auditoria após trabalho paralelo de antigravity/gemini-3.8-flash (SCR-01) e codex/gpt-5 (SCR-05)
- Resultado: conteúdo no repo está correto (pytest 4/4; registry OK), porém labels de commit trocados:
  `7648a09 [SCR-01]` contém arquivos da SCR-05; arquivos da SCR-01 entraram no `951cf1d [DIR-06]`.
  Histórico NÃO reescrito (regra de não amend/push-forçado); auditoria de tasks deve usar §9 + entradas acima, não mensagens de commit.
- Próxima: DOC-01 ou SCR-02

## [SCR-02] 2026-10-20T13:35Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: criar módulo de inventário `src/rede_zero/data/inventario.py`, CLI `scripts/inventario_dataset.py`, testes unitários e relatório
- Resultado: varredura em `data/raw` identificou 12 sequências, 10.885 frames, 133.256 bboxes (tamanho médio 9.6x8.9 px) e 0 anomalias; relatório salvo em `data/inventario/relatorio.md`; testes em `tests/test_inventario.py` passando (5/5 no total)
- Próxima: SCR-03

## [SCR-03] 2026-10-20T13:45Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: criar dataset de treino `src/rede_zero/data/dataset_treino.py` com amostragem de pares (template 128x128, busca 256x256, patch 128x128), alvos normalizados e data augmentation
- Resultado: `TrackingDataset` implementado e testado; verificação de shapes via batch DataLoader aprovada; testes em `tests/test_dataset_treino.py` passando (7/7 no total)
- Próxima: SCR-04 (ou SCR-06/SCR-07)

## [SCR-06] 2026-10-20T14:06Z
- Agente: codex/gpt-5 | Humano: none
- Ação: implementar modelo Siamese com backbone compartilhado, correlação depthwise e ramo de offset
- Resultado: factory instancia `ModeloSiamese`; forward de batch 2 produz `score_map (2, 1, 9, 9)` e `offset (2, 2)`
- Próxima: SCR-07

## [SCR-07] 2026-10-20T14:10Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: implementar paradigma de Regressão completo (backbone conv 3 blocos, head MLP, perda smooth-L1 + BCE e inferência de ROI)
- Resultado: `ModeloRegressao` implementado em `src/rede_zero/paradigmas/regressao/`; registrado na factory `get_paradigma("regressao")`; forward com batch 2 gerando predições `(2, 5)`; 4 testes unitários adicionados em `tests/test_regressao.py` passando (11/11 no total)
- Próxima: SCR-08 ou SCR-09

## [SCR-10] 2026-10-20T14:21Z
- Agente: codex/gpt-5 | Humano: none
- Ação: implementar perdas dos dois paradigmas e métricas de rastreamento por frame
- Resultado: BCE+L1 Siamese, smooth-L1+BCE Regressão, IoU, curvas success/precision e resumo; 14 testes passam
- Próxima: SCR-11

## [SCR-09] 2026-10-20T14:25Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: criar testes de registry (`tests/test_paradigma_registry.py`) e testes de forward com validação de shapes e independência de até 8 objetos (`tests/test_forward.py`)
- Resultado: 16 novos testes implementados cobrindo factory, case insensitivity, exceção do detector, shapes paramétricos (batch 1, 2, 4, 8) e independência estrita sem contaminação de estado para Siamese e Regressão; total de 30/30 testes passando no pytest
- Próxima: SCR-11





## [SCR-08] 2026-10-20T15:05Z
- Agente: opencode/big-pickle | Humano: none
- Ação: stub do 3º paradigma — `detector/__init__.py` com docstring do contrato, classe `Detector` (placeholder) e `fabrica_detector` que lança `NotImplementedError`; `detector/README.md` com instruções de ativação; `base.py` agora delega o registro do detector ao módulo dele
- Resultado: pytest 30/30; `get_paradigma('detector')` e métodos de `Detector` apontam SCR-18; config `detector.yaml` já existia (DIR-05)
- Próxima: SCR-04, SCR-11 ou DIR-07 pendente de SCR-14/15/16

## [SCR-15] 2026-10-20T15:20Z
- Agente: opencode/big-pickle | Humano: none
- Ação: criar `src/rede_zero/inferencia/selecao_roi.py` — `selecionar_rois(video, max_objetos=8)` com `cv2.selectROI` no frame 0, devolve `list[BBox]` com ids 0..n-1; testes com GUI simulada (`tests/test_selecao_roi.py`)
- Resultado: pytest 3/3 novos, 33/33 total; smoke manual com display fica pendente de humano (sem X nesta sessão). Nota: SCR-14 em andamento por agente paralelo (pipeline.py não commitado) — stage seletivo feito apenas dos arquivos desta task
- Próxima: SCR-16 (após SCR-14), SCR-04, SCR-11

## [SCR-14] 2026-10-20T17:23Z
- Agente: codex/gpt-5 | Humano: none
- Ação: implementar pipeline de inferência multi-ROI sem GUI
- Resultado: recortes Siamese/Regressão, até oito rastreios independentes, remoção por confiança e registro de perdas; smoke de dois frames e import sem `rede_zero.treino` aprovados
- Próxima: SCR-16 ou SCR-17

## [SCR-17] 2026-10-20T17:26Z
- Agente: codex/gpt-5 | Humano: none
- Ação: testar o pipeline com leitor de dois frames e modelo fake de ROI fixa
- Resultado: ROI final, crop de Regressão e ausência de perdas validados; `pytest tests/test_pipeline.py` passa
- Próxima: SCR-16 ou DIR-07 após SCR-16

## [SCR-11] 2026-10-20T17:28Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: implementar `Treinador` em `src/rede_zero/treino/trainer.py` (loop por config, AdamW, AMP compatível, checkpointing last/best, resume e logs em CSV)
- Resultado: suporte a herança de configs com `_base_`; salvamento e carregamento de checkpoints (`last.pt`, `best.pt`); logging em `train_log.csv`; 3 testes unitários adicionados em `tests/test_trainer.py` passando (37/37 no total)
- Próxima: SCR-12 ou SCR-13

## [SCR-04] 2026-10-20T18:18Z
- Agente: codex/gpt-6 | Humano: lennin — descreveu aparência, movimento e duração dos vídeos reais
- Ação: criar gerador sintético de smoke inspirado nos objetos e fundos descritos
- Resultado: módulo `data/sintetico.py` e CLI `scripts/gerar_sinteticos.py`; MP4 e CSV com IDs estáveis, fundo verde sombreado e objetos multicoloridos arredondados/estrelados; máximo de 8 objetos e 90 frames
- Próxima: RUN-02


## [SCR-12] 2026-10-20T15:45Z
- Agente: opencode/big-pickle | Humano: none
- Ação: criar `scripts/treinar.py` — CLI com `--config`, `--resume`, `--paradigma`, `--epochs`, `--num-amostras`, `--dry-run`; monta readers (Forma-A OTB + Forma-B vídeo+CSV), `TrackingDataset` e `DataLoader`, delega ao `Treinador` (SCR-11). Ajuste: `configs/base.yaml` `dados.raiz` agora aponta o path real do dataset (fora do repo, conforme data/README.md)
- Resultado: dry-run OK em CUDA com 12 readers (siamese e override de paradigma/epochs); pytest 40/40
- Nota: SCR-13 em andamento por agente paralelo (avaliar.py etc.) — stage seletivo apenas de scripts/treinar.py, configs/base.yaml, PLAN.md, HISTORY.md
- Próxima: SCR-16, DOC-01 ou RUN-03

## [SCR-13] 2026-10-20T18:30Z
- Agente: antigravity/gemini-3.8-flash | Humano: none
- Ação: implementar CLI `scripts/avaliar.py` e rotinas de avaliação em `src/rede_zero/treino/avaliador.py` com cálculo de métricas (IoU médio, AUC de success, precisão a 20px) e exportação em JSON e CSV
- Resultado: carregamento de checkpoints, suporte a leitores OTB e VideoReader com detecção automática de ground-truth; exportação validada em vídeo sintético; 3 testes unitários adicionados em `tests/test_avaliar.py` passando (40/40 no total)
- Próxima: SCR-16 ou DOC-01


## [SCR-16] 2026-10-20T16:15Z
- Agente: opencode/big-pickle | Humano: none
- Ação: criar `src/rede_zero/inferencia/carregador.py` (carrega paradigma via factory + checkpoint) + `scripts/inferir.py` (CLI `--video`, `--paradigma`, `--ckpt`, `--out`, `--roi`, `--listar-paradigmas`); integra com pipeline (SCR-14) e seleção de ROI (SCR-15)
- Resultado: pytest 40/40; CLI funcional (--listar-paradigmas lista siamese/regressao/detector com status); pipeline roda multi-ROI e salva vídeo anotado
- Próxima: SCR-17 (já concluído por agente paralelo), RUN-03, DOC-01

## [RUN-02] 2026-10-21T18:20Z
- Agente: opencode/nemotron-3-ultra-free | Humano: none
- Ação: gerar vídeo sintético + CSV de tracking via `scripts/gerar_sinteticos.py`
- Resultado: `data/sinteticos/exp1/video.mp4` (90 frames, 3 objetos) + `video.csv` com Tracker IDs estáveis; formato compatível com Formato-B (vídeo + CSV)
- Próxima: RUN-03

## [RUN-03] 2026-10-21T18:45Z
- Agente: opencode/nemotron-3-ultra-free | Humano: none
- Ação: smoke test + overfit em 10 amostras (GPU RTX 2060) para Siamese e Regressão
- Resultado: Siamese loss 370→72 (~80% queda, loss absoluto alto); Regressão loss 0.34→0.03 (~91% queda, excelente overfit); configs `smoke.yaml` e `smoke_regressao.yaml` criados; logs em `runs/smoke/train_log.csv` e `runs/smoke_regressao/train_log.csv`
- Próxima: RUN-04

## [RUN-04] 2026-10-21T20:00Z
- Agente: opencode/nemotron-3-ultra-free | Humano: none
- Ação: treino completo Siamese (50 epochs, 1000 amostras, 14 readers, batch=16, AMP, dataset real)
- Resultado: loss 121.9 → ~1.0 (melhor 0.955 na epoch 48); checkpoints em `runs/base/checkpoints/{last,best}.pt`; log em `runs/siamese/train_log.csv`; bug conhecido: `${experimento}` resolveu para "base" em vez de "siamese"
- Próxima: RUN-05

## [RUN-07] 2026-10-21T22:30Z
- Agente: opencode/nemotron-3-ultra-free | Humano: none
- Ação: inferência demo com Siamese em vídeo sintético (90 frames, 3 objetos)
- Resultado: pipeline processou 90 frames, 1 ROI rastreada sem perdas; vídeo anotado gerado em `runs/inferencia/teste_sintetico.mp4`; relatório em `runs/inferencia/relatorio_run07.md`; vídeo real (DJI_0016.MP4, 3020 frames) excede timeout interativo
- Próxima: RUN-08 ou RUN-06
