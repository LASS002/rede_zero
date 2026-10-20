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




