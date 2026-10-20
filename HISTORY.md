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
