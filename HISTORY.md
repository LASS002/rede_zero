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

## [DIR-04] 2026-10-19T20:30Z
- Agente: opencode/big-pickle | Humano: none (resposta pendente)
- Ação: layout `data/{real,inventario,sinteticos}` + README de 1 linha criados em DIR-02; schemas detalhados e integração do dataset real **pendentes**
- Resultado: 🔶 em andamento — perguntas sobre o dataset real enviadas ao humano (paths, formato das anotações, colunas, nº de objetos)
- Próxima: DIR-04 (retomar com respostas do humano)
