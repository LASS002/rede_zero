Esse é um repositório experimental para criar uma rede neural do zero. Exatamente, do zero!

A rede consiste em um rastreador de objetos, simples, testando algumas arquiteturas e rastreando um até 8 abjetos, sem sobreposição.

A ideia é que para a inferencia seja aberto um frame do vídeo, a pessoa selecione um roi que enquadra o objeto e esse roi se desloque, acompanhando o objeto, até o fim do video.

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
- Paradigmas: o registry `rede_zero.paradigmas.get_paradigma()` já define os contratos
  `siamese`, `regressao` e `detector`; Siamese já possui backbone, correlação e offset,
  Regressão já possui backbone e MLP, e o detector permanece pendente. Perdas e métricas
  de avaliação vivem em `treino/`. Consulte `docs/paradigmas.md` e `docs/arquitetura.md`.
