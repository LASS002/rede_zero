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
