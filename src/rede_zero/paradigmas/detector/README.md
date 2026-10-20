# Paradigma C — Detector + associação

**Status: ⏸️ pendente (SCR-18)** · `configs/detector.yaml` já existe com `status: pendente`.

- `get_paradigma("detector")` lança `NotImplementedError` (mensagem aponta a SCR-18).
- `Detector` (classe em `__init__.py`) é o ponto de partida: implementar `forward`,
  `prever_roi` e `perda_fn` do contrato comum (`ParadigmaBase`).
- Registrar a versão final no factory `fabrica_detector` e remover o `NotImplementedError`.
- Referência: contrato completo em [docs/paradigmas.md](../../../docs/paradigmas.md);
  escopo na tarefa SCR-18 do `PLAN.md`.
