# Contrato dos paradigmas

Cada subclasse de `rede_zero.paradigmas.ParadigmaBase` expõe `nome`, `versao`,
`forward(treino_batch)`, `perda_fn(saidas, alvos)` e
`prever_roi(patch, contexto) -> BBox`. `BBox` usa pixels `(x, y, w, h)`; o contexto
mantém a ROI e demais dados entre frames. Cada ROI é independente, até oito sem
sobreposição.

## Siamese

Registry: `siamese`. Recebe template RGB `128x128` e busca RGB `256x256`. A SCR-06
implementará backbone compartilhado, correlação depthwise, score map e offset. A perda
será BCE com rótulo Gaussiano mais L1; o pipeline usará soft-argmax para atualizar a ROI.

## Regressão

Registry: `regressao`. Recebe patch RGB `128x128` da ROI expandida duas vezes. A SCR-07
produzirá `(dx, dy, sw, sh, confianca)` e usará smooth-L1 mais BCE; baixa confiança marca
o objeto como perdido.

## Detector e registry

Registry: `detector`. A interface já reserva esse terceiro paradigma, mas
`get_paradigma("detector")` lança `NotImplementedError` até a SCR-18. O factory aceita
nomes sem distinção de maiúsculas e `listar_paradigmas()` devolve os nomes registrados.
Em SCR-05, Siamese e Regressão são placeholders: só ficarão operacionais nas SCR-06/07.
