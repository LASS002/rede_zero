# Contrato dos paradigmas

Cada subclasse de `rede_zero.paradigmas.ParadigmaBase` expõe `nome`, `versao`,
`forward(treino_batch)`, `perda_fn(saidas, alvos)` e
`prever_roi(patch, contexto) -> BBox`. `BBox` usa pixels `(x, y, w, h)`; o contexto
mantém a ROI e demais dados entre frames. Cada ROI é independente, até oito sem
sobreposição.

## Siamese

Registry: `siamese`. Recebe template RGB `128x128` e busca RGB `256x256`. A SCR-06
implementa backbone compartilhado, correlação depthwise, score map e offset. A SCR-10
combina BCE com rótulo Gaussiano e L1; o modelo usa soft-argmax para atualizar a ROI.

## Regressão

Registry: `regressao`. Recebe patch RGB `128x128` da ROI expandida duas vezes. A SCR-07
produz `(dx, dy, sw, sh, confianca)` e a SCR-10 usa smooth-L1 mais BCE; baixa confiança
marca o objeto como perdido.

## Detector e registry

Registry: `detector`. A interface já reserva esse terceiro paradigma, mas
`get_paradigma("detector")` lança `NotImplementedError` até a SCR-18. O factory aceita
nomes sem distinção de maiúsculas e `listar_paradigmas()` devolve os nomes registrados.
Siamese já está operacional; Regressão permanece placeholder até a SCR-07.
