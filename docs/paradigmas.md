# Contrato dos paradigmas

Todo paradigma é uma subclasse de `rede_zero.paradigmas.ParadigmaBase` e expõe:

- `nome` e `versao`, para identificação em checkpoints e logs;
- `forward(treino_batch)`, que recebe tensores já amostrados e retorna as saídas da rede;
- `perda_fn(saidas, alvos)`, que devolve um escalar `torch.Tensor` para retropropagação;
- `prever_roi(patch, contexto) -> BBox`, usado pela inferência em uma ROI por vez.

`BBox` usa pixels no formato `(x, y, w, h)`. O `contexto` contém os dados mantidos pelo
pipeline entre frames, como a ROI atual e, no Siamese, o template. A inferência processa
cada objeto de forma independente; o limite é oito ROIs sem sobreposição.

## Siamese

Nome de registry: `siamese`. Recebe template RGB `128x128` e região de busca RGB
`256x256`. A SCR-06 implementará backbone compartilhado, correlação depthwise, score map
e deslocamento do centro. A perda será BCE para o score map com rótulo Gaussiano mais L1
para o offset. O pipeline atualiza a ROI pelo soft-argmax do score map.

## Regressão

Nome de registry: `regressao`. Recebe patch RGB `128x128`, recortado da ROI atual
expandida em duas vezes. A SCR-07 produzirá `(dx, dy, sw, sh, confianca)`: os quatro
primeiros valores são normalizados e atualizam a ROI; confiança abaixo do limiar marca o
objeto como perdido. A perda combina smooth-L1 e BCE.

## Detector e associação

Nome de registry: `detector`. O contrato reserva a mesma interface para que a inferência
possa selecioná-lo por nome. Ele permanece indisponível: `get_paradigma("detector")`
lança `NotImplementedError` até a SCR-18 definir detector, associação e treinamento.

## Registry

`get_paradigma(nome, **kwargs)` cria um paradigma registrado e aceita nomes sem distinção
de maiúsculas. `listar_paradigmas()` retorna os nomes registrados em ordem estável.
Durante SCR-05, `siamese` e `regressao` são placeholders de contrato: podem ser
instanciados e identificados, mas `forward`, `perda_fn` e `prever_roi` só estarão
funcionais após SCR-06 e SCR-07, respectivamente.
