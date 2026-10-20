# Arquitetura da rede

## Siamese (SCR-06)

`ModeloSiamese` usa o mesmo `BackboneSiamese` para template RGB `128x128` e busca RGB
`256x256`. O backbone contém quatro blocos `Conv2d(3x3) → BatchNorm2d → ReLU → MaxPool2d`
com canais `32, 64, 128, 256`; portanto produz mapas `8x8` e `16x16`, respectivamente.

`CabecaSiamese` executa correlação depthwise entre esses mapas e soma os canais, gerando
logits `score_map` de forma `(B, 1, 9, 9)`. Em paralelo, médias espaciais dos dois mapas
alimentam um MLP que produz `offset` normalizado `(B, 2)`. Na inferência, um soft-argmax
do score map e esse offset atualizam o centro da ROI.

## Treinamento e avaliação (SCR-10)

`treino/losses.py` cria o rótulo Gaussiano do score map a partir de `alvo_siamese` e
combina BCE com L1 do offset. Para Regressão, combina smooth-L1 dos quatro deltas com BCE
da confiança. `treino/metricas.py` oferece IoU por frame, curva success, curva precision
por erro de centro e um resumo com IoU médio, AUC success e precisão a 20 pixels.
