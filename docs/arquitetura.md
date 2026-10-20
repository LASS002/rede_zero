# Arquitetura da rede

## Siamese (SCR-06)

`ModeloSiamese` usa o mesmo `BackboneSiamese` para template RGB `128x128` e busca RGB
`256x256`. O backbone contém quatro blocos `Conv2d(3x3) → BatchNorm2d → ReLU → MaxPool2d`
com canais `32, 64, 128, 256`; portanto produz mapas `8x8` e `16x16`, respectivamente.

`CabecaSiamese` executa correlação depthwise entre esses mapas e soma os canais, gerando
logits `score_map` de forma `(B, 1, 9, 9)`. Em paralelo, médias espaciais dos dois mapas
alimentam um MLP que produz `offset` normalizado `(B, 2)`. Na inferência, um soft-argmax
do score map e esse offset atualizam o centro da ROI. A perda BCE+L1 será centralizada em
`treino/losses.py` na SCR-10.
