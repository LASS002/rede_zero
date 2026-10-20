# runs/ — experimentos de treino e inferência

Estrutura por experimento:

```
runs/
├── <experimento>/                # nome vem de configs/*.yaml (ex.: siamese)
│   ├── checkpoints/               # GITIGNORED (pesos .pt podem ser grandes)
│   │   ├── last.pt               # modelo+otimizador+epoch, para --resume
│   │   └── best.pt               # melhor IoU de valid
│   ├── metrics.yaml              # VERSIONADO — curvas/epocas finais
│   └── curves/                    # gráficos exportados (versionar se pequenos)
├── smoke/                         # RUN-03 (overfit tiny)
└── inferencia/                    # SCR-16 — vídeos de saída (gitignored) + relatórios
```

Política (DIR-06):

- **Versionar**: `metrics.yaml`, relatórios `.md`/`.json`, configs já estão em `configs/`.
- **Não versionar**: `checkpoints/*.pt`, vídeos (`*.mp4` já está no `.gitignore` raiz).
- Nunca editar `.gitignore` para liberar pesos no git — usar links/artefatos externos.
- Cada task `RUN-` deve ditar o nome da pasta de experimento em sua entrada no `HISTORY.md`.
