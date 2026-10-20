# data/ — dados brutos do projeto (não versionados)

Dataset real fora do repo: `/home/labiocom01/Documents/yolo_cfc_retrain/data/raw/`
(este diretório `data/` aqui guarda apenas READMEs, inventários e dados sintéticos futuros).

## Formatos descobertos (DIR-04, 2026-10-19 — inspeção do dataset real)

### Formato-A — Frames + labels (estilo YOLO / OTB adaptado)

Presente em: `dataset_20260918_112727/` e `dataset_20260918_160136/`

```
<seq>/
├── train|val|test/
│   ├── images/frame_XXXXXX.jpg
│   └── labels/frame_XXXXXX.txt   # YOLO: class cx cy w h (normalizados 0–1)
│                        .json    # {image, width, height, annotations:[{x,y,w,h,label}]} px
├── classes.txt                   # "object" (classe única)
├── data.yaml                     # (160136) config YOLO
└── part1_labeling_project.json   # (112727) projeto: video_source, total_frames,
                                  # bboxes{frame_index:[{x,y,w,h,label}]} px — sem id de objeto
```

- Fonte do vídeo: `part1.mp4` (1511 frames em 112727).
- **Sem id de objeto** nas anotações de detecção (várias bboxes/frame, label "object").
- Frames ≈ 1280×720; ~20+ objetos por frame (>8 do plano → limitar a 8 na amostragem,
  tarefa de SCR-03/RUN-08).

### Formato-B — Vídeo + CSV de tracking (com Tracker ID)

Presente em: `part1/vailatracker_20261002_153148/part1/`

```
part1/
├── part1.mp4                      # vídeo de origem (há também processed_*.mp4)
├── object_id_<N>.csv              # Frame,Tracker ID,Label,X_min,Y_min,X_max,Y_max,
│                                  # Confidence,Color_R,G,B  (bbox em px, id por objeto)
├── all_id_pose.csv                # Frame,Tracker_ID,Label + keypoints de pose
├── part1_id_<NN>_pose.csv         # pose de um único tracker id
```

- Cada `object_id_<N>.csv` = trajetória de **um objeto** (Tracker ID fixo, 1 linha/frame).
- Amostragem original do usuário: **10 frames a cada 60** do vídeo.
- Vídeos na raiz: `DJI_0019_720p.mp4`, `part2.mp4` (sem anotação mapeada ainda).

## Decisões de integração

- Não copiar dados brutos para o repo (gitignored); readers (SCR-01) receberão o path absoluto.
- `data/inventario/` receberá o relatório de SCR-02/RUN-01.
- Formato-JSON do projeto de label também será coberto pelo reader (colunas em px).
