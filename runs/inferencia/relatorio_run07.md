# Relatório de Inferência — RUN-07 (Demo)

**Data:** 2026-10-21
**Paradigma:** Siamese
**Checkpoint:** `runs/base/checkpoints/best.pt` (epoch 48, loss 0.955)
**Vídeo de teste:** `data/sinteticos/exp1/video.mp4` (90 frames, 320x240, 3 objetos coloridos, fundo verde variável)
**ROI inicial:** `x=160, y=120, w=40, h=30` (objeto central no frame 0)
**Vídeo de saída:** `runs/inferencia/teste_sintetico.mp4` (90 frames, 6s)

## Resultados

- Pipeline executou com sucesso em todos os 90 frames
- 1 ROI rastreada do frame 0 ao frame 89
- Nenhum objeto perdido (confiança sempre > 0.5)
- Vídeo de saída gerado corretamente (verificado com ffprobe: 90 frames, 320x240)

## Observações

1. **Vídeo real (DJI_0016.MP4):** 1.1GB, ~3020 frames. Processamento completo excede timeout de 5min. Requer execução em background ou processamento em lotes.

2. **GUI de seleção de ROI:** Implementada em `src/rede_zero/inferencia/selecao_roi.py` (SCR-15). Usa `cv2.selectROI` no frame 0, permite até 8 objetos. Requer display gráfico (não funciona em ambiente headless).

3. **Modo headless:** Use `--roi "x,y,w,h"` (pode repetir até 8 vezes) para definir ROIs sem GUI.

## Próximos passos

- RUN-08: Teste multi-objeto (3–8 objetos) em vídeo sintético com ground-truth conhecido
- RUN-06: Avaliação comparativa Siamese vs Regressão no split de validação
- SCR-18: Implementar GUI para anotação de bboxes no frame inicial (detecção manual + tracking automático)