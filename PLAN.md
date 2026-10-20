# PLAN.md — Rede neural de rastreamento de objetos (rede_zero)

> Plano mestre para agentes de IA. Cada task é pequena, modular e retomável:
> um agente pode pausar em qualquer ponto e outro continua exatamente de onde parou.
> Todo o conteúdo é em PT-BR (decisão registrada nas premissas).

---

## 1. Premissas (decididas com o humano — 2026-10-19)

1. **Frameworks**: PyTorch como padrão (paradigmas, treino, inferência). NumPy puro é
   permitido em tasks educacionais/validação ("do zero", SCR-19). TensorFlow/Keras fica
   disponível como alternativa caso algum estágio justifique (sem uso planejado hoje).
2. **Paradigmas de rastreamento**: o usuário escolhe o paradigma **na inferência**
   (`--paradigma`). Fase 1: **Siamese** e **Regressão** prontos e treinados;
   **Detector + associação** = 3º paradigma plugável (tasks ⏸️ SCR-18/RUN-09).
3. **Dados**: dataset real do usuário em `/home/labiocom01/Documents/yolo_cfc_retrain/data/raw/`.
   Formato-A = frames + labels YOLO/txt + JSON por frame (sem id de objeto);
   Formato-B = vídeo + CSV de tracking **com Tracker ID** (detalhes em `data/README.md`,
   revisados em DIR-04). Dataset público (GOT-10k/LaSOT) = task futura opcional.
4. **Infra**: GPU local NVIDIA (RTX 2060 confirmada); repositório já ligado ao GitHub
   (`LASS002/rede_zero`), push contínuo após cada task.
5. **Gerenciador de pacotes**: `uv` (decidido pelo humano em 2026-10-19 durante DIR-03).
   Dependências em `pyproject.toml`; lockfile `uv.lock` versionado; comandos via `uv run`.

## 2. Convenções

### 2.1 Categorias (para o humano distinguir o tipo de trabalho)

| Prefixo | Categoria |
|---|---|
| `DIR-` | Estruturação de diretórios |
| `SCR-` | Escrita de scripts / código |
| `RUN-` | Execução de treino, inferência ou avaliação |
| `DOC-` | Documentação (README/plano) |

### 2.2 Nível de tokens (modelo necessário)

| Marca | Significado | Modelo sugerido |
|---|---|---|
| `T1` | Tarefa mecânica (mkdir, stubs, CLI trivial, commit) | econômico / rápido |
| `T2` | Implementação com spec pronta (seguir este plano) | intermediário |
| **`T3`** | **Raciocínio de alto nível**: arquitetura, perda, diagnóstico de convergência, análise comparativa | **modelo potente** |

**Tasks T3 (alocar modelo potente):** SCR-05, SCR-06, SCR-07, SCR-10, SCR-18, SCR-19,
RUN-03*, RUN-06, RUN-09. (*RUN-03 só vira T3 se o smoke test falhar e exigir diagnóstico.)

### 2.3 Status das tasks

`⬜` pendente · `🔶` em andamento · `✅` concluída · `⏸️` adiada/futura

### 2.4 Regras de granularidade e fechamento

- **1 task = 1 entregável**, ≲ 150 linhas novas. Se crescer, o agente deve parar,
  deixar `🔶` e registrar em `HISTORY.md` a subdivisão proposta (ex.: `SCR-06a/b`).
- **Toda task fecha com o checklist** (nesta ordem):
  1. Marcar `✅` na task em `PLAN.md` com data (ISO) e identificação do agente;
  2. Gravar entrada no `HISTORY.md` (template em §3);
  3. Atualizar `README.md` se características da rede/estrutura/execução mudaram;
  4. `git add -A && git commit -m "[ID] resumo curto" && git push`.
- **Proibido** fazer trabalho de outra task dentro da sessão (ex.: não treinar dentro de
  uma task `DIR`).

## 3. Protocolo de retomada (token-eficiente)

1. Ler **este arquivo** e localizar a primeira task `⬜`/`⏸️` (ou `🔶`) cujas dependências
   estejam `✅`.
2. Ler as **últimas 2 entradas** de `HISTORY.md` + apenas as seções do `README.md`
   apontadas na task. **Não** varrer o repo inteiro.
3. Executar somente os passos da task.
4. Fechar com o checklist da §2.4 e parar.

**Template de entrada no `HISTORY.md`:**

```markdown
## [SCR-06] 2026-10-19T14:05Z
- Agente: opencode/<model-id> | Humano: none
- Ação: implementar backbone+head do paradigma Siamese
- Resultado: commits abc1234, def5678
- Próxima: SCR-07
```

(Se houver intervenção humana: `Humano: <nome/intervenção — ex.: forneceu path do dataset>`.)

## 4. Árvore de diretórios alvo

Separação obrigatória: **a rede neural vive em `paradigmas/`**; **inferência é script
separado** (`scripts/inferir.py` + `inferencia/`) que importa a rede, mas **nunca**
importa `treino/`.

```
rede_zero/
├── PLAN.md · README.md · HISTORY.md
├── docs/
│   ├── historico_formato.md        # formato do HISTORY (DIR-01)
│   ├── arquitetura.md              # detalhes técnicos (SCR-06/07)
│   └── paradigmas.md               # contrato dos 3 paradigmas (SCR-05)
├── configs/
│   ├── base.yaml
│   ├── siamese.yaml
│   ├── regressao.yaml
│   └── detector.yaml               # placeholder do 3º paradigma
├── src/rede_zero/
│   ├── __init__.py
│   ├── paradigmas/                 # ← SÓ A REDE NEURAL
│   │   ├── base.py                 # interface + factory (SCR-05)
│   │   ├── siamese/{__init__,backbone,head,modelo}.py
│   │   ├── regressao/{__init__,modelo}.py
│   │   └── detector/{__init__,README.md}   # stub (SCR-08)
│   ├── data/
│   │   ├── readers/{__init__,base,otb,video_csv,video_json}.py
│   │   ├── dataset_treino.py
│   │   ├── inventario.py
│   │   └── sintetico.py
│   ├── treino/                     # nunca importado por inferencia/
│   │   ├── losses.py · metricas.py · trainer.py
│   ├── inferencia/
│   │   ├── pipeline.py             # loop de rastreamento, sem GUI (SCR-14)
│   │   ├── selecao_roi.py          # GUI OpenCV (SCR-15)
│   │   └── carregador.py           # ckpt + factory de paradigma (SCR-16)
│   └── utils/{seed,log,io}.py
├── scripts/
│   ├── inventario_dataset.py · gerar_sinteticos.py
│   ├── treinar.py · avaliar.py
│   ├── inferir.py                  # --paradigma siamese|regressao|detector
│   └── listar_paradigmas.py
├── data/                           # brutos — gitignore; só README de schema
│   ├── real/ (formato-a/ e formato-b/)
│   ├── inventario/
│   └── sinteticos/
├── runs/                           # ckpts gitignore; logs/métricas versionados
│   └── <experimento>/{checkpoints/,metrics.yaml,curves/}
├── tests/
│   ├── test_paradigma_registry.py · test_forward.py
│   ├── test_readers.py · test_pipeline.py
├── .gitignore · requirements.txt · pyproject.toml
```

## 5. Arquitetura da rede (spec — detalhada nas tasks T3)

**Comum aos dois paradigmas**: até **8 objetos**, rastreados por ROIs **independentemente**
(sem sobreposição → sem NMS, sem pós-processamento conjunto); imagens em RGB,
redimensionadas preservando proporção; pesos **iniciados do zero** (sem pré-treino).

### Paradigma A — Siamese (template matching) — SCR-06

- **Entrada**: par (template 128×128 colhido na ROI do frame 0 ou atualizado periodicamente,
  região de busca 256×256 centrada na ROI atual).
- **Backbone** (compartilhado p/ template e busca): 4 blocos `Conv→BN→ReLU→MaxPool`,
  canais `[32, 64, 128, 256]`.
- **Head**: correlação em profundidade (depthwise) template×busca → score map;
  ramo paralelo de deslocamento (offset do centro).
- **Perda**: BCE no score map (centro-positivo com Gaussian label) + L1 no offset.
- **Inferência**: soft-argmax sobre o score map → novo centro da ROI.

### Paradigma B — Regressão de deslocamento — SCR-07

- **Entrada**: patch 128×128 = ROI atual expandida ×2.
- **Backbone**: 3 blocos conv `[32, 64, 128]`.
- **Head**: MLP → `(dx, dy, sw, sh)` normalizados + score de confiança.
- **Perda**: smooth-L1 nas 4 saídas + BCE na confiança.
- **Inferência**: aplica Δ à ROI; se confiança < limiar → objeto perdido (reportado).

### Paradigma C — Detector + associação — ⏸️ SCR-18

Contrato definido em SCR-05/08; implementação e treino adiados.

### Modelo de execução (espelhado no README)

- `scripts/treinar.py` e `scripts/avaliar.py` importam `rede_zero.treino` + `rede_zero.paradigmas`.
- `scripts/inferir.py` importa `rede_zero.inferencia` + `rede_zero.paradigmas`.
- `rede_zero.inferencia` **nunca** importa `rede_zero.treino`.

## 6. Tasks

> **Colunas**: `ID` = identificador estável · **`Nível de tokens`** = consumo/modelo
> necessário (T1/T2/T3, ver §2.2) · **`Depende de`** = IDs que precisam estar `✅`
> antes de iniciar · `Entregável` = resultado concreto ao final.

### 6.1 Fase DIR — Estruturação (treino e inferência)

| ID | Nível de tokens | Depende de | Entregável |
|---|---|---|---|
| DIR-01 | T1 | — | `HISTORY.md` inicial + `docs/historico_formato.md` + `.gitignore` |
| DIR-02 | T1 | DIR-01 | esqueleto de pastas + `__init__.py` + READMEs de 1 linha |
| DIR-03 | T1 | DIR-02 | `pyproject.toml` + `uv.lock` (gerenciador: **uv**) |
| DIR-04 | T1 | DIR-02 | layout de `data/` + `data/README.md` com os 2 schemas |
| DIR-05 | T1 | DIR-02 | `configs/base.yaml` + yamls dos 3 paradigmas |
| DIR-06 | T1 | DIR-02 | política de `runs/` (ckpts fora do git, logs versionados) |
| DIR-07 | T1 | SCR-14, SCR-15, SCR-16 | revisão da árvore pós-inferência (task de retorno à estrutura) |

**DIR-01** — *Criar histórico e regras de versionamento.*
Passos: (1) criar `HISTORY.md` com 1ª entrada `[DIR-01]`; (2) `docs/historico_formato.md`
com o template da §3; (3) `.gitignore` com `data/real/`, `runs/*/checkpoints/`,
`__pycache__/`, `*.pt`, `*.mp4`. Verificação: `git status` limpo após commit.

**DIR-02** — *Criar esqueleto.*
Passos: `mkdir -p` da árvore da §4 (sem arquivos de lógica); tocar `__init__.py` vazios;
README de 1 linha em `docs/`, `data/`, `runs/`. Verificação: `python -c "import rede_zero"`
com `PYTHONPATH=src` (ou configuração equivalente no pyproject).

**DIR-03** — *Dependências (uv).* Ajustado em 2026-10-19: sem `requirements.txt`;
usar `pyproject.toml` (PEP 621) + `[dependency-groups] dev` + `uv.lock`.
Passos: `torch`, `torchvision`, `opencv-python`, `numpy`, `pyyaml`; dev: `pytest`.
Verificação: `uv sync` +
`uv run python -c "import torch; print(torch.cuda.is_available())"`.

**DIR-04** — *Layout de dados.*
Passos: pastas `data/real/formato-a/`, `data/real/formato-b/`, `data/inventario/`,
`data/sinteticos/`; `data/README.md` descrevendo: formato-a = `seq/frames/*.jpg` +
`seq/gt.txt` (`frame,x,y,w,h,id`); formato-b = `video.mp4` + `video.csv`/`.json` com
mesmas colunas. Verificação: árvore existe; git não versiona binários.

**DIR-05** — *Configs.*
Passos: `base.yaml` (seed, device, epochs, lr, batch, image sizes 128/256);
`siamese.yaml` e `regressao.yaml` herdam de base; `detector.yaml` com
`status: pendente`. Verificação: `python -c "import yaml; yaml.safe_load(open('configs/base.yaml'))"`.

**DIR-06** — *Política de runs.*
Passos: `runs/README.md` explicando pastas por experimento; `.gitignore` cobre
`runs/**/checkpoints/` e versiona `metrics.yaml`. Verificação: `git check-ignore -v runs/exp1/checkpoints/model.pt` (deve ignorar).

**DIR-07** — *Retorno à estrutura.*
Passos: (1) ler `HISTORY.md` e notar dores registradas nas tasks SCR-14…16;
(2) mover/renomear apenas o que a inferência exigiu; (3) atualizar árvore no
`README.md`. Verificação: `pytest -q` continua verde. Fechamento padrão.

### 6.2 Fase SCR — Escrita de scripts/código

| ID | Nível de tokens | Depende de | Entregável |
|---|---|---|---|
| SCR-01 | T2 | DIR-04 | readers dos 2 formatos sob `ReaderBase` |
| SCR-02 | T2 | SCR-01 | `scripts/inventario_dataset.py` (relatório do dataset) |
| SCR-03 | T2 | SCR-01 | `data/dataset_treino.py` (amostragem p/ ambos paradigmas + augment) |
| SCR-04 | T1 | DIR-04 | gerador de vídeos sintéticos (smoke tests) |
| SCR-05 | **T3** | DIR-05 | interface + factory de paradigmas (`base.py` + `docs/paradigmas.md`) |
| SCR-06 | **T3** | SCR-05 | modelo Siamese completo |
| SCR-07 | **T3** | SCR-05 | modelo de Regressão completo |
| SCR-08 | T1 | SCR-05 | stub `paradigmas/detector/` (contrato + `NotImplementedError`) |
| SCR-09 | T2 | SCR-06, SCR-07 | testes de registry/forward (8 objetos independentes) |
| SCR-10 | **T3** | SCR-06, SCR-07 | losses + métricas (IoU, success/precision) |
| SCR-11 | T2 | SCR-10 | trainer (loop, ckpt, resume, logs CSV) |
| SCR-12 | T1 | SCR-11 | CLI `scripts/treinar.py` (`--config --resume --paradigma`) |
| SCR-13 | T2 | SCR-11 | CLI `scripts/avaliar.py` (métricas JSON/CSV em `runs/`) |
| SCR-14 | T2 | SCR-05 | `inferencia/pipeline.py` (loop multi-ROI, sem GUI, sem import de `treino`) |
| SCR-15 | T1 | DIR-02 | `inferencia/selecao_roi.py` (GUI OpenCV, até 8 objetos) |
| SCR-16 | T2 | SCR-14, SCR-15 | `inferencia/carregador.py` + CLI `scripts/inferir.py` com `--paradigma` |
| SCR-17 | T2 | SCR-14 | `tests/test_pipeline.py` (pipeline c/ modelo fake, 2 frames) |
| SCR-18 | **T3** ⏸️ | SCR-08 | implementação do Detector + associação |
| SCR-19 | **T3** ⏸️ opc. | SCR-06 | validação NumPy-puro do forward/backward do bloco conv |

**SCR-01** — *Leitores de dataset.*
Contrato `ReaderBase`: `n_frames`, `get_frame(i) -> np.ndarray`, `get_gt(i) -> list[BBox]`.
Implementar `OTBReader` (imagens+gt.txt) e `VideoReader` (.mp4 + csv/json). ~120 linhas.
Verificação: `pytest tests/test_readers.py` (sintético mínimo de 2 frames).

**SCR-02** — *Inventário.*
Passos: varrer `data/real/`; relatório `data/inventario/relatorio.md` (nº seqüências,
frames, objetos, tamanho médio de bbox, inconsistências). Verificação: executar em dados
de teste sintéticos criados no próprio teste.

**SCR-03** — *Dataset de treino.*
Passos: amostrador de pares (template, busca, Δbbox) alinhado ao contrato de SCR-05;
augment leve (jitter de escala/pos, flip, ruído). Verificação: `python -c` imprimindo
shapes de 1 batch.

**SCR-04** — *Sintéticos (opcional, smoke).*
Passos: formas em movimento (1–8 objetos, sem sobreposição), gravar vídeo + gt. ~80 linhas.

**SCR-05** *(T3)* — *Interface/factory de paradigmas.*
Passos: definir `ParadigmaBase` (obrigatórios: `forward(treino_batch)`, `prever_roi(patch, contexto) -> BBox`,
`perda_fn`, `nome`, `versão`); registry `get_paradigma(nome)`; `listar_paradigmas()`;
contrato do 3º paradigma em `docs/paradigmas.md`. Verificação: registry instancia
siamese e regressao, detector lança `NotImplementedError` registrado.

**SCR-06** *(T3)* — *Paradigma Siamese.*
Passos: `backbone.py` (4 blocos), `head.py` (correlação depthwise + offset),
`modelo.py` montando o par; shapes conforme §5. Verificação:
`python -c` forward com batch 2 e impressão dos shapes do score map.

**SCR-07** *(T3)* — *Paradigma Regressão.*
Passos: backbone 3 blocos + MLP head; saída `(dx,dy,sw,sh)` + conf. Verificação: forward
com batch 2 → shape final `(B, 5)`.

**SCR-08** — *Stub do detector.*
Passos: `paradigmas/detector/` com docstring do contrato + `NotImplementedError`;
`configs/detector.yaml` já existente. Verificação: `pytest tests/test_paradigma_registry.py`.

**SCR-09** — *Testes da rede.*
Passos: `test_paradigma_registry.py` (nomes, erro do detector); `test_forward.py`
(shapes; 8 modelos/ROI independentes sem estado compartilhado). Verificação: `pytest -q`.

**SCR-10** *(T3)* — *Perdas e métricas.*
Passos: `losses.py` (BCE+L1 siamese; smooth-L1+BCE regressão); `metricas.py`
(IoU por frame, curva success/precision simplificada). Verificação: `pytest` com
casos de IoU conhecidos.

**SCR-11** — *Trainer.*
Passos: loop por config; AdamW; checkpoint (modelo+otimizador+epoch); `--resume`;
log em CSV em `runs/<exp>/`. Verificação: 1 epoch em dados sintéticos salva ckpt e log.

**SCR-12** — *CLI treinar.* Passos: argparse (`--config`, `--resume`, `--paradigma`);
delega ao trainer. Verificação: `python scripts/treinar.py --config configs/base.yaml --dry-run`.

**SCR-13** — *CLI avaliar.* Passos: carrega ckpt, roda pipeline de avaliação sobre
valid, grava `runs/<exp>/metricas.json`. Verificação: execução em 1 vídeo sintético.

**SCR-14** — *Pipeline de inferência (sem GUI).*
Passos: recebe `reader`, lista de ROIs iniciais, `ParadigmaBase`; loop frame-a-frame:
recorta busca/patch → `prever_roi` → atualiza ROI; ≤8 objetos independentes; registra
perda de objeto. **Sem import de `rede_zero.treino`.** Verificação: greps
`python -c "import rede_zero.inferencia"` + teste manual de import cycle.

**SCR-15** — *Seleção de ROI (GUI).* Passos: `cv2.selectROI` repetido (até 8, tecla
Enter/fim), devolve `list[BBox]`. Verificação: import + smoke manual (humano).

**SCR-16** — *CLI inferir.*
Passos: `scripts/inferir.py --video X --paradigma siamese|regressao --ckpt Y --out Z`;
opcional `--roi x,y,w,h` (sem GUI, para runs automatizados); GUI é default interativo;
carregador monta o paradigma via factory e carrega o ckpt. Verificação:
`--listar-paradigmas` imprime os disponíveis e marca detector como pendente.

**SCR-17** — *Testes do pipeline.* Passos: modelo fake que devolve ROI fixa; pipeline em
2 frames; asserção de ROIs finais. Verificação: `pytest tests/test_pipeline.py`.

**SCR-18** ⏸️ *(T3)* — *Detector + associação* (futura; escopo a detalhar na ativação).

**SCR-19** ⏸️ *(T3, opcional)* — *NumPy-puro "do zero"*: reimplementar forward/backward
do bloco conv e comparar gradientes com autograd (tolerância 1e-5).

### 6.3 Fase RUN — Execução (treino / avaliação / inferência)

| ID | Nível de tokens | Depende de | Entregável |
|---|---|---|---|
| RUN-01 | T1 | SCR-02 | inventário do dataset real (**humano fornece o path**) |
| RUN-02 | T1 | SCR-04 | vídeos sintéticos gerados em `data/sinteticos/` |
| RUN-03 | T2 (T3 se falhar) | SCR-09, SCR-12 | smoke + overfit tiny em GPU |
| RUN-04 | T2 | RUN-01, RUN-03 | treino completo Siamese (GPU local) |
| RUN-05 | T2 | RUN-01, RUN-03 | treino completo Regressão (GPU local) |
| RUN-06 | **T3** | RUN-04, RUN-05 | avaliação comparativa → baseline escolhida no README |
| RUN-07 | T1 | SCR-16, RUN-04 ou 05 | inferência demo c/ ROI interativo (vídeo de saída em `runs/`) |
| RUN-08 | T2 | RUN-06, SCR-16 | teste com até 8 objetos + relatório |
| RUN-09 | **T3** ⏸️ | SCR-18 | treino do detector |

**RUN-01** — *Inventário real.* Passos: humano informa o path do dataset real;
rodar `scripts/inventario_dataset.py`; anomalias → `HISTORY.md`. Verificação: relatório gerado.

**RUN-03** — *Smoke.* Passos: 1 batch; overfit em 10 amostras (loss deve cair >80%);
se não cair → marcar `🔶`, diagnosticar (T3). Verificação: curva em `runs/smoke/metrics.yaml`.

**RUN-04/05** — *Treinos completos.* Passos: lançar treino em background na GPU;
o agente só **agenda e monitora** (baixo token); checkpoints em `runs/<exp>/checkpoints/`.
Verificação: `metrics.yaml` com epochs completas.

**RUN-06** *(T3)* — *Comparação.* Passos: rodar `avaliar.py` nos dois modelos no split
de valid; registrar success/precision médios; escolher baseline; documentar no
`README.md` (seção Estado atual) e `docs/arquitetura.md`. Verificação: tabelas no README.

**RUN-07** — *Demo.* Passos: rodar `scripts/inferir.py` em vídeo real; salvar vídeo
anotado; commit apenas do caminho/relatório (vídeo fora do git).

**RUN-08** — *Multiobjeto.* Passos: vídeo com 3–8 objetos; relatório de sucesso por objeto.

### 6.4 Fase DOC — Documentação (README)

| ID | Nível de tokens | Depende de | Entregável |
|---|---|---|---|
| DOC-01 | T1 | DIR-02 | seções do README: visão geral, árvore, modelo de execução, arquitetura, dados, como treinar/inferir, estado atual, ponteiro p/ HISTORY |

> DOC-01 cria a base; as demais tasks **mantêm** o README atualizado no próprio
> fechamento (checklist §2.4).

## 7. Ordem sugerida (fluxo, não é obrigatória — as tasks são independentes)

```
DIR-01 → DIR-02 → DIR-03/04/05/06 (paralelizáveis)
      ↘ SCR-05 → SCR-06 e SCR-07 (T3) → SCR-09 → SCR-10 → SCR-11 → SCR-12/13
      ↘ SCR-01 → SCR-02/03, SCR-04
      ↘ SCR-14 → SCR-15 → SCR-16 → SCR-17 → DIR-07
RUN-01/02 → RUN-03 → RUN-04/05 → RUN-06 (T3) → RUN-07/08
⏸️ SCR-18/RUN-09/SCR-19 quando ativados
```

## 8. Como usar este plano (para humanos e agentes)

- **Humano**: acompanhe o `README.md` (estado da rede) e o `HISTORY.md` (log). Forneça
  o **path do dataset real** quando RUN-01 for iniciada (interpelação esperada).
- **Agente**: siga §3 (retomada). Nunca invente tasks fora deste arquivo; se surgir
  algo novo, registre em `HISTORY.md` e proponha task nova ao humano antes de executar.
- **Versionamento**: um commit por task, mensagem `[ID] …`; push após cada commit.

## 9. Quadro de status

| ID | Status | Data (UTC) | Agente | Obs |
|---|---|---|---|---|
| DIR-01 | ✅ | 2026-10-19 | opencode/big-pickle | HISTORY.md, formato e .gitignore criados |
| DIR-02 | ✅ | 2026-10-19 | opencode/big-pickle | esqueleto `src/`, `scripts/`, `configs/`, `data/`, `runs/`, `tests/`, `docs/` |
| DIR-03 | ✅ | 2026-10-19 | opencode/big-pickle | uv; torch 2.14.1+cu130, CUDA OK (RTX 2060) |
| DIR-04 | ✅ | 2026-10-19 | opencode/big-pickle | schemas reais documentados em `data/README.md`; dataset fora do repo |
| DIR-05 | ✅ | 2026-10-19 | opencode/big-pickle | `configs/{base,siamese,regressao,detector}.yaml` com herança `_base_`; detector `status: pendente` |
| SCR-05 | ✅ | 2026-10-20 | codex/gpt-5 | contrato `ParadigmaBase`, registry e documentação dos três paradigmas |
| DOC-01 | ⬜ | — | — | README base pendente (estrutura mínima já no README) |
| (demais) | ⬜ | — | — | ver §6 |
