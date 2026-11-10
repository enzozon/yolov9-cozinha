# YOLOv9 Cozinha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar YOLOv9-t (3 folds) e YOLOv9-m (fold1, +2/3 se der) do zero no dataset de cozinha, avaliar no teste e gerar slides no estilo da 1ª entrega.

**Architecture:** Repo GitHub privado com scripts Python pequenos e um notebook Colab que os chama. Colab clona `WongKinYiu/yolov9`, aplica as correções do professor, monta os folds em `/content/data`, treina com `train_dual.py` salvando no Drive, avalia com `val_dual.py`. Resultados voltam ao repo como CSV/PNG; slides gerados com python-pptx a partir do pptx da 1ª entrega.

**Tech Stack:** Python 3, YOLOv9 (PyTorch), Google Colab (GPU), Google Drive, python-pptx, matplotlib, gh CLI.

**Spec:** `docs/superpowers/specs/2026-11-10-yolov9-cozinha-design.md`

## Global Constraints

- Classes (id 0..9): garfo, faca, colher, prato, espatula, tigela, frigideira, panela, xicara, taca.
- `--weights ''` (do zero), `data/hyps/hyp.scratch-high.yaml` sem alteração, `--img 640`, 100 épocas.
- Batch 16 (t), 8 (m). Runs no Drive: `/content/drive/MyDrive/yolov9_cozinha/runs`.
- Nomes de run: `t_fold{N}`, `m_fold{N}`.
- Sem pesos `.pt` no git. Repo privado `enzozon/yolov9-cozinha`.
- Slides em pt-BR, notas do apresentador em todo slide; folds = "3 holdouts repetidos", não "k-fold".

## Review Focus

- Rótulo do FOLD3 sem imagem no pool → excluído e listado, não quebra (teste `test_rotulo_sem_imagem_excluido`).
- Mesmo nome em Treino e Teste de um fold → erro alto (teste `test_vazamento_detectado`).
- class_id fora de 0..9 ou linha malformada → erro alto (teste `test_classe_invalida`).
- Imagem `.png`/`.jpeg` no pool (não só `.jpg`) → aceita (teste `test_extensoes_aceitas`).
- Colab cai no meio → retomada com `--resume` do `last.pt` (verificação manual, Task 3).

---

### Task 1: Repo + montagem dos folds

**Files:**
- Create: `scripts/montar_folds.py`, `tests/test_montar_folds.py`, `.gitignore`, `README.md`

**Interfaces:**
- Produces: `montar_fold(origem: Path, pool: dict[str, Path], destino: Path) -> dict` (retorna `{"contagem": {split: {classe: n}}, "excluidos": [str]}`); `indexar_pool(raizes: list[Path]) -> dict[str, Path]`; CLI `python scripts/montar_folds.py <raiz_grupo9> <destino>` que cria `destino/foldN/{train,val,test}` e `destino/foldN.yaml`.
- Splits mapeados: `Treino→train`, `Validacao→val`, `Teste→test`.

- [ ] Step 1: testes com tmp dirs: `test_monta_com_pool` (txt em FOLD3 sem jpg + jpg no pool → par copiado), `test_rotulo_sem_imagem_excluido`, `test_vazamento_detectado` (raises `ValueError`), `test_classe_invalida` (raises `ValueError`), `test_extensoes_aceitas`, `test_yaml_gerado` (`nc: 10`, names na ordem).
- [ ] Step 2: `python -m pytest tests -q` → FAIL.
- [ ] Step 3: implementar (stdlib + shutil; yaml escrito à mão).
- [ ] Step 4: `python -m pytest tests -q` → PASS.
- [ ] Step 5: `gh repo create enzozon/yolov9-cozinha --private`, commit, push.

### Task 2: Notebook Colab

**Files:**
- Create: `notebooks/treino_yolov9_cozinha.ipynb`, `colab/aplicar_fixes.py`, `colab/fixes/` (metrics.py, plots.py, augmentations.py, experimental.py)

**Interfaces:**
- Consumes: `scripts/montar_folds.py` CLI.
- Células: (1) montar Drive; (2) clonar yolov9 + repo do grupo (token via `gh` não vai pro notebook: repo clonado por cópia do Drive `MyDrive/yolov9_cozinha/code`); (3) `pip install -r requirements.txt`; (4) `aplicar_fixes.py`; (5) copiar `grupo 9` do Drive (atalho) para `/content/src` e rodar `montar_folds.py` → imprimir contagens/excluídos; (6) treino parametrizado `MODELO`, `FOLD`, `BATCH` com `--resume` se `last.pt` existir; (7) avaliação `val_dual.py --task test --conf 0.001 --iou 0.7`; (8) detect de exemplo/dia.

- [ ] Step 1: gerar `.ipynb` via script Python (nbformat-free JSON).
- [ ] Step 2: `python -c "import json;json.load(open(...))"` → OK.
- [ ] Step 3: commit + push; copiar `code/` pro Drive.

### Task 3: Treino no Colab

- [ ] Step 1: abrir notebook no Colab (conta enzozon7b), GPU, rodar células 1–5; conferir contagens (~806/101/100 por fold).
- [ ] Step 2: `t_fold1` — medir 1ª época, recalcular cronograma e GPU.
- [ ] Step 3: `t_fold2`, `t_fold3`, `m_fold1`, depois `m_fold2`, `m_fold3` se tempo/unidades.
- [ ] Step 4: em queda, reconectar e rodar a célula de treino (retoma do `last.pt`).

### Task 4: Avaliação e resumo

**Files:**
- Create: `scripts/resumo_resultados.py`, `tests/test_resumo.py`, `results/`

**Interfaces:**
- Produces: `ler_resultados(pasta_runs: Path) -> list[dict]` (run, modelo, fold, P, R, mAP50, mAP50_95, ms_img, melhor_conf_f1); `tabela_agregada(linhas) -> list[dict]` (média ± desvio por modelo); gráficos `results/*.png`.
- [ ] Step 1: teste `test_agregado` com 2 linhas fake → média/desvio exatos.
- [ ] Step 2: FAIL → implementar → PASS.
- [ ] Step 3: rodar val de cada run no Colab, baixar CSVs/PNGs (matriz, F1, PR, exemplos de detecção) para `results/`; commit.

### Task 5: Slides

**Files:**
- Create: `slides/gerar_slides.py`, `slides/Cozinha_yolov9.pptx`

- [ ] Step 1: python-pptx abre `grupo 9/Cozinha_dtset_pptx.pptx`, reaproveita layouts, escreve deck novo: título, roteiro, introdução/aplicações, classes, dataset e folds, YOLOv9 (GELAN/PGI) t × m vs outras técnicas, parâmetros/treino do zero, aumento de dados padrão, threshold, resultados por fold, comparação t × m, matriz de confusão/pares, exemplos, modelo indicado, dificuldades, conclusão. Notas em todos.
- [ ] Step 2: renderizar para PNG (LibreOffice se houver, senão conferir via leitura) e revisar.
- [ ] Step 3: commit + push; copiar pptx para o Drive do grupo.
