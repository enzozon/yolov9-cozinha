# Design — Detecção de utensílios de cozinha com YOLOv9 (2ª entrega, Grupo 9)

Data: 2026-11-10 · Apresentação: quinta 2026-11-12, 07:15 · Turma CC7/CC8-M, Prof. Jairo Moraes (UVV)

## Intenção

Treinar detectores YOLOv9 sobre o dataset próprio do grupo (10 utensílios de cozinha, bounding box, 3 folds 80/10/10), comparar dois tamanhos de modelo e apresentar os resultados em slides no mesmo estilo da 1ª entrega. Nota da 2ª entrega: modelo (7.0) + apresentação (3.0). O professor também avalia o modelo com um dataset novo entregue no dia — o detector precisa estar pronto para rodar em imagens novas.

## Objetivos

1. Treinar **YOLOv9-t** e **YOLOv9-m do zero** (`--weights ''`), como nas aulas.
2. Treinar nos 3 folds; avaliar cada run no **Teste** do próprio fold.
3. Slides (.pptx, estilo do `Cozinha_dtset_pptx.pptx`) cobrindo:
   - introdução, aplicações práticas, as 10 classes;
   - os dois modelos (arquitetura, diferença t × m, comparação com outras técnicas);
   - threshold usado e por quê;
   - resultados e diferenças entre modelos e entre folds (P, R, mAP50, mAP50-95, matriz de confusão, curvas);
   - modelo mais indicado;
   - aumento de dados: usamos o padrão do YOLOv9 (sem rotação; com espelhamento horizontal, mosaico etc.) — dizer explicitamente;
   - principais parâmetros, dificuldades e como foram contornadas;
   - exemplos de detecção em imagens de teste.
4. Notas do apresentador em cada slide (feedback do professor: não ler slides/colinha — notas servem para estudar antes).
5. Tudo em repositório GitHub **privado** na conta `enzozon`.

## Não-objetivos

- Artigo (não pedido nesta entrega).
- Pesos pré-treinados COCO (decisão do grupo: do zero).
- Experimentos de aumento de dados (rotação etc.) — usa só o padrão.
- Segmentação.

## Anti-objetivos (falha mesmo se “funcionar”)

- Vazamento entre treino e teste de um fold (avaliar no Teste com imagem que estava no Treino).
- FOLD3 montado com imagem errada para o rótulo.
- Números nos slides que não vêm dos runs reais.
- Perder runs por queda do Colab (tudo salvo no Drive).

## Restrições

- Sem GPU local. Treino no **Google Colab** (conta `enzozon7b@gmail.com`, plano Google AI Plus: 50 unidades de computação/mês). Claude opera o Colab pelo Chrome; usuário ausente a partir de 18:00 de 2026-11-10, disponível via chat remoto.
- Prazo: treinos até quarta 2026-11-11 à noite; slides prontos quarta à tarde/noite.
- Base de código: repositório oficial `WongKinYiu/yolov9` + correções do professor (`utils/metrics.py`, `utils/plots.py`, `utils/augmentations.py`, `models/experimental.py` dos zips na pasta) e o fix `np.trapz` → `np.trapezoid`. Script de treino: `train_dual.py` (como no notebook das aulas).

## Dados (estado real verificado)

- Classes (ordem do `labels.txt` = class_id 0..9): garfo, faca, colher, prato, espatula, tigela, frigideira, panela, xicara, taca.
- Formato YOLO bbox (`cls xc yc w h`, normalizado), imagem e `.txt` na mesma pasta, nome `NNNNNNN-C.jpg`.
- Drive (pasta do grupo 9, `1gOMJcfLYc0l3CgHWWGT0xlPeZXidbn5K`): FOLD1 e FOLD2 têm imagens + rótulos; **FOLD3 só tem rótulos**.
- A cópia local em `D:\projetoMLdatasetcozinha\grupo 9` está incompleta (~50% dos pares); fonte de verdade = Drive.

### Montagem dos folds (no Colab)

1. Copiar FOLD1/FOLD2/FOLD3 do Drive para o disco local do Colab.
2. Pool de imagens = todas as imagens de FOLD1 + FOLD2 (todas as divisões), indexadas pelo nome sem extensão.
3. Para cada `.txt` de cada split de cada fold, garantir a imagem de mesmo nome (do próprio diretório ou do pool). Rótulo sem imagem → relatar e excluir. Imagem sem rótulo → relatar (tratar como negativa só se for intencional; padrão: excluir e listar).
4. Checagens que falham alto: nenhum nome repetido entre Treino/Validacao/Teste do mesmo fold; contagem por classe por split; todo class_id em 0..9.
5. Gerar `data/foldN.yaml` (train/val/test, `nc: 10`, `names`).

Por que o pool é seguro: os nomes são globalmente únicos (cada integrante recebeu uma faixa exclusiva de números — convenção da 1ª entrega), e os 3 folds são **três sorteios independentes, estratificados por classe, do mesmo conjunto de ~1009 imagens**. Então o mesmo nome é a mesma foto em qualquer fold, e uma imagem de Teste do FOLD3 estar no Treino do FOLD1 é esperado (a regra de vazamento vale só dentro do fold). Rótulo do FOLD3 sem imagem no pool → excluído e listado (não esperar upload).

Consequência para os slides: chamar de **“3 holdouts repetidos (sorteios independentes)”**, não de “validação cruzada k-fold”. A média ± desvio mede a sensibilidade ao sorteio. Possíveis quase-duplicatas (mesma cena, várias fotos) inflam um pouco as métricas — citar como limitação.

## Treino

| Run | Modelo | Folds | Épocas | Batch | img | Prioridade |
|---|---|---|---|---|---|---|
| T | `models/detect/yolov9-t.yaml` | 1, 2, 3 | 100 | 16 | 640 | 1 (obrigatório) |
| M | `models/detect/yolov9-m.yaml` | 1 | 100 | 8 | 640 | 2 (obrigatório) |
| M | `models/detect/yolov9-m.yaml` | 2, 3 | 100 | 8 | 640 | 3 (se houver tempo/unidades) |

- Hiperparâmetros: `data/hyps/hyp.scratch-high.yaml` padrão (fliplr 0.5, mosaic 1.0, scale, translate, HSV; `degrees 0`).
- `--weights ''`, `--project` no Drive (`/content/drive/MyDrive/yolov9_cozinha/runs`) para sobreviver a quedas; retomar com `--resume` se cair.
- WANDB desligado. Medir a 1ª época e recalcular o cronograma; escolher GPU (T4/L4/A100) pelo custo × prazo.
- Batch acima é o alvo; reduzir se der OOM e registrar.
- Se faltar tempo ou unidades, cortar nesta ordem: (1) M nos folds 2–3; (2) épocas do M no fold 1 (mínimo 60). T nos 3 folds com 100 épocas não se corta. Sem comprar unidades extras.
- Convergência baixa do zero é aceitável e vira discussão nos slides.
- Colab caindo com o usuário ausente: Claude reconecta pelo Chrome e retoma com `--resume` a partir do `last.pt` no Drive. Pedir ao app para manter o PC acordado.

## Avaliação

- `val_dual.py --task test` com `best.pt` de cada run sobre o Teste do fold: P, R, mAP50, mAP50-95, por classe, matriz de confusão, curvas PR/F1.
- Agregado: média ± desvio entre folds por modelo.
- Velocidade de inferência (ms/img) e tamanho/parâmetros de cada modelo — entra na escolha do “modelo mais indicado”.
- **Threshold**: NMS IoU 0.7 (padrão); limiar de confiança para uso = o que maximiza F1 na curva F1 da validação (tipicamente ~0.25–0.4); para mAP usa-se conf 0.001 (padrão do cálculo). Explicar a diferença nos slides.
- Imagens de exemplo com detecções (acertos e erros, foco nos pares confusos: garfo/faca/colher, prato/tigela, frigideira/panela, xícara/taça).

## Dia da apresentação

- Célula pronta no notebook: `detect_dual.py --weights <melhor best.pt> --conf <threshold> --source <pasta>` para o dataset de avaliação do professor. Assume as mesmas 10 classes. Pesos = `best.pt` do modelo indicado, no fold com melhor mAP50-95 de teste. Se vier com rótulos, a mesma célula roda também `val_dual.py`.

## Fontes já no projeto

- Estilo: `grupo 9/Cozinha_dtset_pptx.pptx` — reaproveitado como **modelo** (python-pptx sobre o próprio arquivo: mesmo master, fontes e cores).
- Correções do professor: `metrics.zip` (`utils/metrics.py`, `np.trapezoid`), `utils.zip` (`utils/plots.py`, `utils/augmentations.py`), `experimental.zip` (`models/experimental.py`, `torch.load(..., weights_only=False)`). Copiadas por cima do clone do `yolov9` (HEAD do `main`).
- GitHub: `gh` já autenticado como `enzozon`.

## Entregáveis / repositório (`enzozon/yolov9-cozinha`, privado)

- `notebooks/treino_yolov9_cozinha.ipynb` — notebook único: montar Drive, clonar yolov9, aplicar fixes, montar folds, treinar, avaliar, detectar.
- `scripts/montar_folds.py` — montagem/validação dos folds (com auto-checagem).
- `scripts/resumo_resultados.py` — lê os `results.csv`/saídas de val e gera tabela + gráficos para os slides.
- `results/` — tabelas, gráficos e matrizes de confusão (sem pesos `.pt` no git; pesos ficam no Drive).
- `slides/Cozinha_yolov9.pptx` + notas.
- `docs/superpowers/specs/` — este documento.

## Decidido pelo usuário

- YOLOv9-t e YOLOv9-m; do zero; hyp padrão sem experimento de aumento; 3 folds; M pode ficar só no FOLD1 se apertar; slides no mesmo estilo; Colab conta enzozon7b (AI Plus); repo privado no GitHub.

## Decisões minhas (revisar)

- Batch 16 (t) / 8 (m), 100 épocas, `hyp.scratch-high.yaml`.
- FOLD3 reconstruído pelo nome a partir do pool FOLD1+FOLD2.
- Threshold de confiança pelo máximo de F1.
- Imagens sem rótulo excluídas e listadas.
- Ordem: T (3 folds) → M fold1 → M folds 2–3.
- Critério de “modelo mais indicado”: mAP50-95 médio no teste, ponderado por custo de inferência.

## Deixado ao executor

- Escolha da GPU no momento, conforme disponibilidade e medida da 1ª época.
- Layout fino dos slides dentro do estilo existente.
