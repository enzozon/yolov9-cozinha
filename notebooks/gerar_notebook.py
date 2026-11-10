"""Gera notebooks/treino_yolov9_cozinha.ipynb (rodar: python notebooks/gerar_notebook.py)."""
import json
from pathlib import Path

CELULAS = [
    ("md", """# YOLOv9 — utensílios de cozinha (Grupo 9)
Treino **do zero** de YOLOv9-t e YOLOv9-m nos 3 folds, avaliação no Teste e detecção em imagens novas.

Ordem: rode as células 1–4 sempre que abrir o Colab (inclusive depois de uma queda). Depois escolha o treino (5), a avaliação (6) ou a detecção (7).
Pesos e resultados ficam no Drive (`MyDrive/yolov9_cozinha/runs`): se o Colab cair, a célula 5 retoma do `last.pt`."""),
    ("code", """# 1. Drive e caminhos
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'
BASE = '/content/drive/MyDrive/yolov9_cozinha'   # código + runs
# pasta compartilhada do grupo, acessada pelo ID (precisa existir um atalho dela em qualquer lugar do Meu Drive)
GRUPO = '/content/drive/.shortcut-targets-by-id/1gOMJcfLYc0l3CgHWWGT0xlPeZXidbn5K/grupo 9'
RUNS = f'{BASE}/runs'
os.makedirs(RUNS, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. YOLOv9 oficial + correções do professor (metrics/plots/augmentations/experimental)
%cd /content
!test -d yolov9 || git clone -q https://github.com/WongKinYiu/yolov9.git
%cd /content/yolov9
!pip install -q -r requirements.txt
!cp {BASE}/code/metrics.py {BASE}/code/plots.py {BASE}/code/augmentations.py utils/
!cp {BASE}/code/experimental.py models/
!grep -c "np.trapz(" utils/metrics.py || echo "trapz ok"
# PyTorch >= 2.6 mudou o padrão de torch.load (weights_only=True): sem isso o fim do treino quebra no strip_optimizer
!sed -i "s/torch.load(f, map_location=torch.device('cpu'))/torch.load(f, map_location=torch.device('cpu'), weights_only=False)/" utils/general.py
!sed -i "s/map_location='cpu')/map_location='cpu', weights_only=False)/" train_dual.py val_dual.py detect_dual.py"""),
    ("code", """# 3. Dataset: copia do Drive e monta os folds (FOLD3 busca imagens no pool FOLD1+FOLD2)
!test -d /content/src || (mkdir -p /content/src && cp -r "{GRUPO}/FOLD1" "{GRUPO}/FOLD2" "{GRUPO}/FOLD3" /content/src/)
!test -d /content/data || python {BASE}/code/montar_folds.py /content/src /content/data
!cat /content/data/fold1.yaml"""),
    ("code", """# 4. Funções de treino / avaliação / detecção
import subprocess, glob

def sh(cmd, log=None):
    print('$', cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd='/content/yolov9')
    f = open(log, 'a') if log else None
    for linha in p.stdout:
        if '%|' in linha and '100%|' not in linha:
            continue  # barras de progresso parciais do tqdm: só poluem a saída
        print(linha, end='')
        if f: f.write(linha)
    p.wait()
    if f: f.close()
    return p.returncode

def treinar(modelo, fold, batch, epocas=100):
    nome = f'{modelo}_fold{fold}'
    last = f'{RUNS}/{nome}/weights/last.pt'
    csv_ = f'{RUNS}/{nome}/results.csv'
    if os.path.exists(csv_) and sum(1 for _ in open(csv_)) - 1 >= epocas:
        print('já terminou:', nome); return 0
    if os.path.exists(last):
        cmd = f'python train_dual.py --resume {last}'
    else:
        cmd = (f'python train_dual.py --device 0 --workers 4 --batch-size {batch} --epochs {epocas} --img 640 '
               f'--data /content/data/fold{fold}.yaml --cfg models/detect/yolov9-{modelo}.yaml --weights "" '
               f'--hyp data/hyps/hyp.scratch-high.yaml --cache ram --project {RUNS} --name {nome} --exist-ok')
    return sh(cmd, f'{RUNS}/{nome}_train.log')

def avaliar(modelo, fold, split='test'):
    nome = f'{modelo}_fold{fold}'
    return sh(f'python val_dual.py --device 0 --batch-size 16 --img 640 --conf 0.001 --iou 0.7 --task {split} '
              f'--data /content/data/fold{fold}.yaml --weights {RUNS}/{nome}/weights/best.pt --verbose '
              f'--project {RUNS}/val --name {nome}_{split} --exist-ok', f'{RUNS}/val_{nome}_{split}.log')

def detectar(pesos, origem, conf=0.25, nome='detect'):
    return sh(f'python detect_dual.py --device 0 --img 640 --conf {conf} --iou 0.7 --weights {pesos} '
              f'--source "{origem}" --project {RUNS}/detect --name {nome} --exist-ok')"""),
    ("code", """# 5. Treino (um por vez; ordem do plano). Comente o que já terminou.
treinar('t', 1, 16)
treinar('t', 2, 16)
treinar('t', 3, 16)
treinar('m', 1, 8)
treinar('m', 2, 8)
treinar('m', 3, 8)"""),
    ("code", """# 6. Avaliação no Teste de cada fold (e na Validação, para escolher o threshold pela curva F1)
for m in ('t', 'm'):
    for f in (1, 2, 3):
        if os.path.exists(f'{RUNS}/{m}_fold{f}/weights/best.pt'):
            avaliar(m, f, 'test'); avaliar(m, f, 'val')"""),
    ("code", """# 7. Dia da apresentação: detecção nas imagens novas do professor
PESOS = f'{RUNS}/m_fold1/weights/best.pt'   # trocar pelo modelo indicado
CONF = 0.25                                  # trocar pelo threshold escolhido (máx. F1)
PASTA = '/content/drive/MyDrive/yolov9_cozinha/avaliacao'  # subir as imagens aqui
detectar(PESOS, PASTA, CONF, 'avaliacao')"""),
]


def celula(tipo, src):
    linhas = src.splitlines(keepends=True)
    if tipo == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": linhas}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": linhas}


nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [celula(t, s) for t, s in CELULAS],
}
saida = Path(__file__).with_name("treino_yolov9_cozinha.ipynb")
saida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", saida)
