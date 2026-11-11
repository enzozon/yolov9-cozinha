#!/bin/bash
# Pós-treino (rodar no Terminal do Colab depois das células 1–4):
#  1) limiar de confiança que maximiza o F1 na validação, por modelo
#  2) detecções de exemplo no Teste com esse limiar
#  3) zip com tudo o que os slides usam em $RUNS/resultados.zip
set -e
RUNS=/content/drive/MyDrive/yolov9_cozinha/runs
cd /content/yolov9
grep -q CONF_MAX_F1 utils/metrics.py || \
  sed -i "s/^\( *\)i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index/&\n\1print(f'CONF_MAX_F1={px[i]:.3f}')/" utils/metrics.py

mkdir -p /content/exemplos
# 2 imagens de teste por classe (sufixo -C do nome = classe)
for c in 0 1 2 3 4 5 6 7 8 9; do ls /content/data/fold1/test/*-$c.jpg | head -2; done | xargs -I{} cp {} /content/exemplos/

for m in t m; do
  conf=$(python val_dual.py --device 0 --batch-size 16 --img 640 --conf 0.001 --iou 0.7 --task val \
           --data /content/data/fold1.yaml --weights $RUNS/${m}_fold1/weights/best.pt \
           --project /tmp/f1 --name $m --exist-ok 2>&1 | grep -o 'CONF_MAX_F1=[0-9.]*' | tail -1 | cut -d= -f2)
  echo "$m conf_max_f1=$conf" | tee $RUNS/conf_${m}.txt
  python detect_dual.py --device 0 --img 640 --conf $conf --iou 0.7 --weights $RUNS/${m}_fold1/weights/best.pt \
    --source /content/exemplos --project $RUNS/detect --name exemplos_$m --exist-ok > /dev/null 2>&1
done

cd $RUNS
rm -f resultados.zip
zip -qr resultados.zip conf_*.txt *_train.log val_*.log */results.csv */opt.yaml \
  val/*/confusion_matrix.png val/*/F1_curve.png val/*/PR_curve.png detect/exemplos_*
ls -la resultados.zip
