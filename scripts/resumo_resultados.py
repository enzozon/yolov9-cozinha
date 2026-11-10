"""Resume os logs do val_dual.py (copiados do Drive para results/logs) em tabelas e gráficos.

Uso: python scripts/resumo_resultados.py results/logs results
Espera arquivos val_{modelo}_fold{N}_{split}.log e, opcionalmente, {modelo}_fold{N}/results.csv.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

NUM = r"([\d.]+)"
LINHA = re.compile(rf"^\s*(\S+)\s+\d+\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$")
METRICAS = ("P", "R", "mAP50", "mAP50_95")


def ler_log_val(caminho):
    r = {"all": None, "classes": {}, "ms_img": None}
    for linha in Path(caminho).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINHA.match(linha)
        if m:
            vals = dict(zip(METRICAS, map(float, m.groups()[1:])))
            if m.group(1) == "all":
                r["all"] = vals  # última ocorrência vence (log em modo append)
                r["classes"] = {}
            else:
                r["classes"][m.group(1)] = vals
        s = re.search(rf"{NUM}ms inference", linha)
        if s:
            r["ms_img"] = float(s.group(1))
    return r


def tabela_agregada(linhas):
    ag = {}
    for modelo in sorted({l["modelo"] for l in linhas}):
        grupo = [l for l in linhas if l["modelo"] == modelo]
        ag[modelo] = {"n": len(grupo)}
        for k in ("mAP50", "mAP50_95"):
            v = [l[k] for l in grupo]
            ag[modelo][k] = (round(statistics.mean(v), 4),
                             round(statistics.stdev(v), 4) if len(v) > 1 else 0.0)
    return ag


def main(pasta_logs, saida):
    pasta_logs, saida = Path(pasta_logs), Path(saida)
    linhas = []
    for log in sorted(pasta_logs.glob("val_*_fold*_test.log")):
        modelo, fold = re.match(r"val_(\w+?)_fold(\d)_test", log.stem).groups()
        r = ler_log_val(log)
        if not r["all"]:
            continue
        linhas.append({"modelo": modelo, "fold": int(fold), **r["all"], "ms_img": r["ms_img"],
                       **{f"{c}_mAP50": v["mAP50"] for c, v in r["classes"].items()}})
    saida.mkdir(parents=True, exist_ok=True)
    campos = sorted({k for l in linhas for k in l}, key=lambda k: (k not in ("modelo", "fold"), k))
    with open(saida / "metricas_teste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, campos)
        w.writeheader()
        w.writerows(linhas)
    for modelo, v in tabela_agregada(linhas).items():
        print(modelo, v)
    graficos(pasta_logs, linhas, saida)


def graficos(pasta_logs, linhas, saida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # curvas de treino (mAP50-95 na validação por época)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for csvp in sorted(pasta_logs.glob("*_fold*/results.csv")):
        with open(csvp, encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        chave = next(k for k in rows[0] if "mAP_0.5:0.95" in k)
        ax.plot([float(r[chave]) for r in rows], label=csvp.parent.name)
    ax.set_xlabel("época"); ax.set_ylabel("mAP50-95 (validação)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(saida / "curvas_treino.png", dpi=150); plt.close(fig)

    # mAP por classe (média dos folds) por modelo
    if linhas:
        classes = [k[:-6] for k in linhas[0] if k.endswith("_mAP50")]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        modelos = sorted({l["modelo"] for l in linhas})
        larg = 0.8 / len(modelos)
        for i, m in enumerate(modelos):
            g = [l for l in linhas if l["modelo"] == m]
            vals = [statistics.mean(l.get(f"{c}_mAP50", 0) for l in g) for c in classes]
            ax.bar([j + i * larg for j in range(len(classes))], vals, larg, label=f"YOLOv9-{m}")
        ax.set_xticks([j + larg * (len(modelos) - 1) / 2 for j in range(len(classes))], classes, rotation=30)
        ax.set_ylabel("mAP50 (teste, média dos folds)"); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(saida / "map_por_classe.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:3])
