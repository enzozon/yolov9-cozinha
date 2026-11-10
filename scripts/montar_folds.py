"""Monta os 3 folds no formato que o YOLOv9 lê.

O FOLD3 do Drive tem só os rótulos (.txt). Como os nomes das imagens são
únicos no dataset inteiro (faixa de números por integrante), a imagem de
cada rótulo é buscada pelo nome num "pool" com as imagens de FOLD1 e FOLD2.

Uso:
    python montar_folds.py "<pasta grupo 9>" <destino>
Cria <destino>/foldN/{train,val,test}/ (imagem + .txt juntos) e <destino>/foldN.yaml.
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

NOMES = ["garfo", "faca", "colher", "prato", "espatula",
         "tigela", "frigideira", "panela", "xicara", "taca"]
SPLITS = {"Treino": "train", "Validacao": "val", "Teste": "test"}
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def indexar_pool(raizes):
    """nome sem extensão -> caminho da imagem, procurando recursivamente."""
    pool = {}
    for raiz in raizes:
        for p in Path(raiz).rglob("*"):
            if p.suffix.lower() in EXT_IMG:
                pool.setdefault(p.stem, p)
    return pool


def _validar_rotulo(txt):
    for n, linha in enumerate(txt.read_text().split("\n"), 1):
        if not linha.strip():
            continue
        partes = linha.split()
        if len(partes) != 5:
            raise ValueError(f"linha malformada {txt}:{n}: {linha!r}")
        cls = int(partes[0])
        if not 0 <= cls < len(NOMES):
            raise ValueError(f"classe invalida {cls} em {txt}:{n}")
        yield cls


def montar_fold(origem, pool, destino):
    origem, destino = Path(origem), Path(destino)
    stems = {s: {t.stem for t in (origem / pasta).glob("*.txt")}
             for pasta, s in SPLITS.items()}
    nomes = list(stems)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            comum = stems[a] & stems[b]
            if comum:
                raise ValueError(f"vazamento {a}/{b} em {origem}: {sorted(comum)[:5]}")

    contagem, excluidos = {}, []
    for pasta, split in SPLITS.items():
        saida = destino / split
        saida.mkdir(parents=True, exist_ok=True)
        cont = Counter()
        for txt in sorted((origem / pasta).glob("*.txt")):
            classes = list(_validar_rotulo(txt))
            local = [p for p in txt.parent.glob(txt.stem + ".*") if p.suffix.lower() in EXT_IMG]
            img = local[0] if local else pool.get(txt.stem)
            if img is None:
                excluidos.append(f"{split}/{txt.stem}")
                continue
            shutil.copy2(img, saida / (txt.stem + img.suffix.lower()))
            shutil.copy2(txt, saida / txt.name)
            cont.update(NOMES[c] for c in classes)
        contagem[split] = {n: cont.get(n, 0) for n in NOMES}

    (destino.parent / f"{destino.name}.yaml").write_text(
        f"path: {destino.as_posix()}\ntrain: train\nval: val\ntest: test\n"
        f"nc: {len(NOMES)}\nnames: {NOMES}\n", encoding="utf-8")
    return {"contagem": contagem, "excluidos": excluidos}


def main(raiz_grupo, destino):
    raiz_grupo, destino = Path(raiz_grupo), Path(destino)
    pool = indexar_pool([raiz_grupo / "FOLD1", raiz_grupo / "FOLD2"])
    print(f"pool: {len(pool)} imagens")
    for n in (1, 2, 3):
        r = montar_fold(raiz_grupo / f"FOLD{n}", pool, destino / f"fold{n}")
        print(f"\n== fold{n}")
        for split, c in r["contagem"].items():
            print(f"  {split:5s} total={sum(c.values()):4d} ", c)
        print(f"  excluidos ({len(r['excluidos'])}): {r['excluidos']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
