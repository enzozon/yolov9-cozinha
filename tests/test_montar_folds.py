import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from montar_folds import NOMES, indexar_pool, montar_fold  # noqa: E402


def _par(pasta: Path, stem: str, cls: int = 0, img: bool = True, ext: str = ".jpg"):
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{stem}.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n")
    if img:
        (pasta / f"{stem}{ext}").write_bytes(b"\xff\xd8\xff fake")


def _fold(raiz: Path, treino, val, teste, img=True):
    for split, stems in (("Treino", treino), ("Validacao", val), ("Teste", teste)):
        for s, c in stems:
            _par(raiz / split, s, c, img=img)


def test_monta_com_pool(tmp_path):
    _fold(tmp_path / "FOLD1", [("a-0", 0)], [("b-1", 1)], [("c-2", 2)])
    _fold(tmp_path / "FOLD3", [("c-2", 2)], [("a-0", 0)], [("b-1", 1)], img=False)
    pool = indexar_pool([tmp_path / "FOLD1"])
    r = montar_fold(tmp_path / "FOLD3", pool, tmp_path / "out")
    assert (tmp_path / "out/train/c-2.jpg").exists()
    assert (tmp_path / "out/train/c-2.txt").exists()
    assert r["contagem"]["test"]["faca"] == 1
    assert r["excluidos"] == []


def test_rotulo_sem_imagem_excluido(tmp_path):
    _fold(tmp_path / "FOLD3", [("x-0", 0)], [("y-1", 1)], [("z-2", 2)], img=False)
    _par(tmp_path / "pool", "x-0", 0)
    _par(tmp_path / "pool", "y-1", 1)
    r = montar_fold(tmp_path / "FOLD3", indexar_pool([tmp_path / "pool"]), tmp_path / "out")
    assert r["excluidos"] == ["test/z-2"]
    assert not (tmp_path / "out/test/z-2.txt").exists()


def test_vazamento_detectado(tmp_path):
    _fold(tmp_path / "F", [("a-0", 0)], [("b-1", 1)], [("a-0", 0)])
    with pytest.raises(ValueError, match="vazamento"):
        montar_fold(tmp_path / "F", indexar_pool([tmp_path / "F"]), tmp_path / "out")


def test_classe_invalida(tmp_path):
    _fold(tmp_path / "F", [("a-0", 10)], [("b-1", 1)], [("c-2", 2)])
    with pytest.raises(ValueError, match="classe"):
        montar_fold(tmp_path / "F", indexar_pool([tmp_path / "F"]), tmp_path / "out")


def test_extensoes_aceitas(tmp_path):
    _par(tmp_path / "pool", "p-3", 3, ext=".png")
    _par(tmp_path / "pool", "q-4", 4, ext=".jpeg")
    pool = indexar_pool([tmp_path / "pool"])
    assert pool["p-3"].suffix == ".png" and pool["q-4"].suffix == ".jpeg"


def test_contagem_e_nomes(tmp_path):
    _fold(tmp_path / "F", [("a-0", 0), ("d-9", 9)], [("b-1", 1)], [("c-2", 2)])
    r = montar_fold(tmp_path / "F", indexar_pool([tmp_path / "F"]), tmp_path / "out")
    assert r["contagem"]["train"]["garfo"] == 1
    assert r["contagem"]["train"]["taca"] == 1
    assert NOMES == ["garfo", "faca", "colher", "prato", "espatula",
                     "tigela", "frigideira", "panela", "xicara", "taca"]
