import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resumo_resultados import ler_log_val, tabela_agregada  # noqa: E402

LOG = """\
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%
                   all        100        102      0.612      0.553      0.580      0.401
                 garfo        100         10      0.500      0.400      0.450      0.300
                  taca        100         11      0.700      0.650      0.690      0.500
Speed: 0.3ms pre-process, 12.5ms inference, 1.1ms NMS per image at shape (16, 3, 640, 640)
"""


def test_ler_log_val(tmp_path):
    p = tmp_path / "val_t_fold1_test.log"
    p.write_text(LOG)
    r = ler_log_val(p)
    assert r["all"] == {"P": 0.612, "R": 0.553, "mAP50": 0.580, "mAP50_95": 0.401}
    assert r["classes"]["taca"]["mAP50"] == 0.690
    assert r["ms_img"] == 12.5


def test_agregado():
    linhas = [{"modelo": "t", "mAP50": 0.5, "mAP50_95": 0.3},
              {"modelo": "t", "mAP50": 0.7, "mAP50_95": 0.5}]
    ag = tabela_agregada(linhas)
    assert ag["t"]["mAP50"] == (0.6, 0.1414)  # desvio amostral
    assert ag["t"]["n"] == 2
