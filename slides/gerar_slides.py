"""Gera slides/Cozinha_yolov9.pptx no estilo da 1ª entrega.

Uso: python slides/gerar_slides.py
Lê os números de results/ (gerados por scripts/resumo_resultados.py e pelos runs do Colab).
"""
import csv
import json
import statistics
from pathlib import Path

from pptx.enum.text import PP_ALIGN

from estilo import (CINZA, CLARO, ESCURO, LARANJA, MONO, TEAL, TEAL2, VERDE, VERMELHO, card, imagem,
                    novo_deck, ret, slide_conteudo, slide_escuro, tabela, texto)

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results"
NOMES = ["Garfo", "Faca", "Colher", "Prato", "Espátula", "Tigela", "Frigideira", "Panela", "Xícara", "Taça"]


def carregar():
    linhas = []
    p = RES / "metricas_teste.csv"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                linhas.append({k: (v if k == "modelo" else float(v or 0)) for k, v in r.items()})
    extra = json.loads((RES / "extra.json").read_text(encoding="utf-8")) if (RES / "extra.json").exists() else {}
    return linhas, extra


def media(linhas, modelo, k):
    v = [l[k] for l in linhas if l["modelo"] == modelo]
    if not v:
        return "—"
    return f"{statistics.mean(v):.3f}" + (f" ± {statistics.stdev(v):.3f}" if len(v) > 1 else "")


def main():
    linhas, ex = carregar()
    d = novo_deck()

    # 1 capa
    s = slide_escuro(d, "modelo", "Abertura. Apresentar o grupo e dizer em uma frase o que fizemos: pegamos o dataset "
                     "da 1ª etapa e treinamos duas versões do YOLOv9, do zero, para detectar os 10 utensílios.")
    for x, y, w, h in [(2.1, 2.63, .68, .07), (2.1, 2.63, .07, .68), (17.24, 2.63, .68, .07), (17.85, 2.63, .07, .68),
                       (2.1, 7.14, .68, .07), (2.1, 6.53, .07, .68), (17.24, 7.14, .68, .07), (17.85, 6.53, .07, .68)]:
        ret(s, x, y, w, h, LARANJA)
    texto(s, 2.85, 3.3, 14.3, 1.6, "Detectando utensílios", 80, "FFFFFF", alinha=PP_ALIGN.CENTER)
    texto(s, 2.85, 5.0, 14.3, 0.6, "YOLOv9-t × YOLOv9-m treinados do zero sobre o nosso dataset de cozinha",
          25, CLARO, alinha=PP_ALIGN.CENTER)
    texto(s, 1.28, 7.96, 17.46, 1.9, [
        "Enzo Faroni Zon    Filippo Ungarelli    Guilherme Fagundes",
        "Leonardo Pupin    Matheus Soares Fernandes",
        [("Universidade Vila Velha  ·  Ciência da Computação CC7/CC8  ·  Prof. Jairo Moraes", {"tam": 15, "cor": CLARO})]],
        21, "FFFFFF", alinha=PP_ALIGN.CENTER)
    texto(s, 1.28, 10.04, 17.46, 0.5, "10 classes   ·   3 folds   ·   YOLOv9-t × YOLOv9-m   ·   100 épocas do zero",
          17, CLARO, MONO, alinha=PP_ALIGN.CENTER)

    # 2 roteiro
    s = slide_conteudo(d, "roteiro", "O que vamos mostrar", "Só ler os tópicos rapidamente; não explicar ainda.")
    itens = [("01", "Projeto", "o problema, aplicações e as dez classes"),
             ("02", "Dados", "os três folds e como foram montados"),
             ("03", "Modelos", "YOLO, YOLOv9 e as versões t e m"),
             ("04", "Treino", "parâmetros, aumento de dados e threshold"),
             ("05", "Resultados", "métricas por fold, por classe e confusões"),
             ("06", "Escolha", "qual modelo indicamos e por quê")]
    for i, (n, t, desc) in enumerate(itens):
        y = 2.75 + i * 1.18
        texto(s, 1.3, y, 1.0, 0.7, n, 30, LARANJA, MONO, True)
        texto(s, 2.6, y, 4.5, 0.7, t, 28, TEAL, negrito=True)
        texto(s, 7.0, y + 0.08, 11.5, 0.7, desc, 22, CINZA)

    # 3 introdução
    s = slide_conteudo(d, "1 · projeto", "Da etapa 1 para a etapa 2",
                       "Lembrar que na 1ª etapa montamos e rotulamos o dataset com bounding box. Agora a pergunta é: uma "
                       "rede treinada só com essas ~800 imagens consegue achar e nomear os utensílios? Detecção = o que é + onde está.")
    card(s, 1.28, 2.76, 8.4, 3.6, TEAL, "Etapa 1 · o dataset",
         "1.009 imagens, 10 classes, cada uma com ~100 exemplos. Rotulação em bounding box com ferramenta própria. "
         "Três sorteios 80/10/10 estratificados por classe.")
    card(s, 10.33, 2.76, 8.4, 3.6, LARANJA, "Etapa 2 · o modelo",
         "Treinar um detector que, dada uma foto, devolve a caixa e a classe de cada utensílio. "
         "Dois tamanhos de YOLOv9, comparados nos três folds.")
    texto(s, 1.28, 7.0, 17.4, 1.5, [[("Detecção responde duas perguntas: ", {}),
                                     ("o que é ", {"negrito": True}), ("e ", {}), ("onde está.", {"negrito": True}),
                                     (" Classificação responde só a primeira.", {"cor": CINZA})]], 22)

    # 4 aplicações
    s = slide_conteudo(d, "1 · aplicações", "Onde um detector de utensílios serve",
                       "Exemplos concretos: robôs domésticos e braços de cozinha precisam saber onde pegar; inventário de "
                       "louça em restaurante/hotel; acessibilidade para pessoas com deficiência visual (app que fala o que está na mesa).")
    card(s, 1.28, 2.76, 5.52, 4.2, TEAL, "Robótica doméstica",
         "Robô ou braço de cozinha precisa saber onde está a espátula antes de pegá-la — classe e posição.")
    card(s, 7.25, 2.76, 5.52, 4.2, VERDE, "Inventário",
         "Contar pratos, taças e talheres em restaurantes, hotéis e cozinhas industriais por câmera.")
    card(s, 13.22, 2.76, 5.52, 4.2, VERMELHO, "Acessibilidade",
         "App que descreve em voz alta os objetos sobre a mesa para pessoas com deficiência visual.")

    # 5 classes
    s = slide_conteudo(d, "1 · classes", "Dez classes — e quatro pares que se confundem",
                       "class_id 0 a 9 na ordem do labels.txt. Os quatro pares são a nossa hipótese de onde a rede vai errar; "
                       "vamos voltar a eles na matriz de confusão.")
    for i, n in enumerate(NOMES):
        x, y = (1.28 if i < 5 else 4.8), 2.76 + (i % 5) * 0.78
        ret(s, x, y, 0.63, 0.57, TEAL)
        texto(s, x, y + 0.04, 0.63, 0.5, str(i), 19.5, "FFFFFF", MONO, True, PP_ALIGN.CENTER)
        texto(s, x + 0.87, y + 0.02, 2.6, 0.58, n, 22.5)
    ret(s, 8.93, 2.61, 9.81, 4.43, "F6EBE9")
    texto(s, 9.46, 2.92, 8.3, 0.6, "Os pares problemáticos", 22.5, VERMELHO, negrito=True)
    pares = [("garfo · faca · colher", "mesmo eixo alongado, mesmo metal — muda só a ponta"),
             ("prato · tigela", "ambos circulares; separa a aba plana"),
             ("frigideira · panela", "parede baixa e inclinada × parede alta e reta"),
             ("xícara · taça", "alça lateral × haste com base")]
    for i, (a, b) in enumerate(pares):
        texto(s, 9.46, 3.6 + i * 0.83, 8.1, 0.45, a, 18, ESCURO, MONO, True)
        texto(s, 9.46, 4.0 + i * 0.83, 8.1, 0.45, b, 18, CINZA)

    # 6 dados / folds
    cont = ex.get("contagem", {})
    s = slide_conteudo(d, "2 · dados", "Três sorteios independentes, 80 / 10 / 10",
                       "Cada fold é um sorteio novo e estratificado por classe; a mesma foto pode estar no treino de um fold e no "
                       "teste de outro — o que importa é não vazar dentro do fold, e o script verifica isso. O FOLD3 no Drive "
                       "só tem os .txt; as imagens são buscadas pelo nome nos folds 1 e 2 (os nomes são únicos). "
                       "Não chamamos de validação cruzada k-fold: são holdouts repetidos.")
    linhas_t = [["", "treino", "validação", "teste"]]
    for f in (1, 2, 3):
        c = cont.get(f"fold{f}", {})
        linhas_t.append([f"FOLD{f}"] + [str(sum(c.get(sp, {}).values())) if c else "—" for sp in ("train", "val", "test")])
    tabela(s, 1.28, 2.76, 8.4, linhas_t, [2.1, 2.1, 2.1, 2.1], 19)
    card(s, 10.33, 2.76, 8.4, 2.2, TEAL, "Sem vazamento dentro do fold",
         "Antes de treinar, o script confere que nenhum nome aparece em dois conjuntos do mesmo fold.", 17)
    card(s, 10.33, 5.2, 8.4, 2.2, LARANJA, "FOLD3 só com rótulos",
         "Para não duplicar 1 GB de fotos, o FOLD3 guarda só os .txt. A imagem vem do FOLD1/FOLD2 pelo nome.", 17)
    texto(s, 1.28, 8.0, 17.4, 1.2, [[("Três sorteios mostram se a métrica é real ou sorte do sorteio: ", {}),
                                     ("reportamos média ± desvio.", {"negrito": True})]], 21)

    # 7 detector de um estágio
    s = slide_conteudo(d, "3 · modelos", "YOLO: detectar em uma passada só",
                       "Explicar: a imagem passa uma vez pela rede, que já devolve caixas + classe + confiança. Two-stage "
                       "(Faster R-CNN) primeiro propõe regiões e depois classifica — mais lento. Transformers (DETR) "
                       "precisam de muito mais dados e épocas, ruim para 800 imagens.")
    card(s, 1.28, 2.76, 5.52, 4.6, TEAL, "Um estágio · YOLO",
         "Uma passada: caixas, classes e confiança de uma vez. Rápido, roda em tempo real. Nossa escolha.", 17.5)
    card(s, 7.25, 2.76, 5.52, 4.6, CINZA, "Dois estágios · Faster R-CNN",
         "Primeiro propõe regiões, depois classifica cada uma. Preciso, mas várias vezes mais lento.", 17.5)
    card(s, 13.22, 2.76, 5.52, 4.6, VERMELHO, "Transformer · DETR",
         "Sem NMS e sem âncoras, mas exige muito mais dados e épocas para convergir — ruim para ~800 fotos.", 17.5)
    texto(s, 1.28, 7.9, 17.4, 1.2, "Depois da rede, o NMS remove caixas repetidas sobre o mesmo objeto.", 21, CINZA)

    # 8 YOLOv9
    s = slide_conteudo(d, "3 · YOLOv9", "O que o YOLOv9 traz de novo",
                       "Duas ideias do artigo (Wang et al., 2024). GELAN: blocos que reaproveitam caminhos de gradiente — mais "
                       "precisão por parâmetro. PGI: um ramo auxiliar reversível que só existe no treino e ajuda o gradiente a "
                       "chegar nas camadas iniciais; na inferência ele é removido. Por isso usamos train_dual.py.")
    card(s, 1.28, 2.76, 8.4, 4.3, TEAL, "GELAN · a espinha",
         "Rede de agregação de camadas eficiente: combina caminhos curtos e longos de gradiente. "
         "Mais precisão com menos parâmetros e operações.", 18)
    card(s, 10.33, 2.76, 8.4, 4.3, LARANJA, "PGI · só no treino",
         "Ramo auxiliar que preserva a informação até a perda (contra o 'gargalo de informação'). "
         "É descartado na inferência: custo zero na hora de detectar.", 18)
    texto(s, 1.28, 7.6, 17.4, 1.2, [[("train_dual.py ", {"fonte": MONO, "negrito": True}),
                                     ("treina com o ramo auxiliar (dual); o modelo final é só o ramo principal.", {})]], 21)

    # 9 t x m
    pt, pm = ex.get("params", {}).get("t", "≈2,0 M"), ex.get("params", {}).get("m", "≈20 M")
    gt, gm = ex.get("gflops", {}).get("t", "≈7,7"), ex.get("gflops", {}).get("m", "≈76")
    s = slide_conteudo(d, "3 · modelos", "Dois tamanhos: t (tiny) e m (medium)",
                       "Mesma arquitetura, escala diferente: o m tem ~10× mais parâmetros e operações. Mais capacidade, mas "
                       "com só ~800 imagens de treino isso também significa mais risco de decorar o treino. Tempo de treino "
                       "real medido no Colab T4.")
    tabela(s, 1.28, 2.76, 17.4, [["", "YOLOv9-t", "YOLOv9-m"],
                                 ["parâmetros", pt, pm],
                                 ["GFLOPs (640 px)", gt, gm],
                                 ["batch no T4", "16", "8"],
                                 ["tempo de treino (100 épocas)", ex.get("tempo", {}).get("t", "—"), ex.get("tempo", {}).get("m", "—")],
                                 ["inferência (ms/imagem, T4)", ex.get("ms", {}).get("t", "—"), ex.get("ms", {}).get("m", "—")]],
           [7.4, 5.0, 5.0], 20)

    # 10 treino
    s = slide_conteudo(d, "4 · treino", "Treino do zero, igual às aulas",
                       "Do zero = --weights '' (sem pesos pré-treinados no COCO). Toda a 'visão' vem das nossas fotos. "
                       "Hiperparâmetros do arquivo padrão hyp.scratch-high.yaml, sem alterações. Correções do professor em "
                       "metrics.py, plots.py, augmentations.py e experimental.py aplicadas antes de treinar.")
    tabela(s, 1.28, 2.76, 8.4, [["parâmetro", "valor"], ["pesos iniciais", "nenhum (do zero)"], ["épocas", "100"],
                                ["imagem", "640 × 640"], ["otimizador", "SGD, lr0 0.01"], ["lr final", "lr0 × 0.01"],
                                ["momentum", "0.937"], ["warmup", "3 épocas"], ["GPU", "Colab T4 16 GB"]],
           [4.2, 4.2], 17)
    card(s, 10.33, 2.76, 8.4, 2.4, TEAL, "Por que do zero",
         "É o que fizemos em aula. Custa mais épocas, mas mostra o que a rede aprende só com o nosso dataset.", 17)
    card(s, 10.33, 5.4, 8.4, 2.4, VERMELHO, "O custo",
         "Sem conhecimento prévio de bordas e formas, 800 imagens são pouco: a métrica tende a ser modesta.", 17)

    # 11 aumento de dados
    s = slide_conteudo(d, "4 · aumento de dados", "Usamos o aumento padrão — sem rotação",
                       "Pergunta comum: usamos parâmetros diferentes de rotação/espelhamento? Não: ficamos com o padrão do "
                       "YOLOv9. Ele já espelha na horizontal (50%), faz mosaico de 4 imagens, mixup, escala e cor. Rotação "
                       "(degrees) e espelhamento vertical ficam em 0 — utensílio de cabeça para baixo não é o caso comum.")
    on = [("fliplr", "0.5", "espelho horizontal em 50% das imagens"), ("mosaic", "1.0", "junta 4 fotos numa só"),
          ("mixup", "0.15", "sobrepõe duas imagens"), ("scale", "0.9", "zoom de ±90%"), ("translate", "0.1", "desloca ±10%"),
          ("hsv h/s/v", ".015/.7/.4", "matiz, saturação, brilho")]
    off = [("degrees", "0", "rotação"), ("flipud", "0", "espelho vertical"), ("shear", "0", "cisalhamento"),
           ("perspective", "0", "perspectiva")]
    texto(s, 1.28, 2.7, 8.4, 0.6, "ligados", 22.5, VERDE, negrito=True)
    for i, (k, v, desc) in enumerate(on):
        texto(s, 1.28, 3.35 + i * 0.66, 2.6, 0.5, k, 18, ESCURO, MONO, True)
        texto(s, 3.9, 3.35 + i * 0.66, 1.6, 0.5, v, 18, VERDE, MONO)
        texto(s, 5.5, 3.35 + i * 0.66, 4.4, 0.5, desc, 17, CINZA)
    texto(s, 10.9, 2.7, 8.0, 0.6, "desligados", 22.5, VERMELHO, negrito=True)
    for i, (k, v, desc) in enumerate(off):
        texto(s, 10.9, 3.35 + i * 0.66, 2.6, 0.5, k, 18, ESCURO, MONO, True)
        texto(s, 13.5, 3.35 + i * 0.66, 1.0, 0.5, v, 18, VERMELHO, MONO)
        texto(s, 14.5, 3.35 + i * 0.66, 4.4, 0.5, desc, 17, CINZA)
    texto(s, 1.28, 8.0, 17.4, 1.2, [[("Não usamos parâmetros diferentes do padrão. ", {"negrito": True}),
                                     ("Toda variação de ângulo e rotação vem das próprias fotos.", {"cor": CINZA})]], 21)

    # 12 threshold
    conf = ex.get("conf_f1", "—")
    s = slide_conteudo(d, "4 · threshold", "Três limiares diferentes, três papéis",
                       "Confusão comum: 'o threshold'. São três. (1) IoU para contar acerto: 0.5 no mAP50 e de 0.5 a 0.95 no "
                       "mAP50-95. (2) IoU do NMS = 0.7: caixas mais sobrepostas que isso viram uma só. (3) Confiança: para "
                       "calcular mAP usamos 0.001 (a curva inteira); para usar o modelo escolhemos o valor que maximiza o F1 "
                       "na validação.")
    card(s, 1.28, 2.76, 5.52, 4.4, TEAL, "IoU de acerto",
         "Caixa prevista conta como acerto se sobrepõe a real em ≥ 0.5 (mAP50) — ou na média de 0.5 a 0.95 (mAP50-95).", 17)
    card(s, 7.25, 2.76, 5.52, 4.4, VERDE, "IoU do NMS = 0.7",
         "Caixas da mesma classe com sobreposição acima de 0.7 são fundidas: um objeto, uma caixa.", 17)
    card(s, 13.22, 2.76, 5.52, 4.4, LARANJA, f"Confiança = {conf}",
         "Para medir mAP: 0.001 (curva completa). Para usar o modelo: o valor que maximiza o F1 na validação.", 17)

    # 13 resultados por fold
    s = slide_conteudo(d, "5 · resultados", "Resultados no teste, por fold",
                       "Ler a linha da média primeiro. O desvio entre folds diz quanto o resultado depende do sorteio. "
                       "Cada fold tem ~100 imagens de teste.")
    tab = [["modelo", "fold", "P", "R", "mAP50", "mAP50-95"]]
    for l in sorted(linhas, key=lambda l: (l["modelo"], l["fold"])):
        tab.append([f"YOLOv9-{l['modelo']}", str(int(l["fold"])), f"{l['P']:.3f}", f"{l['R']:.3f}",
                    f"{l['mAP50']:.3f}", f"{l['mAP50_95']:.3f}"])
    for m in ("t", "m"):
        if any(l["modelo"] == m for l in linhas):
            tab.append([f"YOLOv9-{m} · média", "", "", "", media(linhas, m, "mAP50"), media(linhas, m, "mAP50_95")])
    if len(tab) == 1:
        tab.append(["—"] * 6)
    tabela(s, 1.28, 2.6, 17.4, tab, [4.4, 1.6, 2.2, 2.2, 3.5, 3.5], 17)

    # 14 curvas
    s = slide_conteudo(d, "5 · resultados", "Como a rede aprendeu ao longo das épocas",
                       "Curva de mAP50-95 na validação por época. Do zero, as primeiras épocas ficam perto de zero até a rede "
                       "aprender formas básicas. Comentar se a curva ainda subia no fim (mais épocas ajudariam) ou estabilizou.")
    if (RES / "curvas_treino.png").exists():
        imagem(s, RES / "curvas_treino.png", 1.6, 2.5, h=7.9)
    texto(s, 15.4, 3.0, 3.4, 5.0, ex.get("obs_curvas", ""), 17, CINZA)

    # 15 por classe
    s = slide_conteudo(d, "5 · resultados", "Desempenho por classe",
                       "mAP50 por classe, média dos folds. Com ~10 imagens de teste por classe, um acerto move a barra em "
                       "~10 pontos: olhar tendência, não décimos. Comparar com a hipótese da Tigela (46% internet).")
    if (RES / "map_por_classe.png").exists():
        imagem(s, RES / "map_por_classe.png", 1.6, 2.5, h=7.9)
    texto(s, 15.4, 3.0, 3.4, 5.0, ex.get("obs_classes", ""), 17, CINZA)

    # 16 matriz de confusão
    s = slide_conteudo(d, "5 · confusões", "Onde a rede erra",
                       "Matriz de confusão do melhor modelo no teste. Diagonal = acerto. Fora dela, ver se os erros caem nos "
                       "pares previstos (garfo/faca/colher, prato/tigela, frigideira/panela, xícara/taça). 'background' = "
                       "objeto não detectado ou detecção em fundo vazio.")
    if (RES / "confusion_matrix.png").exists():
        imagem(s, RES / "confusion_matrix.png", 1.28, 2.45, h=8.3)
    texto(s, 13.0, 3.0, 5.8, 6.0, ex.get("obs_confusao", ""), 18, ESCURO)

    # 17 exemplos
    s = slide_conteudo(d, "5 · exemplos", "Detecções em imagens de teste",
                       "Imagens que a rede nunca viu no treino. Mostrar um acerto limpo e um erro entre classes parecidas.")
    exemplos = sorted((RES / "exemplos").glob("*.jpg"))[:4] if (RES / "exemplos").exists() else []
    for i, e in enumerate(exemplos):
        imagem(s, e, 1.28 + i * 4.45, 2.6, w=4.2)

    # 18 escolha
    s = slide_conteudo(d, "6 · escolha", f"Modelo indicado: {ex.get('indicado', '—')}",
                       ex.get("nota_indicado", "Justificar com mAP50-95 médio e custo de inferência."))
    card(s, 1.28, 2.76, 5.52, 4.6, TEAL, "Precisão", ex.get("escolha_precisao", "—"), 17.5)
    card(s, 7.25, 2.76, 5.52, 4.6, VERDE, "Custo", ex.get("escolha_custo", "—"), 17.5)
    card(s, 13.22, 2.76, 5.52, 4.6, LARANJA, "Na prática", ex.get("escolha_pratica", "—"), 17.5)

    # 19 dificuldades
    s = slide_conteudo(d, "dificuldades", "O que deu trabalho — e como resolvemos",
                       "Contar as dificuldades reais: FOLD3 sem imagens, bibliotecas novas quebrando o código do YOLOv9 "
                       "(np.trapz, torch.load), tempo de GPU para treinar do zero.")
    card(s, 1.28, 2.76, 5.52, 4.8, TEAL, "FOLD3 sem imagens",
         "No Drive só os .txt. Script monta o fold buscando cada imagem pelo nome nos folds 1 e 2 e confere vazamento.", 17)
    card(s, 7.25, 2.76, 5.52, 4.8, VERMELHO, "Código quebrado",
         "NumPy 2 removeu np.trapz e o PyTorch novo mudou torch.load. Aplicamos as correções do professor.", 17)
    card(s, 13.22, 2.76, 5.52, 4.8, LARANJA, "Tempo de GPU",
         ex.get("dif_tempo", "Do zero, o m leva horas por fold no T4. Treinamos o t nos 3 folds e o m conforme o tempo."), 17)

    # 20 conclusão
    s = slide_escuro(d, "conclusão", "Fechar com os números principais e o modelo indicado. Próximos passos: pesos "
                     "pré-treinados no COCO, mais épocas, mais fotos das classes que mais erraram.")
    texto(s, 1.28, 1.57, 17.5, 1.3, "Onde chegamos", 54, "FFFFFF")
    pontos = ex.get("conclusao", [("—", "")])
    for i, (a, b) in enumerate(pontos[:3]):
        y = 3.3 + i * 1.18
        ret(s, 1.28, y + 0.17, 0.08, 0.75, LARANJA)
        texto(s, 1.7, y, 8.2, 0.5, a, 24, "FFFFFF", negrito=True)
        texto(s, 1.7, y + 0.52, 8.2, 0.6, b, 18.75, CLARO)
    texto(s, 10.88, 3.13, 7.9, 0.7, "Próximos passos", 19.5, LARANJA, MONO, True)
    for i, t in enumerate(["Partir de pesos pré-treinados (COCO)", "Mais épocas para o m",
                           "Mais fotos das classes que mais erraram"]):
        texto(s, 10.88, 4.0 + i * 0.83, 0.5, 0.5, "→", 19.5, LARANJA)
        texto(s, 11.45, 4.0 + i * 0.83, 7.1, 0.8, t, 19.5, "FFFFFF")

    out = RAIZ / "slides" / "Cozinha_yolov9.pptx"
    d.save(out)
    print("ok", out, len(d.slides), "slides")


if __name__ == "__main__":
    main()
