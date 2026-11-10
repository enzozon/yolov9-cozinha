## Rotina para separar os rotulos de uma arquivo json criando um arquivo  separado para cada objeto.



import json

from pathlib import Path

from collections import defaultdict



# ==========================================

# CONFIGURAÇÕES

# ==========================================



JSON_ENTRADA = "/content/lote5.json"

PASTA_SAIDA = "/content/labels"



# Classes que não devem ser utilizadas

CLASSES_EXCLUIR = {"Excluir"}



# ==========================================

# CARREGAMENTO

# ==========================================



with open(JSON_ENTRADA, "r", encoding="utf-8") as f:

    coco = json.load(f)



saida = Path(PASTA_SAIDA)

saida.mkdir(parents=True, exist_ok=True)



# Mapear categorias COCO para YOLO (0, 1, 2...)

categorias = [

    cat for cat in coco["categories"]

    if cat["name"] not in CLASSES_EXCLUIR

]



categorias.sort(key=lambda c: c["id"])



mapa_classes = {

    cat["id"]: i

    for i, cat in enumerate(categorias)

}



print("Classes YOLO:")

for cat in categorias:

    print(f"{mapa_classes[cat['id']]}: {cat['name']}")



# Organizar anotações por imagem

anotacoes = defaultdict(list)



for ann in coco["annotations"]:

    anotacoes[ann["image_id"]].append(ann)



total_objetos = 0

ignorados = 0



# ==========================================

# CONVERSÃO

# ==========================================



for imagem in coco["images"]:



    image_id = imagem["id"]

    largura = imagem["width"]

    altura = imagem["height"]



    nome = Path(imagem["file_name"]).stem + ".txt"

    caminho_saida = saida / nome



    linhas = []



    for ann in anotacoes[image_id]:



        categoria = ann["category_id"]



        if categoria not in mapa_classes:

            ignorados += 1

            continue



        classe = mapa_classes[categoria]



        segmentos = ann.get("segmentation", [])



        # Segmentação por polígonos COCO

        if not isinstance(segmentos, list):

            ignorados += 1

            continue



        for poligono in segmentos:



            if len(poligono) < 6 or len(poligono) % 2:

                ignorados += 1

                continue



            coordenadas = []



            for i in range(0, len(poligono), 2):



                x = float(poligono[i]) / largura

                y = float(poligono[i + 1]) / altura



                # Garantir valores entre 0 e 1

                x = max(0.0, min(1.0, x))

                y = max(0.0, min(1.0, y))



                coordenadas.extend([x, y])



            linha = (

                str(classe) + " " +

                " ".join(f"{v:.6f}" for v in coordenadas)

            )



            linhas.append(linha)

            total_objetos += 1



    # Um TXT para cada imagem

    caminho_saida.write_text(

        "\n".join(linhas) + ("\n" if linhas else ""),

        encoding="utf-8"

    )



print("\nConversão concluída!")

print("Imagens:", len(coco["images"]))

print("Polígonos convertidos:", total_objetos)

print("Anotações/polígonos ignorados:", ignorados)

print("Diretório:", PASTA_SAIDA)