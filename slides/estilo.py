"""Primitivas visuais copiadas do deck da 1ª entrega (Cozinha_dtset_pptx.pptx)."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

W, H = 20.0, 11.25  # polegadas (16:9 do deck original)
TEAL, TEAL2, LARANJA = "0F4A57", "176B7C", "E08A17"
ESCURO, CINZA, CLARO = "11161D", "5B6674", "9FB8BF"
VERDE, VERMELHO = "1D7A5F", "A33B33"
TINTA = {TEAL: "EEF3F6", VERDE: "E3F0EB", VERMELHO: "F6EBE9", LARANJA: "FBF0E1"}
SANS, MONO = "Segoe UI", "Consolas"


def rgb(h):
    return RGBColor.from_string(h)


def novo_deck():
    p = Presentation()
    p.slide_width, p.slide_height = Inches(W), Inches(H)
    return p


def ret(slide, x, y, w, h, cor, forma=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(cor)
    s.line.fill.background(); s.shadow.inherit = False
    return s


def texto(slide, x, y, w, h, conteudo, tam=18.75, cor=ESCURO, fonte=SANS, negrito=False,
          alinha=PP_ALIGN.LEFT, ancora=MSO_ANCHOR.TOP, espaco=1.15):
    """conteudo: str ou lista de parágrafos; cada parágrafo str ou lista de (texto, {opções})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    pars = conteudo if isinstance(conteudo, list) else [conteudo]
    for i, par in enumerate(pars):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = alinha
        p.line_spacing = espaco
        runs = par if isinstance(par, list) else [(par, {})]
        for t, op in runs:
            r = p.add_run(); r.text = t
            f = r.font
            f.name = op.get("fonte", fonte); f.size = Pt(op.get("tam", tam))
            f.bold = op.get("negrito", negrito); f.color.rgb = rgb(op.get("cor", cor))
    return tb


def fundo(slide, cor):
    ret(slide, 0, 0, W, H, cor)


def slide_conteudo(deck, etiqueta, titulo, notas):
    s = deck.slides.add_slide(deck.slide_layouts[6])
    fundo(s, "FFFFFF")
    larg = 0.16 * len(etiqueta) + 0.4
    ret(s, 1.28, 0.78, larg, 0.41, TEAL)
    texto(s, 1.28, 0.80, larg, 0.4, etiqueta, 15.75, "FFFFFF", MONO, True, PP_ALIGN.CENTER)
    ret(s, 1.28, 1.47, 0.08, 0.63, LARANJA)
    texto(s, 1.61, 1.25, 17.1, 1.1, titulo, 46, ESCURO)
    s.notes_slide.notes_text_frame.text = notas
    return s


def slide_escuro(deck, etiqueta, notas):
    s = deck.slides.add_slide(deck.slide_layouts[6])
    fundo(s, TEAL)
    ret(s, 0, 0, W, 0.2, LARANJA)
    larg = 0.16 * len(etiqueta) + 0.4
    ret(s, 1.28, 0.93, larg, 0.41, LARANJA)
    texto(s, 1.28, 0.95, larg, 0.4, etiqueta, 15.75, TEAL, MONO, True, PP_ALIGN.CENTER)
    s.notes_slide.notes_text_frame.text = notas
    return s


def card(slide, x, y, w, h, cor, titulo, corpo, tam=18):
    ret(slide, x, y, w, h, TINTA.get(cor, "EEF3F6"))
    ret(slide, x, y, 0.08, h, cor)
    texto(slide, x + 0.42, y + 0.28, w - 0.75, 0.6, titulo, 22.5, cor, negrito=True)
    texto(slide, x + 0.42, y + 0.95, w - 0.75, h - 1.1, corpo, tam, ESCURO)


def imagem(slide, caminho, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(caminho), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


def tabela(slide, x, y, w, linhas, larguras=None, tam=17, destaque_linha=None):
    n, m = len(linhas), len(linhas[0])
    alt = 0.55
    t = slide.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(alt * n)).table
    for j in range(m):
        if larguras:
            t.columns[j].width = Inches(larguras[j])
    for i, linha in enumerate(linhas):
        for j, v in enumerate(linha):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(TEAL if i == 0 else ("FBF0E1" if i == destaque_linha else ("FFFFFF" if i % 2 else "EEF3F6")))
            c.margin_left = c.margin_right = Inches(0.12)
            tf = c.text_frame
            r = tf.paragraphs[0].add_run(); r.text = str(v)
            r.font.size = Pt(tam); r.font.name = MONO if j else SANS
            r.font.bold = i == 0 or i == destaque_linha
            r.font.color.rgb = rgb("FFFFFF" if i == 0 else ESCURO)
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
    return t
