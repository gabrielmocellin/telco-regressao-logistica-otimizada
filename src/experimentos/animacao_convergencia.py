"""
Gera animações (.gif e .mp4) comparando a convergência por iteração do
Método de Cauchy e do Método de Newton Puro (Experimento 1).

Uso direto:
    python -m src.experimentos.animacao_convergencia
"""
from pathlib import Path
import shutil
import subprocess
from typing import List, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.otimizacao.metodo_cauchy import otimizar_metodo_cauchy
from src.otimizacao.metodo_newton import otimizar_metodo_newton
from src.otimizacao.resultado import ResultadoOtimizacao
from src.preprocessamento.processamento_dados import (
    carregar_dados_brutos,
    carregar_variaveis_json,
    preparar_treino_teste,
)


PASTA_GRAFICOS = Path("docs") / "imgs" / "experimentos"
COR_CAUCHY = "#1f77b4"
COR_NEWTON = "#ff7f0e"
COR_TOL = "#d62728"
COR_OK = "#1e8e3e"


def _construir_cronograma_quadros(
    iter_newton: int,
    iter_cauchy: int,
    zoom_inicial: float = 14.0,
) -> List[Tuple[int, float, int]]:
    """
    Monta a sequência de quadros únicos (k_atual, xlim_max, repeticoes) em dois ritmos:
      1. Início lento (1 iteração por etapa) com zoom em [0, zoom_inicial], destacando
         a convergência quadrática de Newton e o zigue-zague inicial de Cauchy.
      2. Aceleração progressiva com expansão suave do eixo X até Cauchy convergir.

    Guardar `repeticoes` evita re-renderizar quadros idênticos nas pausas e reduz
    drasticamente o tempo de geração e o tamanho do arquivo GIF.
    """
    cronograma: List[Tuple[int, float, int]] = []
    xlim_zoom = max(zoom_inicial, float(iter_newton + 6))

    # Pausa inicial em k = 0 (ponto de partida comum w^0 = 0)
    cronograma.append((0, xlim_zoom, 10))

    # Avanço lento nas iterações de Newton (k = 1 .. iter_newton - 1)
    for k in range(1, iter_newton):
        cronograma.append((k, xlim_zoom, 8))

    # Pausa dramática no instante em que Newton converge
    cronograma.append((iter_newton, xlim_zoom, 34))

    # Zigue-zague inicial de Cauchy ainda dentro da janela de zoom
    fim_zoom_k = min(iter_cauchy, int(xlim_zoom))
    for k in range(iter_newton + 1, fim_zoom_k + 1):
        cronograma.append((k, xlim_zoom, 3))

    # Aceleração suave com zoom-out dinâmico até Cauchy convergir
    xlim_final = max(xlim_zoom, iter_cauchy * 1.055)
    if iter_cauchy > fim_zoom_k:
        n_frames_acel = 88
        t = np.linspace(0.0, 1.0, n_frames_acel)
        progresso = 0.25 * t + 0.75 * (t ** 1.8)
        ks_float = fim_zoom_k + progresso * (iter_cauchy - fim_zoom_k)

        for kf in ks_float[:-1]:
            k_int = min(iter_cauchy - 1, max(fim_zoom_k + 1, int(round(kf))))
            xlim_dyn = min(xlim_final, max(xlim_zoom, kf * 1.055 + 1.0))
            cronograma.append((k_int, xlim_dyn, 1))

    # Pausa final no estado completo antes de reiniciar o loop
    cronograma.append((iter_cauchy, xlim_final, 56))
    return cronograma


def _formatar_card(
    titulo: str,
    k_exibido: int,
    perda: float,
    grad: float,
    convergiu_agora: bool,
) -> str:
    status = f"CONVERGIU (k = {k_exibido})" if convergiu_agora else "OTIMIZANDO..."
    return (
        f"{titulo}  •  {status}\n"
        f"Iteração: {k_exibido:3d}   |   f(w): {perda:.6f}   |   ||∇f(w)||: {grad:.2e}"
    )


def _capturar_quadro_rgb(fig: plt.Figure) -> np.ndarray:
    """Extrai o buffer RGB (H, W, 3) diretamente do canvas Agg da figura."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def _salvar_gif(
    quadros_rgb: List[np.ndarray],
    repeticoes: List[int],
    caminho_gif: Path,
    fps: int,
    largura_max: int = 1080,
) -> None:
    """
    Salva GIF otimizado com paleta global compartilhada, sem dithering (ideal para
    gráficos vetoriais com fundo branco) e durações variáveis por quadro.
    """
    ms_por_frame = int(round(1000.0 / fps))
    duracoes_ms = [rep * ms_por_frame for rep in repeticoes]

    altura_orig, largura_orig, _ = quadros_rgb[0].shape
    if largura_orig > largura_max:
        escala = largura_max / largura_orig
        tamanho_alvo = (largura_max, int(round(altura_orig * escala)))
        imagens_rgb = [
            Image.fromarray(arr).resize(tamanho_alvo, Image.Resampling.LANCZOS)
            for arr in quadros_rgb
        ]
    else:
        imagens_rgb = [Image.fromarray(arr) for arr in quadros_rgb]

    # Paleta global (256 cores via FastOctree preserva tons raros e vivos como vermelho/verde)
    w, h = imagens_rgb[0].size
    amostra = Image.new("RGB", (w, h * 2))
    amostra.paste(imagens_rgb[0], (0, 0))
    amostra.paste(imagens_rgb[-1], (0, h))
    paleta_global = amostra.quantize(colors=256, method=Image.Quantize.FASTOCTREE)

    imagens_pil = [
        img.quantize(palette=paleta_global, dither=Image.Dither.NONE)
        for img in imagens_rgb
    ]
    imagens_pil[0].save(
        caminho_gif,
        save_all=True,
        append_images=imagens_pil[1:],
        duration=duracoes_ms,
        loop=0,
        optimize=True,
    )


def _salvar_mp4(
    quadros_rgb: List[np.ndarray],
    repeticoes: List[int],
    caminho_mp4: Path,
    fps: int,
) -> None:
    """Codifica MP4 (H.264 / yuv420p) via ffmpeg para compatibilidade total em slides."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg não encontrado no PATH; não foi possível gerar o arquivo .mp4."
        )

    altura, largura, _ = quadros_rgb[0].shape
    comando = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{largura}x{altura}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(caminho_mp4),
    ]

    proc = subprocess.Popen(
        comando,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for arr, rep in zip(quadros_rgb, repeticoes):
            raw = arr.tobytes()
            for _ in range(rep):
                proc.stdin.write(raw)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        codigo = proc.wait()
        if codigo != 0:
            raise RuntimeError(f"Erro ao gerar MP4 com ffmpeg (código {codigo}):\n{stderr}")
    finally:
        if proc.poll() is None:
            proc.kill()


def gerar_animacao_convergencia(
    res_cauchy: ResultadoOtimizacao,
    res_newton: ResultadoOtimizacao,
    pasta_saida: Path = PASTA_GRAFICOS,
    nome_base: str = "01_exp1_convergencia_animacao",
    tolerancia: float = 1e-6,
    fps: int = 18,
    dpi: int = 110,
    gerar_gif: bool = True,
    gerar_mp4: bool = True,
    pasta_preview: Path | None = None,
) -> List[Path]:
    """
    Gera a animação da corrida de convergência entre Cauchy e Newton Puro
    em formato .gif e/ou .mp4.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)

    perda_c = np.asarray(res_cauchy.historico_perda, dtype=float)
    grad_c = np.asarray(res_cauchy.historico_norma_gradiente, dtype=float)
    perda_n = np.asarray(res_newton.historico_perda, dtype=float)
    grad_n = np.asarray(res_newton.historico_norma_gradiente, dtype=float)

    iter_c = len(perda_c) - 1
    iter_n = len(perda_n) - 1
    xs_c = np.arange(len(perda_c))
    xs_n = np.arange(len(perda_n))
    razao_str = f"{iter_c / max(iter_n, 1):.1f}".replace(".", ",")

    cronograma = _construir_cronograma_quadros(iter_n, iter_c)

    # 12.8 x 6.4 in @ 110 dpi = 1408 x 704 px (múltiplos de 16 -> ótimo para H.264)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=dpi, facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.695, bottom=0.105, wspace=0.22)

    # Cabeçalho e placar ao vivo
    fig.text(
        0.5,
        0.952,
        "Convergência por Iteração: Método de Cauchy vs. Newton Puro",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1a1a1a",
    )

    badge_iter = fig.text(
        0.5,
        0.893,
        "Iteração k = 0",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#2c3e50", edgecolor="none"),
    )

    card_cauchy = fig.text(
        0.265,
        0.812,
        "",
        ha="center",
        va="center",
        fontsize=9.5,
        fontfamily="DejaVu Sans Mono",
        color="#103b5c",
        bbox=dict(
            boxstyle="round,pad=0.48",
            facecolor="#eef6fc",
            edgecolor=COR_CAUCHY,
            linewidth=1.5,
        ),
    )

    card_newton = fig.text(
        0.735,
        0.812,
        "",
        ha="center",
        va="center",
        fontsize=9.5,
        fontfamily="DejaVu Sans Mono",
        color="#6e3200",
        bbox=dict(
            boxstyle="round,pad=0.48",
            facecolor="#fff6ec",
            edgecolor=COR_NEWTON,
            linewidth=1.5,
        ),
    )

    # ---------------- Subplot 1: Função Objetivo f(w^k) ----------------
    f_otimo = float(min(perda_c[-1], perda_n[-1]))
    y1_min = f_otimo - 0.015
    y1_max = float(max(perda_c[0], perda_n[0])) + 0.02

    ax1.axhline(
        f_otimo,
        color="#7f8c8d",
        linestyle=":",
        linewidth=1.2,
        label=f"Ótimo f(w*) ≈ {f_otimo:.4f}",
        zorder=1,
    )
    (linha_c1,) = ax1.plot([], [], color=COR_CAUCHY, linewidth=2.2, label="Cauchy (1ª ordem)", zorder=3)
    (linha_n1,) = ax1.plot(
        [], [], "o-", color=COR_NEWTON, linewidth=2.4, markersize=5.5, label="Newton Puro (2ª ordem)", zorder=4
    )
    (ponto_c1,) = ax1.plot(
        [], [], "o", color=COR_CAUCHY, markersize=8, markeredgecolor="white", markeredgewidth=1.4, zorder=5
    )
    (ponto_n1,) = ax1.plot(
        [], [], "o", color=COR_NEWTON, markersize=8.5, markeredgecolor="white", markeredgewidth=1.4, zorder=6
    )
    (estrela_n1,) = ax1.plot(
        [], [], "*", color=COR_OK, markersize=15, markeredgecolor="white", markeredgewidth=1.0, zorder=7
    )
    (estrela_c1,) = ax1.plot(
        [], [], "*", color=COR_CAUCHY, markersize=15, markeredgecolor="white", markeredgewidth=1.0, zorder=7
    )

    ax1.set_ylim(y1_min, y1_max)
    ax1.set_title("Função Objetivo f(w^k) por Iteração", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Iteração (k)", fontsize=10.5)
    ax1.set_ylabel("Perda f(w^k)", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ---------------- Subplot 2: Norma do Gradiente ||∇f(w^k)|| ----------------
    y2_min = min(float(grad_n.min()), tolerancia) * 0.35
    y2_max = max(float(grad_c[0]), float(grad_n[0])) * 2.5

    ax2.axhline(
        tolerancia,
        color=COR_TOL,
        linestyle="--",
        linewidth=1.4,
        label=f"Tolerância ({tolerancia:.0e})",
        zorder=2,
    )
    (linha_c2,) = ax2.semilogy([], [], color=COR_CAUCHY, linewidth=2.0, label="Cauchy (1ª ordem)", zorder=3)
    (linha_n2,) = ax2.semilogy(
        [], [], "o-", color=COR_NEWTON, linewidth=2.4, markersize=5.5, label="Newton Puro (2ª ordem)", zorder=4
    )
    (ponto_c2,) = ax2.semilogy(
        [], [], "o", color=COR_CAUCHY, markersize=8, markeredgecolor="white", markeredgewidth=1.4, zorder=5
    )
    (ponto_n2,) = ax2.semilogy(
        [], [], "o", color=COR_NEWTON, markersize=8.5, markeredgecolor="white", markeredgewidth=1.4, zorder=6
    )
    (estrela_n2,) = ax2.semilogy(
        [], [], "*", color=COR_OK, markersize=15, markeredgecolor="white", markeredgewidth=1.0, zorder=7
    )
    (estrela_c2,) = ax2.semilogy(
        [], [], "*", color=COR_CAUCHY, markersize=15, markeredgecolor="white", markeredgewidth=1.0, zorder=7
    )

    ax2.set_ylim(y2_min, y2_max)
    ax2.set_title("Norma do Gradiente ||∇f(w^k)|| (Escala Log)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Iteração (k)", fontsize=10.5)
    ax2.set_ylabel("||∇f(w^k)||", fontsize=10.5)
    ax2.grid(True, which="both", linestyle="--", alpha=0.30)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # Anotações de destaque quando cada método converge
    anot_newton = ax2.annotate(
        f"Newton convergiu!\nk = {iter_n}  (||∇f|| = {grad_n[-1]:.2e})",
        xy=(iter_n, grad_n[-1]),
        xycoords="data",
        xytext=(0.48, 0.045),
        textcoords="axes fraction",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#145a28",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#e8f7ec", edgecolor=COR_OK, linewidth=1.3),
        arrowprops=dict(arrowstyle="->", color=COR_OK, lw=1.6, connectionstyle="arc3,rad=-0.10"),
        visible=False,
        zorder=8,
    )

    anot_cauchy = ax2.annotate(
        f"Cauchy convergiu!\nk = {iter_c}  ({razao_str}× mais iter.)",
        xy=(iter_c, grad_c[-1]),
        xycoords="data",
        xytext=(0.45, 0.56),
        textcoords="axes fraction",
        fontsize=9.2,
        fontweight="bold",
        color="#103b5c",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#eef6fc", edgecolor=COR_CAUCHY, linewidth=1.3),
        arrowprops=dict(arrowstyle="->", color=COR_CAUCHY, lw=1.6, connectionstyle="arc3,rad=-0.15"),
        visible=False,
        zorder=8,
    )

    def atualizar_estado(k: int, xlim_max: float) -> None:
        kc = min(k, iter_c)
        kn = min(k, iter_n)
        newton_ok = k >= iter_n
        cauchy_ok = k >= iter_c

        linha_c1.set_data(xs_c[: kc + 1], perda_c[: kc + 1])
        linha_n1.set_data(xs_n[: kn + 1], perda_n[: kn + 1])
        linha_c2.set_data(xs_c[: kc + 1], grad_c[: kc + 1])
        linha_n2.set_data(xs_n[: kn + 1], grad_n[: kn + 1])

        if cauchy_ok:
            ponto_c1.set_data([], [])
            ponto_c2.set_data([], [])
            estrela_c1.set_data([iter_c], [perda_c[-1]])
            estrela_c2.set_data([iter_c], [grad_c[-1]])
        else:
            ponto_c1.set_data([kc], [perda_c[kc]])
            ponto_c2.set_data([kc], [grad_c[kc]])
            estrela_c1.set_data([], [])
            estrela_c2.set_data([], [])

        if newton_ok:
            ponto_n1.set_data([], [])
            ponto_n2.set_data([], [])
            estrela_n1.set_data([iter_n], [perda_n[-1]])
            estrela_n2.set_data([iter_n], [grad_n[-1]])
        else:
            ponto_n1.set_data([kn], [perda_n[kn]])
            ponto_n2.set_data([kn], [grad_n[kn]])
            estrela_n1.set_data([], [])
            estrela_n2.set_data([], [])

        x_esq = -max(0.4, xlim_max * 0.03)
        ax1.set_xlim(x_esq, xlim_max)
        ax2.set_xlim(x_esq, xlim_max)

        anot_newton.set_visible(newton_ok)
        anot_cauchy.set_visible(cauchy_ok)

        badge_iter.set_text(f"Iteração k = {k:3d}")

        card_cauchy.set_text(
            _formatar_card("Cauchy", kc, perda_c[kc], grad_c[kc], cauchy_ok)
        )
        if cauchy_ok:
            card_cauchy.get_bbox_patch().set_facecolor("#e8f7ec")
            card_cauchy.get_bbox_patch().set_edgecolor(COR_OK)
            card_cauchy.set_color("#145a28")
        else:
            card_cauchy.get_bbox_patch().set_facecolor("#eef6fc")
            card_cauchy.get_bbox_patch().set_edgecolor(COR_CAUCHY)
            card_cauchy.set_color("#103b5c")

        card_newton.set_text(
            _formatar_card("Newton Puro", kn, perda_n[kn], grad_n[kn], newton_ok)
        )
        if newton_ok:
            card_newton.get_bbox_patch().set_facecolor("#e8f7ec")
            card_newton.get_bbox_patch().set_edgecolor(COR_OK)
            card_newton.set_color("#145a28")
        else:
            card_newton.get_bbox_patch().set_facecolor("#fff6ec")
            card_newton.get_bbox_patch().set_edgecolor(COR_NEWTON)
            card_newton.set_color("#6e3200")

    # Renderiza cada quadro único uma única vez
    quadros_rgb: List[np.ndarray] = []
    repeticoes: List[int] = []
    ks_preview = {0, iter_n, 14, iter_c}
    previews_salvos = set()

    if pasta_preview is not None:
        pasta_preview.mkdir(parents=True, exist_ok=True)

    for k, xlim_max, rep in cronograma:
        atualizar_estado(k, xlim_max)
        rgb = _capturar_quadro_rgb(fig)
        quadros_rgb.append(rgb)
        repeticoes.append(rep)

        if pasta_preview is not None and k in ks_preview and k not in previews_salvos:
            Image.fromarray(rgb).save(pasta_preview / f"preview_v2_k{k:03d}.png")
            previews_salvos.add(k)

    plt.close(fig)

    arquivos_gerados: List[Path] = []
    if gerar_gif:
        caminho_gif = pasta_saida / f"{nome_base}.gif"
        _salvar_gif(quadros_rgb, repeticoes, caminho_gif, fps=fps)
        arquivos_gerados.append(caminho_gif)

    if gerar_mp4:
        if shutil.which("ffmpeg") is not None:
            caminho_mp4 = pasta_saida / f"{nome_base}.mp4"
            _salvar_mp4(quadros_rgb, repeticoes, caminho_mp4, fps=fps)
            arquivos_gerados.append(caminho_mp4)
        else:
            print(
                "[Aviso] ffmpeg não encontrado no PATH; "
                "a animação foi salva apenas em .gif."
            )

    return arquivos_gerados


def main() -> None:
    X_tr_df, _, y_tr_s, _ = preparar_treino_teste(
        carregar_dados_brutos(), carregar_variaveis_json()
    )
    X_treino = X_tr_df.to_numpy(dtype=float)
    y_treino = y_tr_s.to_numpy(dtype=float)

    res_cauchy = otimizar_metodo_cauchy(X_treino, y_treino, tolerancia_gradiente=1e-6)
    res_newton = otimizar_metodo_newton(
        X_treino, y_treino, tolerancia_gradiente=1e-6, maximo_iteracoes=1000
    )

    arquivos = gerar_animacao_convergencia(res_cauchy, res_newton)
    print("\nAnimações salvas em:")
    for arq in arquivos:
        tamanho_kb = arq.stat().st_size / 1024.0
        print(f"  - {arq} ({tamanho_kb:.1f} KB)")


if __name__ == "__main__":
    main()
