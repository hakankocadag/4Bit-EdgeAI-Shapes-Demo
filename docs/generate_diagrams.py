"""
Generate architectural and gate-level hardware diagrams for README.md:
  1. docs/fsm_state_diagram.png
  2. docs/fsm_gate_schematic.png
  3. docs/top_block_diagram.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, PathPatch
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).resolve().parent

BG = "#FFFFFF"
NAVY = "#1B365D"
BLUE_FILL = "#EBF3FA"
GREEN_FILL = "#E8F5E9"
GREEN_EDGE = "#2E7D32"
AMBER_FILL = "#FFF8E1"
AMBER_EDGE = "#EF6C00"
PURPLE_FILL = "#F3E5F5"
PURPLE_EDGE = "#6A1B9A"
SLATE_FILL = "#F4F6F8"
SLATE_EDGE = "#455A64"
WIRE = "#263238"
RAIL_COLORS = ["#1565C0", "#00838F", "#2E7D32", "#6A1B9A", "#C62828", "#EF6C00"]


def draw_box(ax, x, y, w, h, title, subtitle=None, fc=BLUE_FILL, ec=NAVY, lw=1.8, title_size=10, sub_size=8.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.0",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=ec, zorder=4)
        ax.plot([x + 0.8, x + w - 0.8], [y + h * 0.50, y + h * 0.50], color=ec, lw=0.9, alpha=0.5, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=sub_size, color="#1A1A1A", linespacing=1.28, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=ec, zorder=4)


def draw_arrow(ax, p1, p2, label=None, label_pos=None, rad=0.0, color=WIRE, lw=1.5, ls="-"):
    arr = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6.5,head_width=3.8",
        color=color, linewidth=lw, linestyle=ls, zorder=2
    )
    ax.add_patch(arr)
    if label:
        lx, ly = label_pos if label_pos else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(
            lx, ly, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CFD8DC", lw=0.8),
            zorder=5
        )


def dot(ax, x, y, color=WIRE, r=0.32):
    ax.add_patch(Circle((x, y), radius=r, facecolor=color, edgecolor="none", zorder=6))


def wire(ax, pts, color=WIRE, lw=1.25):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2)


# ============================================================================
# 1. FSM STATE DIAGRAM
# ============================================================================
def make_fsm_state_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 8.2), facecolor=BG)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(52, 63.2, "6-State One-Hot Moore FSM State Diagram (control_fsm)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(52, 60.4, "Outputs depend strictly on current state | Asynchronous active-low reset (rst_n = 0) -> S_IDLE",
            ha="center", va="center", fontsize=9.5, color="#455A64")

    draw_box(ax, 10, 38.5, 18, 12, "S_IDLE [000001]", "agu_clr = 1\nacc_clr = 1",
             fc=SLATE_FILL, ec=SLATE_EDGE)
    draw_box(ax, 42, 38.5, 18, 12, "S_FETCH [000010]", "mem_en = 1",
             fc=BLUE_FILL, ec=NAVY)
    draw_box(ax, 74, 38.5, 18, 12, "S_MAC [000100]", "mac_en = 1\nagu_inc = 1",
             fc=BLUE_FILL, ec=NAVY)

    draw_box(ax, 10, 12.5, 18, 12, "S_DONE [100000]", "done = 1",
             fc=AMBER_FILL, ec=AMBER_EDGE)
    draw_box(ax, 41.5, 12.5, 19, 12, "S_OUTPUT [010000]", "output_valid = 1\nacc_clr = 1",
             fc=GREEN_FILL, ec=GREEN_EDGE)
    draw_box(ax, 74, 12.5, 18, 12, "S_RELU [001000]", "relu_en = 1",
             fc=PURPLE_FILL, ec=PURPLE_EDGE)

    draw_arrow(ax, (2.2, 44.5), (9.7, 44.5), "rst_n = 0", (5.3, 47.6), color="#C62828", lw=1.8)
    draw_arrow(ax, (15.0, 50.8), (23.0, 50.8), "start = 0", (19.0, 56.8), rad=-0.75)
    draw_arrow(ax, (28.3, 44.5), (41.7, 44.5), "start = 1", (35.0, 46.8))
    draw_arrow(ax, (60.3, 44.5), (73.7, 44.5), "always", (67.0, 46.8))
    draw_arrow(ax, (83.0, 50.8), (51.0, 50.8), "act_idx != 783", (67.0, 57.6), rad=0.35, color="#1565C0")
    draw_arrow(ax, (83.0, 38.2), (83.0, 24.8), "act_idx == 783", (92.2, 31.5), color="#6A1B9A")
    draw_arrow(ax, (73.7, 18.5), (60.8, 18.5), "always", (67.2, 20.8))
    draw_arrow(ax, (51.0, 24.8), (51.0, 38.2), "neuron_cnt != 2", (59.5, 31.5), color="#2E7D32")
    draw_arrow(ax, (41.2, 18.5), (28.3, 18.5), "neuron_cnt == 2", (34.8, 20.8), color=AMBER_EDGE)
    draw_arrow(ax, (19.0, 24.8), (19.0, 38.2), "start = 0", (12.2, 31.5))
    draw_arrow(ax, (15.0, 12.2), (23.0, 12.2), "start = 1", (19.0, 6.2), rad=0.75)

    ax.text(
        52, 2.2,
        "Inference Latency (VECTOR_LENGTH = 784, NUM_NEURONS = 3):  "
        "1 (IDLE->FETCH) + 3 \u00d7 (784\u00d72 + 2) = 4,711 clock cycles  =  94.22 \u03bcs @ 50 MHz",
        ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECEFF1", edgecolor="#B0BEC5")
    )

    out_path = OUT_DIR / "fsm_state_diagram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# 2. GATE-LEVEL LOGIC SCHEMATIC
# ============================================================================
def draw_and_gate(ax, x, y, w=4.6, h=3.6):
    verts = [
        (x, y - h / 2),
        (x + w * 0.55, y - h / 2),
        (x + w, y - h / 2),
        (x + w, y),
        (x + w, y + h / 2),
        (x + w * 0.55, y + h / 2),
        (x, y + h / 2),
        (x, y - h / 2),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="#E3F2FD", edgecolor=NAVY, lw=1.4, zorder=4))
    return x + w, y


def draw_or_gate(ax, x, y, w=4.8, h=3.8):
    verts = [
        (x, y - h / 2),
        (x + w * 0.45, y - h / 2),
        (x + w * 0.85, y - h * 0.28),
        (x + w, y),
        (x + w * 0.85, y + h * 0.28),
        (x + w * 0.45, y + h / 2),
        (x, y + h / 2),
        (x + w * 0.28, y),
        (x, y - h / 2),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
        MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="#E8F5E9", edgecolor=GREEN_EDGE, lw=1.4, zorder=4))
    return x + w, y


def draw_not_gate(ax, x, y, w=2.2, h=2.0):
    tri = Polygon([(x, y - h / 2), (x, y + h / 2), (x + w * 0.78, y)],
                  closed=True, facecolor="#FFF3E0", edgecolor=AMBER_EDGE, lw=1.3, zorder=4)
    ax.add_patch(tri)
    bub = Circle((x + w * 0.90, y), radius=0.36, facecolor="white", edgecolor=AMBER_EDGE, lw=1.3, zorder=5)
    ax.add_patch(bub)
    return x + w + 0.22, y


def draw_dff(ax, x, y, idx, set_on_reset=False):
    w, h = 9.0, 7.6
    rect = patches.Rectangle((x, y - h / 2), w, h, facecolor="#F5F7FA", edgecolor=NAVY, lw=1.6, zorder=4)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + 1.5, f"DFF{idx}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY, zorder=5)
    ax.text(x + 1.2, y + 0.0, "D", ha="left", va="center", fontsize=8, fontweight="bold", color="#333", zorder=5)
    ax.text(x + w - 1.2, y + 0.0, f"Q{idx}", ha="right", va="center", fontsize=8, fontweight="bold", color=NAVY, zorder=5)
    clk_y = y - 2.1
    tri = Polygon([(x, clk_y - 0.7), (x, clk_y + 0.7), (x + 1.1, clk_y)],
                  closed=True, facecolor="none", edgecolor=NAVY, lw=1.1, zorder=5)
    ax.add_patch(tri)
    pin_lbl = "PRE" if set_on_reset else "CLR"
    ax.text(x + w / 2, y - h / 2 + 1.0, pin_lbl, ha="center", va="center",
            fontsize=6.8, fontweight="bold", color="#C62828", zorder=5)
    bub = Circle((x + w / 2, y - h / 2 - 0.35), radius=0.35, facecolor="white", edgecolor="#C62828", lw=1.2, zorder=5)
    ax.add_patch(bub)
    return (x, y), (x + w, y), (x, clk_y), (x + w / 2, y - h / 2 - 0.7)


def make_fsm_gate_schematic():
    fig, ax = plt.subplots(figsize=(17.0, 11.2), facecolor=BG)
    ax.set_xlim(0, 146)
    ax.set_ylim(0, 102)
    ax.axis("off")

    ax.text(73, 99.0, "Gate-Level Logic Schematic -- 6-Bit One-Hot Moore FSM (control_fsm)",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)
    ax.text(73, 96.3, "Inputs: S = start,  Ea = (act_idx == 783),  En = (neuron_count == 2)   |   State Flip-Flops: Q5..Q0 (Reset = 000001)",
            ha="center", va="center", fontsize=9.5, color="#455A64")

    for bx, bw, title in [
        (3.5, 20.0, "1. Input & Inverter Rails"),
        (24.5, 14.5, "2. State Feedback"),
        (40.0, 28.0, "3. Next-State Combinational Logic"),
        (69.5, 24.5, "4. State Register (6 DFFs)"),
        (96.0, 46.0, "5. Moore Output Decoding"),
    ]:
        ax.add_patch(patches.Rectangle((bx, 91.5), bw, 2.8, facecolor="#ECEFF1", edgecolor="#B0BEC5", lw=0.9, zorder=1))
        ax.text(bx + bw / 2, 92.9, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=NAVY)

    in_rails = {
        "S": 6.0,
        "~S": 9.2,
        "Ea": 12.2,
        "~Ea": 15.4,
        "En": 18.4,
        "~En": 21.6,
    }
    for name, rx in in_rails.items():
        if "~" not in name:
            ax.text(rx, 89.8, name, ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY)
            wire(ax, [(rx, 89.2), (rx, 10.0)], color="#37474F", lw=1.3)
        else:
            src_x = in_rails[name[1:]]
            inv_y = {"~S": 87.5, "~Ea": 85.0, "~En": 82.5}[name]
            wire(ax, [(src_x, inv_y), (src_x + 0.5, inv_y)], color="#37474F")
            dot(ax, src_x, inv_y, "#37474F")
            ox, oy = draw_not_gate(ax, src_x + 0.5, inv_y, w=2.0, h=1.8)
            wire(ax, [(ox, oy), (rx, oy), (rx, 10.0)], color=AMBER_EDGE, lw=1.2)
            ax.text(rx, 89.8, name, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AMBER_EDGE)

    q_rails = {f"Q{i}": 26.0 + i * 2.3 for i in range(6)}
    for i in range(6):
        qn = f"Q{i}"
        qx = q_rails[qn]
        col = RAIL_COLORS[i]
        ax.text(qx, 89.8, qn, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)
        wire(ax, [(qx, 89.2), (qx, 2.0 + i * 1.1)], color=col, lw=1.15)

    row_y = [83, 68, 53, 40, 27, 15]
    dff_x = 79.0
    clk_x = 71.5
    rst_x = 75.5

    ax.text(clk_x, 89.8, "clk", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=NAVY)
    ax.text(rst_x, 89.8, "rst_n", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#C62828")
    wire(ax, [(clk_x, 89.2), (clk_x, row_y[5] - 2.1)], color=NAVY, lw=1.2)
    wire(ax, [(rst_x, 89.2), (rst_x, row_y[5] - 4.8)], color="#C62828", lw=1.2)

    dff_ports = []
    for i, ry in enumerate(row_y):
        d_pt, q_pt, c_pt, r_pt = draw_dff(ax, dff_x, ry, i, set_on_reset=(i == 0))
        dff_ports.append((d_pt, q_pt))
        wire(ax, [(clk_x, c_pt[1]), c_pt], color=NAVY, lw=1.1)
        dot(ax, clk_x, c_pt[1], NAVY)
        wire(ax, [(rst_x, r_pt[1]), r_pt], color="#C62828", lw=1.1)
        dot(ax, rst_x, r_pt[1], "#C62828")

    def tap(rail_name, gy, gx_end):
        if rail_name in in_rails:
            rx = in_rails[rail_name]
            col = AMBER_EDGE if "~" in rail_name else "#37474F"
        else:
            rx = q_rails[rail_name]
            col = RAIL_COLORS[int(rail_name[1])]
        dot(ax, rx, gy, col)
        wire(ax, [(rx, gy), (gx_end, gy)], color=col, lw=1.15)

    # D0 = (Q0 + Q5) & ~S
    or0_x, or0_y = draw_or_gate(ax, 44, row_y[0] + 1.2, w=4.6, h=3.4)
    tap("Q0", row_y[0] + 2.0, 44.3)
    tap("Q5", row_y[0] + 0.4, 44.3)
    and0_x, and0_y = draw_and_gate(ax, 56, row_y[0], w=4.6, h=3.6)
    wire(ax, [(or0_x, or0_y), (53, or0_y), (53, row_y[0] + 0.9), (56, row_y[0] + 0.9)], color=WIRE)
    tap("~S", row_y[0] - 0.9, 56)
    wire(ax, [(and0_x, and0_y), dff_ports[0][0]], color=NAVY, lw=1.4)
    ax.text(63.8, row_y[0] + 1.4, "D0", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # D1 = (Q0 & S) + (Q2 & ~Ea) + (Q4 & ~En)
    a1a_x, a1a_y = draw_and_gate(ax, 44, row_y[1] + 4.5, w=4.4, h=3.2)
    tap("Q0", row_y[1] + 5.3, 44)
    tap("S", row_y[1] + 3.7, 44)
    a1b_x, a1b_y = draw_and_gate(ax, 44, row_y[1], w=4.4, h=3.2)
    tap("Q2", row_y[1] + 0.8, 44)
    tap("~Ea", row_y[1] - 0.8, 44)
    a1c_x, a1c_y = draw_and_gate(ax, 44, row_y[1] - 4.5, w=4.4, h=3.2)
    tap("Q4", row_y[1] - 3.7, 44)
    tap("~En", row_y[1] - 5.3, 44)
    or1_x, or1_y = draw_or_gate(ax, 56, row_y[1], w=5.0, h=4.8)
    wire(ax, [(a1a_x, a1a_y), (52.5, a1a_y), (52.5, row_y[1] + 1.4), (56.4, row_y[1] + 1.4)], color=WIRE)
    wire(ax, [(a1b_x, a1b_y), (56.5, a1b_y)], color=WIRE)
    wire(ax, [(a1c_x, a1c_y), (52.5, a1c_y), (52.5, row_y[1] - 1.4), (56.4, row_y[1] - 1.4)], color=WIRE)
    wire(ax, [(or1_x, or1_y), dff_ports[1][0]], color=NAVY, lw=1.4)
    ax.text(63.8, row_y[1] + 1.4, "D1", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # D2 = Q1
    tap("Q1", row_y[2], dff_ports[2][0][0])
    ax.text(63.8, row_y[2] + 1.4, "D2 = Q1", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # D3 = Q2 & Ea
    a3_x, a3_y = draw_and_gate(ax, 50, row_y[3], w=4.6, h=3.6)
    tap("Q2", row_y[3] + 0.9, 50)
    tap("Ea", row_y[3] - 0.9, 50)
    wire(ax, [(a3_x, a3_y), dff_ports[3][0]], color=NAVY, lw=1.4)
    ax.text(63.8, row_y[3] + 1.4, "D3", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # D4 = Q3
    tap("Q3", row_y[4], dff_ports[4][0][0])
    ax.text(63.8, row_y[4] + 1.4, "D4 = Q3", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # D5 = (Q4 & En) + (Q5 & S)
    a5a_x, a5a_y = draw_and_gate(ax, 44, row_y[5] + 2.5, w=4.4, h=3.2)
    tap("Q4", row_y[5] + 3.3, 44)
    tap("En", row_y[5] + 1.7, 44)
    a5b_x, a5b_y = draw_and_gate(ax, 44, row_y[5] - 2.5, w=4.4, h=3.2)
    tap("Q5", row_y[5] - 1.7, 44)
    tap("S", row_y[5] - 3.3, 44)
    or5_x, or5_y = draw_or_gate(ax, 56, row_y[5], w=4.8, h=4.0)
    wire(ax, [(a5a_x, a5a_y), (52.5, a5a_y), (52.5, row_y[5] + 1.0), (56.4, row_y[5] + 1.0)], color=WIRE)
    wire(ax, [(a5b_x, a5b_y), (52.5, a5b_y), (52.5, row_y[5] - 1.0), (56.4, row_y[5] - 1.0)], color=WIRE)
    wire(ax, [(or5_x, or5_y), dff_ports[5][0]], color=NAVY, lw=1.4)
    ax.text(63.8, row_y[5] + 1.4, "D5", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)

    # Feedback return buses
    for i in range(6):
        qx, qy = dff_ports[i][1]
        col = RAIL_COLORS[i]
        tap_x = 90.0 + i * 1.1
        bot_y = 2.0 + i * 1.1
        wire(ax, [(qx, qy), (tap_x, qy)], color=col, lw=1.2)
        dot(ax, tap_x, qy, col)
        wire(ax, [(tap_x, qy), (tap_x, bot_y), (q_rails[f"Q{i}"], bot_y)], color=col, lw=1.15)

    # Moore Output Decoding Stage
    def draw_out_badge(x, y, text, sub=""):
        txt = f"{text}  ({sub})" if sub else text
        ax.text(x, y, txt, ha="left", va="center", fontsize=9.2, fontweight="bold", color=NAVY,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="#F4F6F8", edgecolor=NAVY, lw=1.1), zorder=5)

    wire(ax, [(90.0, row_y[0]), (124.0, row_y[0])], color=RAIL_COLORS[0], lw=1.4)
    draw_out_badge(124.5, row_y[0], "agu_clr", "= Q0")

    or_acc_x, or_acc_y = draw_or_gate(ax, 110, 75.5, w=4.8, h=3.8)
    dot(ax, 100.5, row_y[0], RAIL_COLORS[0])
    wire(ax, [(100.5, row_y[0]), (100.5, 76.5), (110.3, 76.5)], color=RAIL_COLORS[0], lw=1.2)
    dot(ax, 103.5, row_y[4], RAIL_COLORS[4])
    wire(ax, [(103.5, row_y[4]), (103.5, 74.5), (110.3, 74.5)], color=RAIL_COLORS[4], lw=1.2)
    wire(ax, [(or_acc_x, or_acc_y), (124.0, or_acc_y)], color=GREEN_EDGE, lw=1.4)
    draw_out_badge(124.5, or_acc_y, "acc_clr", "= Q0 + Q4")

    wire(ax, [(91.1, row_y[1]), (124.0, row_y[1])], color=RAIL_COLORS[1], lw=1.4)
    draw_out_badge(124.5, row_y[1], "mem_en", "= Q1")

    wire(ax, [(92.2, row_y[2]), (117.0, row_y[2])], color=RAIL_COLORS[2], lw=1.4)
    dot(ax, 117.0, row_y[2], RAIL_COLORS[2])
    wire(ax, [(117.0, row_y[2]), (117.0, row_y[2] + 2.4), (124.0, row_y[2] + 2.4)], color=RAIL_COLORS[2], lw=1.4)
    wire(ax, [(117.0, row_y[2]), (117.0, row_y[2] - 2.4), (124.0, row_y[2] - 2.4)], color=RAIL_COLORS[2], lw=1.4)
    draw_out_badge(124.5, row_y[2] + 2.4, "mac_en", "= Q2")
    draw_out_badge(124.5, row_y[2] - 2.4, "agu_inc", "= Q2")

    wire(ax, [(93.3, row_y[3]), (124.0, row_y[3])], color=RAIL_COLORS[3], lw=1.4)
    draw_out_badge(124.5, row_y[3], "relu_en", "= Q3")

    wire(ax, [(94.4, row_y[4]), (124.0, row_y[4])], color=RAIL_COLORS[4], lw=1.4)
    draw_out_badge(124.5, row_y[4], "output_valid", "= Q4")

    wire(ax, [(95.5, row_y[5]), (124.0, row_y[5])], color=RAIL_COLORS[5], lw=1.4)
    draw_out_badge(124.5, row_y[5], "done", "= Q5")

    out_path = OUT_DIR / "fsm_gate_schematic.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# 3. TOP-LEVEL BLOCK DIAGRAM & DATAPATH
# ============================================================================
def make_top_block_diagram():
    fig, ax = plt.subplots(figsize=(16, 9.8), facecolor=BG)
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 80)
    ax.axis("off")

    ax.text(64, 77.2, "Top-Level Hardware Architecture & Datapath (top_accelerator)",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)
    ax.text(64, 74.4, "4-Bit Signed Weights & Activations | 24-Bit Saturating MAC + Bias | Hardware ReLU + Sequential Argmax",
            ha="center", va="center", fontsize=9.5, color="#455A64")

    # Plane background boxes
    ax.add_patch(FancyBboxPatch((15, 46), 96, 25, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#F8FAFC", edgecolor="#CFD8DC", lw=1.2, ls="--", zorder=1))
    ax.text(17, 69.2, "CONTROL PLANE", fontsize=9.5, fontweight="bold", color="#546E7A")

    ax.add_patch(FancyBboxPatch((15, 2.5), 96, 39.5, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="#FAFCFF", edgecolor="#CFD8DC", lw=1.2, ls="--", zorder=1))
    ax.text(17, 40.3, "DATAPATH & MEMORY PLANE", fontsize=9.5, fontweight="bold", color="#546E7A")

    # Control Plane blocks (control_fsm taller on top so done/output_valid route cleanly ABOVE agu!)
    draw_box(ax, 21, 48, 25, 19, "control_fsm",
             "6-State One-Hot Moore FSM\nVECTOR_LENGTH = 784\nNUM_NEURONS = 3",
             fc=AMBER_FILL, ec=AMBER_EDGE, title_size=11)

    draw_box(ax, 57, 48, 22, 13.5, "agu",
             "Address Generation Unit\nact_addr [9:0]\nweight_addr [11:0]",
             fc=BLUE_FILL, ec=NAVY, title_size=11)

    # External Inputs
    for py, pname in [(62, "clk"), (57.5, "rst_n"), (53, "start")]:
        ax.text(3.0, py, pname, ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECEFF1", edgecolor=NAVY))
        draw_arrow(ax, (10.5, py), (20.7, py), lw=1.5)

    # FSM <-> AGU connections
    draw_arrow(ax, (46.3, 57.5), (56.7, 57.5), "agu_clr, agu_inc", (51.5, 59.8))
    draw_arrow(ax, (56.7, 51.5), (46.3, 51.5), "act_addr [9:0]", (51.5, 49.2), color="#1565C0")

    # Memory blocks
    draw_box(ax, 19, 25, 22, 11, "weight_mem (ROM)",
             "memory_block (2352 x 4b)\nweights.hex",
             fc=SLATE_FILL, ec=SLATE_EDGE, title_size=10)
    draw_box(ax, 19, 8.5, 22, 11, "act_mem (RAM/ROM)",
             "memory_block (784 x 4b)\nacts.hex (28x28 pixels)",
             fc=SLATE_FILL, ec=SLATE_EDGE, title_size=10)

    # Address lines from AGU to memories (routed along y=43.8 above datapath)
    wire(ax, [(79.3, 54.5), (82.5, 54.5), (82.5, 43.8), (17.0, 43.8), (17.0, 14.0)], color="#1565C0", lw=1.4)
    ax.annotate("", xy=(18.7, 30.5), xytext=(17.0, 30.5), arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.4))
    ax.annotate("", xy=(18.7, 14.0), xytext=(17.0, 14.0), arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.4))
    ax.text(83.2, 50.0, "weight_addr [11:0]\n& act_addr [9:0]", ha="left", va="center", fontsize=8.2,
            fontweight="bold", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CFD8DC"))

    # mem_en from FSM to both memories
    wire(ax, [(28.0, 47.7), (28.0, 36.3)], color=AMBER_EDGE, lw=1.4)
    ax.annotate("", xy=(28.0, 36.3), xytext=(28.0, 39.5), arrowprops=dict(arrowstyle="-|>", color=AMBER_EDGE, lw=1.4))
    ax.text(28.0, 42.8, "mem_en", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CFD8DC"))
    wire(ax, [(30.0, 24.7), (30.0, 19.8)], color=AMBER_EDGE, lw=1.3)
    ax.annotate("", xy=(30.0, 19.8), xytext=(30.0, 23.0), arrowprops=dict(arrowstyle="-|>", color=AMBER_EDGE, lw=1.3))

    # MAC Unit
    draw_box(ax, 49, 17, 20, 13, "mac_unit",
             "4b x 4b Signed Mult\n24b Saturating Accum.\nmac_out [23:0]",
             fc=BLUE_FILL, ec=NAVY, title_size=10.5)

    draw_arrow(ax, (41.3, 30.5), (48.7, 26.5), "weight [3:0]", (45.0, 31.8))
    draw_arrow(ax, (41.3, 14.0), (48.7, 20.5), "act [3:0]", (45.0, 13.8))

    # mac_en, acc_clr from FSM to mac_unit
    wire(ax, [(36.0, 47.7), (36.0, 34.5), (56.0, 34.5)], color=AMBER_EDGE, lw=1.3)
    draw_arrow(ax, (56.0, 34.5), (56.0, 30.3), "mac_en, acc_clr", (47.5, 36.4), color=AMBER_EDGE)

    # Bias memory & Adder
    draw_box(ax, 50, 4.0, 18, 9, "bias_mem",
             "3 x 24b Signed\nbiases.hex",
             fc=SLATE_FILL, ec=SLATE_EDGE, title_size=10)

    # neuron_count[1:0] from FSM to bias_mem
    wire(ax, [(40.5, 47.7), (40.5, 44.8), (74.2, 44.8), (74.2, 8.5), (68.3, 8.5)], color="#6A1B9A", lw=1.2)
    ax.annotate("", xy=(68.3, 8.5), xytext=(71.0, 8.5), arrowprops=dict(arrowstyle="-|>", color="#6A1B9A", lw=1.2))
    ax.text(74.6, 14.5, "neuron_count\n[1:0]", ha="left", va="center", fontsize=8, fontweight="bold", color="#6A1B9A",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CFD8DC"))

    # Adder circle (+)
    add_c = Circle((76, 23.5), radius=2.6, facecolor="#FFF8E1", edgecolor=AMBER_EDGE, lw=1.8, zorder=4)
    ax.add_patch(add_c)
    ax.text(76, 23.5, "+", ha="center", va="center", fontsize=15, fontweight="bold", color=AMBER_EDGE, zorder=5)

    draw_arrow(ax, (69.3, 23.5), (73.2, 23.5))
    wire(ax, [(68.3, 6.2), (76.0, 6.2), (76.0, 20.7)], color=WIRE, lw=1.4)
    ax.annotate("", xy=(76.0, 20.7), xytext=(76.0, 18.0), arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.4))
    ax.text(72.0, 4.6, "bias [23:0]", fontsize=8, fontweight="bold", color=WIRE)

    # ReLU Unit
    draw_box(ax, 82, 18, 13, 11, "relu_unit",
             "24b MUX\nmax(0, z)",
             fc=PURPLE_FILL, ec=PURPLE_EDGE, title_size=10)
    draw_arrow(ax, (78.8, 23.5), (81.7, 23.5))

    # relu_en routed cleanly below agu along y=42.8 -> x=88.5 -> relu_unit
    wire(ax, [(44.0, 47.7), (44.0, 42.6), (88.5, 42.6)], color=AMBER_EDGE, lw=1.25)
    draw_arrow(ax, (88.5, 42.6), (88.5, 29.3), "relu_en", (88.5, 36.2), color=AMBER_EDGE)

    # Argmax Unit
    draw_box(ax, 98.5, 18, 12, 11, "argmax_unit",
             "Sequential\nMax Index",
             fc=GREEN_FILL, ec=GREEN_EDGE, title_size=9.5)

    # final_result[23:0] split to output & argmax
    wire(ax, [(95.3, 23.5), (98.2, 23.5)], color=WIRE, lw=1.5)
    ax.annotate("", xy=(98.2, 23.5), xytext=(96.5, 23.5), arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.5))
    dot(ax, 96.6, 23.5, WIRE)
    wire(ax, [(96.6, 23.5), (96.6, 10.5), (114.5, 10.5)], color=NAVY, lw=1.5)
    ax.annotate("", xy=(114.5, 10.5), xytext=(111.5, 10.5), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.5))

    def draw_out_pin(x, y, name):
        ax.text(x, y, name, ha="left", va="center", fontsize=9.5, fontweight="bold", color=GREEN_EDGE,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=GREEN_FILL, edgecolor=GREEN_EDGE))

    # done & output_valid routed ABOVE agu (y=65.5 and y=63.0) so no wire passes behind agu!
    wire(ax, [(46.3, 65.8), (106.5, 65.8)], color=GREEN_EDGE, lw=1.4)
    draw_arrow(ax, (106.5, 65.8), (114.5, 65.8), "done", (110.5, 67.6), color=GREEN_EDGE)
    dot(ax, 106.5, 65.8, GREEN_EDGE)
    wire(ax, [(106.5, 65.8), (106.5, 29.3)], color=GREEN_EDGE, lw=1.2)
    ax.annotate("", xy=(106.5, 29.3), xytext=(106.5, 34.0), arrowprops=dict(arrowstyle="-|>", color=GREEN_EDGE, lw=1.2))
    ax.text(107.2, 36.2, "done_in", ha="left", fontsize=8, fontweight="bold", color=GREEN_EDGE)

    wire(ax, [(46.3, 63.2), (101.2, 63.2)], color=GREEN_EDGE, lw=1.4)
    draw_arrow(ax, (101.2, 63.2), (114.5, 63.2), "output_valid", (109.5, 61.2), color=GREEN_EDGE)
    dot(ax, 101.2, 63.2, GREEN_EDGE)
    wire(ax, [(101.2, 63.2), (101.2, 29.3)], color=GREEN_EDGE, lw=1.2)
    ax.annotate("", xy=(101.2, 29.3), xytext=(101.2, 34.0), arrowprops=dict(arrowstyle="-|>", color=GREEN_EDGE, lw=1.2))
    ax.text(100.5, 33.5, "score_valid", ha="right", fontsize=8, fontweight="bold", color=GREEN_EDGE)

    draw_arrow(ax, (110.8, 23.5), (114.5, 23.5), lw=1.6, color=GREEN_EDGE)

    draw_out_pin(115.0, 65.8, "done")
    draw_out_pin(115.0, 63.2, "output_valid")
    draw_out_pin(115.0, 23.5, "predicted_class [1:0]")
    draw_out_pin(115.0, 10.5, "final_result [23:0]")

    out_path = OUT_DIR / "top_block_diagram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_fsm_state_diagram()
    make_fsm_gate_schematic()
    make_top_block_diagram()
