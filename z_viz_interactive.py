"""Generate interactive Plotly HTML for z-distribution visualization on 3D sphere."""
import argparse, os, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def make_sphere_mesh(r, n=30):
    """Create wireframe sphere coordinates."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def build_figure(npz_path, output_html):
    data = np.load(npz_path)
    r = math.sqrt(3)  # z_dim=3, sphere radius = sqrt(3)

    colors = {"Q1": "blue", "Q2": "orange", "Q3": "green"}
    stages = ["S1", "S2", "S3"]
    quads = ["Q1", "Q2", "Q3"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"After Stage {i+1}" for i in range(3)],
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.02,
    )

    for col, stage in enumerate(stages, 1):
        # Add wireframe sphere
        sx, sy, sz = make_sphere_mesh(r, n=20)
        for i in range(sx.shape[0]):
            fig.add_trace(go.Scatter3d(
                x=sx[i], y=sy[i], z=sz[i],
                mode="lines", line=dict(color="lightgray", width=0.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)
        for j in range(sx.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=sx[:, j], y=sy[:, j], z=sz[:, j],
                mode="lines", line=dict(color="lightgray", width=0.5),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

        # Add z-vectors for each quadrant
        for q in quads:
            key_z = f"{stage}_{q}_zs"
            key_g = f"{stage}_{q}_goals"
            if key_z not in data:
                continue
            zs = data[key_z]   # (N, 3)
            goals = data[key_g]  # (N, 2)

            hover_text = [f"{q} goal=({g[0]:.2f}, {g[1]:.2f})<br>z=({z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f})"
                         for g, z in zip(goals, zs)]

            fig.add_trace(go.Scatter3d(
                x=zs[:, 0], y=zs[:, 1], z=zs[:, 2],
                mode="markers",
                marker=dict(size=4, color=colors[q], opacity=0.8,
                           line=dict(width=0.5, color="black")),
                name=f"{q}" if col == 1 else None,
                showlegend=(col == 1),
                legendgroup=q,
                text=hover_text,
                hoverinfo="text",
            ), row=1, col=col)

    # Configure 3D scenes
    scene_cfg = dict(
        xaxis=dict(range=[-r*1.1, r*1.1], title="z₁"),
        yaxis=dict(range=[-r*1.1, r*1.1], title="z₂"),
        zaxis=dict(range=[-r*1.1, r*1.1], title="z₃"),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    fig.update_layout(
        scene=scene_cfg, scene2=scene_cfg, scene3=scene_cfg,
        title=dict(text="z-Distribution on Sphere (z_dim=3) — Q1🔵 Q2🟠 Q3🟢",
                   font=dict(size=16)),
        height=600, width=1600,
        legend=dict(x=0.01, y=0.99, font=dict(size=14)),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(output_html, include_plotlyjs=True)
    print(f"saved {output_html}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="path to z_viz_data.npz")
    p.add_argument("--output", default=None, help="output HTML path")
    args = p.parse_args()

    output = args.output or args.npz.replace(".npz", ".html")
    build_figure(args.npz, output)


if __name__ == "__main__":
    main()
