import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def objective(x, y):
    return -((x - 1.5) ** 2 + 0.5 * (y + 0.5) ** 2) + 0.8 * np.sin(2 * x) * np.cos(y)

rng = np.random.default_rng(1)

# Simulated optimizer state
steps = 12
base_step = 0.5
shrink_iters = {10, 5}  # simulate shrink events; recenter on best then
trajectory = []
best_so_far = None
curr = np.array([-3.0, -3.0])
cand_scores = []
for i in range(steps):
    # shrink logic
    if i in shrink_iters:
        base_step *= 0.5
        curr = best_so_far.copy()  # recenter on best found so far

    # propose neighbors around current
    cand = curr + rng.normal(scale=base_step, size=(20, 2))
    adj = cand + rng.normal(scale=0.5 * base_step, size=(20, 2))
    scores = np.array([objective(x, y) for x, y in cand])
    peak_idx = scores.argmax()
    peak = cand[peak_idx]

    cand_scores.append(scores)
    curr = peak.copy()
    trajectory.append((cand, adj, peak, curr, base_step))
    if (best_so_far is None) or (objective(*curr) > objective(*best_so_far)):
        best_so_far = curr.copy()

# Build arrays for plotting accepted path
accepted = np.array([t[3] for t in trajectory])
values = np.array([objective(x, y) for x, y in accepted])
running_best = np.maximum.accumulate(values)

# Colors
colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory)))

# Plot
fig = plt.figure(figsize=(8, 11))
fig.suptitle("Sampling, moves, and objective trend", fontsize=14, y=0.98)
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0])
ax = fig.add_subplot(gs[0])

xs = np.linspace(-5, 2.0, 220)
ys = np.linspace(-5, 2.0, 220)
X, Y = np.meshgrid(xs, ys)
Z = objective(X, Y)
cs = ax.contourf(X, Y, Z, levels=40, cmap="viridis")
ax.contour(X, Y, Z, levels=20, colors="k", alpha=0.12, linewidths=0.5)
fig.colorbar(cs, ax=ax, label="Objective (higher is better)")

for i, (cand, adj, peak, curr, _) in enumerate(trajectory):
    c = colors[i]
    ax.scatter(adj[:, 0], adj[:, 1], s=14, color=c, alpha=0.25, edgecolors="none")
    ax.scatter(cand[:, 0], cand[:, 1], s=28, color=c, alpha=0.65, edgecolors="k", linewidths=0.4)
    ax.scatter(peak[0], peak[1], s=70, color="#ff7043", edgecolors="k", linewidths=0.8, zorder=5)
    ax.plot(accepted[: i + 1, 0], accepted[: i + 1, 1], "-o", color="#ff7043", lw=2, ms=6)
    if i > 0:
        ax.add_patch(FancyArrowPatch(accepted[i - 1], accepted[i],
                                     arrowstyle="->", color="#ff7043", lw=1.8, mutation_scale=12))

# Final best marker
ax.scatter(best_so_far[0], best_so_far[1], s=110, facecolor="#ffd54f", edgecolor="k", linewidths=1.2, zorder=6, label="Best so far")

ax.set_xlabel("Parameter x")
ax.set_ylabel("Parameter y")
ax.set_title("Neighbor sets and accepted path")
ax.legend(loc="lower right")

ax1 = fig.add_subplot(gs[1])
steps_idx = np.arange(len(values))
# ax1.scatter(steps_idx - 0.05, values, s=46, marker="s", facecolors="none", edgecolors="#42a5f5",
#             linewidths=1.4, alpha=0.9, zorder=2, label="Accepted (raw)")
ax1.plot(steps_idx, running_best, "-o", color="#3949ab", lw=2.4, ms=7, zorder=3, label="Running best")

# Show all neighbor scores per iteration (faint gray scatter)
rng_plot = np.random.default_rng(123)
for i, sc in enumerate(cand_scores):
    jitter_x = i + rng_plot.normal(scale=0.035, size=len(sc))
    ax1.scatter(jitter_x, sc, s=10, color="#9e9e9e", alpha=0.9, edgecolors="none", zorder=1)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Objective")
ax1.set_title("Objective per iteration")
ax1.grid(alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.savefig("optimizer_path.png", dpi=220)
plt.show()