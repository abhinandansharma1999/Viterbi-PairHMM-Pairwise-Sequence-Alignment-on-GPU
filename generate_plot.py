import matplotlib.pyplot as plt

O          = [12,      20,      28,      56,      64]
runtime    = [17.9611, 17.9638, 17.9703, 17.9736, 17.9743]
score_loss = [1.499,   0.713,   0.403,   0.1832,  0.180]

fig, ax1 = plt.subplots(figsize=(8, 5))

color_runtime = "#4f86c6"
color_loss    = "#e07b39"

ax1.plot(O, runtime, color=color_runtime, marker='o', linewidth=2, label="Runtime (s)")
ax1.set_xlabel("Overlap Size (O)", fontsize=12)
ax1.set_ylabel("Runtime (s)", color=color_runtime, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_runtime)
ax1.set_ylim(16, 19)

# Annotate runtime points
for x, y in zip(O, runtime):
    ax1.annotate(f"({x}, {y})", xy=(x, y), xytext=(0, 8),
                 textcoords="offset points", ha='center', fontsize=8,
                 color=color_runtime)

ax2 = ax1.twinx()
ax2.plot(O, score_loss, color=color_loss, marker='s', linewidth=2, label="Avg Score Loss (%)")
ax2.set_ylabel("Avg Score Loss (%)", color=color_loss, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_loss)
ax2.set_ylim(0, 1.6)

# Annotate score loss points
for x, y in zip(O, score_loss):
    ax2.annotate(f"({x}, {y})", xy=(x, y), xytext=(0, -16),
                 textcoords="offset points", ha='center', fontsize=8,
                 color=color_loss)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

plt.title("Effect of Overlap Size (O) on Runtime and Score Loss\n(T=200, maxPairs=5000)", fontsize=13)
plt.xticks(O)
plt.tight_layout()
# plt.savefig("/mnt/user-data/outputs/overlap_annotated.png", dpi=150)
plt.show()