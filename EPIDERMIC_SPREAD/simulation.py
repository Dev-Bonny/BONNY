"""
Agent-Based SIR Disease Transmission Model
==========================================
Full-featured simulation with the following extensions:
  1. Agent heterogeneity (varied susceptibility & recovery times)
  2. Vaccination (portion of agents start immune)
  3. Quarantine/isolation (infected agents stop moving)
  4. Social distancing (reduced movement probability)
  5. Stochastic infection variability

Usage:
    python sir_simulation.py

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random
import argparse


# ─────────────────────────────────────────────
# Agent Definition
# ─────────────────────────────────────────────

SUSCEPTIBLE = "S"
INFECTED    = "I"
RECOVERED   = "R"
VACCINATED  = "V"   # starts immune


@dataclass
class Agent:
    id: int
    x: int
    y: int
    state: str = SUSCEPTIBLE

    # Heterogeneity fields
    susceptibility: float = 1.0     # multiplier on infection probability
    recovery_time: int = 7          # steps until recovery

    # Internals
    infected_timer: int = 0
    quarantined: bool = False

    def is_infected(self):
        return self.state == INFECTED

    def is_susceptible(self):
        return self.state == SUSCEPTIBLE

    def infect(self):
        self.state = INFECTED
        self.infected_timer = 0

    def step_infection(self, quarantine_enabled: bool):
        if self.state == INFECTED:
            self.infected_timer += 1
            if quarantine_enabled:
                self.quarantined = True          # infected agents stop moving
            if self.infected_timer >= self.recovery_time:
                self.state = RECOVERED
                self.quarantined = False


# ─────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────

class SIRSimulation:
    def __init__(
        self,
        grid_size: int = 30,
        n_agents: int = 100,
        infection_prob: float = 0.4,
        recovery_time: int = 7,
        initial_infected: int = 5,
        steps: int = 150,
        # Extensions
        vaccination_rate: float = 0.0,
        quarantine_enabled: bool = False,
        social_distancing: float = 0.0,   # 0.0 = no distancing, 1.0 = no movement
        heterogeneity: bool = False,
        seed: int = 42,
        label: str = "Baseline",
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.grid_size        = grid_size
        self.n_agents         = n_agents
        self.infection_prob   = infection_prob
        self.recovery_time    = recovery_time
        self.initial_infected = initial_infected
        self.steps            = steps
        self.vaccination_rate = vaccination_rate
        self.quarantine_enabled = quarantine_enabled
        self.social_distancing  = social_distancing
        self.heterogeneity      = heterogeneity
        self.label              = label

        self.agents: List[Agent] = []
        self.history: Dict[str, List[int]] = {SUSCEPTIBLE: [], INFECTED: [], RECOVERED: [], VACCINATED: []}

        self._init_agents()

    # ── Initialisation ──────────────────────

    def _init_agents(self):
        positions = set()

        n_vaccinated = int(self.n_agents * self.vaccination_rate)
        n_start_infected = min(self.initial_infected, self.n_agents - n_vaccinated)

        for i in range(self.n_agents):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in positions:
                    positions.add(pos)
                    break

            # Heterogeneity: randomise susceptibility and recovery time
            if self.heterogeneity:
                susc  = np.clip(np.random.normal(1.0, 0.3), 0.1, 2.0)
                recov = max(3, int(np.random.normal(self.recovery_time, 2)))
            else:
                susc  = 1.0
                recov = self.recovery_time

            agent = Agent(
                id=i,
                x=pos[0],
                y=pos[1],
                susceptibility=susc,
                recovery_time=recov,
            )

            if i < n_vaccinated:
                agent.state = VACCINATED
            elif i < n_vaccinated + n_start_infected:
                agent.infect()

            self.agents.append(agent)

    # ── Movement ────────────────────────────

    def _move_agent(self, agent: Agent):
        if agent.quarantined:
            return                                   # quarantine: no movement
        if random.random() < self.social_distancing: # social distancing: skip move
            return

        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
        nx = (agent.x + dx) % self.grid_size
        ny = (agent.y + dy) % self.grid_size
        agent.x = nx
        agent.y = ny

    # ── Infection ───────────────────────────

    def _try_infect(self, susceptible: Agent, infected: Agent):
        # Stochastic: effective probability modified by susceptibility
        eff_prob = self.infection_prob * susceptible.susceptibility
        # Gaussian noise for realism
        eff_prob += np.random.normal(0, 0.05)
        eff_prob = float(np.clip(eff_prob, 0, 1))
        if random.random() < eff_prob:
            susceptible.infect()

    def _contact_pairs(self):
        """Build a spatial index and find neighbouring agent pairs."""
        grid: Dict[Tuple[int,int], List[Agent]] = {}
        for a in self.agents:
            grid.setdefault((a.x, a.y), []).append(a)

        pairs = []
        for (x, y), cell_agents in grid.items():
            # Neighbours: same cell + 4 adjacent (Moore distance 1)
            neighbours: List[Agent] = list(cell_agents)
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = (x+dx) % self.grid_size, (y+dy) % self.grid_size
                neighbours.extend(grid.get((nx, ny), []))

            infected_here   = [a for a in neighbours if a.is_infected()]
            susceptible_here = [a for a in cell_agents  if a.is_susceptible()]

            for s in susceptible_here:
                for inf in infected_here:
                    if inf in cell_agents or True:  # neighbour contact counts
                        pairs.append((s, inf))
        return pairs

    # ── Record State ────────────────────────

    def _record(self):
        counts = {SUSCEPTIBLE: 0, INFECTED: 0, RECOVERED: 0, VACCINATED: 0}
        for a in self.agents:
            counts[a.state] += 1
        for k in self.history:
            self.history[k].append(counts[k])

    # ── Main Loop ───────────────────────────

    def run(self) -> Dict[str, List[int]]:
        self._record()
        for _ in range(self.steps):
            # 1. Move
            for agent in self.agents:
                self._move_agent(agent)

            # 2. Infection
            for s, inf in self._contact_pairs():
                if s.is_susceptible():
                    self._try_infect(s, inf)

            # 3. Recovery / quarantine update
            for agent in self.agents:
                agent.step_infection(self.quarantine_enabled)

            # 4. Record
            self._record()

        return self.history

    # ── Snapshot Grid ───────────────────────

    def grid_snapshot(self) -> np.ndarray:
        """Return a 2D colour-coded grid for visualisation."""
        colour_map = {SUSCEPTIBLE: 0, INFECTED: 1, RECOVERED: 2, VACCINATED: 3}
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for a in self.agents:
            grid[a.y][a.x] = colour_map[a.state]
        return grid


# ─────────────────────────────────────────────
# Plotting Helpers
# ─────────────────────────────────────────────

COLOUR = {
    SUSCEPTIBLE: "#3A86FF",
    INFECTED:    "#FF006E",
    RECOVERED:   "#06D6A0",
    VACCINATED:  "#FFD166",
}

def plot_sir_curves(histories: List[Dict], labels: List[str], title: str, ax, steps: int):
    t = list(range(steps + 1))
    for hist, lbl in zip(histories, labels):
        ax.plot(t, hist[INFECTED], lw=2.5, label=f"I – {lbl}")
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Number of Agents", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_full_sir(history: Dict, label: str, steps: int, ax):
    t = list(range(steps + 1))
    ax.stackplot(t,
                 history[SUSCEPTIBLE],
                 history[INFECTED],
                 history[RECOVERED],
                 history[VACCINATED],
                 labels=["Susceptible", "Infected", "Recovered", "Vaccinated"],
                 colors=[COLOUR[SUSCEPTIBLE], COLOUR[INFECTED],
                         COLOUR[RECOVERED],   COLOUR[VACCINATED]],
                 alpha=0.8)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Agents", fontsize=11)
    ax.set_title(f"SIR Curves – {label}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────

def run_experiments():
    STEPS = 150
    BASE = dict(grid_size=30, n_agents=30, infection_prob=0.4,
                recovery_time=7, initial_infected=5, steps=STEPS)

    def make(label, **overrides):
        cfg = dict(**BASE)
        cfg.update(overrides)
        cfg["label"] = label
        return cfg

    scenarios = [
        make("Baseline"),
        make("High Infection (p=0.7)",       infection_prob=0.7),
        make("Large Population (N=200)",      n_agents=200),
        make("Vaccination 30%",              vaccination_rate=0.30),
        make("Quarantine",                   quarantine_enabled=True),
        make("Social Distancing 50%",        social_distancing=0.50),
        make("Heterogeneity",                heterogeneity=True),
        make("Vacc + Quarantine (combo)",    vaccination_rate=0.25, quarantine_enabled=True),
    ]

    print("Running simulations …")
    results = []
    for cfg in scenarios:
        sim = SIRSimulation(**cfg)
        hist = sim.run()
        results.append((hist, cfg["label"]))
        peak = max(hist[INFECTED])
        total_infected = sim.n_agents - hist[SUSCEPTIBLE][-1] - hist.get(VACCINATED, [0])[-1]
        print(f"  [{cfg['label']:35s}]  Peak I = {peak:3d}   Total infected ≈ {total_infected:3d}")

    # ── Figure layout ───────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="#0F0F1A")
    fig.suptitle("Agent-Based SIR Disease Transmission Model",
                 color="white", fontsize=18, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.04, left=0.07, right=0.97)

    ax_style = dict(facecolor="#1A1A2E", facecolor_alpha=1)

    # ── Panel 1: Full SIR stack (baseline) ──
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#1A1A2E")
    baseline_hist, _ = results[0]
    plot_full_sir(baseline_hist, "Baseline", STEPS, ax0)
    _style_ax(ax0)

    # ── Panel 2: Infection curves – infection probability ──
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#1A1A2E")
    sel = [results[0], results[1]]          # baseline vs high p
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Infection Probability", ax1, STEPS)
    _style_ax(ax1)

    # ── Panel 3: Population size ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1A1A2E")
    sel = [results[0], results[2]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Population Size", ax2, STEPS)
    _style_ax(ax2)

    # ── Panel 4: Vaccination ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1A1A2E")
    sel = [results[0], results[3]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Vaccination (30%)", ax3, STEPS)
    _style_ax(ax3)

    # ── Panel 5: Quarantine ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#1A1A2E")
    sel = [results[0], results[4]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Quarantine", ax4, STEPS)
    _style_ax(ax4)

    # ── Panel 6: Social distancing ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#1A1A2E")
    sel = [results[0], results[5]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Social Distancing (50%)", ax5, STEPS)
    _style_ax(ax5)

    # ── Panel 7: Heterogeneity ──
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor("#1A1A2E")
    sel = [results[0], results[6]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Effect of Agent Heterogeneity", ax6, STEPS)
    _style_ax(ax6)

    # ── Panel 8: Combined intervention ──
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor("#1A1A2E")
    sel = [results[0], results[7]]
    plot_sir_curves([r[0] for r in sel], [r[1] for r in sel],
                    "Combined: Vaccination + Quarantine", ax7, STEPS)
    _style_ax(ax7)

    out = "sir_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out}")
    plt.close(fig)

    # ── Grid snapshot animation frames ──────
    print("\nGenerating grid snapshot figure …")
    _grid_snapshots()


def _style_ax(ax):
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#AAAACC")
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#1A1A2E")
        legend.get_frame().set_edgecolor("#444466")
        for text in legend.get_texts():
            text.set_color("white")


def _grid_snapshots():
    """Show 4 snapshots of the grid at different time steps."""
    sim = SIRSimulation(grid_size=30, n_agents=150, infection_prob=0.5,
                        recovery_time=7, initial_infected=5, steps=150,
                        heterogeneity=True, quarantine_enabled=False,
                        vaccination_rate=0.1, label="Snapshot run")

    snap_times = [0, 25, 75, 149]
    snaps = []

    sim._record()
    for step in range(150):
        for agent in sim.agents:
            sim._move_agent(agent)
        for s, inf in sim._contact_pairs():
            if s.is_susceptible():
                sim._try_infect(s, inf)
        for agent in sim.agents:
            agent.step_infection(sim.quarantine_enabled)
        sim._record()
        if step in snap_times:
            snaps.append((step + 1, sim.grid_snapshot()))

    cmap = plt.cm.colors.ListedColormap(
        [COLOUR[SUSCEPTIBLE], COLOUR[INFECTED], COLOUR[RECOVERED], COLOUR[VACCINATED]]
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0F0F1A")
    fig.suptitle("Grid Snapshots – Agent States Over Time",
                 color="white", fontsize=15, fontweight="bold")
    labels = [mpatches.Patch(color=COLOUR[s], label=n)
              for s, n in [(SUSCEPTIBLE,"Susceptible"),(INFECTED,"Infected"),
                           (RECOVERED,"Recovered"),(VACCINATED,"Vaccinated")]]
    for ax, (t, grid) in zip(axes, snaps):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"Step {t}", color="white", fontsize=12)
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    fig.legend(handles=labels, loc="lower center", ncol=4,
               facecolor="#1A1A2E", edgecolor="#444466",
               labelcolor="white", fontsize=11, bbox_to_anchor=(0.5, -0.02))

    out = "sir_grid_snapshots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Grid snapshots saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_experiments()
    print("\nDone. ✓")