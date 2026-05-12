# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:07:26 2026

@author: squac
"""

# -*- coding: utf-8 -*-
"""
Infection Lab — Wesley-Niles airborne transmission model
P = 1 - exp(-I * q * p * t / Q)

Parameters
----------
I : number of active infectors (tracked live from simulation)
q : quantum generation rate (quanta/h)  — slider
Q : room ventilation rate (m³/h)        — slider
p : pulmonary ventilation rate per person = 0.3 m³/h (fixed, typical seated adult)
t : elapsed simulation time (hours), advances each frame

Default values chosen so P ≈ 0.25 at t ≈ 0.5 h with I = 1:
  q = 30 quanta/h  (mid-range COVID-19 estimate: 14-48 /h)
  Q = 480 m³/h     (moderate HVAC for a medium room)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from matplotlib.widgets import Slider
from itertools import combinations


# ── Wells-Riley ────────────────────────────────────────────────────────────────
P_PULMONARY = 0.3   # m³/h — fixed breathing rate per susceptible

def wells_riley(I, q, Q, t):
    """Return per-contact infection probability via Wells-Riley equation."""
    if Q <= 0 or t <= 0:
        return 0.0
    exponent = -(I * q * P_PULMONARY * t) / Q
    return 1.0 - np.exp(exponent)


# ── Particle ───────────────────────────────────────────────────────────────────
class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius
        self.infected = False
        self.styles = styles or {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}

    # Convenience properties
    @property
    def x(self): return self.r[0]
    @x.setter
    def x(self, v): self.r[0] = v

    @property
    def y(self): return self.r[1]
    @y.setter
    def y(self, v): self.r[1] = v

    @property
    def vx(self): return self.v[0]
    @vx.setter
    def vx(self, v): self.v[0] = v

    @property
    def vy(self): return self.v[1]
    @vy.setter
    def vy(self, v): self.v[1] = v

    def overlaps(self, other):
        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def advance(self, dt):
        self.r += self.v * dt
        if self.x - self.radius < 0:
            self.x = self.radius;  self.vx = -self.vx
        if self.x + self.radius > 1:
            self.x = 1 - self.radius; self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius;  self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1 - self.radius; self.vy = -self.vy


# ── Simulation ─────────────────────────────────────────────────────────────────
class Simulation:
    """Hard-circle particle simulation with Wells-Riley infection model."""

    # Animation timestep: each frame = 0.04 sim-units.
    # We map sim-units → hours: DT_HOURS controls how fast the clock runs.
    # 1 sim-unit ≈ 1 minute  →  DT_HOURS = 1/60
    DT_ANIM   = 0.04          # sim-units per frame (unchanged from original)
    DT_HOURS  = 0.04 / 60.0  # hours per frame  (~0.04 min per frame)

    def __init__(self, n, q=30.0, Q=480.0, radius=0.01, styles=None):
        """
        Parameters
        ----------
        n       : number of particles
        q       : initial quantum generation rate (quanta/h)
        Q       : initial room ventilation rate (m³/h)
        radius  : particle radius (scalar or length-n sequence)
        styles  : matplotlib Circle kwargs for healthy particles
        """
        self.q = q
        self.Q = Q
        self.elapsed_hours = 0.0     # simulation clock
        self.init_particles(n, radius, styles)

    # ── initialisation ──────────────────────────────────────────────────────
    def init_particles(self, n, radius, styles=None):
        try:
            assert n == len(radius)
        except TypeError:
            def r_gen(n, r):
                for _ in range(n): yield r
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []
        default_styles = styles or {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}

        for rad in radius:
            while True:
                x, y = rad + (1 - 2*rad) * np.random.random(2)
                vr   = 0.1 * np.random.random() + 0.1
                vphi = 2 * np.pi * np.random.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                p = Particle(x, y, vx, vy, rad,
                             dict(default_styles))   # each gets its own dict
                for p2 in self.particles:
                    if p2.overlaps(p):
                        break
                else:
                    self.particles.append(p)
                    break

        # Patient zero
        self.particles[0].infected = True
        self.particles[0].styles = {'edgecolor': 'red', 'linewidth': 2, 'fill': None}

    # ── infection count ─────────────────────────────────────────────────────
    @property
    def n_infected(self):
        return sum(1 for p in self.particles if p.infected)

    # ── collision / infection ───────────────────────────────────────────────
    def handle_collisions(self):
        """Elastic collisions + probabilistic Wells-Riley infection on contact."""

        # Current per-collision infection probability
        p_infect = wells_riley(self.n_infected, self.q, self.Q,
                               self.elapsed_hours)
        p_infect = np.clip(p_infect, 0.0, 1.0)

        def maybe_infect(src, dst):
            if src.infected and not dst.infected:
                if np.random.random() < p_infect:
                    dst.infected = True

        def change_velocities(p1, p2):
            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d  = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            p1.v = v1 - 2*m2/M * np.dot(v1-v2, r1-r2) / d * (r1-r2)
            p2.v = v2 - 2*m1/M * np.dot(v2-v1, r2-r1) / d * (r2-r1)

        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                change_velocities(self.particles[i], self.particles[j])
                maybe_infect(self.particles[i], self.particles[j])
                maybe_infect(self.particles[j], self.particles[i])

    # ── animation step ──────────────────────────────────────────────────────
    def advance_animation(self, dt):
        self.elapsed_hours += self.DT_HOURS

        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
            self.circles[i].set_edgecolor('red' if p.infected else 'C0')

        self.handle_collisions()
        return self.circles

    def init(self):
        self.circles = []
        for p in self.particles:
            self.circles.append(p.draw(self.ax))
        return self.circles

    def animate(self, i):
        self.advance_animation(self.DT_ANIM)

        # Live stats for title
        I = self.n_infected
        P = wells_riley(I, self.q, self.Q, self.elapsed_hours)
        self.title.set_text(
            f'Infected: {I}/{self.n}   '
            f'P(infect) = {P:.3f}   '
            f't = {self.elapsed_hours*60:.1f} min'
        )
        return self.circles

    # ── main entry point ────────────────────────────────────────────────────
    def do_animation(self, save=False):
        fig, self.ax = plt.subplots(figsize=(7, 7))

        # Make room for two sliders at the bottom
        plt.subplots_adjust(bottom=0.25)

        for s in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        self.title = self.ax.set_title(
            f'Infected: 1/{self.n}   P(infect) = {wells_riley(1, self.q, self.Q, 1e-6):.3f}   t = 0.0 min'
        )

        # ── Slider: q (emission rate) ──────────────────────────────────────
        # Range: 1 – 200 quanta/h  (1 = weak pathogen, 200 = highly infectious)
        ax_q = fig.add_axes([0.15, 0.12, 0.70, 0.03])
        self.slider_q = Slider(
            ax=ax_q,
            label='q  emission rate (quanta/h)',
            valmin=1.0,
            valmax=200.0,
            valinit=self.q,
            valstep=1.0,
            color='tomato',
        )

        # ── Slider: Q (ventilation rate) ──────────────────────────────────
        # Range: 10 – 2000 m³/h  (10 = stuffy room, 2000 = heavy HVAC / outdoors)
        ax_Q = fig.add_axes([0.15, 0.05, 0.70, 0.03])
        self.slider_Q = Slider(
            ax=ax_Q,
            label='Q  ventilation (m³/h)',
            valmin=10.0,
            valmax=2000.0,
            valinit=self.Q,
            valstep=10.0,
            color='steelblue',
        )

        def update_q(val):
            self.q = self.slider_q.val

        def update_Q(val):
            self.Q = self.slider_Q.val

        self.slider_q.on_changed(update_q)
        self.slider_Q.on_changed(update_Q)

        # ── run ────────────────────────────────────────────────────────────
        self.anim = animation.FuncAnimation(
            fig, self.animate,
            init_func=self.init,
            frames=2000,
            interval=20,
            blit=False,
        )

        if save:
            self.anim.save('infection_wellsriley.mp4', fps=30)
        else:
            plt.show()


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    nparticles = 128
    radii      = 0.01
    styles     = {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}

    # Default q=30, Q=480  →  Wells-Riley P ≈ 0.25 at t ≈ 0.5 h (I=1)
    sim = Simulation(nparticles, q=30.0, Q=480.0, radius=radii, styles=styles)
    sim.do_animation(save=False)