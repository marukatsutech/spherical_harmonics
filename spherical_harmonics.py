# Spherical Harmonics

import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from scipy.special import sph_harm


def on_move(event):
    if event.inaxes == ax0:
        ax1.view_init(elev=ax0.elev, azim=ax0.azim)
    elif event.inaxes == ax1:
        ax1.view_init(elev=ax1.elev, azim=ax1.azim)
    fig.canvas.draw_idle()


def phase_color(phase):
    colors = np.zeros((phase.shape[0], phase.shape[1], 3))
    if m > 0:
        colors[(phase >= 0) & (phase < np.pi)] = [0, 0, 1]  # Red
        colors[(phase >= np.pi) & (phase < 2 * np.pi)] = [1, 0, 0]  # Blue
    else:
        colors[(phase >= 0) & (phase < np.pi)] = [1, 0, 0]  # Red
        colors[(phase >= np.pi) & (phase < 2 * np.pi)] = [0, 0, 1]  # Blue
    return colors


def update_plot():
    global title_ax0, title_ax1
    global l, m, Y_lm
    global Y_lm_real, Y_lm_imag
    global x, y, z
    global x_real, x_imag, y_real, y_imag, z_real, z_imag
    global magnitude_real, magnitude_imag
    global colors_real
    global plt_sph_harm0, plt_sph_harm1
    title_ax0 = f'l={l}, m={m}\nPolar plots'
    title_ax1 = f'l={l}, m={m}\nPolar plots with magnitude as radius'
    ax0.set_title(title_ax0)
    ax1.set_title(title_ax1)
    # Spherical harmonics
    Y_lm = sph_harm(m, l, phi, theta)
    # Real basis set calculations
    if m > 0:
        Y_lm_real = np.sqrt(2) * (-1) ** m * np.real(Y_lm)
        Y_lm_imag = np.sqrt(2) * (-1) ** m * np.imag(Y_lm)
    else:
        Y_lm_real = np.real(Y_lm)
        Y_lm_imag = np.imag(Y_lm)
    # Color setting
    colors_real = phase_color(np.angle(Y_lm_real))

    # ax0: Polar plots
    # Conversion from spherical coordinates to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    plt_sph_harm0.remove()
    plt_sph_harm0 = ax0.plot_surface(x, y, z, facecolors=colors_real,
                                     rstride=2, cstride=2, edgecolor='black', linewidth=0.1)

    # ax1: Polar plots with magnitude as radius
    # Absolute value of spherical harmonics calculations
    magnitude_real = np.abs(Y_lm_real)
    magnitude_imag = np.abs(Y_lm_imag)
    # Conversion from spherical coordinates to Cartesian coordinates
    x_real = magnitude_real * np.sin(theta) * np.cos(phi)
    y_real = magnitude_real * np.sin(theta) * np.sin(phi)
    z_real = magnitude_real * np.cos(theta)
    x_imag = magnitude_imag * np.sin(theta) * np.cos(phi)
    y_imag = magnitude_imag * np.sin(theta) * np.sin(phi)
    z_imag = magnitude_imag * np.cos(theta)
    plt_sph_harm1.remove()
    plt_sph_harm1 = ax1.plot_surface(x_real, y_real, z_real, facecolors=colors_real,
                                     rstride=2, cstride=2, edgecolor='black', linewidth=0.1)


def set_l(value):
    global l
    global title_ax0, title_ax1
    l = int(value)
    update_plot()


def set_m(value):
    global m
    m = int(value)
    update_plot()


# Animation control
def step():
    global cnt
    cnt += 1


def reset():
    global is_play, cnt
    is_play = False
    cnt = 0


def switch():
    global is_play
    if is_play:
        is_play = False
    else:
        is_play = True


def update(f):
    global cnt
    # txt_step.set_text("dummy" + str(cnt))
    if is_play:
        cnt += 1
        step()


# Global variables

# Animation control
cnt = 0
is_play = False

# Data structure
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Spherical Harmonics
l = 0  # Principal quantum number
m = 0  # Magnetic quantum number

# Spherical harmonics
Y_lm = sph_harm(m, l, phi, theta)

# Real basis set calculations
if m > 0:
    Y_lm_real = np.sqrt(2) * (-1)**m * np.real(Y_lm)
    Y_lm_imag = np.sqrt(2) * (-1)**m * np.imag(Y_lm)
else:
    Y_lm_real = np.real(Y_lm)
    Y_lm_imag = np.imag(Y_lm)

# ax0: Polar plots
# Conversion from spherical coordinates to Cartesian coordinates
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# ax1: Polar plots with magnitude as radius
# Absolute value of spherical harmonics calculations
magnitude_real = np.abs(Y_lm_real)
magnitude_imag = np.abs(Y_lm_imag)

# Conversion from spherical coordinates to Cartesian coordinates
x_real = magnitude_real * np.sin(theta) * np.cos(phi)
y_real = magnitude_real * np.sin(theta) * np.sin(phi)
z_real = magnitude_real * np.cos(theta)

x_imag = magnitude_imag * np.sin(theta) * np.cos(phi)
y_imag = magnitude_imag * np.sin(theta) * np.sin(phi)
z_imag = magnitude_imag * np.cos(theta)

# Generate figure and axes
range_xyz = 1.
title_tk = 'Spherical Harmonics'
title_ax0 = f'l={l}, m={m}\nPolar plots'
title_ax1 = f'l={l}, m={m}\nPolar plots with magnitude as radius'

x_min0 = - range_xyz
x_max0 = range_xyz
y_min0 = - range_xyz
y_max0 = range_xyz
z_min0 = - range_xyz
z_max0 = range_xyz

x_min1 = - range_xyz
x_max1 = range_xyz
y_min1 = - range_xyz
y_max1 = range_xyz
z_min1 = - range_xyz
z_max1 = range_xyz

fig = Figure()
ax0 = fig.add_subplot(121, projection='3d')
ax0.set_box_aspect((1, 1, 1))
ax0.grid()
ax0.set_title(title_ax0)
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_zlabel('z')
ax0.set_xlim(x_min0, x_max0)
ax0.set_ylim(y_min0, y_max0)
ax0.set_zlim(z_min0, z_max0)

ax1 = fig.add_subplot(122, projection='3d')
ax1.set_box_aspect((1, 1, 1))
ax1.grid()
ax1.set_title(title_ax1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim(x_min1, x_max1)
ax1.set_ylim(y_min1, y_max1)
ax1.set_zlim(z_min1, z_max1)

# Text items
# txt_step = ax0.text2D(x_min0, y_max0, "dummy" + str(cnt))
# xz, yz, _ = proj3d.proj_transform(x_min0, y_max0, z_max0, ax0.get_proj())
# txt_step.set_position((xz, yz))

# Plot items
colors_real = phase_color(np.angle(Y_lm_real))
plt_sph_harm0 = ax0.plot_surface(x, y, z, facecolors=colors_real,
                                 rstride=2, cstride=2, edgecolor='black', linewidth=0.1)

plt_sph_harm1 = ax1.plot_surface(x_real, y_real, z_real, facecolors=colors_real,
                                 rstride=2, cstride=2, edgecolor='black', linewidth=0.1)


# Legend
# ax0.legend(loc='lower right')

# Embed in Tkinter
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill='both')

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

fig.canvas.mpl_connect('motion_notify_event', on_move)

# Animation
'''
frm_anim = ttk.Labelframe(root, relief='ridge', text='Animation', labelanchor='n')
frm_anim.pack(side='left', fill=tk.Y)
btn_play = tk.Button(frm_anim, text='Play/Pause', command=switch)
btn_play.pack(side='left')
btn_step = tk.Button(frm_anim, text='Step', command=step)
btn_step.pack(side='left')
btn_reset = tk.Button(frm_anim, text='Reset', command=reset)
btn_reset.pack(side='left')
'''
# Quantum number
frm_qn = ttk.Labelframe(root, relief='ridge', text='Quantum number', labelanchor='n')
frm_qn.pack(side='left', fill=tk.Y)

lbl_l = tk.Label(frm_qn, text='l (principal quantum number):')
lbl_l.pack(side='left')
var_l = tk.StringVar(root)
var_l.set(str(l))
spn_l = tk.Spinbox(
    frm_qn, textvariable=var_l, format='%.0f', from_=-6, to=6, increment=1,
    command=lambda: set_l(var_l.get()), width=6
    )
spn_l.pack(side='left')

lbl_m = tk.Label(frm_qn, text='m (magnetic quantum number):')
lbl_m.pack(side='left')
var_m = tk.StringVar(root)
var_m.set(str(m))
spn_m = tk.Spinbox(
    frm_qn, textvariable=var_m, format='%.0f', from_=-6, to=6, increment=1,
    command=lambda: set_m(var_m.get()), width=6
    )
spn_m.pack(side='left')

# main loop
anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
root.mainloop()
