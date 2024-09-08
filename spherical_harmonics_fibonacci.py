# Spherical Harmonics on fibonacci sphere
# (Wave simulation on fibonacci sphere)

import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import Normalize
from scipy.spatial import KDTree


def fibonacci_sphere(num_points):
    global points
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(num_points):
        z = 1. - (i / float(num_points - 1)) * 2.  # z goes from 1 to -1
        radius = np.sqrt(1 - z ** 2)  # radius at z
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)


def update_arrow():
    global x_arw0, y_arw0, z_arw0, u_arw0, v_arw0, w_arw0
    global qvr_gaussian_center0
    global x_arw1, y_arw1, z_arw1, u_arw1, v_arw1, w_arw1
    global qvr_gaussian_center1
    # Arrow 0
    x_arw0 = np.sin(theta_arrow0) * np.cos(phi_arrow0)
    y_arw0 = np.sin(theta_arrow0) * np.sin(phi_arrow0)
    z_arw0 = np.cos(theta_arrow0)
    u_arw0 = x_arw0 * 0.5
    v_arw0 = y_arw0 * 0.5
    w_arw0 = z_arw0 * 0.5
    qvr_gaussian_center0.remove()
    qvr_gaussian_center0 = ax0.quiver(x_arw0, y_arw0, z_arw0, u_arw0, v_arw0, w_arw0,
                                      length=1, color='red', normalize=False, label='Arrow0')
    # Arrow 1
    x_arw1 = np.sin(theta_arrow1) * np.cos(phi_arrow1)
    y_arw1 = np.sin(theta_arrow1) * np.sin(phi_arrow1)
    z_arw1 = np.cos(theta_arrow1)
    u_arw1 = x_arw1 * 0.5
    v_arw1 = y_arw1 * 0.5
    w_arw1 = z_arw1 * 0.5
    qvr_gaussian_center1.remove()
    qvr_gaussian_center1 = ax0.quiver(x_arw1, y_arw1, z_arw1, u_arw1, v_arw1, w_arw1,
                                      length=1, color='blue', normalize=False, label='Arrow1')


def get_force(i_self):
    global tree
    dist, idx = tree.query(base_points[i_self], k=21)  # Set k=21 and find 21 points including itself.
    neighbors = idx[1:]  # 20 points excluding itself.
    dists = dist[1:]  # Distance of 20 points excluding itself.
    # force by magnitude
    force_by_magnitude = 0.
    for i in range(len(neighbors)):
        # effective_magnitude = (magnitude[neighbors[i]] - magnitude[i_self])
        effective_magnitude = (magnitude[neighbors[i]] - magnitude[i_self]) * (dists.mean() / dists[i]) ** 2.
        force_by_magnitude += k_spring * effective_magnitude
    return force_by_magnitude


def calc_wave():
    global points, xs, ys, zs
    global scat_sphere, magnitude, magnitude_buffer
    global velocity
    for i in range(number_of_points):
        force = get_force(i)
        a = force / mass
        velocity[i] = velocity[i] + a * 1.
        magnitude_buffer[i] = magnitude[i] + velocity[i] * 1.
    magnitude = magnitude_buffer.copy()


def update_scat():
    global points, xs, ys, zs
    global scat_sphere
    if var_plots.get() == 1:
        points[:, 0] = 1. * base_points[:, 0]
        points[:, 1] = 1. * base_points[:, 1]
        points[:, 2] = 1. * base_points[:, 2]
    elif var_plots.get() == 2:
        points[:, 0] = (1. + magnitude) * base_points[:, 0]
        points[:, 1] = (1. + magnitude) * base_points[:, 1]
        points[:, 2] = (1. + magnitude) * base_points[:, 2]
    elif var_plots.get() == 3:
        points[:, 0] = np.abs(magnitude) * base_points[:, 0] * 10
        points[:, 1] = np.abs(magnitude) * base_points[:, 1] * 10
        points[:, 2] = np.abs(magnitude) * base_points[:, 2] * 10
    else:
        points[:, 0] = 1. * base_points[:, 0]
        points[:, 1] = 1. * base_points[:, 1]
        points[:, 2] = 1. * base_points[:, 2]
    xs, ys, zs = zip(*points)
    scat_sphere.remove()
    if var_plots.get() == 4:
        size_mag = np.abs(magnitude) * 60.
        scat_sphere = ax0.scatter(xs, ys, zs, c=magnitude, cmap=cmap_scat, s=size_mag, norm=norm)
    else:
        scat_sphere = ax0.scatter(xs, ys, zs, c=magnitude, cmap=cmap_scat, s=size_scat, norm=norm)


def get_angle(point1, point2):
    magnitude1 = np.linalg.norm(point1)
    magnitude2 = np.linalg.norm(point2)
    dot_product = np.dot(point1, point2)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    return np.arccos(cos_angle)


def apply_gaussian():
    global points, xs, ys, zs
    global scat_sphere, magnitude
    # Gaussian 0 (red arrow)
    x0 = np.sin(theta_arrow0) * np.cos(phi_arrow0)
    y0 = np.sin(theta_arrow0) * np.sin(phi_arrow0)
    z0 = np.cos(theta_arrow0)
    # Gaussian 1 (blue arrow)
    x1 = np.sin(theta_arrow1) * np.cos(phi_arrow1)
    y1 = np.sin(theta_arrow1) * np.sin(phi_arrow1)
    z1 = np.cos(theta_arrow1)
    for i in range(number_of_points):
        # Gaussian 0 (red arrow)
        if var_ch_arrow0.get():
            angle0 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x0, y0, z0))
            gauss0 = scale_gaussian * (1. / (np.sqrt(2. * np.pi) * sigma) *
                                       np.exp(- (angle0 ** 2.) / (2. * sigma ** 2.)))
        else:
            gauss0 = 0.
        # Gaussian 1 (blue arrow)
        if var_ch_arrow1.get():
            angle1 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x1, y1, z1))
            gauss1 = scale_gaussian * (1. / (np.sqrt(2. * np.pi) * sigma) *
                                       np.exp(- (angle1 ** 2.) / (2. * sigma ** 2.)))
        else:
            gauss1 = 0.
        magnitude[i] = gauss0 + gauss1
    update_scat()


def apply_sin_cos_latitude():
    global points, xs, ys, zs
    global scat_sphere, magnitude
    # Axis 0 (red arrow)
    x0 = np.sin(theta_arrow0) * np.cos(phi_arrow0)
    y0 = np.sin(theta_arrow0) * np.sin(phi_arrow0)
    z0 = np.cos(theta_arrow0)
    # Axis 1 (blue arrow)
    x1 = np.sin(theta_arrow1) * np.cos(phi_arrow1)
    y1 = np.sin(theta_arrow1) * np.sin(phi_arrow1)
    z1 = np.cos(theta_arrow1)
    for i in range(number_of_points):
        # Axis 0 (red arrow)
        angle0 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x0, y0, z0))
        if var_ch_arrow0.get():
            if var_sin_cos.get() == 1:
                sin_cos0 = 0.1 * np.sin(k_wave_lat * angle0)
            else:
                sin_cos0 = 0.1 * np.cos(k_wave_lat * angle0)
        else:
            sin_cos0 = 0. * np.cos(k_wave_lat * angle0)
        # Axis 1 (blue arrow)
        angle1 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x1, y1, z1))
        if var_ch_arrow1.get():
            if var_sin_cos.get() == 1:
                sin_cos1 = 0.1 * np.sin(k_wave_lat * angle1)
            else:
                sin_cos1 = 0.1 * np.cos(k_wave_lat * angle1)
        else:
            sin_cos1 = 0. * np.cos(k_wave_lat * angle1)
        magnitude[i] = sin_cos0 + sin_cos1
        points[i, 0] = (1. + sin_cos0 + sin_cos1) * base_points[i, 0]
        points[i, 1] = (1. + sin_cos0 + sin_cos1) * base_points[i, 1]
        points[i, 2] = (1. + sin_cos0 + sin_cos1) * base_points[i, 2]
    update_scat()


def apply_sin_cos_longitude():
    global points, xs, ys, zs
    global scat_sphere, magnitude
    for i in range(number_of_points):
        angle0 = np.arctan2(base_points[i, 1], base_points[i, 0])
        if var_ch_arrow0.get():
            if var_sin_cos.get() == 1:
                sin_cos0 = 0.1 * np.sin(k_wave_long * angle0)
            else:
                sin_cos0 = 0.1 * np.cos(k_wave_long * angle0)
        else:
            sin_cos0 = 0. * np.cos(k_wave_long * angle0)
        angle1 = np.arctan2(base_points[i, 2], base_points[i, 0])
        if var_ch_arrow1.get():
            if var_sin_cos.get() == 1:
                sin_cos1 = 0.1 * np.sin(k_wave_long * angle1)
            else:
                sin_cos1 = 0.1 * np.cos(k_wave_long * angle1)
        else:
            sin_cos1 = 0. * np.cos(k_wave_long * angle1)
        magnitude[i] = sin_cos0 + sin_cos1
        points[i, 0] = (1. + sin_cos0 + sin_cos1) * base_points[i, 0]
        points[i, 1] = (1. + sin_cos0 + sin_cos1) * base_points[i, 1]
        points[i, 2] = (1. + sin_cos0 + sin_cos1) * base_points[i, 2]
    update_scat()


def apply_sin_cos_latitude_longitude():
    global points, xs, ys, zs
    global scat_sphere, magnitude
    # Axis 0 (red arrow)
    x0 = np.sin(theta_arrow0) * np.cos(phi_arrow0)
    y0 = np.sin(theta_arrow0) * np.sin(phi_arrow0)
    z0 = np.cos(theta_arrow0)
    # Axis 1 (blue arrow)
    x1 = np.sin(theta_arrow1) * np.cos(phi_arrow1)
    y1 = np.sin(theta_arrow1) * np.sin(phi_arrow1)
    z1 = np.cos(theta_arrow1)
    for i in range(number_of_points):
        # Axis 0 (red arrow)
        angle_lat0 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x0, y0, z0))
        angle_long0 = np.arctan2(base_points[i, 1], base_points[i, 0])
        if var_ch_arrow0.get():
            if var_sin_cos.get() == 1:
                sin_cos_lat0 = 0.1 * np.sin(k_wave_lat * angle_lat0)
                sin_cos_long0 = 0.1 * np.sin(k_wave_long * angle_long0)
            else:
                sin_cos_lat0 = 0.1 * np.cos(k_wave_lat * angle_lat0)
                sin_cos_long0 = 0.1 * np.cos(k_wave_long * angle_long0)
        else:
            sin_cos_lat0 = 0. * np.cos(k_wave_lat * angle_lat0)
            sin_cos_long0 = 0. * np.cos(k_wave_long * angle_long0)
        # Axis 1 (blue arrow)
        angle_lat1 = get_angle((base_points[i, 0], base_points[i, 1], base_points[i, 2]), (x1, y1, z1))
        angle_long1 = np.arctan2(base_points[i, 2], base_points[i, 0])
        if var_ch_arrow1.get():
            if var_sin_cos.get() == 1:
                sin_cos_lat1 = 0.1 * np.sin(k_wave_lat * angle_lat1)
                sin_cos_long1 = 0.1 * np.sin(k_wave_long * angle_long1)
            else:
                sin_cos_lat1 = 0.1 * np.cos(k_wave_lat * angle_lat1)
                sin_cos_long1 = 0.1 * np.cos(k_wave_long * angle_long1)
        else:
            sin_cos_lat1 = 0. * np.cos(k_wave_lat * angle_lat1)
            sin_cos_long1 = 0. * np.cos(k_wave_long * angle_long1)
        mag = (sin_cos_lat0 + sin_cos_lat1) * (sin_cos_long0 + sin_cos_long1) * 10
        magnitude[i] = mag
        points[i, 0] = (1. + mag) * base_points[i, 0]
        points[i, 1] = (1. + mag) * base_points[i, 1]
        points[i, 2] = (1. + mag) * base_points[i, 2]
    update_scat()


def apply_sin_cos():
    global points, xs, ys, zs
    global scat_sphere, magnitude
    if var_dir.get() == 1:
        apply_sin_cos_latitude()
    if var_dir.get() == 2:
        apply_sin_cos_longitude()
    if var_dir.get() == 3:
        apply_sin_cos_latitude_longitude()


# Setter at tkinter
def set_theta0(value):
    global theta_gaussian_deg0, theta_arrow0
    theta_gaussian_deg0 = int(value)
    theta_arrow0 = np.deg2rad(theta_gaussian_deg0)
    update_arrow()


def set_phi0(value):
    global phi_gaussian_deg0, phi_arrow0
    phi_gaussian_deg0 = int(value)
    phi_arrow0 = np.deg2rad(phi_gaussian_deg0)
    update_arrow()


def set_theta1(value):
    global theta_gaussian_deg1, theta_arrow1
    theta_gaussian_deg1 = int(value)
    theta_arrow1 = np.deg2rad(theta_gaussian_deg1)
    update_arrow()


def set_phi1(value):
    global phi_gaussian_deg1, phi_arrow1
    phi_gaussian_deg1 = int(value)
    phi_arrow1 = np.deg2rad(phi_gaussian_deg1)
    update_arrow()


def set_sigma(value):
    global sigma
    sigma = float(value)


def set_scale_gaussian(value):
    global scale_gaussian
    scale_gaussian = float(value)


def set_mass(value):
    global mass
    reset()
    mass = float(value)


def set_k_spring(value):
    global k_spring
    reset()
    k_spring = float(value)


def set_k_wave_lat(value):
    global k_wave_lat
    reset()
    k_wave_lat = float(value)


def set_k_wave_long(value):
    global k_wave_long
    reset()
    k_wave_long = float(value)


# Animation control
def step():
    global cnt
    global txt_step
    cnt += 1
    txt_step.set_text("Step=" + str(cnt))
    calc_wave()
    update_scat()


def reset():
    global is_play, cnt, txt_step
    global xs, ys, zs
    global velocity
    is_play = False
    cnt = 0
    txt_step.set_text("Step=" + str(cnt))
    velocity = np.zeros(number_of_points)
    for i in range(number_of_points):
        magnitude[i] = 0.
        points[i, 0] = base_points[i, 0]
        points[i, 1] = base_points[i, 1]
        points[i, 2] = base_points[i, 2]
    xs, ys, zs = zip(*points)
    update_scat()


def switch():
    global is_play
    if is_play:
        is_play = False
    else:
        is_play = True


def update(f):
    global cnt
    # global txt_step
    if is_play:
        step()
        # cnt += 1


# Global variables

# Animation control
cnt = 0
is_play = False

# Gaussian
sigma = 0.30
scale_gaussian = 0.05

theta_gaussian_deg0 = 0.
phi_gaussian_deg0 = 0.
theta_arrow0 = np.deg2rad(theta_gaussian_deg0)
phi_arrow0 = np.deg2rad(phi_gaussian_deg0)

theta_gaussian_deg1 = 90.
phi_gaussian_deg1 = - 90.
theta_arrow1 = np.deg2rad(theta_gaussian_deg1)
phi_arrow1 = np.deg2rad(phi_gaussian_deg1)

# Sin, Cos
k_wave = 2.
k_wave_lat = k_wave
k_wave_long = - k_wave

# Mass point and Spring constant
mass = 30.
k_spring = 2.
number_of_points = 4000

# Data structure
base_points = fibonacci_sphere(number_of_points)
points = base_points.copy()
xs, ys, zs = zip(*points)
magnitude = np.zeros(number_of_points)
magnitude_buffer = np.zeros(number_of_points)
velocity = np.zeros(number_of_points)

# Tree for search points
tree = KDTree(base_points)

# Generate figure and axes
title_tk = 'Spherical Harmonics on fibonacci sphere'
title_ax0 = title_tk

range_xyz = 1.1
x_min0 = - range_xyz
x_max0 = range_xyz
y_min0 = - range_xyz
y_max0 = range_xyz
z_min0 = - range_xyz
z_max0 = range_xyz

fig = Figure()
ax0 = fig.add_subplot(111, projection='3d')
ax0.set_box_aspect((1, 1, 1))

ax0.set_xlim(x_min0, x_max0)
ax0.set_ylim(y_min0, y_max0)
ax0.set_zlim(z_min0, z_max0)
ax0.set_title(title_ax0)
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_zlabel('z')
ax0.grid()
# ax0.set_facecolor('lightblue')
ax0.xaxis.pane.set_facecolor('darkgray')
ax0.yaxis.pane.set_facecolor('darkgray')
ax0.zaxis.pane.set_facecolor('darkgray')

# Generate items
# Text items
txt_step = ax0.text2D(x_min0, y_max0, "Step=" + str(cnt))
xz, yz, _ = proj3d.proj_transform(x_min0, y_max0, z_max0, ax0.get_proj())
txt_step.set_position((xz, yz))

# Plot items
size_scat = 8
cmap_scat = 'seismic'
norm = Normalize(vmin=-0.1, vmax=0.1)
scat_sphere = ax0.scatter(xs, ys, zs, c=magnitude, cmap=cmap_scat, s=size_scat, norm=norm)

# Arrow
x_arw0 = np.sin(theta_arrow0) * np.cos(phi_arrow0)
y_arw0 = np.sin(theta_arrow0) * np.sin(phi_arrow0)
z_arw0 = np.cos(theta_arrow0)
u_arw0 = x_arw0 * 0.5
v_arw0 = y_arw0 * 0.5
w_arw0 = z_arw0 * 0.5
qvr_gaussian_center0 = ax0.quiver(x_arw0, y_arw0, z_arw0, u_arw0, v_arw0, w_arw0,
                                  length=1, color='red', normalize=False, label='Arrow0')

x_arw1 = np.sin(theta_arrow1) * np.cos(phi_arrow1)
y_arw1 = np.sin(theta_arrow1) * np.sin(phi_arrow1)
z_arw1 = np.cos(theta_arrow1)
u_arw1 = x_arw1 * 0.5
v_arw1 = y_arw1 * 0.5
w_arw1 = z_arw1 * 0.5
qvr_gaussian_center1 = ax0.quiver(x_arw1, y_arw1, z_arw1, u_arw1, v_arw1, w_arw1,
                                  length=1, color='blue', normalize=False, label='Arrow1')

# Tkinter
root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill='both')

toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()

# Animation
frm_anim = ttk.Labelframe(root, relief='ridge', text='Animation', labelanchor='n')
frm_anim.pack(side='left', fill=tk.Y)
btn_play = tk.Button(frm_anim, text='Play/Pause', command=switch)
btn_play.pack(fill=tk.X)
btn_step = tk.Button(frm_anim, text='Step', command=step)
btn_step.pack(fill=tk.X)
btn_reset = tk.Button(frm_anim, text='Reset', command=reset)
btn_reset.pack(fill=tk.X)

# Parameters
frm_parameters = ttk.Labelframe(root, relief='ridge', text='k ,Mass', labelanchor='n', width=100)
frm_parameters.pack(side='left', anchor=tk.N)
lbl_ks = tk.Label(frm_parameters, text='k(spring)')
lbl_ks.pack(anchor=tk.W)
var_ks = tk.StringVar(root)
var_ks.set(str(k_spring))
spn_ks = tk.Spinbox(
    frm_parameters, textvariable=var_ks, format='%.2f', from_=1., to=10.0, increment=1.,
    command=lambda: set_k_spring(float(var_ks.get())), width=5
)
spn_ks.pack(anchor=tk.W)
lbl_mass = tk.Label(frm_parameters, text='Mass')
lbl_mass.pack(anchor=tk.W)
var_mass = tk.StringVar(root)
var_mass.set(str(mass))
spn_mass = tk.Spinbox(
    frm_parameters, textvariable=var_mass, format='%.2f', from_=1., to=50., increment=1.,
    command=lambda: set_mass(float(var_mass.get())), width=5
)
spn_mass.pack(anchor=tk.W)

# Plots method
frm_plots = ttk.Labelframe(root, relief='ridge', text='Plots of magnitude', labelanchor='n', width=100)
frm_plots.pack(side='left')

var_plots = tk.IntVar(root)
r_plots = tk.Radiobutton(frm_plots, text='Only color', value=1, variable=var_plots)
r_plots.pack(anchor=tk.W)

r_plots = tk.Radiobutton(frm_plots, text='Radius=(1 + Mag.)', value=2, variable=var_plots)
r_plots.pack(anchor=tk.W)

r_plots = tk.Radiobutton(frm_plots, text='Radius=(Abs(Mag.)(*10)', value=3, variable=var_plots)
r_plots.pack(anchor=tk.W)

r_plots = tk.Radiobutton(frm_plots, text='Points size', value=4, variable=var_plots)
r_plots.pack(anchor=tk.W)
var_plots.set(1)

# gaussian curve center (arrow position)
# Arrow 0
frm_arw0 = ttk.Labelframe(root, relief='ridge', text='Red arrow(center of Gauss.)', labelanchor='n', width=100)
frm_arw0.pack(side='left')

var_ch_arrow0 = tk.BooleanVar(frm_arw0)
var_ch_arrow0.set(True)
chk0 = tk.Checkbutton(frm_arw0, text='', variable=var_ch_arrow0)
chk0.pack(side='left')

lbl_theta0 = tk.Label(frm_arw0, text='Theta')
lbl_theta0.pack(side='left')
var_theta0 = tk.StringVar(root)
var_theta0.set(str(theta_gaussian_deg0))
spn_theta0 = tk.Spinbox(
    frm_arw0, textvariable=var_theta0, format='%.0f', from_=-180, to=180, increment=1,
    command=lambda: set_theta0(var_theta0.get()), width=4
)
spn_theta0.pack(side='left')

lbl_phi0 = tk.Label(frm_arw0, text='Phi')
lbl_phi0.pack(side='left')
var_phi0 = tk.StringVar(root)
var_phi0.set(str(phi_gaussian_deg0))
spn_phi0 = tk.Spinbox(
    frm_arw0, textvariable=var_phi0, format='%.0f', from_=-360, to=360, increment=1,
    command=lambda: set_phi0(var_phi0.get()), width=4
)
spn_phi0.pack(side='left')

# Arrow 1
frm_arw1 = ttk.Labelframe(root, relief='ridge', text='Blue arrow(center of Gauss)', labelanchor='n', width=100)
frm_arw1.pack(side='left')

var_ch_arrow1 = tk.BooleanVar(frm_arw1)
var_ch_arrow1.set(False)
chk1 = tk.Checkbutton(frm_arw1, text='', variable=var_ch_arrow1)
chk1.pack(side='left')

lbl_theta1 = tk.Label(frm_arw1, text='Theta')
lbl_theta1.pack(side='left')
var_theta1 = tk.StringVar(root)
var_theta1.set(str(theta_gaussian_deg1))
spn_theta1 = tk.Spinbox(
    frm_arw1, textvariable=var_theta1, format='%.0f', from_=-180, to=180, increment=1,
    command=lambda: set_theta1(var_theta1.get()), width=4
)
spn_theta1.pack(side='left')

lbl_phi1 = tk.Label(frm_arw1, text='Phi')
lbl_phi1.pack(side='left')
var_phi1 = tk.StringVar(root)
var_phi1.set(str(phi_gaussian_deg1))
spn_phi1 = tk.Spinbox(
    frm_arw1, textvariable=var_phi1, format='%.0f', from_=-360, to=360, increment=1,
    command=lambda: set_phi1(var_phi1.get()), width=4
)
spn_phi1.pack(side='left')

# gaussian curve
frm_gaussian = ttk.Labelframe(root, relief='ridge', text='Gaussian', labelanchor='n', width=100)
frm_gaussian.pack(side='left', fill=tk.Y)

lbl_sigma = tk.Label(frm_gaussian, text='Sigma')
lbl_sigma.pack(anchor=tk.W)
var_sigma = tk.StringVar(root)
var_sigma.set(str(sigma))
spn_sigma = tk.Spinbox(
    frm_gaussian, textvariable=var_sigma, format='%.2f', from_=0.01, to=1.0, increment=0.01,
    command=lambda: set_sigma(float(var_sigma.get())), width=5
)
spn_sigma.pack(anchor=tk.W)

lbl_scale = tk.Label(frm_gaussian, text='Scale')
lbl_scale.pack(anchor=tk.W)
var_scale = tk.StringVar(root)
var_scale.set(str(scale_gaussian))
spn_scale = tk.Spinbox(
    frm_gaussian, textvariable=var_scale, format='%.3f', from_=0.001, to=2.0, increment=0.001,
    command=lambda: set_scale_gaussian(var_scale.get()), width=5
)
spn_scale.pack(anchor=tk.W)

btn_gaussian = tk.Button(frm_gaussian, text='Apply', command=lambda: apply_gaussian())
btn_gaussian.pack(fill=tk.X)

# sine cosine curve
frm_sin_cos = ttk.Labelframe(root, relief='ridge', text='Trigonometric', labelanchor='n', width=100)
frm_sin_cos.pack(side='left')

frm_function = ttk.Labelframe(frm_sin_cos, relief='ridge', text='Function', labelanchor='n', width=100)
frm_function.pack(side='left', fill=tk.Y)

var_sin_cos = tk.IntVar(root)
r_sin_cos1 = tk.Radiobutton(frm_function, text='Sin', value=1, variable=var_sin_cos)
r_sin_cos1.pack(anchor=tk.W)

r_sin_cos2 = tk.Radiobutton(frm_function, text='Cos', value=2, variable=var_sin_cos)
r_sin_cos2.pack(anchor=tk.W)
var_sin_cos.set(2)

frm_direction = ttk.Labelframe(frm_sin_cos, relief='ridge', text='Direction', labelanchor='n', width=100)
frm_direction.pack(side='left')

var_dir = tk.IntVar(root)
r_dir_lat = tk.Radiobutton(frm_direction, text='Latitude', value=1, variable=var_dir)
r_dir_lat.pack(anchor=tk.W)

r_dir_long = tk.Radiobutton(frm_direction, text='Longitude', value=2, variable=var_dir)
r_dir_long.pack(anchor=tk.W)

r_dir_lat_long = tk.Radiobutton(frm_direction, text='Lat. * Long.', value=3, variable=var_dir)
r_dir_lat_long.pack(anchor=tk.W)

var_dir.set(1)

frm_wn = ttk.Labelframe(frm_sin_cos, relief='ridge', text='Wave number', labelanchor='n', width=100)
frm_wn.pack(side='left', fill=tk.Y)

lbl_wn_lat = tk.Label(frm_wn, text='k(latitude)')
lbl_wn_lat.pack(anchor=tk.W)
var_wn_lat = tk.StringVar(root)
var_wn_lat.set(str(k_wave_lat))
spn_wn_lat = tk.Spinbox(
    frm_wn, textvariable=var_wn_lat, format='%.0f', from_=-10, to=10, increment=1,
    command=lambda: set_k_wave_lat(var_wn_lat.get()), width=3
)
spn_wn_lat.pack(anchor=tk.W)

lbl_wn_long = tk.Label(frm_wn, text='k(longitude)')
lbl_wn_long.pack(anchor=tk.W)
var_wn_long = tk.StringVar(root)
var_wn_long.set(str(k_wave_long))
spn_wn_long = tk.Spinbox(
    frm_wn, textvariable=var_wn_long, format='%.0f', from_=-10, to=10, increment=1,
    command=lambda: set_k_wave_lat(var_wn_long.get()), width=3
)
spn_wn_long.pack(anchor=tk.W)

btn_sin_cos = tk.Button(frm_sin_cos, text='Apply', command=lambda: apply_sin_cos())
btn_sin_cos.pack(side='left')

# Draw animation
anim = animation.FuncAnimation(fig, update, interval=50, save_count=100)
root.mainloop()
