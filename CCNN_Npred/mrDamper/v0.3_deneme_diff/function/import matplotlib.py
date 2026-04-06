import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

outdir = "/mnt/data"

def add_box(ax, xy, w, h, text, fontsize=11):
    rect = Rectangle(xy, w, h, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def add_arrow(ax, p1, p2, text=None, text_offset=(0,0), fontsize=10):
    arr = FancyArrowPatch(p1, p2, arrowstyle='->', mutation_scale=15, linewidth=1.8)
    ax.add_patch(arr)
    if text:
        mx = (p1[0]+p2[0])/2 + text_offset[0]
        my = (p1[1]+p2[1])/2 + text_offset[1]
        ax.text(mx, my, text, fontsize=fontsize, ha='center', va='center')

# Top-level diagram
fig, ax = plt.subplots(figsize=(16,9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

add_box(ax, (0.5, 6.3), 2.2, 1.2, "From Workspace\nδf, δr, Tf, Tr", 12)
add_box(ax, (3.3, 6.1), 2.5, 1.6, "Kinematics &\nGeometry", 12)
add_box(ax, (6.5, 6.1), 2.7, 1.6, "Linear Tire Model\n(Front + Rear)", 12)
add_box(ax, (10.0, 6.1), 2.5, 1.6, "Tire-Chassis\nProjection", 12)
add_box(ax, (13.2, 6.1), 2.1, 1.6, "Dynamics\n(V, β, ψ̇)", 12)

add_box(ax, (10.0, 3.2), 2.5, 1.4, "Longitudinal\nResistive Force", 12)
add_box(ax, (13.2, 3.2), 2.1, 1.4, "Trajectory\n(X, Y, ψ)", 12)

add_arrow(ax, (2.7, 6.9), (3.3, 6.9), "δf, δr, Tf, Tr", (0,0.3))
add_arrow(ax, (5.8, 6.9), (6.5, 6.9), "βf, βr, V, ψ̇")
add_arrow(ax, (9.2, 6.9), (10.0, 6.9), "Fxf, Fyf, Fxr, Fyr")
add_arrow(ax, (12.5, 6.9), (13.2, 6.9), "Fx, Fy, Mz")
add_arrow(ax, (14.25, 6.1), (14.25, 4.6), "V, β, ψ̇", (0.5,0))
add_arrow(ax, (12.5, 3.9), (13.2, 3.9), "Fload")
add_arrow(ax, (14.25, 3.2), (14.25, 2.1), "Integrate", (0.5,0))
ax.text(14.25, 1.6, "Outputs:\nX, Y, ψ", ha='center', va='center', fontsize=12)

# feedback lines
add_arrow(ax, (13.2, 6.2), (5.8, 5.1), "V, β, ψ̇", (0,-0.2))
add_arrow(ax, (13.0, 6.1), (12.6, 4.6))
add_arrow(ax, (10.0, 3.9), (8.8, 5.9), "V")
ax.text(8, 8.3, "Önerilen Top-Level Simulink Şeması", fontsize=16, ha='center')

top_path = os.path.join(outdir, "simulink_top_level_schema.png")
plt.tight_layout()
plt.savefig(top_path, dpi=200, bbox_inches='tight')
plt.close(fig)

# Subsystem diagram
fig, ax = plt.subplots(figsize=(16,10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')

add_box(ax, (0.6, 8.6), 2.6, 1.2, "Inputs\nδf, δr, Tf, Tr", 12)
add_box(ax, (4.0, 8.3), 3.0, 1.8, "1) Kinematics & Geometry\nβf, βr, αf, αr,\nVfx,Vfy,Vrx,Vry", 12)
add_box(ax, (8.0, 8.3), 3.2, 1.8, "2) Linear Tire Force\nFxf,Fxr from slip\nFyf,Fyr from slip angle", 12)
add_box(ax, (12.3, 8.3), 3.2, 1.8, "3) Projection to Body Axes\nFx, Fy, Mz", 12)

add_box(ax, (4.0, 5.4), 3.0, 1.6, "4) Resistive Forces\nFaero, Rx, Fgrade", 12)
add_box(ax, (8.0, 5.4), 3.2, 1.6, "5) Vehicle Dynamics\nV̇, β̇, ψ̈", 12)
add_box(ax, (12.3, 5.4), 3.2, 1.6, "6) Wheel Dynamics\nω̇f, ω̇r", 12)

add_box(ax, (8.0, 2.3), 3.2, 1.8, "7) Trajectory Calc\nψ̇→ψ\nV,β,ψ → Ẋ,Ẏ → X,Y", 12)
add_box(ax, (12.3, 2.3), 3.2, 1.8, "Scopes / To Workspace\nX,Y,ψ,V,β,\nαf,αr,sf,sr, forces", 12)

for p1, p2, txt in [
    ((3.2,9.2),(4.0,9.2),""),
    ((7.0,9.2),(8.0,9.2),""),
    ((11.2,9.2),(12.3,9.2),""),
    ((10.7,8.3),(10.3,7.0),""),
    ((6.7,5.4),(8.0,6.2),"Fload"),
    ((11.2,6.2),(12.3,6.2),"Tf,Tr"),
    ((9.6,5.4),(9.6,4.1),""),
    ((11.2,3.2),(12.3,3.2),""),
]:
    add_arrow(ax,p1,p2,txt)

# feedback loops
add_arrow(ax,(8.0,6.1),(6.8,8.1),"V, β, ψ̇")
add_arrow(ax,(12.3,6.1),(10.2,8.1),"ωf, ωr")
add_arrow(ax,(15.5,3.2),(16.7,3.2))
ax.text(9, 10.4, "Subsystem İç Yapısı", fontsize=16, ha='center')

sub_path = os.path.join(outdir, "simulink_subsystems_schema.png")
plt.tight_layout()
plt.savefig(sub_path, dpi=200, bbox_inches='tight')
plt.close(fig)

print(top_path)
print(sub_path)