import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

C = 299_792_458  # m/s

def wavelength_m(freq_mhz: float) -> float:
    return C / (freq_mhz * 1e6)

def current_distribution(x, L_elec_lambda):
    """
    Simple standing-wave current approximation for a center-fed straight dipole.
    x is position in wavelengths (λ units), centered at 0.
    L_elec_lambda is total electrical length in wavelengths.
    """
    half = L_elec_lambda / 2.0
    inside = np.abs(x) <= half

    # piecewise: I = sin(k*(half - |x|)), with k = 2π/λ and λ normalized to 1
    # so k = 2π. This forces I=0 at ends and max near center for ~0.5λ
    I = np.zeros_like(x)
    I[inside] = np.sin(2*np.pi * (half - np.abs(x[inside])))
    return I

def lengths(freq_mhz, vf, mult):
    lam = wavelength_m(freq_mhz)
    L_total = mult * lam * vf
    return lam, L_total, L_total/2

# ---- Figure setup ----
plt.figure(figsize=(10, 6))
ax = plt.axes([0.08, 0.30, 0.88, 0.65])

# x in wavelengths, show up to ±1.5λ for context
x = np.linspace(-1.5, 1.5, 2000)

# Defaults
freq0 = 100.0   # MHz
vf0 = 0.95
mult0 = 0.5     # 0.5λ dipole (classic)

I0 = current_distribution(x, mult0)

(line,) = ax.plot(x, I0, linewidth=2)
ax.set_xlabel("Position along antenna (in wavelengths, λ)")
ax.set_ylabel("Normalized current (arb.)")
ax.set_title("Dipole current distribution (simple standing-wave model)")
ax.grid(True)

# ---- UI elements ----
axcolor = "lightgoldenrodyellow"

# Slider: velocity factor
ax_vf = plt.axes([0.12, 0.20, 0.80, 0.03], facecolor=axcolor)
s_vf = Slider(ax_vf, "VF", 0.80, 1.00, valinit=vf0, valstep=0.005)

# Slider: length multiplier in wavelengths
ax_mult = plt.axes([0.12, 0.15, 0.80, 0.03], facecolor=axcolor)
s_mult = Slider(ax_mult, "Length (×λ)", 0.25, 2.0, valinit=mult0, valstep=0.01)

# TextBox: frequency MHz
ax_freq = plt.axes([0.12, 0.08, 0.20, 0.05])
tb_freq = TextBox(ax_freq, "Freq (MHz)", initial=str(freq0))

# Button: reset
ax_reset = plt.axes([0.80, 0.08, 0.12, 0.05])
btn_reset = Button(ax_reset, "Reset")

# Status text
status = ax.text(
    0.02, 0.95, "", transform=ax.transAxes, va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    fontsize=10
)

def update(_=None):
    # Read UI
    try:
        freq_mhz = float(tb_freq.text.strip())
        if freq_mhz <= 0:
            raise ValueError
    except ValueError:
        freq_mhz = freq0  # fallback

    vf = float(s_vf.val)
    mult = float(s_mult.val)

    # Update current curve
    I = current_distribution(x, mult)
    line.set_ydata(I)

    # Update lengths
    lam, L_total, L_arm = lengths(freq_mhz, vf, mult)

    status.set_text(
        f"f = {freq_mhz:.3f} MHz\n"
        f"λ = {lam:.3f} m\n"
        f"Total length = {L_total:.3f} m  ({mult:.2f}×λ × VF {vf:.3f})\n"
        f"Each arm = {L_arm:.3f} m"
    )

    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    plt.draw()

def submit_freq(_):
    update()

def reset(_):
    tb_freq.set_val(str(freq0))
    s_vf.reset()
    s_mult.reset()
    update()

# Wire callbacks
s_vf.on_changed(update)
s_mult.on_changed(update)
tb_freq.on_submit(submit_freq)
btn_reset.on_clicked(reset)

# Initial render
update()
plt.show()
