import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
sin = np.sin
sign = np.sign

# Create a meshgrid for x and y
w1 = np.linspace(-2, 2, 400)
w2 = np.linspace(-2, 2, 400)
w1, w2 = np.meshgrid(w1, w2)


def y(w1, w2, x):
    return sin(w1 * x) + sin(w2 * x)


def binarzed_y(w1, w2, x):
    return sign(sign(sin(sign(w1) * x)) + sign(sin(sign(w2) * x)))


def half_binarzed_y(w1, w2, x):
    return sign(sign(sin(w1 * x)) + sign(sin(w2 * x)))


# Calculate z based on the given function
z = ((y(w1, w2, pi / 2) - 1) ** 2
     + (y(w1, w2, pi)) ** 2
     + (y(w1, w2, 3 / 2 * pi) + 1) ** 2
     ) / 4

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))
levels = np.linspace(0, 5, 25)
contour = ax.contourf(w1, w2, z, levels=levels, cmap='viridis')
contour_lines = ax.contour(w1, w2, z, levels=levels,
                           colors='black', linewidths=0.5)
ax.clabel(contour_lines, inline=True, fontsize=8)

plt.colorbar(contour)

# Labels and title
plt.xlabel('W_1')
plt.ylabel('W_2')
plt.title('Contour plot of MSE with 4 inputs and 2 weights.  ')
plt.figtext(0.5, 0.01,
            'y_target(t) = sin(t) + sin(2t).  '
            'y_pred(t) = sin(W_1 * t) + sin(W_2 * t).  '
            't = 0, pi/2, pi, 3/2*pi.',
            ha="center", fontsize=10,
            bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

# Show the plot
plt.show()
