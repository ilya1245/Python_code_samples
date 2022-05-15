# https://www.geeksforgeeks.org/python-scroll-through-plots/

# Importing Libraries using import function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Setting Plot and Axis variables as subplots()
# function returns tuple(fig, ax)
fig, ax = plt.subplots()

# Adjust the bottom size according to the
# requirement of the user
plt.subplots_adjust(bottom = 0.25)

# Set the x and y axis to some dummy data
t = np.arange(0.0, 1.0, 0.001)
# Initial values of amplitude and frequency
# are denoted by a0 and f0
a0 = 6
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)

# plot the x and y using plot function
Plot, = plt.plot(t, s, lw = 3, color = 'green')
plt.axis([0, 1, -10, 10])

# Choose the Slider color
axcolor = "White"

# Set the frequency and amplitude axis
frequency_axis = plt.axes([0.25, 0.1, 0.65, 0.03],
                          facecolor = axcolor)
amplitude_axis = plt.axes([0.25, 0.15, 0.65, 0.03],
                          facecolor = axcolor)

# Set the slider for frequency and amplitude
frequency_slider = Slider(frequency_axis, 'Freq',
                          0.1, 30.0, valinit = f0)
amplitude_slider = Slider(amplitude_axis, 'Amp',
                          0.1, 10.0, valinit = a0)

# update() function to change the graph when the
# slider is in use
def update(val):
    amp = amplitude_slider.val
    freq = frequency_slider.val
    Plot.set_ydata(amp*np.sin(2*np.pi*freq*t))

# update function called using on_changed() function
# for both frequency and amplitude
frequency_slider.on_changed(update)
amplitude_slider.on_changed(update)

# Display the plot
plt.show()
