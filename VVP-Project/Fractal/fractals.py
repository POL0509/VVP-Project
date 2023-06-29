import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from numba import jit
from math import exp

class Fractals:
    """
    Class that holds all functions and initialization
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, c: complex, n: int, k: int, width: int, height: int, fractal_type: str, cmap: str, threshold: int = None):
        """
        Initialize Fractals
        """
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._n = n
        self._k = k
        self._zoom = 0
        self._c = c
        self._fractal_type = fractal_type
        self._width = width
        self._height = height
        self._cmap = cmap
        self._threshold = threshold
        self._cbar = None

        self._fig, self._ax = plt.subplots(figsize=(self._width, self._height))
        plt.title('Fractal Visualization')

        self._ax_iters = plt.axes([0.08, 0.1, 0.65, 0.03])
        self._slider_iter = Slider(self._ax_iters, 'Iterations', 1, 500, valinit=self._k, valstep=5)

        self._ax_zoom = plt.axes([0.08, 0.075, 0.65, 0.03])
        self._slider_zoom = Slider(self._ax_zoom, 'Zoom', -1, 1, valinit=self._zoom, valstep=0.1)

        self._ax_button = plt.axes([0.08, 0, 0.65, 0.03])
        self._button_update = Button(self._ax_button, 'Update')

    def mandelbrot(self, c: complex, k: int) -> int:
        """
        Algorithm that recognizes how many iterations are needed. Used for Mandelbrot set.

        Args:
            c: Complex number
        
        Return:
            int: Number of iterations needed
        """
        z = c
        for i in range(k):
            z = z**2 + c
            if abs(z) > 2:
                return i
        return k
    
    def julia(self, z: complex, c: complex, k: int) -> int:
        """
        Algorithm that recognizes how many iterations are needed. Used for Julia set.

        Args:
            c: Complex number
        
        Return:
            int: Number of iterations needed
        """
        for i in range(k):
            z = z**2 + c
            if abs(z) > 2:
                return i
        return k
    
    @jit
    def generate_fractal(self):
        """
        Algorithm for generating image we can later plot.
        
        Return:
            array-like
        """
        zoom = np.exp(self._zoom)
        x = np.linspace(self._x_min/zoom, self._x_max/zoom, self._n)
        y = np.linspace(self._y_min/zoom, self._y_max/zoom, self._n)
        image = np.empty((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
                if self._fractal_type == 'mandelbrot':
                    c1 = x[j] + y[i]*1j
                    image[i, j] = self.mandelbrot(c1, self._k)
                elif self._fractal_type == 'julia':
                    z = x[j] + y[i]*1j
                    image[i, j] = self.julia(z, self._c, self._k)
        return image
    
    def plot_fractal(self):
        """
        Algoritm that plots our fractal. We differentiate possibilites where we either use threshold or not.
        """
        zoom = np.exp(self._zoom)
        image = self.generate_fractal()
        if self._threshold is not None:
            image = np.where(image > self._threshold, self._threshold, image)
        im = self._ax.imshow(image, self._cmap, extent=(self._x_min/zoom, self._x_max/zoom, self._y_min/zoom, self._y_max/zoom))
        if self._cbar is None:
            self._cbar = plt.colorbar(im)
        else:
            self._cbar.mappable.set_clim(vmin=0, vmax=self._k)
            self._cbar.update_normal(im)
        #plt.colorbar(im)

    def update_iter(self, val: int):
        """
        Updates our iteration value with a value on a slider.
        """
        self._k = val
        #self.update_plot()

    def update_zoom(self, val: float):
        self._zoom = val

    def update_button_clicked(self, event):
        self.update_plot()

    def update_plot(self):
        """
        Updates plot if changes are made.
        """
        self.plot_fractal()


    def show(self):
        """
        Main function to plot viewing.
        """
        self._button_update.on_clicked(self.update_button_clicked)
        self._slider_zoom.on_changed(self.update_zoom)
        self._slider_iter.on_changed(self.update_iter)

        self.update_plot()
        plt.show()

