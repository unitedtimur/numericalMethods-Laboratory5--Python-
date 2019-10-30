import numpy as np
import matplotlib.pyplot as plt

class Splines():
    x = None
    y = None
    h = None
    g = None
    shape = None

    def find_parabolic_c(self):
        c = []
        c.append(self.g[-1] / self.h[-1])
        for i in range(self.shape - 3, -1, -1):
            c.append((self.g[i] - c[self.shape - 3 - i] * self.h[i + 1]) / self.h[i])
        c.reverse()
        return c

    def find_parabolic_b(self, c):
        b = []
        for i in range(0, self.shape - 1):
            b.append((self.y[i + 1] - self.y[i]) / self.h[i] - self.h[i] * c[i])
        return b

    def find_parabolic_a(self):
        return [self.y[i - 1] for i in range(1, self.shape)]

    def find_cubic_a(self):
        return self.find_parabolic_a()

    def find_cubic_c(self, xI, zeta):
        c = [0]
        for i in range(1, self.shape - 1):
            c.append(c[i - 1] * xI[-i] + zeta[-i])
        c.reverse()
        c = [0] + c
        return c

    def __find_cubic_b(self, c):
        b = []
        for i in range(1, self.shape):
            b.append((self.y[i] - self.y[i - 1]) / self.h[i - 1] - (self.h[i - 1] * (c[i] + 2 * c[i - 1])) / 3)
        return b

    def __find_cubic_d(self, c):
        d = []
        for i in range(1, self.shape):
            d.append((c[i] - c[i - 1]) / (3 * self.h[i - 1]))
        return d

    def find_zeta_xI_coff(self, g):
        zeta = [0]
        xI = [0]
        for i in range(1, self.shape - 1):
            zeta.append(-self.h[i] / (self.h[i - 1] * zeta[i - 1] + 2 * (self.h[i - 1] + self.h[i])))
            xI.append((g[i - 1] - self.h[i - 1] * xI[i - 1]) / (self.h[i - 1] * zeta[i - 1] + 2 * (self.h[i - 1] + self.h[i])))
        return zeta, xI

    def __linear(self, X):
        result = []
        for x in X:
            index = self.__arg_low(x)
            a = self.y[index - 1]
            b = (self.y[index] - self.y[index - 1]) / self.h[index - 1]
            result.append(a + (x - self.x[index - 1]) * b)
        return np.array(result)

    def __parabolic(self, X):
        result = []
        c = self.find_parabolic_c()
        b = self.find_parabolic_b(c)
        a = self.find_parabolic_a()
        for x in X:
            index = self.__arg_low(x) - 1
            result.append(c[index] * (x - self.x[index]) ** 2 + b[index] * (x - self.x[index]) + a[index])
        return result

    def __cubic(self, X):
        g = 3 * np.array([(self.y[i + 1] - self.y[i]) / self.h[i] - (self.y[i] - self.y[i - 1]) / self.h[i - 1] for i in range(1, self.shape - 1)])
        xI, zeta = self.find_zeta_xI_coff(g)
        a = self.find_cubic_a()
        c = self.find_cubic_c(xI, zeta)
        b = self.__find_cubic_b(c)
        d = self.__find_cubic_d(c)
        result = []
        for x in X:
            index = self.__arg_low(x) - 1
            result.append(d[index] * (x - self.x[index]) ** 3 + c[index] * (x - self.x[index]) ** 2 + b[index] * (x - self.x[index]) + a[index])
        return result

    def __ermits(self, X, Y):
        a = self.find_cubic_a()
        b = Y[:-1]
        c = [(3 * self.y[i] - 3 * self.y[i - 1] - 2 * self.h[i - 1] * Y[i - 1] - self.h[i - 1] * Y[i]) / self.h[i - 1] ** 2 for i in range(1, self.shape)]
        d = [(2 * self.y[i - 1] - 2 * self.y[i] + self.h[i - 1] * Y[i - 1] + self.h[i - 1] * Y[i]) / self.h[i - 1] ** 3 for i in range(1, self.shape)]
        result = []
        for x in X:
            index = self.__arg_low(x) - 1
            result.append(d[index] * (x - self.x[index]) ** 3 + c[index] * (x - self.x[index]) ** 2 + b[index] * (x - self.x[index]) + a[index])
        return result

    def fit(self, X, Y):
        self.x = np.array(X)
        self.y = np.array(Y)
        self.shape = self.x.shape[0]
        self.h = [(X[i] - X[i - 1]) for i in range(1, len(X))]
        self.g = [(Y[i + 1] - Y[i]) / (X[i + 1] - X[i]) - (Y[i] - Y[i - 1]) / (X[i] - X[i - 1]) for i in range(1, len(X) - 1)]
        self.g.append((Y[-2] - Y[-1]) / (X[-1] - X[-2]))
        return self

    def __arg_low(self, x):
        arg = 1
        while self.x[arg] < x and arg < self.shape - 1:
            arg += 1
        return arg

    def transform(self, x, interpolation: str, *argv):
        if interpolation == 'linear':
            return self.__linear(x)
        elif interpolation == 'parabolic':
            return self.__parabolic(x)
        elif interpolation == 'cubic':
            return self.__cubic(x)
        elif interpolation == 'ermit':
            return self.__ermits(x, argv[0])

table = {
    'x' : [0.000, 1.250, 2.350, 3.000, 5.500],
    'y' : [3.000, -1.513, 2.872, -2.592, -2.813],
    'f(x)': lambda x: np.sin(x) + 3 * np.cos(3 * x)
    }

#print(Splines().fit(table['x'], table['y']).transform([0], 'ermit', table['y']))

#from pylab import pcParams
#pcParams['figure.figsize'] = 15, 10

x = np.linspace(table['x'][0] + 1e-10, table['x'][-1], 100)

#Points
plt.scatter(table['x'], table['y'], c = 'blue', label = 1)

#Original function
plt.plot(x, table['f(x)'](x), '--', color = 'black', label = 2)

#Linear spline
plt.plot(x, Splines().fit(table['x'], table['y']).transform(x, 'linear'), color = 'purple', label = 3)

#Parabolic spline
plt.plot(x, Splines().fit(table['x'], table['y']).transform(x, 'parabolic'), color = 'red', label = 4)

#Cubic spline
plt.plot(x, Splines().fit(table['x'], table['y']).transform(x, 'cubic'), color = 'green', label = 5)

#Ermit spline
plt.plot(x, Splines().fit(table['x'], table['y']).transform(x, 'ermit', table['y']), color = 'yellow', label = 6)

#Title
plt.title("Splines functions")

#Legends
plt.legend(['Points', 'Original function', 'Linear spline', 'Parabolic spline', 'Cubic spline', 'Ermits spline'])

plt.grid()
plt.show()