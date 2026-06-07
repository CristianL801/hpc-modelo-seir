# Archivo: solver_numerico.py
import numpy as np

def modelo_seir(t, Y, beta, sigma, gamma, N):
    """
    Calcula las derivadas parciales del modelo epidemiológico SEIR en redes corporativas.
    """
    S, E, I, P = Y
    dSdt = -beta * (S * I) / N
    dEdt = (beta * (S * I) / N) - (sigma * E)
    dIdt = (sigma * E) - (gamma * I)
    dPdt = gamma * I
    return np.array([dSdt, dEdt, dIdt, dPdt])

def rk4_step(f, t, Y, h, *args):
    """
    Avanza un paso discreto en el tiempo utilizando el método de Runge-Kutta de 4to orden (RK4).
    """
    k1 = f(t, Y, *args)
    k2 = f(t + h/2, Y + (h/2)*k1, *args)
    k3 = f(t + h/2, Y + (h/2)*k2, *args)
    k4 = f(t + h, Y + h*k3, *args)
    return Y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)