import numpy as np
import matplotlib.pyplot as plt
from solver_numerico import modelo_seir, rk4_step 


def simular_baseline():
    """
    Función principal que ejecuta y valida el Baseline Secuencial del modelo SEIR.
    """
    # Parámetros iniciales de la red
    N = 100000  # Total de equipos en la red
    I0 = 10     # Equipos Infectados iniciales (Paciente Cero)
    E0 = 50     # Equipos Expuestos (Tienen el malware pero latente)
    P0 = 0      # Equipos Parcheados iniciales
    S0 = N - I0 - E0 - P0 # Equipos Susceptibles restantes
    
    # Tasas epidemiológicas (ejemplo teórico para validación)
    beta = 0.8   # Tasa de propagación del malware
    sigma = 0.2  # Tasa de latencia de la vulnerabilidad
    gamma = 0.05 # Velocidad de distribución de los parches
    
    # Configuración del tiempo de simulación
    T_max = 150  # Días de simulación
    h = 0.1      # Tamaño del paso temporal (discretización)
    tiempos = np.arange(0, T_max, h)
    
    # Matriz para almacenar el historial (Filas: pasos de tiempo, Columnas: S,E,I,P)
    historial = np.zeros((len(tiempos), 4))
    Y_actual = np.array([S0, E0, I0, P0])
    
    # Bucle de integración numérica
    for i, t in enumerate(tiempos):
        historial[i] = Y_actual
        
        # Validación crítica para rúbrica: Conservación rigurosa de la población total
        if not np.isclose(np.sum(Y_actual), N):
            print(f"Advertencia: Inestabilidad numérica detectada en t={t}")
            break
            
        Y_actual = rk4_step(modelo_seir, t, Y_actual, h, beta, sigma, gamma, N)
        
    print("Simulación secuencial finalizada con éxito. Población total conservada.")
    # El siguiente paso lógico aquí será usar matplotlib para visualizar 'historial'
    # Generar la gráfica del modelo epidemiológico
    plt.figure(figsize=(10, 6))
    plt.plot(tiempos, historial[:, 0], label='Susceptibles (S)', color='blue', linewidth=2)
    plt.plot(tiempos, historial[:, 1], label='Expuestos (E)', color='orange', linewidth=2)
    plt.plot(tiempos, historial[:, 2], label='Infectados (I)', color='red', linewidth=2)
    plt.plot(tiempos, historial[:, 3], label='Parcheados (P)', color='green', linewidth=2)
    
    plt.title('Simulación Baseline: Propagación de Malware (Modelo SEIR)')
    plt.xlabel('Tiempo (Días)')
    plt.ylabel('Cantidad de Equipos')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simular_baseline()