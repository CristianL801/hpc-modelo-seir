# Archivo: main_montecarlo.py
import time
import numpy as np
import ipyparallel as ipp

def simulacion_aislada(params):
    """
    Worker Function: Ejecuta una simulación completa del modelo SEIR para una mutación.
    Como los workers corren en entornos aislados, debemos importar las dependencias aquí dentro.
    """
    import numpy as np
    from solver_numerico import modelo_seir, rk4_step
    
    # Desempaquetamos la mutación generada por el fuzzing
    beta, sigma, gamma = params
    
    # Parámetros fijos de la red corporativa
    N = 100000 
    I0 = 10 
    E0 = 50 
    P0 = 0 
    S0 = N - I0 - E0 - P0 
    
    T_max = 150 
    h = 0.5 # Aumentamos un poco el paso (h) para que 50,000 iteraciones sean manejables
    tiempos = np.arange(0, T_max, h)
    
    Y_actual = np.array([S0, E0, I0, P0])
    
    # Variable para almacenar el peor momento de esta mutación
    pico_maximo_infectados = 0
    
    # Bucle de integración RK4
    for t in tiempos:
        Y_actual = rk4_step(modelo_seir, t, Y_actual, h, beta, sigma, gamma, N)
        
        # Registramos si esta iteración superó el pico histórico de infectados (índice 2)
        if Y_actual[2] > pico_maximo_infectados:
            pico_maximo_infectados = Y_actual[2]
            
    return pico_maximo_infectados

def ejecutar_montecarlo():
    """
    Función principal (Cliente): Genera el fuzzing y delega las tareas a la cola.
    """
    import ipyparallel as ipp
    import time
    import numpy as np
    
    print("1. Encendiendo el clúster local (esto puede tomar unos segundos)...")
    
    # ¡Magia pura! Iniciamos 4 motores directamente desde el script
    with ipp.Cluster(n=4) as cluster:
        # Nos conectamos al cliente una vez que los motores están listos
        cliente = cluster.client
        cola_trabajo = cliente.load_balanced_view()
        
        print(f"¡Éxito! Conectado a {len(cliente.ids)} workers asíncronos.")

        print("\n2. Generando Fuzzing (50,000 mutaciones aleatorias)...")
        num_simulaciones = 50000
        
        betas = np.random.uniform(0.5, 1.2, num_simulaciones)  
        sigmas = np.random.uniform(0.1, 0.4, num_simulaciones) 
        gammas = np.random.uniform(0.01, 0.1, num_simulaciones) 
        
        mutaciones = list(zip(betas, sigmas, gammas))

        print("\n3. Inyectando tareas a la cola de trabajo...")
        inicio_tiempo = time.time()
        
        # Enviamos las tareas
        resultado_async = cola_trabajo.map_async(simulacion_aislada, mutaciones)
        resultado_async.wait_interactive()
        
        # Recolectamos resultados
        resultados_finales = resultado_async.get()
        fin_tiempo = time.time()

        print("\n--- ANÁLISIS DE RESULTADOS MONTE CARLO ---")
        print(f"Tiempo total de cómputo: {fin_tiempo - inicio_tiempo:.2f} segundos")
        print(f"Peor escenario mutado: {max(resultados_finales):.0f} equipos infectados simultáneamente.")
        print(f"Escenario promedio: {np.mean(resultados_finales):.0f} equipos infectados simultáneamente.")
        
    print("\nSimulación finalizada. Clúster apagado correctamente.")

if __name__ == '__main__':
    ejecutar_montecarlo()