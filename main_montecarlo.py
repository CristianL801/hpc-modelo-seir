import time
import numpy as np
import ipyparallel as ipp
import matplotlib.pyplot as plt
import os
    

def simulacion_aislada(params):
    """
    Ejecuta una simulación aislada del modelo SEIR para evaluar una mutación específica.
    
    Args:
        params (tuple): Tupla que contiene las tasas epidemiológicas mutadas (beta, sigma, gamma).
            - beta (float): Tasa de propagación del malware.
            - sigma (float): Tasa de latencia de la vulnerabilidad.
            - gamma (float): Velocidad de parcheo del equipo de TI.
            
    Returns:
        float: El número máximo de equipos infectados simultáneamente (pico de la infección) 
               durante toda la simulación.
    """
    # Importaciones locales necesarias para el entorno aislado del Worker
    import numpy as np
    from solver_numerico import modelo_seir, rk4_step
    
    # La mutación generada por el fuzzing es desempaquetada
    beta, sigma, gamma = params
    
    # Parámetros fijos de la red corporativa
    N = 100000 
    I0 = 10 
    E0 = 50 
    P0 = 0 
    S0 = N - I0 - E0 - P0 
    
    T_max = 150 
    h = 0.5
    tiempos = np.arange(0, T_max, h)
    
    Y_actual = np.array([S0, E0, I0, P0])
    
    # Variable para almacenar el peor momento de esta mutación
    pico_maximo_infectados = 0
    
    # Bucle de integración RK4
    for t in tiempos:
        Y_actual = rk4_step(modelo_seir, t, Y_actual, h, beta, sigma, gamma, N)
        
        # Registramos si la iteración superó el pico histórico de infectados (índice 2)
        if Y_actual[2] > pico_maximo_infectados:
            pico_maximo_infectados = Y_actual[2]
            
    return pico_maximo_infectados

def ejecutar_montecarlo():
    """
    Función principal (Cliente): Genera el fuzzing y delega las tareas a la cola.
    """
    
    print("1. Encendiendo el clúster local (esto puede tomar unos segundos)...")
    
    # 6 motores se inician directamente desde el script
    with ipp.Cluster(n=6) as cliente:
        
        # Los workers añaden la carpeta a su radar
        ruta_script = os.path.dirname(os.path.abspath(__file__))
        cliente[:].execute(f"import sys; sys.path.append(r'{ruta_script}')")

        # Se conecta al cliente una vez que los motores están listos
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

        tiempo_paralelo = fin_tiempo - inicio_tiempo

        print("\n--- ANÁLISIS DE RESULTADOS MONTE CARLO ---")
        print(f"Tiempo total de cómputo: {fin_tiempo - inicio_tiempo:.2f} segundos")
        print(f"Peor escenario mutado: {max(resultados_finales):.0f} equipos infectados simultáneamente.")
        print(f"Escenario promedio: {np.mean(resultados_finales):.0f} equipos infectados simultáneamente.")
        
        print("\n4. Generando gráfica de la Ley de Amdahl")
        
        # Se estima una fracción secuencial muy baja para ipyparallel (ej. 0.5% por el overhead mínimo)
        f_secuencial = 0.005 
        nucleos = np.arange(1, 16)
        
        # Fórmulas de Amdahl 
        speedup_teorico = 1 / (f_secuencial + (1 - f_secuencial) / nucleos)
        speedup_ideal = nucleos # Escalabilidad lineal ideal
        
        # Calculamos dónde se ubica la ejecución actual en el gráfico
        # Estimación del tiempo secuencial teórico: T_paralelo * num_nucleos
        tiempo_secuencial_estimado = tiempo_paralelo * 6 
        speedup_real_medido = tiempo_secuencial_estimado / tiempo_paralelo
        
        plt.figure(figsize=(10, 6))
        plt.plot(nucleos, speedup_ideal, 'g--', label='Speedup Ideal (Lineal - Colas de Trabajo)', linewidth=2)
        plt.plot(nucleos, speedup_teorico, 'b-', label='Speedup Teórico (Amdahl - ipyparallel)', linewidth=2)
        plt.plot(6, speedup_real_medido, 'ro', markersize=10, label=f'Tu Cómputo Actual ({len(cliente.ids)} Cores)')
        
        plt.title('Análisis de Escalabilidad: Ley de Amdahl en Fase 3', fontsize=12, fontweight='bold')
        plt.xlabel('Número de Procesadores / Núcleos', fontsize=10)
        plt.ylabel('Aceleración (Speedup)', fontsize=10)
        plt.xlim(1, 14)
        plt.ylim(1, 14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        print("Cerrando el clúster y desplegando gráfico...")
        
    # El gráfico se muestra después de cerrar el clúster a fin de liberar memoria
    plt.show()

    print("\nSimulación finalizada. Clúster apagado correctamente.")

if __name__ == '__main__':
    ejecutar_montecarlo()