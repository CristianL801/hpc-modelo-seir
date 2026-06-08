import numpy as np
from mpi4py import MPI

def modelo_seir_acoplado(t, Y, beta, sigma, gamma, N_local, I_vecinos):
    """
    Calcula las derivadas del modelo SEIR integrando los infectados de VLANs vecinas.
    (Cumple PEP 257).
    
    Args:
        t (float): Tiempo actual.
        Y (np.ndarray): Estado local [S, E, I, P].
        beta, sigma, gamma (float): Parámetros epidemiológicos.
        N_local (int): Población de la VLAN local.
        I_vecinos (float): Suma de infectados en las VLANs vecinas.
    """
    S, E, I, P = Y
    
    # El contagio local SÍ se ve afectado por los infectados de las redes vecinas
    I_total_expuesto = I + (I_vecinos * 0.1) # Asumimos un 10% de filtración entre VLANs
    
    dSdt = -beta * (S * I_total_expuesto) / N_local
    dEdt = (beta * (S * I_total_expuesto) / N_local) - (sigma * E)
    dIdt = (sigma * E) - (gamma * I)
    dPdt = gamma * I
    
    return np.array([dSdt, dEdt, dIdt, dPdt])

def intercambiar_halo(comm, I_local, vecino_izq, vecino_der):
    """
    Barrera de sincronización MPI. Intercambia la variable 'I' con los nodos adyacentes.
    Usa sendrecv para evitar Deadlocks (Interbloqueos).
    """
    # 1. Enviar hacia la izquierda, recibir desde la derecha
    I_recibido_der = comm.sendrecv(sendobj=I_local, dest=vecino_izq, source=vecino_der)
    
    # 2. Enviar hacia la derecha, recibir desde la izquierda
    I_recibido_izq = comm.sendrecv(sendobj=I_local, dest=vecino_der, source=vecino_izq)
    
    # Si el vecino era PROC_NULL (extremos), mpi4py devuelve None. Lo pasamos a 0.
    I_der = I_recibido_der if I_recibido_der is not None else 0
    I_izq = I_recibido_izq if I_recibido_izq is not None else 0
    
    return I_der + I_izq

def rk4_step_mpi(f, t, Y, h, comm, vecino_izq, vecino_der, *args):
    """
    Integrador RK4 modificado para entornos distribuidos.
    Requiere comunicación de red (Halo) para cada pendiente intermedia.
    """
    # k1: Estado inicial del paso
    I_vecinos = intercambiar_halo(comm, Y[2], vecino_izq, vecino_der)
    k1 = f(t, Y, *args, I_vecinos)
    
    # k2: Mitad del paso
    Y_k2 = Y + (h/2)*k1
    I_vecinos = intercambiar_halo(comm, Y_k2[2], vecino_izq, vecino_der)
    k2 = f(t + h/2, Y_k2, *args, I_vecinos)
    
    # k3: Mitad del paso (refinado)
    Y_k3 = Y + (h/2)*k2
    I_vecinos = intercambiar_halo(comm, Y_k3[2], vecino_izq, vecino_der)
    k3 = f(t + h/2, Y_k3, *args, I_vecinos)
    
    # k4: Fin del paso
    Y_k4 = Y + h*k3
    I_vecinos = intercambiar_halo(comm, Y_k4[2], vecino_izq, vecino_der)
    k4 = f(t + h, Y_k4, *args, I_vecinos)
    
    return Y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def simular_red_distribuida():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N_global = 100000
    N_local = N_global // size
    
    vecino_izq = rank - 1 if rank > 0 else MPI.PROC_NULL
    vecino_der = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    
    # Configuración epidemiológica
    beta, sigma, gamma = 0.8, 0.2, 0.05
    T_max, h = 150, 0.1
    tiempos = np.arange(0, T_max, h)
    
    # Solo el núcleo maestro (VLAN 0) sufre el ataque inicial (Paciente Cero)
    I0 = 10 if rank == 0 else 0 
    Y_actual = np.array([N_local - I0, 0, I0, 0], dtype=float)
    
    if rank == 0:
        print(f"[*] Iniciando simulación HPC distribuida con {size} núcleos...")
        
    for i, t in enumerate(tiempos):
        Y_actual = rk4_step_mpi(modelo_seir_acoplado, t, Y_actual, h, comm, vecino_izq, vecino_der, beta, sigma, gamma, N_local)
    
    # Al final, recolectamos (Gather) los datos de todos los núcleos en el núcleo 0
    I_final_local = Y_actual[2]
    I_final_red = comm.reduce(I_final_local, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"[*] Simulación MPI finalizada.")
        print(f"[*] Total de infectados en la red al día {T_max}: {I_final_red:.2f} equipos.")

if __name__ == "__main__":
    simular_red_distribuida()