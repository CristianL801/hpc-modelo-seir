import numpy as np
import matplotlib.pyplot as plt

def generar_grafico_rk4_overhead():
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(12, 7))

    # Tiempos discretos del paso RK4
    t0 = 0.0      # t_n
    tm = 0.5      # t_n + h/2 (Medio camino)
    t1 = 1.0      # t_n + h   (Paso completo)

    # Estado inicial de la variable (Ej. Nodos Infectados)
    y0 = 1.0

    # Pendientes (k) seleccionadas para que el gráfico sea didáctico y claro
    k1 = 1.2
    k2 = 2.0
    k3 = 1.5
    k4 = 0.5

    # --- TRAZADO DE LAS 4 PENDIENTES ---
    
    # Paso 1: k1 (Azul) - Se calcula localmente con el estado actual
    y_k1 = y0 + k1 * tm
    ax.plot([t0, tm], [y0, y_k1], color='#1f77b4', linestyle='-', lw=2.5, label=r'Pendiente $k_1$ (Cálculo Local)')
    ax.plot(tm, y_k1, marker='o', color='#1f77b4', markersize=8)
    ax.text(tm - 0.02, y_k1 + 0.03, r'Evaluar $k_2$ aquí', color='#1f77b4', ha='right', fontsize=11)

    # Paso 2: k2 (Verde) - Usa la pendiente estimada para proyectar a medio camino
    y_k2 = y0 + k2 * tm
    ax.plot([t0, tm], [y0, y_k2], color='#2ca02c', linestyle='-', lw=2.5, label=r'Pendiente $k_2$ (Medio camino)')
    ax.plot(tm, y_k2, marker='o', color='#2ca02c', markersize=8)
    ax.text(tm - 0.02, y_k2 + 0.03, r'Evaluar $k_3$ aquí', color='#2ca02c', ha='right', fontsize=11)

    # Paso 3: k3 (Naranja) - Corrige el tiro a medio camino y proyecta al final del paso
    y_k3 = y0 + k3 * t1
    ax.plot([t0, t1], [y0, y_k3], color='#ff7f0e', linestyle='-', lw=2.5, label=r'Pendiente $k_3$ (Medio camino corregido)')
    ax.plot(t1, y_k3, marker='o', color='#ff7f0e', markersize=8)
    ax.text(t1 - 0.02, y_k3 + 0.03, r'Evaluar $k_4$ aquí', color='#ff7f0e', ha='right', fontsize=11)

    # Paso 4: k4 (Púrpura) - Usa la pendiente final para proyectar
    y_k4 = y0 + k4 * t1
    ax.plot([t0, t1], [y0, y_k4], color='#9467bd', linestyle='-', lw=2.5, label=r'Pendiente $k_4$ (Al final del paso)')
    ax.plot(t1, y_k4, marker='o', color='#9467bd', markersize=8)

    # Resultado Final (Rojo) - El promedio de todo
    k_promedio = (k1 + 2*k2 + 2*k3 + k4) / 6
    y_final = y0 + k_promedio * t1
    ax.plot([t0, t1], [y0, y_final], color='#d62728', linestyle='-', lw=4, label=r'Avance Promediado Real ($Y_{n+1}$)')
    ax.plot(t1, y_final, marker='o', color='#d62728', markersize=10, zorder=5)

    # --- LAS BARRERAS DE OVERHEAD MPI (LO CRÍTICO DE LA PPT) ---
    
    # Barrera en t_n + h/2 (Afecta el cálculo de k2 y k3)
    ax.axvline(x=tm, color='red', linestyle='--', lw=2, alpha=0.8, zorder=1)
    ax.text(tm + 0.02, 2.2, 'BARRERA MPI 1:\nEl proceso pausa y espera\nlos datos de la VLAN\nvecina (Zonas de Halo)\npara evaluar $k_2$ y $k_3$',
            color='white', fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='red', edgecolor='darkred', alpha=0.85, boxstyle='round,pad=0.5'))

    # Barrera en t_n + h (Afecta el cálculo de k4)
    ax.axvline(x=t1, color='red', linestyle='--', lw=2, alpha=0.8, zorder=1)
    ax.text(t1 + 0.02, 1.6, 'BARRERA MPI 2:\nNueva pausa de red\npara evaluar $k_4$',
            color='white', fontsize=10, fontweight='bold', 
            bbox=dict(facecolor='red', edgecolor='darkred', alpha=0.85, boxstyle='round,pad=0.5'))

    # --- MARCADORES BASE Y ESTILOS ---
    ax.plot(t0, y0, 'ko', markersize=10, zorder=10)
    ax.text(t0 - 0.03, y0, '$Y_n$', fontsize=14, va='center', ha='right', fontweight='bold')
    ax.text(t1 + 0.03, y_final, '$Y_{n+1}$', fontsize=14, va='center', ha='left', color='#d62728', fontweight='bold')

    # Configuración de Ejes
    ax.set_xticks([t0, tm, t1])
    ax.set_xticklabels(['$t_n$\n(Inicio)', '$t_n + h/2$\n(Medio Camino)', '$t_n + h$\n(Fin del Paso)'], fontsize=12)
    ax.set_xlim(-0.1, 1.35)
    ax.set_ylim(0.8, 2.75)
    
    ax.set_ylabel('Nodos Infectados (I)', fontsize=12, fontweight='bold')
    ax.set_title('El Cuello de Botella RK4: Overhead de Comunicación (Fase 2 - MPI)', fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generar_grafico_rk4_overhead()