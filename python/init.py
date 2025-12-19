import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURACIÓN DE LA MÁQUINA ---
TABLA_PREMIOS = {
    "manzana": 5, "sandia": 20, "estrella": 30, "77": 40,
    "bar": 100, "campana": 20, "kiwi": 15, "naranja": 10, "cereza": 2
}
FRUTAS = list(TABLA_PREMIOS.keys())
N_FRUTAS = len(FRUTAS)
VENTANA = 10
APUESTA_MAXIMA = 9
ARCHIVO_MODELO = "cerebro_pikachu_final.pth"

# --- 2. ARQUITECTURA LSTM ---
class AgentePikachu(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AgentePikachu, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out) * APUESTA_MAXIMA

# --- 3. INICIALIZACIÓN ---
modelo = AgentePikachu(N_FRUTAS, 64, N_FRUTAS)
optimizer = optim.Adam(modelo.parameters(), lr=0.002)
historial_tiradas = []
saldo_historico = [0]

if os.path.exists(ARCHIVO_MODELO):
    checkpoint = torch.load(ARCHIVO_MODELO)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    historial_tiradas = checkpoint.get('historial', [])
    print(f">>> Cerebro cargado: {len(historial_tiradas)} jugadas previas.")

# --- 4. FUNCIÓN DE VISUALIZACIÓN DE RED (BOLITAS Y PALITOS) ---
def dibujar_red_en_subtrama(ax, apuesta_ia):
    ax.clear()
    ax.set_facecolor('#1a1a1a') # Fondo oscuro técnico
    capas = [N_FRUTAS, 8, 8, N_FRUTAS] 
    x_pos = [0, 1, 2, 3]
    
    # Dibujar Conexiones (Palitos)
    for i in range(len(capas) - 1):
        y_curr_space = np.linspace(0.1, 0.9, capas[i])
        y_next_space = np.linspace(0.1, 0.9, capas[i+1])
        for y_c in y_curr_space:
            for y_n in y_next_space:
                # Las conexiones hacia la salida se iluminan si hay apuesta
                color_link = '#444444' # Gris por defecto
                alpha_link = 0.1
                if i == 2: # Conexión hacia salida
                    color_link = 'red'
                    alpha_link = 0.05
                ax.plot([x_pos[i], x_pos[i+1]], [y_c, y_n], color=color_link, alpha=alpha_link, lw=0.5)

    # Dibujar Neuronas (Bolitas)
    for i, n_neuronas in enumerate(capas):
        y_space = np.linspace(0.1, 0.9, n_neuronas)
        for j, y in enumerate(y_space):
            color_node = '#777777' # Gris medio
            if i == 3: # Capa de Salida
                valor = apuesta_ia[j]
                intensidad = valor / 9.0
                color_node = (intensidad, 0.1, 0.1) # Gradiente de Rojo
                if valor > 0: # Brillo de actividad
                    ax.scatter(x_pos[i], y, s=200, color='red', alpha=intensidad*0.4, zorder=2)
            
            ax.scatter(x_pos[i], y, s=120, color=color_node, edgecolors='white', linewidth=0.5, zorder=5)
    
    ax.set_title("Flujo de Decisión Neuronal", color='red', fontsize=10, fontweight='bold')
    ax.axis('off')

# --- 5. ACTUALIZACIÓN DEL DASHBOARD ---
def actualizar_dashboard(apuesta_ia, estado_actual):
    plt.clf()
    plt.gcf().set_facecolor('#f0f0f0')

    # [1] Gráfico de Saldo (Rojo sobre Gris)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(saldo_historico, color='red', linewidth=2, label='Saldo Neto')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Rendimiento Acumulado", fontweight='bold')
    ax1.set_facecolor('#e6e6e6')

    # [2] Bolitas y Palitos
    ax2 = plt.subplot(2, 2, 2)
    dibujar_red_en_subtrama(ax2, apuesta_ia)

    # [3] Barras de Apuesta
    ax3 = plt.subplot(2, 2, 3)
    colores = [(x/9.0, 0.1, 0.1) for x in apuesta_ia] # Barras también en gradiente rojo
    ax3.bar(FRUTAS, apuesta_ia, color=colores, edgecolor='black')
    ax3.set_ylim(0, 10)
    plt.xticks(rotation=45)
    ax3.set_title("Apuestas Propuestas (0-9)", fontweight='bold')

    # [4] Memoria LSTM (Gris)
    ax4 = plt.subplot(2, 2, 4)
    if len(historial_tiradas) > 0:
        sns.heatmap(estado_actual, xticklabels=FRUTAS, cmap="Greys", cbar=False, ax=ax4)
    ax4.set_title("Memoria Temporal (Input Histórico)", fontweight='bold')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# --- 6. BUCLE PRINCIPAL ---
plt.ion()
fig = plt.figure(figsize=(15, 8))

while True:
    # Preparar datos de entrada
    if len(historial_tiradas) < VENTANA:
        estado = [np.zeros(N_FRUTAS) for _ in range(VENTANA)]
    else:
        estado = historial_tiradas[-VENTANA:]
    
    input_tensor = torch.FloatTensor(estado).unsqueeze(0)
    
    # IA propone apuesta
    with torch.no_grad():
        pred = modelo(input_tensor).numpy()[0]
        apuesta_ia = np.round(pred).astype(int)
    
    total_apostado = np.sum(apuesta_ia)
    actualizar_dashboard(apuesta_ia, estado)

    print("\n" + "—"*50)
    print(">>> PROPUESTA DE LA IA (Modo Autónomo)")
    hay_apuesta = False
    for i, f in enumerate(FRUTAS):
        if apuesta_ia[i] > 0:
            print(f"    {f.upper()}: {apuesta_ia[i]} monedas")
            hay_apuesta = True
    if not hay_apuesta: print("    IA decide no apostar en este turno.")
    
    print(f"\nInversión total: {total_apostado}")
    input("Haz la apuesta física y presiona ENTER para continuar...")

    # Reportar resultado
    print("\n¿Qué fruta salió?")
    for i, f in enumerate(FRUTAS):
        print(f"{i}:{f[:3]}", end=" | ")
    
    try:
        res_idx = int(input("\nID Resultado: "))
        fruta_ganadora = FRUTAS[res_idx]
        
        # Calcular Recompensa
        ganancia_bruta = apuesta_ia[res_idx] * TABLA_PREMIOS[fruta_ganadora]
        recompensa_neta = ganancia_bruta - total_apostado
        saldo_historico.append(saldo_historico[-1] + recompensa_neta)

        # Entrenar Red
        optimizer.zero_grad()
        pred_train = modelo(input_tensor)
        # Loss: Maximizamos recompensa mediante Policy Gradient
        loss = -torch.mean(torch.log(pred_train + 1e-5) * recompensa_neta)
        loss.backward()
        optimizer.step()

        # Guardar historial
        vec_res = np.zeros(N_FRUTAS)
        vec_res[res_idx] = 1
        historial_tiradas.append(vec_res)
        
        # Persistencia
        torch.save({
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'historial': historial_tiradas
        }, ARCHIVO_MODELO)

        print(f"\n[!] RESULTADO: {fruta_ganadora.upper()} | Neto: {recompensa_neta}")

    except Exception as e:
        print(f"Error en entrada: {e}")