import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from maquina import MaquinaPikachu 

# --- 1. CONFIGURACIÓN ---
TABLA_PREMIOS = {
    "manzana": 5, "sandia": 20, "estrella": 30, "77": 40,
    "bar": 100, "campana": 20, "kiwi": 15, "naranja": 10, "cereza": 2
}
FRUTAS = list(TABLA_PREMIOS.keys())
N_FRUTAS = len(FRUTAS)
VENTANA = 10
APUESTA_MAXIMA = 9
ARCHIVO_MODELO = "cerebro_pikachu_final.pth"

# Simulador con reserva inicial para evitar el cero absoluto al arranque
simulador = MaquinaPikachu(retencion=0.15)

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
    print(f">>> Cerebro cargado: {len(historial_tiradas)} jugadas.")

# --- 4. FUNCIONES DE VISUALIZACIÓN (ROJO Y GRIS) ---
def dibujar_red_en_subtrama(ax, apuesta_ia):
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    capas = [N_FRUTAS, 8, 8, N_FRUTAS] 
    x_pos = [0, 1, 2, 3]
    for i in range(len(capas) - 1):
        y_curr = np.linspace(0.1, 0.9, capas[i])
        y_next = np.linspace(0.1, 0.9, capas[i+1])
        for yc in y_curr:
            for yn in y_next:
                color_l = 'red' if i == 2 else '#333333'
                alpha_l = 0.05 if i == 2 else 0.05
                ax.plot([x_pos[i], x_pos[i+1]], [yc, yn], color=color_l, alpha=alpha_l, lw=0.5)
    for i, n in enumerate(capas):
        y_space = np.linspace(0.1, 0.9, n)
        for j, y in enumerate(y_space):
            color_n = '#555555'
            if i == 3:
                intensidad = apuesta_ia[j] / 9.0
                color_n = (intensidad, 0.0, 0.0)
                if apuesta_ia[j] > 0:
                    ax.scatter(x_pos[i], y, s=150, color='red', alpha=intensidad*0.3, zorder=2)
            ax.scatter(x_pos[i], y, s=100, color=color_n, edgecolors='white', lw=0.5, zorder=5)
    ax.axis('off')

def actualizar_dashboard(apuesta_ia, estado_actual, nivel_tanque):
    plt.clf()
    plt.gcf().set_facecolor('#f0f0f0')
    # Saldo
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(saldo_historico, color='red', lw=1.2)
    ax1.set_title(f"Saldo Neto | Tanque Máquina: {nivel_tanque}", fontsize=10, fontweight='bold')
    ax1.set_facecolor('#e0e0e0')
    # Red
    ax2 = plt.subplot(2, 2, 2)
    dibujar_red_en_subtrama(ax2, apuesta_ia)
    # Apuestas
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(FRUTAS, apuesta_ia, color='red', alpha=0.8)
    ax3.set_ylim(0, 10)
    plt.xticks(rotation=45, fontsize=8)
    # Memoria
    ax4 = plt.subplot(2, 2, 4)
    if len(historial_tiradas) > 0:
        sns.heatmap(estado_actual, xticklabels=FRUTAS, cmap="Greys", cbar=False, ax=ax4)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

# --- 5. BUCLE AUTOMÁTICO CON EXPLORACIÓN ---
plt.ion()
fig = plt.figure(figsize=(15, 8))

while True:
    if len(historial_tiradas) < VENTANA:
        estado = [np.zeros(N_FRUTAS) for _ in range(VENTANA)]
    else:
        estado = historial_tiradas[-VENTANA:]
    
    input_tensor = torch.FloatTensor(estado).unsqueeze(0)
    
    # IA propone apuesta
    with torch.no_grad():
        pred = modelo(input_tensor).numpy()[0]
        
        # --- MECANISMO DE EXPLORACIÓN (Anti-Miedo) ---
        # 5% de probabilidad de apuesta aleatoria para rellenar tanque
        if np.random.rand() < 0.05:
            apuesta_ia = np.random.randint(0, 2, size=N_FRUTAS) 
        else:
            apuesta_ia = np.round(pred).astype(int)
    
    total_apostado = np.sum(apuesta_ia)
    
    # --- PROTECCIÓN ANTI-ESTANCAMIENTO ---
    # Si la IA decide no jugar nada, forzamos 1 moneda a la cereza
    # Esto asegura que la máquina siempre reciba algo y el tanque suba
    if total_apostado == 0:
        apuesta_ia[FRUTAS.index("cereza")] = 1
        total_apostado = 1

    # Ejecutar en simulador
    resultado_nombre, nivel_tanque = simulador.girar(total_apostado)
    res_idx = FRUTAS.index(resultado_nombre)
    
    # Calcular Recompensa
    ganancia_bruta = apuesta_ia[res_idx] * TABLA_PREMIOS[resultado_nombre]
    recompensa_neta = ganancia_bruta - total_apostado
    saldo_historico.append(saldo_historico[-1] + recompensa_neta)

    # Entrenamiento
    optimizer.zero_grad()
    pred_train = modelo(input_tensor)
    loss = -torch.mean(torch.log(pred_train + 1e-5) * recompensa_neta)
    loss.backward()
    optimizer.step()

    # Historial
    vec_res = np.zeros(N_FRUTAS)
    vec_res[res_idx] = 1
    historial_tiradas.append(vec_res)
    
    # Dashboard (cada 10 jugadas para mayor velocidad de entrenamiento)
    if len(historial_tiradas) % 10 == 0:
        actualizar_dashboard(apuesta_ia, estado, nivel_tanque)
        
        if len(historial_tiradas) % 100 == 0:
            torch.save({
                'model_state_dict': modelo.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'historial': historial_tiradas
            }, ARCHIVO_MODELO)
            print(f"Jugada: {len(historial_tiradas)} | Saldo: {round(saldo_historico[-1], 2)} | Tanque: {nivel_tanque}")