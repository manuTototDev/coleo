import os
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure


class AgentePikachu(nn.Module):
    """Modelo LSTM compacto para asignar fichas a cada símbolo."""

    def __init__(self, input_size, hidden_size, output_size, apuesta_maxima):
        super().__init__()
        self.apuesta_maxima = apuesta_maxima
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out) * self.apuesta_maxima


class AutoInitAgent:
    """Agente autosuficiente que aprende solo a partir de resultados observados."""

    def __init__(
        self,
        simbolos,
        ventana=10,
        apuesta_maxima=9,
        archivo_modelo="cerebro_pikachu_final.pth",
    ):
        self.simbolos = list(simbolos)
        self.n_frutas = len(self.simbolos)
        self.ventana = ventana
        self.apuesta_maxima = apuesta_maxima
        self.archivo_modelo = archivo_modelo
        self.modelo = AgentePikachu(self.n_frutas, 64, self.n_frutas, apuesta_maxima)
        self.optimizer = optim.Adam(self.modelo.parameters(), lr=0.002)
        self.historial_tiradas = []
        self.saldo_historico = [0]
        self._ultimo_input = None
        self._prob_exploracion = 0.05

        self._cargar()

    def _cargar(self):
        if os.path.exists(self.archivo_modelo):
            checkpoint = torch.load(self.archivo_modelo)
            self.modelo.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.historial_tiradas = checkpoint.get("historial", [])

    def _guardar(self):
        torch.save(
            {
                "model_state_dict": self.modelo.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "historial": self.historial_tiradas,
            },
            self.archivo_modelo,
        )

    def _estado_actual(self):
        if len(self.historial_tiradas) < self.ventana:
            faltantes = self.ventana - len(self.historial_tiradas)
            estado = [np.zeros(self.n_frutas) for _ in range(faltantes)] + self.historial_tiradas
        else:
            estado = self.historial_tiradas[-self.ventana :]
        return np.array(estado)

    def elegir_apuesta(self):
        estado = self._estado_actual()
        input_tensor = torch.FloatTensor(estado).unsqueeze(0)

        with torch.no_grad():
            pred = self.modelo(input_tensor).numpy()[0]

        if np.random.rand() < self._prob_exploracion:
            apuesta_ia = np.random.randint(0, 2, size=self.n_frutas)
        else:
            apuesta_ia = np.clip(np.round(pred), 0, self.apuesta_maxima).astype(int)

        if np.sum(apuesta_ia) == 0:
            apuesta_ia[0] = 1

        self._ultimo_input = input_tensor
        return apuesta_ia

    def registrar_resultado(self, resultado_nombre, recompensa_neta):
        if resultado_nombre not in self.simbolos:
            return

        vec_res = np.zeros(self.n_frutas)
        res_idx = self.simbolos.index(resultado_nombre)
        vec_res[res_idx] = 1
        self.historial_tiradas.append(vec_res)
        self.saldo_historico.append(self.saldo_historico[-1] + recompensa_neta)

        if self._ultimo_input is None:
            return

        self.optimizer.zero_grad()
        pred_train = self.modelo(self._ultimo_input)
        loss = -torch.mean(torch.log(pred_train + 1e-5) * recompensa_neta)
        loss.backward()
        self.optimizer.step()

        if len(self.historial_tiradas) % 100 == 0:
            self._guardar()

    def estado_memoria(self):
        return self._estado_actual()

    def saldo(self):
        return self.saldo_historico


class AutoInitDashboard:
    """Panel de control reutilizable para incrustar dentro de Tkinter."""

    def __init__(self, simbolos, apuesta_maxima):
        self.simbolos = simbolos
        self.apuesta_maxima = apuesta_maxima
        self.fig = Figure(figsize=(6, 5), facecolor="#f0f0f0")
        self.ax_saldo = self.fig.add_subplot(2, 2, 1)
        self.ax_red = self.fig.add_subplot(2, 2, 2)
        self.ax_apuestas = self.fig.add_subplot(2, 2, 3)
        self.ax_memoria = self.fig.add_subplot(2, 2, 4)

    def _dibujar_red(self, apuesta_ia):
        self.ax_red.clear()
        self.ax_red.set_facecolor("#1a1a1a")
        capas = [len(self.simbolos), 8, 8, len(self.simbolos)]
        x_pos = [0, 1, 2, 3]
        for i in range(len(capas) - 1):
            y_curr = np.linspace(0.1, 0.9, capas[i])
            y_next = np.linspace(0.1, 0.9, capas[i + 1])
            for yc in y_curr:
                for yn in y_next:
                    color_l = "red" if i == 2 else "#333333"
                    alpha_l = 0.05 if i == 2 else 0.05
                    self.ax_red.plot(
                        [x_pos[i], x_pos[i + 1]], [yc, yn], color=color_l, alpha=alpha_l, lw=0.5
                    )
        for i, n in enumerate(capas):
            y_space = np.linspace(0.1, 0.9, n)
            for j, y in enumerate(y_space):
                color_n = "#555555"
                if i == 3:
                    intensidad = apuesta_ia[j] / float(self.apuesta_maxima)
                    color_n = (intensidad, 0.0, 0.0)
                    if apuesta_ia[j] > 0:
                        self.ax_red.scatter(
                            x_pos[i], y, s=150, color="red", alpha=intensidad * 0.3, zorder=2
                        )
                self.ax_red.scatter(
                    x_pos[i], y, s=100, color=color_n, edgecolors="white", lw=0.5, zorder=5
                )
        self.ax_red.axis("off")

    def actualizar(self, apuesta_ia, estado_actual, saldo_historico, nivel_tanque, canvas=None):
        self.ax_saldo.clear()
        self.ax_saldo.plot(saldo_historico, color="red", lw=1.2)
        self.ax_saldo.set_title(
            f"Saldo Neto | Tanque Máquina: {round(nivel_tanque, 2)}",
            fontsize=10,
            fontweight="bold",
        )
        self.ax_saldo.set_facecolor("#e0e0e0")

        self._dibujar_red(apuesta_ia)

        self.ax_apuestas.clear()
        self.ax_apuestas.bar(self.simbolos, apuesta_ia, color="red", alpha=0.8)
        self.ax_apuestas.set_ylim(0, self.apuesta_maxima)
        self.ax_apuestas.tick_params(axis="x", labelrotation=45, labelsize=8)

        self.ax_memoria.clear()
        if len(estado_actual) > 0:
            sns.heatmap(
                estado_actual,
                xticklabels=self.simbolos,
                cmap="Greys",
                cbar=False,
                ax=self.ax_memoria,
            )

        self.fig.tight_layout()
        if canvas is not None:
            canvas.draw_idle()


__all__ = ["AutoInitAgent", "AutoInitDashboard"]
