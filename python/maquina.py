import tkinter as tk
from tkinter import messagebox
import random
import time
import threading

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from init import AutoInitAgent, AutoInitDashboard

# --- CONFIGURACIÓN Y CONSTANTES ---

# Valores de los símbolos de apuesta (según el panel inferior de la imagen)
VALORES_APUESTA = {
    'Cereza': 2,
    'Manzana': 5,
    'Naranja': 10,
    'Campana': 20,
    'Sandia': 15,
    'Estrella': 30,
    '77': 40,
    'BAR': 100
}
SIMBOLOS_APOSTABLES = list(VALORES_APUESTA.keys())

# Símbolos especiales en el tablero
ESPECIALES = {'Diamante': 'RESPIN'}

# El orden del recorrido cuadrado (Sentido horario, empezando arriba a la izquierda)
# Esta es una aproximación basada en la imagen y diseños estándar de 24 paradas.
RECORRIDO_TABLERO = [
    'Naranja', 'Campana', 'BAR', 'Manzana', 'Sandia', 'Cereza', # Fila Superior
    'Diamante', 'Manzana', 'Estrella', '77', 'Manzana',         # Columna Derecha bajando
    'Campana', 'Naranja', 'Cereza', 'Sandia', 'Manzana',        # Fila Inferior izquierda
    'Diamante', 'Cereza', '77', 'Estrella', 'Manzana', 'BAR'    # Columna Izquierda subiendo
]
# Ajustamos a 24 paradas para cerrar el cuadrado simétricamente
RECORRIDO_TABLERO = [
    'Naranja', 'Campana', 'BAR', 'Manzana', 'Sandia', 'Cereza', # Top 0-5
    'Diamante', 'Manzana', 'Estrella', '77', 'Manzana', 'Cereza', # Right 6-11
    'Campana', 'Naranja', 'Sandia', 'Manzana', 'Diamante', 'Cereza', # Bottom 12-17
    '77', 'Estrella', 'Manzana', 'BAR', 'Campana', 'Naranja' # Left 18-23
]

# --- LÓGICA "REAL" DE LA MÁQUINA (Sistema de Pozo/Reservorio) ---
class LogicaMaquina:
    def __init__(self):
        self.creditos = 1000  # Dinero del jugador
        self.pozo_maquina = 200  # Dinero acumulado dentro de la máquina (para pagar premios)
        self.retorno_objetivo = 0.85 # La máquina intenta devolver el 85% a largo plazo
        self.apuestas_actuales = {s: 0 for s in SIMBOLOS_APOSTABLES}

    def apostar(self, simbolo):
        if self.creditos > 0:
            self.creditos -= 1
            self.apuestas_actuales[simbolo] += 1
            return True
        return False

    def total_apostado(self):
        return sum(self.apuestas_actuales.values())

    def determinar_resultado(self):
        total_bet = self.total_apostado()
        if total_bet == 0:
            return None, 0, False, None

        # 1. Alimentar el Pozo: La máquina se queda una parte, el resto va al pozo para premios
        ganancia_casa = total_bet * (1 - self.retorno_objetivo)
        aporte_al_pozo = total_bet - ganancia_casa
        self.pozo_maquina += aporte_al_pozo
        
        print(f"DEBUG: Pozo actual: {self.pozo_maquina:.2f}")

        # 2. Determinar qué premios son PAGABLES según el pozo actual
        posibles_ganadores = []
        pesos = []

        for simbolo in RECORRIDO_TABLERO:
            ganancia_potencial = 0
            if simbolo in VALORES_APUESTA:
                 # Cuánto ganaría el jugador si sale este símbolo
                ganancia_potencial = self.apuestas_actuales[simbolo] * VALORES_APUESTA[simbolo]
            
            # Peso base (frecuencia natural): Cerezas salen mucho, BAR poco.
            peso_base = 10
            if simbolo in ['BAR', '77']: peso_base = 2
            elif simbolo in ['Estrella', 'Campana']: peso_base = 5
            elif simbolo == 'Diamante': peso_base = 8
            elif simbolo in ['Cereza', 'Manzana']: peso_base = 20
            
            # Factor de realidad: ¿Tiene la máquina dinero para pagar esto?
            if simbolo in ESPECIALES:
                 # Diamante siempre es posible, no cuesta dinero del pozo
                 posibles_ganadores.append(simbolo)
                 pesos.append(peso_base)
            elif ganancia_potencial <= self.pozo_maquina:
                # Sí puede pagarlo. Aumentamos ligeramente la prob si el pozo está muy lleno.
                factor_pozo = 1 + (self.pozo_maquina / 1000) 
                posibles_ganadores.append(simbolo)
                pesos.append(peso_base * factor_pozo)
            else:
                # No puede pagarlo. Probabilidad casi nula (pero no cero absoluto para "casi gano")
                posibles_ganadores.append(simbolo)
                pesos.append(0.1)

        # 3. Selección ponderada
        resultado_simbolo = random.choices(posibles_ganadores, weights=pesos, k=1)[0]
        
        # Encuentra el índice en el tablero (si hay repetidos, toma uno al azar de los que coinciden)
        indices_posibles = [i for i, x in enumerate(RECORRIDO_TABLERO) if x == resultado_simbolo]
        resultado_indice = random.choice(indices_posibles)

        ganancia_total = 0
        es_respin = False

        if resultado_simbolo in VALORES_APUESTA:
            ganancia_total = self.apuestas_actuales[resultado_simbolo] * VALORES_APUESTA[resultado_simbolo]
            # Descontar del pozo si hubo ganancia
            self.pozo_maquina -= ganancia_total
            if self.pozo_maquina < 0: self.pozo_maquina = 0 # Seguridad
        elif resultado_simbolo == 'Diamante':
            es_respin = True

        self.creditos += ganancia_total
        
        # Limpiar apuestas si no es respin
        if not es_respin:
             self.apuestas_actuales = {s: 0 for s in SIMBOLOS_APOSTABLES}

        return resultado_indice, ganancia_total, es_respin, resultado_simbolo

# --- INTERFAZ GRÁFICA (GUI) ---
class PikachuSlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pikachu Virtual Slot - Lógica Real")
        self.root.geometry("1200x780")
        self.root.configure(bg='#f0e68c') # Un color amarillento base

        self.logica = LogicaMaquina()
        self.girando = False
        self.velocidad_rapida = tk.BooleanVar(value=False)
        self.auto_corriendo = False
        self.agente = AutoInitAgent(simbolos=SIMBOLOS_APOSTABLES, apuesta_maxima=9)
        self.dashboard = AutoInitDashboard(SIMBOLOS_APOSTABLES, apuesta_maxima=9)
        self._apuesta_total_actual = 0
        self._ultimo_simbolo = None
        self._canvas_dashboard = None
        self._auto_thread = None
        self.labels_tablero = [] # Almacenará los widgets del recorrido

        self.setup_gui()
        self.actualizar_info()
        self._render_dashboard(np.zeros(len(SIMBOLOS_APOSTABLES)), self.agente.estado_memoria())

    def setup_gui(self):
        # Contenedor principal: dashboard a la izquierda, máquina a la derecha
        contenedor = tk.Frame(self.root, bg='#f0e68c')
        contenedor.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_dashboard = tk.Frame(contenedor, bg='#f0e68c')
        frame_dashboard.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._canvas_dashboard = FigureCanvasTkAgg(self.dashboard.fig, master=frame_dashboard)
        self._canvas_dashboard.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        frame_maquina = tk.Frame(contenedor, bg='#f0e68c')
        frame_maquina.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Marco Superior (El Tablero Giratorio)
        frame_tablero = tk.Frame(frame_maquina, bg='black', bd=5, relief=tk.RIDGE)
        frame_tablero.pack(pady=10)

        # Mapeo de índices lineales a coordenadas de cuadrícula (Grid) para hacer un cuadrado hueco
        # Tamaño del grid: 8x8.
        grid_map = {}
        # Top row (col 1 to 6)
        for i in range(6): grid_map[i] = (0, i + 1)
        # Right col (row 1 to 6)
        for i in range(6): grid_map[i+6] = (i + 1, 7)
        # Bottom row (col 6 to 1 backwards)
        for i in range(6): grid_map[i+12] = (7, 6 - i)
        # Left col (row 6 to 1 backwards)
        for i in range(6): grid_map[i+18] = (6 - i, 0)

        self.labels_tablero = [None] * 24
        
        for i, simbolo in enumerate(RECORRIDO_TABLERO):
            row, col = grid_map[i]
            # Colores para diferenciar símbolos
            bg_color = '#ddd'
            fg_color = 'black'
            if simbolo == 'BAR': bg_color = 'red'; fg_color='white'
            elif simbolo == '77': bg_color = 'blue'; fg_color='white'
            elif simbolo == 'Diamante': bg_color = 'cyan'
            elif simbolo == 'Estrella': bg_color = 'gold'
            
            lbl = tk.Label(frame_tablero, text=simbolo, width=8, height=3, 
                           bg=bg_color, fg=fg_color, bd=2, relief=tk.RAISED, font=('Arial', 8, 'bold'))
            lbl.grid(row=row, column=col, padx=1, pady=1)
            self.labels_tablero[i] = lbl

        # Imagen central (Representador de Pikachu)
        center_label = tk.Label(frame_tablero, text="PIKACHU\nSLOT", bg='yellow', font=('Arial', 14, 'bold'), width=16, height=8)
        center_label.grid(row=1, column=1, rowspan=6, columnspan=6)

        # 2. Marco Medio (Información y Controles de Giro)
        frame_info = tk.Frame(frame_maquina, bg='#f0e68c')
        frame_info.pack(pady=5)

        self.lbl_creditos = tk.Label(frame_info, text="Créditos: 0", font=('Arial', 12), bg='#f0e68c')
        self.lbl_creditos.grid(row=0, column=0, padx=10)
        
        self.lbl_total_bet = tk.Label(frame_info, text="Apuesta Total: 0", font=('Arial', 12), bg='#f0e68c')
        self.lbl_total_bet.grid(row=0, column=1, padx=10)

        self.btn_girar = tk.Button(frame_info, text="¡GIRAR!", bg='orange', font=('Arial', 12, 'bold'), command=self.iniciar_giro_thread)
        self.btn_girar.grid(row=0, column=2, padx=20)

        chk_velocidad = tk.Checkbutton(frame_info, text="Velocidad Turbo", variable=self.velocidad_rapida, bg='#f0e68c')
        chk_velocidad.grid(row=0, column=3)

        tk.Label(frame_info, text="Tiradas/seg", bg='#f0e68c').grid(row=0, column=4, padx=(10, 0))
        self.slider_tiradas = tk.Scale(frame_info, from_=1, to=10, orient=tk.HORIZONTAL, bg='#f0e68c', highlightthickness=0)
        self.slider_tiradas.set(2)
        self.slider_tiradas.grid(row=0, column=5, padx=5)

        self.btn_auto = tk.Button(frame_info, text="Auto INIT", bg='green', fg='white', command=self.toggle_auto)
        self.btn_auto.grid(row=0, column=6, padx=10)
        
        self.lbl_mensaje = tk.Label(frame_maquina, text="¡Haz tus apuestas!", font=('Arial', 12, 'bold'), bg='#f0e68c', fg='blue')
        self.lbl_mensaje.pack(pady=5)

        # 3. Marco Inferior (Panel de Apuestas)
        frame_apuestas = tk.Frame(frame_maquina, bg='#8B4513', bd=5, relief=tk.SUNKEN) # Marrón tipo madera arcade
        frame_apuestas.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.labels_apuesta_individual = {}

        for i, simbolo in enumerate(SIMBOLOS_APOSTABLES):
            col_frame = tk.Frame(frame_apuestas, bg='#8B4513')
            col_frame.pack(side=tk.LEFT, expand=True, padx=2)

            # Valor del premio
            tk.Label(col_frame, text=str(VALORES_APUESTA[simbolo]), bg='black', fg='yellow', width=5).pack()
            
            # Botón con el nombre del símbolo para apostar
            btn = tk.Button(col_frame, text=simbolo[0:4], width=5, height=2, bg='#cd853f', 
                            command=lambda s=simbolo: self.realizar_apuesta(s))
            btn.pack(pady=2)
            
            # Cantidad apostada actualmente
            lbl_cant = tk.Label(col_frame, text="0", bg='white', width=5, relief=tk.SUNKEN)
            lbl_cant.pack()
            self.labels_apuesta_individual[simbolo] = lbl_cant

    def actualizar_info(self):
        self.lbl_creditos.config(text=f"Créditos: {self.logica.creditos}")
        self.lbl_total_bet.config(text=f"Apuesta Total: {self.logica.total_apostado()}")
        for simbolo, cantidad in self.logica.apuestas_actuales.items():
            self.labels_apuesta_individual[simbolo].config(text=str(cantidad))

    def realizar_apuesta(self, simbolo):
        if self.girando: return
        if self.logica.apostar(simbolo):
            self.actualizar_info()
            self.lbl_mensaje.config(text="Apostando...", fg='blue')
        else:
            messagebox.showwarning("Sin Créditos", "¡No tienes suficientes créditos!")

    # Usamos threading para que la animación no congele la interfaz gráfica
    def iniciar_giro_thread(self):
        if self.girando or self.logica.total_apostado() == 0:
            if self.logica.total_apostado() == 0:
                 self.lbl_mensaje.config(text="¡Debes apostar algo primero!", fg='red')
            return
        
        self._apuesta_total_actual = self.logica.total_apostado()
        self.girando = True
        self.btn_girar.config(state=tk.DISABLED)
        self.lbl_mensaje.config(text="¡Girando!", fg='black')
        
        # Determinar el resultado ANTES de empezar a girar (como las máquinas reales)
        indice_destino, ganancia, es_respin, simbolo = self.logica.determinar_resultado()
        self._ultimo_simbolo = simbolo
        
        hilo = threading.Thread(target=self.animar_giro, args=(indice_destino, ganancia, es_respin, simbolo))
        hilo.start()

    def animar_giro(self, destino_idx, ganancia, es_respin, simbolo):
        velocidad_base = 0.05 if self.velocidad_rapida.get() else 0.1
        vueltas_minimas = 2
        total_pasos = (len(RECORRIDO_TABLERO) * vueltas_minimas) + destino_idx
        
        idx_actual = 0
        for i in range(total_pasos):
            # Iluminar actual
            idx_actual = i % len(RECORRIDO_TABLERO)
            self.labels_tablero[idx_actual].config(bg='yellow', relief=tk.SUNKEN)
            
            # Cálculo de velocidad (empieza rápido, frena al final)
            sleep_time = velocidad_base
            pasos_restantes = total_pasos - i
            if pasos_restantes < 10:
                sleep_time += (10 - pasos_restantes) * 0.03 # Frena progresivamente

            time.sleep(sleep_time)
            
            # Apagar actual (volver a su color original)
            simbolo = RECORRIDO_TABLERO[idx_actual]
            bg_color = '#ddd'
            if simbolo == 'BAR': bg_color = 'red'
            elif simbolo == '77': bg_color = 'blue'
            elif simbolo == 'Diamante': bg_color = 'cyan'
            elif simbolo == 'Estrella': bg_color = 'gold'
            
            self.labels_tablero[idx_actual].config(bg=bg_color, relief=tk.RAISED)

        # Fin del giro: dejar el ganador iluminado
        self.labels_tablero[destino_idx].config(bg='lime green', relief=tk.SUNKEN)
        
        # Actualizar interfaz en el hilo principal
        self.root.after(0, lambda: self.finalizar_giro(ganancia, es_respin, simbolo))

    def finalizar_giro(self, ganancia, es_respin, simbolo):
        self.actualizar_info()
        self.girando = False
        self.btn_girar.config(state=tk.NORMAL)

        if ganancia > 0:
            self.lbl_mensaje.config(text=f"¡GANASTE {ganancia} CRÉDITOS!", fg='green', font=('Arial', 14, 'bold'))
            # Efecto visual simple de parpadeo
            for _ in range(3):
                 self.root.update()
                 time.sleep(0.1)
                 self.lbl_creditos.config(bg='lime')
                 self.root.update()
                 time.sleep(0.1)
                 self.lbl_creditos.config(bg='#f0e68c')
        elif es_respin:
             self.lbl_mensaje.config(text="¡DIAMANTE! GIRA DE NUEVO GRATIS", fg='purple', font=('Arial', 14, 'bold'))
             self.root.update()
             time.sleep(1)
             # Lanzar giro automático (respin)
             self.girando = True # Bloquear manual
             indice_destino, ganancia_re, es_respin_re, simbolo_re = self.logica.determinar_resultado()
             self._ultimo_simbolo = simbolo_re
             hilo_re = threading.Thread(
                 target=self.animar_giro, args=(indice_destino, ganancia_re, es_respin_re, simbolo_re)
             )
             hilo_re.start()
             return # Salimos para no resetear el estado aún
        else:
            self.lbl_mensaje.config(text="Suerte la próxima...", fg='black')

        # Apagar la luz final después de un momento
        time.sleep(0.5)
        simbolo_final = simbolo or RECORRIDO_TABLERO[0]
        bg_color = '#ddd'
        if simbolo_final == 'BAR': bg_color = 'red'
        elif simbolo_final == '77': bg_color = 'blue'
        elif simbolo_final == 'Diamante': bg_color = 'cyan'
        elif simbolo_final == 'Estrella': bg_color = 'gold'
        for lbl in self.labels_tablero:
             if lbl.cget('bg') == 'lime green':
                 lbl.config(bg=bg_color, relief=tk.RAISED)

        self._retroalimentar_agente(simbolo_final, ganancia)

    def _retroalimentar_agente(self, simbolo, ganancia):
        recompensa_neta = ganancia - self._apuesta_total_actual
        self.agente.registrar_resultado(simbolo.lower() if simbolo else "", recompensa_neta)
        self._render_dashboard(np.array(list(self.logica.apuestas_actuales.values())), self.agente.estado_memoria())

    def _render_dashboard(self, apuesta_ia, estado):
        nivel_tanque = self.logica.pozo_maquina
        self.dashboard.actualizar(apuesta_ia, estado, self.agente.saldo(), nivel_tanque, self._canvas_dashboard)

    def aplicar_apuesta_agente(self):
        apuesta_ia = self.agente.elegir_apuesta()
        total_necesario = int(np.sum(apuesta_ia))
        if total_necesario == 0 or self.logica.creditos < total_necesario:
            return False

        self.logica.apuestas_actuales = {s: 0 for s in SIMBOLOS_APOSTABLES}
        for simbolo, cantidad in zip(SIMBOLOS_APOSTABLES, apuesta_ia):
            for _ in range(int(cantidad)):
                self.logica.apostar(simbolo)
        self.actualizar_info()
        return True

    def toggle_auto(self):
        if self.auto_corriendo:
            self.auto_corriendo = False
            self.btn_auto.config(text="Auto INIT", bg='green')
            return

        self.auto_corriendo = True
        self.btn_auto.config(text="Detener Auto", bg='red')
        self._auto_thread = threading.Thread(target=self._loop_auto)
        self._auto_thread.daemon = True
        self._auto_thread.start()

    def _loop_auto(self):
        while self.auto_corriendo:
            if self.girando:
                time.sleep(0.05)
                continue

            if not self.aplicar_apuesta_agente():
                self.lbl_mensaje.config(text="Auto detenido: sin créditos", fg='red')
                self.auto_corriendo = False
                self.btn_auto.config(text="Auto INIT", bg='green')
                break

            self.iniciar_giro_thread()
            while self.girando and self.auto_corriendo:
                time.sleep(0.05)

            delay = max(0.05, 1.0 / max(1, self.slider_tiradas.get()))
            time.sleep(delay)


# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PikachuSlotGUI(root)
    # Iniciar con algunos créditos de prueba
    root.mainloop()
