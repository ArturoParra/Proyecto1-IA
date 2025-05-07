import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import gymnasium as gym
from collections import deque
import threading

class MainApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto de IA - Parra Rivas Arturo Gabriel")
        self.root.geometry("900x700")  # Tamaño inicial de la ventana
        self.root.resizable(False, False)
        
        # Inicializar variables para FrozenLake
        self.env = None
        self.solution_path = []
        self.current_step = 0
        self.is_solving = False
        self.selected_algorithm = tk.StringVar(value="BFS")  # Algoritmo seleccionado por defecto
        
        # Crear frames
        self.pantalla1 = tk.Frame(root)
        self.pantalla2 = tk.Frame(root)
        self.pantalla3 = tk.Frame(root)
        self.main_frame = tk.Frame(root)
        
        self.create_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.centrar_ventana()

    def centrar_ventana(self):
        self.root.update_idletasks()
        ancho = self.root.winfo_width()
        alto = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.root.winfo_screenheight() // 2) - (alto // 2)
        self.root.geometry(f"{ancho}x{alto}+{x}+{y}")

    def mostrar_pantalla(self, pantalla):
        # Ocultar todas las pantallas incluyendo el main_frame
        for p in (self.pantalla1, self.pantalla2, self.pantalla3, self.main_frame):
            p.pack_forget()
        
        # Mostrar la pantalla seleccionada
        pantalla.pack(fill='both', expand=True)
        
        # Si es la pantalla del FrozenLake, inicializar el entorno
        if pantalla == self.pantalla1 and self.env is None:
            self.initialize_frozen_lake()

    def initialize_frozen_lake(self):
        """Inicializa el entorno FrozenLake"""
        self.env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
        self.observation, info = self.env.reset()
        
        # Crear y actualizar la visualización
        self.update_frozen_lake_display()
        
    def update_frozen_lake_display(self):
        """Actualiza la visualización del entorno FrozenLake"""
        if self.env is None:
            return
            
        # Obtener la representación visual del entorno
        img = self.env.render()
        
        # Usar matplotlib para mostrar la imagen
        self.frozen_lake_fig.clear()
        ax = self.frozen_lake_fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        self.frozen_lake_canvas.draw()
        
    def bfs_solve(self):
        """Soluciona el FrozenLake usando BFS"""
        if self.is_solving:
            return
            
        self.is_solving = True
        self.solution_text.config(text="Buscando solución con BFS...")
        
        # Ejecutar BFS en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._bfs_worker).start()
    
    def _bfs_worker(self):
        """Implementa el algoritmo BFS en un hilo separado"""
        # Reiniciar el entorno
        self.observation, _ = self.env.reset()
        self.update_frozen_lake_display()
        
        # Estructura del mapa de FrozenLake (4x4 por defecto)
        desc = self.env.unwrapped.desc
        nrow, ncol = desc.shape
        
        # Diccionario para mapear coordenadas a estados
        coords_to_state = {}
        for s in range(nrow * ncol):
            row, col = s // ncol, s % ncol
            coords_to_state[(row, col)] = s
        
        # Estado inicial y meta
        start_coords = (0, 0)  # Esquina superior izquierda
        goal_coords = None
        
        # Encontrar la meta (posición de 'G')
        for i in range(nrow):
            for j in range(ncol):
                if desc[i][j] == b'G':
                    goal_coords = (i, j)
        
        if not goal_coords:
            self.solution_text.config(text="Error: No se encontró la meta")
            self.is_solving = False
            return
        
        # Direcciones: arriba, derecha, abajo, izquierda
        actions = [3, 2, 1, 0]  # UP, RIGHT, DOWN, LEFT
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Implementar BFS
        queue = deque([(start_coords, [])])  # (posición, camino)
        visited = set([start_coords])
        
        while queue:
            (row, col), path = queue.popleft()
            
            # Si llegamos a la meta
            if (row, col) == goal_coords:
                self.solution_path = path
                self.solution_text.config(text=f"Solución encontrada en {len(path)} pasos")
                self.execute_solution()
                return
            
            # Explorar vecinos
            for action, (dr, dc) in zip(actions, directions):
                new_row, new_col = row + dr, col + dc
                
                # Verificar límites del mapa
                if 0 <= new_row < nrow and 0 <= new_col < ncol:
                    # Verificar que no sea un agujero
                    if desc[new_row][new_col] != b'H' and (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))
                        queue.append(((new_row, new_col), path + [action]))
        
        # Si no hay solución
        self.solution_text.config(text="No se encontró solución")
        self.is_solving = False

    def dfs_solve(self):
        """Soluciona el FrozenLake usando DFS"""
        if self.is_solving:
            return
            
        self.is_solving = True
        self.solution_text.config(text="Buscando solución con DFS...")
        
        # Ejecutar DFS en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._dfs_worker).start()

    def _dfs_worker(self):
        """Implementa el algoritmo DFS en un hilo separado"""
        # Reiniciar el entorno
        self.observation, _ = self.env.reset()
        self.update_frozen_lake_display()
        
        # Estructura del mapa de FrozenLake (4x4 por defecto)
        desc = self.env.unwrapped.desc
        nrow, ncol = desc.shape
        
        # Diccionario para mapear coordenadas a estados
        coords_to_state = {}
        for s in range(nrow * ncol):
            row, col = s // ncol, s % ncol
            coords_to_state[(row, col)] = s
        
        # Estado inicial y meta
        start_coords = (0, 0)  # Esquina superior izquierda
        goal_coords = None
        
        # Encontrar la meta (posición de 'G')
        for i in range(nrow):
            for j in range(ncol):
                if desc[i][j] == b'G':
                    goal_coords = (i, j)
        
        if not goal_coords:
            self.solution_text.config(text="Error: No se encontró la meta")
            self.is_solving = False
            return
        
        # Direcciones: arriba, derecha, abajo, izquierda
        actions = [3, 2, 1, 0]  # UP, RIGHT, DOWN, LEFT
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Implementar DFS
        stack = [(start_coords, [])]  # (posición, camino)
        visited = set([start_coords])
        
        while stack:
            (row, col), path = stack.pop()
            
            # Si llegamos a la meta
            if (row, col) == goal_coords:
                self.solution_path = path
                self.solution_text.config(text=f"Solución encontrada en {len(path)} pasos")
                self.execute_solution()
                return
            
            # Explorar vecinos
            for action, (dr, dc) in zip(actions, directions):
                new_row, new_col = row + dr, col + dc
                
                # Verificar límites del mapa
                if 0 <= new_row < nrow and 0 <= new_col < ncol:
                    # Verificar que no sea un agujero
                    if desc[new_row][new_col] != b'H' and (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))
                        stack.append(((new_row, new_col), path + [action]))
        
        # Si no hay solución
        self.solution_text.config(text="No se encontró solución")
        self.is_solving = False
    
    def execute_solution(self):
        """Ejecuta la solución paso a paso"""
        self.current_step = 0
        self._execute_next_step()
    
    def _execute_next_step(self):
        """Ejecuta el siguiente paso de la solución"""
        if self.current_step < len(self.solution_path):
            action = self.solution_path[self.current_step]
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.update_frozen_lake_display()
            
            self.current_step += 1
            # Esperar 500ms antes del siguiente paso
            self.root.after(500, self._execute_next_step)
        else:
            self.solution_text.config(text="Solución ejecutada con éxito")
            self.is_solving = False

    def on_close(self):
        """Maneja el evento de cierre de la ventana"""
        if messagebox.askokcancel("Salir", "¿Estás seguro de que deseas cerrar el programa?"):
            self.root.destroy()  # Cierra la ventana
            sys.exit(0)  # Termina el programa

    def create_layout(self):
        # Configurar el estilo para botones más grandes
        style = ttk.Style()
        style.configure('Big.TButton', font=('Helvetica', 12, 'bold'), padding=10)
        
        # Frame central para contener los botones
        center_frame = tk.Frame(self.main_frame)
        center_frame.pack(expand=True, fill='both')
        
        # Título para el menú principal
        tk.Label(center_frame, text="Visualizador de Algoritmos de IA", 
                font=('Helvetica', 16, 'bold')).pack(pady=20)
        
        # Botones del frame principal con estilo personalizado
        btn1 = ttk.Button(center_frame, text="FrozenLake", 
                        command=lambda: self.mostrar_pantalla(self.pantalla1),
                        style='Big.TButton', width=25)
        btn1.pack(pady=15)

        btn2 = ttk.Button(center_frame, text="Gráfo de Rumania", 
                        command=lambda: self.mostrar_pantalla(self.pantalla2),
                        style='Big.TButton', width=25)
        btn2.pack(pady=15)

        btn3 = ttk.Button(center_frame, text="Gato Minimax", 
                        command=lambda: self.mostrar_pantalla(self.pantalla3),
                        style='Big.TButton', width=25)
        btn3.pack(pady=15)

        # Mostrar el frame principal inicialmente
        self.main_frame.pack(fill='both', expand=True)

        # Pantalla 1 - FrozenLake con BFS y DFS
        frozen_lake_frame = tk.Frame(self.pantalla1)
        frozen_lake_frame.pack(pady=20, fill='both', expand=True)
        
        # Crear figura y canvas para mostrar FrozenLake
        self.frozen_lake_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.frozen_lake_canvas = FigureCanvasTkAgg(self.frozen_lake_fig, master=frozen_lake_frame)
        self.frozen_lake_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Panel de control
        control_frame = tk.Frame(self.pantalla1)
        control_frame.pack(pady=10)
        
        # Menú desplegable para seleccionar algoritmo
        algorithm_menu = ttk.OptionMenu(control_frame, self.selected_algorithm, "BFS", "BFS", "DFS")
        algorithm_menu.pack(pady=5)
        
        # Botón para resolver
        solve_btn = ttk.Button(control_frame, text="Resolver", command=self.solve)
        solve_btn.pack(pady=5)
        
        # Texto de estado
        self.solution_text = tk.Label(control_frame, text="Algoritmo BFS para FrozenLake")
        self.solution_text.pack(pady=5)
        
        # Botón para reiniciar
        reset_btn = ttk.Button(control_frame, text="Reiniciar", command=self.initialize_frozen_lake)
        reset_btn.pack(pady=5)
        
        # Botón para volver al menú principal
        volver_btn = ttk.Button(self.pantalla1, text="Volver al Menú Principal", command=lambda: self.mostrar_pantalla(self.main_frame))
        volver_btn.pack(pady=10)

        # Pantalla 2 - Corregir el botón Volver
        tk.Label(self.pantalla2, text="Gráfo de Rumania").pack(pady=20)
        tk.Button(self.pantalla2, text="Volver", command=lambda: self.mostrar_pantalla(self.main_frame)).pack(pady=10)

        # Pantalla 3 - Corregir el botón Volver
        tk.Label(self.pantalla3, text="Gato Minimax").pack(pady=20)
        tk.Button(self.pantalla3, text="Volver", command=lambda: self.mostrar_pantalla(self.main_frame)).pack(pady=10)

    def solve(self):
        """Resuelve el problema usando el algoritmo seleccionado"""
        algorithm = self.selected_algorithm.get()
        if algorithm == "BFS":
            self.bfs_solve()
        elif algorithm == "DFS":
            self.dfs_solve()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()