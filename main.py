# Programa: Visualizador de Algoritmos de IA
# Autor: Parra Rivas Arturo Gabriel
# Descripción: Aplicación que visualiza diferentes algoritmos de IA:
#   1. BFS y DFS aplicados al entorno FrozenLake de Gymnasium
#   2. A*, Greedy y Hill Climbing aplicados al problema del camino más corto en el mapa de Rumania
#   3. Minimax aplicado al juego de Gato (Tic-Tac-Toe)

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
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque
import threading

class MainApp:
    """Clase principal de la aplicación
    
    Esta clase maneja toda la lógica de la aplicación, incluyendo:
    - La interfaz gráfica con tkinter
    - El entorno FrozenLake con algoritmos BFS y DFS
    - El grafo de Rumania con algoritmos A*, Greedy y Hill Climbing
    - El juego de Gato (Tic-Tac-Toe) con algoritmo Minimax
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto de IA - Parra Rivas Arturo Gabriel")
        self.root.geometry("1100x900")  # Tamaño inicial de la ventana
        self.root.resizable(False, False)
        
        # Inicializar variables para FrozenLake
        self.env = None  # Entorno de Gymnasium para FrozenLake
        self.solution_path = []  # Almacenará la secuencia de acciones de la solución
        self.current_step = 0  # Índice del paso actual durante la ejecución de la solución
        self.is_solving = False  # Bandera para evitar ejecutar múltiples soluciones simultáneamente
        self.algorithm_FL = tk.StringVar(value="BFS")  # Algoritmo seleccionado por defecto para Frozen Lake

        # Inicializar variables para el Grafo de Rumania
        self.algorithm_GR = tk.StringVar(value="A*")  # Algoritmo seleccionado por defecto para el grafo
        self.graph = nx.Graph()  # Grafo de NetworkX para representar el mapa de Rumania
        self.pos = None  # Posiciones de los nodos en el grafo para visualización
        self.start_node = tk.StringVar(value="Arad")  # Nodo inicial para el algoritmo
        self.end_node = tk.StringVar(value="Bucharest")  # Nodo final para el algoritmo
        
        # Crear frames
        self.pantalla1 = tk.Frame(root)
        self.pantalla2 = tk.Frame(root)
        self.pantalla3 = tk.Frame(root)
        self.main_frame = tk.Frame(root)
        
        self.create_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.centrar_ventana()
        self.load_file()  # Cargar el grafo al iniciar la aplicación

    def centrar_ventana(self):
        """Centra la ventana principal en la pantalla
        
        Calcula la posición adecuada para que la ventana aparezca en el centro
        de la pantalla según las dimensiones de la ventana y la resolución del monitor.
        """
        self.root.update_idletasks()
        ancho = self.root.winfo_width()
        alto = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.root.winfo_screenheight() // 2) - (alto // 2)
        self.root.geometry(f"{ancho}x{alto}+{x}+{y}")

    def mostrar_pantalla(self, pantalla):
        """Cambia entre las diferentes pantallas de la aplicación
        
        Oculta todas las pantallas y muestra únicamente la pantalla seleccionada.
        Si la pantalla seleccionada es FrozenLake y el entorno aún no está inicializado,
        se inicializa automáticamente.
        
        Args:
            pantalla: El frame de tkinter que se quiere mostrar
        """
        # Ocultar todas las pantallas incluyendo el main_frame
        for p in (self.pantalla1, self.pantalla2, self.pantalla3, self.main_frame):
            p.pack_forget()
        
        # Mostrar la pantalla seleccionada
        pantalla.pack(fill='both', expand=True)
        
        # Si es la pantalla del FrozenLake, inicializar el entorno
        if pantalla == self.pantalla1 and self.env is None:
            self.initialize_frozen_lake()
    
    def initialize_frozen_lake(self):
        """Inicializa el entorno FrozenLake
        
        Crea un entorno FrozenLake-v1 con el modo de renderizado rgb_array.
        El parámetro is_slippery=False hace que el agente se mueva de forma determinista
        (sin deslizamientos aleatorios en el hielo).
        Utilizamos map_name="8x8" para crear un mapa más grande y variado cada vez.
        """
        self.env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False, desc=generate_random_map(size=8), map_name="8x8")
        self.observation, info = self.env.reset()  # Reinicia el entorno y obtiene el estado inicial
        
        # Crear y actualizar la visualización del entorno
        self.update_frozen_lake_display()
        
    def update_frozen_lake_display(self):
        """Actualiza la visualización del entorno FrozenLake
        
        Obtiene la representación visual del entorno actual y la muestra usando matplotlib.
        Esta función es llamada después de cada acción para actualizar la interfaz gráfica.
        """
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
        """Soluciona el FrozenLake usando BFS (Breadth-First Search)
        
        El algoritmo BFS explora el entorno por niveles, examinando primero todos los
        vecinos inmediatos antes de avanzar al siguiente nivel. Esto garantiza encontrar
        la solución con el menor número de pasos posible.
        """
        if self.is_solving:
            return
            
        self.is_solving = True
        self.solution_text.config(text="Buscando solución con BFS...")
        
        # Ejecutar BFS en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._bfs_worker).start()
    
    def _bfs_worker(self):
        """Implementa el algoritmo BFS (Breadth-First Search) en un hilo separado
        
        Utiliza una cola (FIFO) para explorar los estados del entorno FrozenLake
        nivel por nivel, garantizando encontrar la solución óptima (camino más corto).
        El algoritmo funciona de la siguiente manera:
        1. Inicia en la posición (0,0)
        2. Explora todos los vecinos válidos (no agujeros y dentro de límites)
        3. Para cada vecino, almacena la secuencia de acciones para llegar a él
        4. Continúa explorando por niveles hasta encontrar la meta
        """
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
            return        # Direcciones: arriba, derecha, abajo, izquierda
        actions = [3, 2, 1, 0]  # UP, RIGHT, DOWN, LEFT en el formato de acciones de gymnasium
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Cambios en coordenadas (fila, columna)
        
        # Implementar BFS usando una cola para explorar nivel por nivel
        queue = deque([(start_coords, [])])  # (posición, camino) - Estructura FIFO
        visited = set([start_coords])  # Conjunto para evitar ciclos
        
        while queue:
            (row, col), path = queue.popleft()  # Extraer el primer elemento (FIFO)
            
            # Si llegamos a la meta
            if (row, col) == goal_coords:
                self.solution_path = path
                self.solution_text.config(text=f"Solución encontrada en {len(path)} pasos")
                self.execute_solution()
                return
            
            # Explorar vecinos en las cuatro direcciones posibles
            for action, (dr, dc) in zip(actions, directions):
                new_row, new_col = row + dr, col + dc
                
                # Verificar límites del mapa
                if 0 <= new_row < nrow and 0 <= new_col < ncol:
                    # Verificar que no sea un agujero y que no haya sido visitado antes
                    if desc[new_row][new_col] != b'H' and (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))  # Marcar como visitado
                        queue.append(((new_row, new_col), path + [action]))  # Añadir a la cola con el camino actualizado
        
        # Si no hay solución
        self.solution_text.config(text="No se encontró solución")
        self.is_solving = False

    def dfs_solve(self):
        """Soluciona el FrozenLake usando DFS (Depth-First Search)
        
        El algoritmo DFS explora el entorno en profundidad, siguiendo un camino
        hasta donde sea posible antes de retroceder. Puede encontrar una solución
        más rápidamente, pero no garantiza que sea la solución óptima.
        """
        if self.is_solving:
            return
            
        self.is_solving = True
        self.solution_text.config(text="Buscando solución con DFS...")
        
        # Ejecutar DFS en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._dfs_worker).start()

    def _dfs_worker(self):
        """Implementa el algoritmo DFS (Depth-First Search) en un hilo separado
        
        Utiliza una pila (LIFO) para explorar los estados del entorno FrozenLake
        priorizando la profundidad sobre la amplitud. A diferencia de BFS, DFS
        sigue un camino tan lejos como sea posible antes de retroceder.
        El algoritmo funciona de la siguiente manera:
        1. Inicia en la posición (0,0)
        2. Explora un vecino y continúa desde ese punto (profundidad)
        3. Cuando no puede avanzar más, retrocede y prueba otra dirección
        4. Continúa hasta encontrar la meta o explorar todo el espacio
        """
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
            (row, col), path = stack.pop()  # Extraer el último elemento (LIFO)
            
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
                    # Verificar que no sea un agujero y que no haya sido visitado antes
                    if desc[new_row][new_col] != b'H' and (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))  # Marcar como visitado
                        stack.append(((new_row, new_col), path + [action]))  # Añadir a la pila con el camino actualizado
        
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
        """Crea la interfaz gráfica de la aplicación
        
        Configura todos los elementos visuales de la aplicación, incluyendo:
        - El menú principal con los tres botones de selección de algoritmo
        - La pantalla de FrozenLake con sus controles
        - La pantalla del Grafo de Rumania con sus controles
        - La pantalla del juego de Gato (Tic Tac Toe)
        
        Se definen estilos, se crean widgets y se organizan en frames.
        """
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
        algorithm_menu = ttk.OptionMenu(control_frame, self.algorithm_FL, "BFS", "BFS", "DFS")
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

        # Pantalla 2 - Grafo de Rumania
        graph_panel = tk.Frame(self.pantalla2)
        graph_panel.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Panel superior para los controles
        controls_frame = tk.Frame(graph_panel)
        controls_frame.pack(side=tk.TOP, fill='x', pady=10)
        
        # Título
        tk.Label(controls_frame, text="Gráfo de Rumania", font=('Helvetica', 14, 'bold')).pack(side=tk.TOP, pady=5)
        
        # Frame para la selección de algoritmo y nodos
        options_frame = tk.Frame(controls_frame)
        options_frame.pack(side=tk.TOP, fill='x', pady=5)
          # Dropdown de Algoritmos
        alg_frame = tk.Frame(options_frame)
        alg_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(alg_frame, text="Algoritmo:").pack(anchor="w")
        opciones = ["A*", "Greedy", "Hill Climbing"]
        self.algorithm_GR.set("A*")  # Establecer el algoritmo por defecto explícitamente
        dropdown = ttk.OptionMenu(alg_frame, self.algorithm_GR, "A*", *opciones)
        dropdown.pack(fill="x", pady=5)

        # Botón para ejecutar
        exec_btn = ttk.Button(controls_frame, text="Ejecutar Algoritmo", 
                           command=self.run_algorithm,
                           style='Big.TButton')
        exec_btn.pack(pady=10)
        
        # Frame para el grafo
        graph_frame = tk.Frame(graph_panel, bg="white")
        graph_frame.pack(fill="both", expand=True, pady=10)
        
        # Crear figura y canvas para el grafo
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_facecolor("white")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Botón para volver al menú principal
        volver_btn = ttk.Button(self.pantalla2, text="Volver al Menú Principal", 
                              command=lambda: self.mostrar_pantalla(self.main_frame))
        volver_btn.pack(pady=10)

        # Pantalla 3 - Crear juego de Gato (Tic Tac Toe)
        self.create_tictactoe_game()

    def solve(self):
        """Resuelve el problema de Frozen Lake usando el algoritmo seleccionado
        
        Obtiene el algoritmo seleccionado en la interfaz (BFS o DFS) y ejecuta
        la función correspondiente para encontrar la solución al problema.
        """
        algorithm = self.algorithm_FL.get()
        if algorithm == "BFS":
            self.bfs_solve()
        elif algorithm == "DFS":
            self.dfs_solve()

    def load_file(self):
        """Carga un archivo de texto con la estructura del grafo de Rumania
        
        Lee el archivo 'graforumania.txt' que contiene las conexiones entre ciudades
        con el formato: 'ciudad_origen ciudad_destino distancia'.
        Crea el grafo usando NetworkX y calcula las posiciones para su visualización.
        """
        file_path = "graforumania.txt"
        if file_path:
            self.graph.clear()
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) == 3:
                                node1, node2, weight = parts
                                self.graph.add_edge(node1, node2, weight=float(weight))
                            else:
                                messagebox.showerror("Error", "Formato de archivo incorrecto. Debe ser 'nodo1 nodo2 peso' o 'nodo1 nodo2'.")
                                return
                self.pos = nx.spring_layout(self.graph)  # Calcular posiciones solo una vez
                self.draw_graph()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def draw_graph(self):
        """Dibuja el grafo sin rutas resaltadas
        
        Visualiza el grafo de Rumania usando NetworkX y matplotlib.
        Muestra todas las ciudades como nodos y las conexiones como aristas,
        etiquetando cada conexión con su distancia.
        """
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(False)

        nx.draw(self.graph, self.pos, with_labels=True, node_color='skyblue',
                node_size=500, font_size=10, font_weight='bold', ax=self.ax)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels, ax=self.ax)

        self.canvas.draw()

    def run_algorithm(self):
        """Ejecuta el algoritmo seleccionado en el grafo
        
        Dependiendo del algoritmo seleccionado (A*, Greedy o Hill Climbing),
        encuentra un camino entre el nodo inicial y final, y visualiza
        la ruta encontrada en el grafo.
        """
        algorithm = self.algorithm_GR.get()
        try:
            if algorithm == "A*":
                path = nx.astar_path(self.graph, self.start_node.get(), self.end_node.get(), weight='weight')
                self.visualize_path(path)
            elif algorithm == "Greedy":
                path = self.greedy_algorithm(self.start_node.get(), self.end_node.get())
                self.visualize_path(path)
            elif algorithm == "Hill Climbing":
                path = self.hill_climbing_algorithm(self.start_node.get(), self.end_node.get())
                self.visualize_path(path)
                
            messagebox.showinfo("Éxito", f"Ruta encontrada utilizando {algorithm}!")
        except nx.NetworkXNoPath:
            messagebox.showerror("Error", "No se encontró un camino entre los nodos seleccionados.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al ejecutar el algoritmo: {str(e)}")

    def greedy_algorithm(self, start_node, end_node):
        """Implementación del algoritmo Greedy
        
        Encuentra un camino desde el nodo inicial hasta el nodo objetivo usando
        una estrategia voraz, seleccionando siempre el vecino con menor peso.
        Este algoritmo no garantiza encontrar el camino óptimo, pero suele ser
        más rápido que A*.
        
        Args:
            start_node: Nodo inicial del camino
            end_node: Nodo objetivo
            
        Returns:
            Una lista de nodos que representan el camino encontrado
        """
        visited = set()  # Conjunto de nodos visitados
        path = [start_node]  # Ruta inicial
        current_node = start_node

        while current_node != end_node:
            visited.add(current_node)
            neighbors = self.graph[current_node]  # Vecinos del nodo actual

            # Encontrar el vecino con el menor peso que no haya sido visitado
            next_node = None
            min_weight = float('inf')
            for neighbor, attributes in neighbors.items():
                if neighbor not in visited and attributes.get('weight', 1) < min_weight:
                    next_node = neighbor
                    min_weight = attributes.get('weight', 1)

            if next_node is None:
                raise nx.NetworkXNoPath("No hay camino al destino")

            path.append(next_node)
            current_node = next_node

        return path

    def hill_climbing_algorithm(self, start_node, end_node):
        """Implementación del algoritmo Hill Climbing
        
        Utiliza una estrategia de escalada simple, eligiendo siempre
        el vecino que tiene la menor distancia euclidiana al objetivo.
        Este algoritmo puede quedarse atrapado en óptimos locales, pero es
        eficiente para problemas con espacios de búsqueda bien formados.
        
        Args:
            start_node: Nodo inicial del camino
            end_node: Nodo objetivo
            
        Returns:
            Una lista de nodos que representan el camino encontrado
        """
        # Calcular heurística si no existe (distancia directa a la meta)
        if not hasattr(self, 'heuristics'):
            self.calculate_heuristics(end_node)
            
        visited = set()  # Conjunto de nodos visitados
        path = [start_node]  # Ruta inicial
        current_node = start_node

        while current_node != end_node:
            visited.add(current_node)
            neighbors = self.graph[current_node]  # Vecinos del nodo actual

            # Seleccionar el vecino más prometedor según la heurística
            next_node = None
            best_heuristic = float('inf')
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Obtener la heurística del nodo vecino
                    heuristic = self.heuristics.get(neighbor, float('inf'))
                    if heuristic < best_heuristic:
                        next_node = neighbor
                        best_heuristic = heuristic

            if next_node is None:
                raise nx.NetworkXNoPath("No hay camino al destino")

            path.append(next_node)
            current_node = next_node

        return path
        
    def calculate_heuristics(self, goal_node):
        """Calcula la heurística (distancia euclidiana) para cada nodo hasta el objetivo
        
        Para cada nodo del grafo, calcula la distancia euclidiana (línea recta)
        hasta el nodo objetivo. Esta heurística es admisible para A* ya que nunca
        sobreestima el costo real del camino.
        
        Args:
            goal_node: El nodo objetivo al que se calculará la distancia
        """
        self.heuristics = {}
        goal_pos = self.pos[goal_node]
        
        for node in self.graph.nodes():
            node_pos = self.pos[node]
            # Calcular distancia euclidiana
            dist = math.sqrt((node_pos[0] - goal_pos[0])**2 + (node_pos[1] - goal_pos[1])**2)
            self.heuristics[node] = dist

    def visualize_path(self, path):
        """Visualiza el camino encontrado en el grafo, resaltado en rojo
        
        Toma una lista de nodos que forman un camino y los resalta en el grafo
        para mostrar visualmente la ruta encontrada por el algoritmo.
        
        Args:
            path: Lista de nodos que forman el camino encontrado
        """
        # Limpiar el grafo
        self.draw_graph()
        
        # Dibujar el camino encontrado en rojo
        path_edges = list(zip(path, path[1:]))
        
        # Dibujar aristas del camino en rojo
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=path_edges, 
                              width=3, edge_color='red', ax=self.ax)
        
        # Resaltar nodos del camino
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=path, 
                              node_color='red', node_size=500, ax=self.ax)
        
        # Actualizar el canvas
        self.canvas.draw()
    
    def create_tictactoe_game(self):
        """Crea el juego de gato (Tic Tac Toe) con algoritmo Minimax
        
        Configura la interfaz gráfica para el juego, incluyendo:
        - El tablero de 3x3 con botones interactivos
        - Las variables de estado del juego
        - Los controles para reiniciar la partida y volver al menú principal
        
        El juego enfrenta al jugador humano (X) contra la IA (O) que usa
        el algoritmo Minimax para tomar decisiones óptimas.
        """
        # Frame para el juego
        game_frame = tk.Frame(self.pantalla3)
        game_frame.pack(pady=20, fill='both', expand=True)
        
        # Título
        tk.Label(game_frame, text="Gato - Algoritmo Minimax", 
                font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # Inicializar variables del juego
        self.current_player = 'X'  # El jugador humano es X
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.buttons = []
        self.use_pruning = tk.BooleanVar(value=False)  # Variable para la poda alfa-beta
        self.last_move_time = 0  # Tiempo que tardó el último movimiento
        
        # Crear el tablero con botones
        board_frame = tk.Frame(game_frame)
        board_frame.pack(pady=10)
        
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(board_frame, text=' ', font=('Helvetica', 20), 
                                width=5, height=2,
                                command=lambda r=i, c=j: self.make_move(r, c))
                button.grid(row=i, column=j, padx=5, pady=5)
                row.append(button)
            self.buttons.append(row)
        
        # Estado del juego
        self.game_status = tk.Label(game_frame, text="Tu turno (X)", font=('Helvetica', 12))
        self.game_status.pack(pady=10)
        
        # Botones de control
        control_frame = tk.Frame(game_frame)
        control_frame.pack(pady=10)
        
        reset_btn = ttk.Button(control_frame, text="Reiniciar Juego", 
                             command=self.reset_game)
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Checkbox para habilitar/deshabilitar poda alfa-beta
        pruning_check = ttk.Checkbutton(control_frame, text="Usar poda alfa-beta", 
                                      variable=self.use_pruning)
        pruning_check.pack(side=tk.LEFT, padx=10)
        
        # Etiqueta para mostrar el tiempo de cálculo
        self.time_label = tk.Label(control_frame, text="Tiempo: 0 ms", font=('Helvetica', 10))
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        # Botón para volver al menú principal
        volver_btn = ttk.Button(self.pantalla3, text="Volver al Menú Principal", 
                              command=lambda: self.mostrar_pantalla(self.main_frame))
        volver_btn.pack(pady=10)
        
    def make_move(self, row, col):
        """Realiza un movimiento en el tablero de gato
        
        Gestiona el movimiento del jugador humano, verifica si el juego ha terminado,
        y si no, pasa el turno a la IA.
        
        Args:
            row: Fila del tablero (0-2)
            col: Columna del tablero (0-2)
        """
        if self.game_over or self.board[row][col] != ' ':
            return
            
        # Movimiento del jugador humano
        self.board[row][col] = self.current_player
        self.buttons[row][col].config(text=self.current_player)
        
        # Verificar si el juego terminó
        if self.check_winner():
            self.game_status.config(text=f"¡Ganador: {self.current_player}!")
            self.game_over = True
            return
        elif self.is_board_full():
            self.game_status.config(text="¡Empate!")
            self.game_over = True
            return
            
        # Cambiar al turno de la IA
        self.current_player = 'O'
        self.game_status.config(text="Turno de la IA (O)")
        
        # Permitir que la interfaz se actualice
        self.root.update()
        self.root.after(500, self.ai_move)
        
    def ai_move(self):
        """Realiza el movimiento de la IA usando Minimax
        
        Utiliza el algoritmo Minimax para encontrar el mejor movimiento posible,
        asumiendo que el jugador humano jugará de manera óptima.
        Evalúa todos los posibles movimientos y selecciona el que maximiza la
        probabilidad de victoria para la IA.
        """
        # Iniciar cronómetro para medir el tiempo
        start_time = time.time()
        
        # Encontrar el mejor movimiento usando Minimax
        best_score = float('-inf')
        best_move = None
        
        # Valores iniciales para alfa-beta si se usa poda
        alpha = float('-inf')
        beta = float('inf')
        
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    self.board[i][j] = 'O'
                    
                    # Decidir si usar poda alfa-beta o minimax regular
                    if self.use_pruning.get():
                        score = self.minimax_with_pruning(self.board, 0, False, alpha, beta)
                    else:
                        score = self.minimax(self.board, 0, False)
                        
                    self.board[i][j] = ' '
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
                        
                    # Actualizar alfa si estamos usando poda
                    if self.use_pruning.get():
                        alpha = max(alpha, best_score)
        
        # Calcular tiempo transcurrido en milisegundos
        end_time = time.time()
        self.last_move_time = (end_time - start_time) * 1000  # Convertir a milisegundos
        self.time_label.config(text=f"Tiempo: {self.last_move_time:.2f} ms")
        
        if best_move:
            row, col = best_move
            self.board[row][col] = 'O'
            self.buttons[row][col].config(text='O')
            
            # Verificar si el juego terminó
            if self.check_winner():
                self.game_status.config(text="¡Ganador: O!")
                self.game_over = True
                return
            elif self.is_board_full():
                self.game_status.config(text="¡Empate!")
                self.game_over = True
                return
        
        # Volver al turno del jugador
        self.current_player = 'X'
        self.game_status.config(text="Tu turno (X)")
        
    def minimax(self, board, depth, is_maximizing):
        """Implementación del algoritmo Minimax
        
        Algoritmo recursivo que simula todos los posibles movimientos del juego
        para determinar la mejor estrategia. Asigna valores a los estados terminales:
        1 para victoria de la IA, -1 para victoria del jugador humano, 0 para empate.
        
        Args:
            board: Estado actual del tablero
            depth: Profundidad actual en el árbol de recursión
            is_maximizing: Booleano que indica si se está maximizando (turno de la IA)
                          o minimizando (turno del jugador)
                          
        Returns:
            El valor óptimo del tablero para el jugador actual
        """
        # Verificar si hay un ganador
        if self.check_winner_board(board, 'O'):
            return 1
        elif self.check_winner_board(board, 'X'):
            return -1
        elif self.is_board_full_board(board):
            return 0
            
        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'O'
                        score = self.minimax(board, depth + 1, False)
                        board[i][j] = ' '
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'X'
                        score = self.minimax(board, depth + 1, True)
                        board[i][j] = ' '
                        best_score = min(score, best_score)
            return best_score
            
    def minimax_with_pruning(self, board, depth, is_maximizing, alpha, beta):
        """Implementación del algoritmo Minimax con poda alfa-beta
        
        Similar al algoritmo Minimax estándar, pero utiliza la poda alfa-beta
        para reducir el número de nodos evaluados, mejorando la eficiencia.
        
        Args:
            board: Estado actual del tablero
            depth: Profundidad actual en el árbol de recursión
            is_maximizing: Booleano que indica si se está maximizando o minimizando
            alpha: Mejor valor para el maximizador
            beta: Mejor valor para el minimizador
            
        Returns:
            El valor óptimo del tablero para el jugador actual
        """
        # Verificar si hay un ganador
        if self.check_winner_board(board, 'O'):
            return 1
        elif self.check_winner_board(board, 'X'):
            return -1
        elif self.is_board_full_board(board):
            return 0
            
        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'O'
                        score = self.minimax_with_pruning(board, depth + 1, False, alpha, beta)
                        board[i][j] = ' '
                        best_score = max(score, best_score)
                        alpha = max(alpha, best_score)
                        # Poda beta
                        if beta <= alpha:
                            break
                # Poda alfa adicional entre filas
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == ' ':
                        board[i][j] = 'X'
                        score = self.minimax_with_pruning(board, depth + 1, True, alpha, beta)
                        board[i][j] = ' '
                        best_score = min(score, best_score)
                        beta = min(beta, best_score)
                        # Poda alfa
                        if beta <= alpha:
                            break
                # Poda beta adicional entre filas
                if beta <= alpha:
                    break
            return best_score
            
    def check_winner(self):
        """Verifica si hay un ganador en el tablero actual"""
        return self.check_winner_board(self.board, self.current_player)
        
    def check_winner_board(self, board, player):
        """Verifica si hay un ganador en un tablero dado
        
        Comprueba todas las combinaciones ganadoras posibles (filas, columnas y diagonales)
        para determinar si el jugador especificado ha ganado.
        
        Args:
            board: El tablero a verificar
            player: El jugador ('X' o 'O') para el que se busca la victoria
            
        Returns:
            True si el jugador ha ganado, False en caso contrario
        """
        # Revisar filas
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == player:
                return True
                
        # Revisar columnas
        for i in range(3):
            if board[0][i] == board[1][i] == board[2][i] == player:
                return True
                
        # Revisar diagonales
        if board[0][0] == board[1][1] == board[2][2] == player:
            return True
        if board[0][2] == board[1][1] == board[2][0] == player:
            return True
            
        return False
        
    def is_board_full(self):
        """Verifica si el tablero está lleno
        
        Utiliza la función auxiliar is_board_full_board para verificar si
        todas las celdas del tablero actual están ocupadas.
        
        Returns:
            True si el tablero está lleno, False si hay al menos una celda vacía
        """
        return self.is_board_full_board(self.board)
        
    def is_board_full_board(self, board):
        """Verifica si un tablero dado está lleno
        
        Recorre todas las celdas del tablero para verificar si hay alguna vacía.
        
        Args:
            board: El tablero a verificar
            
        Returns:
            True si todas las celdas están ocupadas, False en caso contrario
        """
        for row in board:
            for cell in row:
                if cell == ' ':
                    return False
        return True
        
    def reset_game(self):
        """Reinicia el juego de gato
        
        Restablece el tablero a su estado inicial, limpiando todas las celdas,
        reiniciando las variables de control y actualizando la interfaz gráfica.
        """
        self.current_player = 'X'
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        
        # Reiniciar botones
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=' ')
                
        self.game_status.config(text="Tu turno (X)")
        self.time_label.config(text="Tiempo: 0 ms")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
