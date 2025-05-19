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
        self.root.geometry("1100x900")  # Tamaño inicial de la ventana
        self.root.resizable(False, False)
        
        # Inicializar variables para FrozenLake
        self.env = None
        self.solution_path = []
        self.current_step = 0
        self.is_solving = False
        self.algorithm_FL = tk.StringVar(value="BFS")  # Algoritmo seleccionado por defecto para Frozen Lake

        self.algorithm_GR = tk.StringVar(value="A*")  # Algoritmo seleccionado por defecto para el grafo
        self.graph = nx.Graph()
        self.pos = None  # Posiciones de los nodos en el grafo
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
        """Resuelve el problema de Frozen Lake usando el algoritmo seleccionado"""
        algorithm = self.algorithm_FL.get()
        if algorithm == "BFS":
            self.bfs_solve()
        elif algorithm == "DFS":
            self.dfs_solve()

    def load_file(self):
        """Carga un archivo de texto con la estructura del grafo"""
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
        """Dibuja el grafo sin rutas resaltadas"""
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
        """Ejecuta el algoritmo seleccionado en el grafo"""
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
        """Implementación del algoritmo Greedy"""
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
        """Implementación sencilla de Hill Climbing con distancia euclidiana como heurística"""
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
        """Calcula la heurística (distancia euclidiana) para cada nodo hasta el objetivo"""
        self.heuristics = {}
        goal_pos = self.pos[goal_node]
        
        for node in self.graph.nodes():
            node_pos = self.pos[node]
            # Calcular distancia euclidiana
            dist = math.sqrt((node_pos[0] - goal_pos[0])**2 + (node_pos[1] - goal_pos[1])**2)
            self.heuristics[node] = dist

    def visualize_path(self, path):
        """Visualiza el camino encontrado en el grafo, resaltado en rojo"""
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
        """Crea el juego de gato (Tic Tac Toe) con algoritmo Minimax"""
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
        
        # Botón para volver al menú principal
        volver_btn = ttk.Button(self.pantalla3, text="Volver al Menú Principal", 
                              command=lambda: self.mostrar_pantalla(self.main_frame))
        volver_btn.pack(pady=10)
        
    def make_move(self, row, col):
        """Realiza un movimiento en el tablero"""
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
        """Realiza el movimiento de la IA usando Minimax"""
        # Encontrar el mejor movimiento usando Minimax
        best_score = float('-inf')
        best_move = None
        
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    self.board[i][j] = 'O'
                    score = self.minimax(self.board, 0, False)
                    self.board[i][j] = ' '
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        
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
        """Implementación del algoritmo Minimax"""
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
            
    def check_winner(self):
        """Verifica si hay un ganador en el tablero actual"""
        return self.check_winner_board(self.board, self.current_player)
        
    def check_winner_board(self, board, player):
        """Verifica si hay un ganador en un tablero dado"""
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
        """Verifica si el tablero está lleno"""
        return self.is_board_full_board(self.board)
        
    def is_board_full_board(self, board):
        """Verifica si un tablero dado está lleno"""
        for row in board:
            for cell in row:
                if cell == ' ':
                    return False
        return True
        
    def reset_game(self):
        """Reinicia el juego de gato"""
        self.current_player = 'X'
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.game_over = False
        
        # Reiniciar botones
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=' ')
                
        self.game_status.config(text="Tu turno (X)")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
