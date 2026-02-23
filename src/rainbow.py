import numpy as np
import torch
from collections import deque

# Prioritized Experience Replay: SumTree implementation
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity # Número de hojas (experiencias)
        self.tree = np.zeros(2 * capacity - 1) # Árbol binario completo
        self.data = np.zeros(capacity, dtype=object) # Almacena experiencias
        self.write = 0 # Índice para escribir la próxima experiencia
        self.n_entries = 0 # Número de experiencias almacenadas
    
    def _propagate(self, idx: int, change: float):
        # Propaga el cambio hacia arriba en el árbol
        parent = (idx - 1) // 2
        # Actualiza el nodo padre con el cambio
        self.tree[parent] += change
        if parent != 0:
            # Si no hemos llegado a la raíz, seguimos propagando (recursividad)
            self._propagate(parent, change)
    
    @property
    def total(self):
        # El valor total de prioridades es el valor de la raíz del árbol
        return self.tree[0]
    
    def update(self, data_idx: int, priority: float):
        # Actualiza la prioridad de una experiencia dada su posición en el buffer
        # data_idx es el índice de la experiencia en el buffer (0 a capacity-1)
        tree_idx = data_idx + self.capacity - 1 # Índice en el árbol correspondiente a la experiencia
        change = priority - self.tree[tree_idx] # Calcula el cambio en prioridad
        self.tree[tree_idx] = priority # Actualiza la prioridad en el nodo hoja
        self._propagate(tree_idx, change) # Propaga el cambio hacia arriba
    
    def add(self, priority: float, data): # DATA?
        # Agrega una nueva experiencia con su prioridad
        data_idx = self.write # Índice donde se escribirá la nueva experiencia
        self.data[data_idx] = data # Almacena la experiencia en el buffer ¡!
        self.update(data_idx, priority) # Actualiza el árbol con la nueva prioridad
        self.write = (self.write + 1) % self.capacity # Mueve el índice de escritura (circular)
        self.n_entries = min(self.n_entries + 1, self.capacity) # Actualiza el conteo de experiencias
        return data_idx # Devuelve el índice de la experiencia agregada (útil para actualizaciones futuras)
    
    def _retrieve(self, idx: int, s: float):
        # Busca en el árbol la experiencia correspondiente a la prioridad acumulada s
        left = 2 * idx + 1 # Índice del hijo izquierdo
        right = left + 1 # Índice del hijo derecho
        
        if left >= len(self.tree): # Si llegamos a una hoja (no hay hijos)
            return idx # Devolvemos el índice de la hoja
        
        if s <= self.tree[left]: # Si s es menor o igual a la prioridad acumulada del hijo izquierdo
            return self._retrieve(left, s) # Buscamos en el hijo izquierdo
        else: # Si s es mayor, buscamos en el hijo derecho, ajustando s restando la prioridad del hijo izquierdo
            return self._retrieve(right, s - self.tree[left]) # Buscamos en el hijo derecho, ajustando s
        
    def get(self, s: float):
        # Devuelve la experiencia correspondiente a la prioridad acumulada s
        idx = self._retrieve(0, s) # Comienza la búsqueda desde la raíz
        data_idx = idx - self.capacity + 1 # Calcula el índice de la experiencia en el buffer
        return (idx, self.tree[idx], self.data[data_idx]) # Devuelve el índice en el árbol, la prioridad y la experiencia
        # ¿QUÉ TIENE QUE DEVOLVER ESTO?
 