from base import BaseMAEnv
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np

from typing import List
import random
from collections import Counter

import cv2
import time

class TruckTrajectoryMAEnv(BaseMAEnv):
    def __init__(self, num_agents: int = 2, cities_left = [], render=False):
        # Aquí se inicializan todos los parámetros necesarios.

        # El límite de agentes es 3.
        if num_agents > 3:
            num_agents = 3

        self.num_agents = num_agents

        # Lista de ciudades objetivo. Si es una lista vacía luego se asignan todas las ciudades disponibles.
        self.cities_left = cities_left

        # Opción para indicar si se quiere ver el render.
        self.render = render

        # Determinamos el número máximo de steps por episodio.
        self.current_step = 1
        self.max_steps = 25

        # Asignamos a cada ciudad los índices deseados.
        self.all_cities = {'Madrid': 0, 'Barcelona': 1, 'Valencia': 2, 'Sevilla': 3,
                           'Bilbao': 4, 'A Coruna': 5, 'Leon': 6, 'Salamanca': 7, 'Murcia': 8}

        # Asignamos una acción a cada movimiento posible en cada ciudad.
        self.cities = {
            'A Coruna': {'Leon': 0, 'Salamanca': 1, 'Bilbao': 2},
            'Leon': {'A Coruna': 0, 'Salamanca': 1, 'Bilbao': 2, 'Madrid': 3},
            'Salamanca': {'A Coruna': 0, 'Leon': 1, 'Madrid': 2, 'Sevilla': 3},
            'Sevilla': {'Salamanca': 0, 'Madrid': 1, 'Murcia': 2},
            'Murcia': {'Sevilla': 0, 'Madrid': 1, 'Valencia': 2},
            'Valencia': {'Murcia': 0, 'Madrid': 1, 'Barcelona': 2},
            'Barcelona': {'Valencia': 0, 'Madrid': 1, 'Bilbao': 2},
            'Bilbao': {'Barcelona': 0, 'Madrid': 1, 'Leon': 2},
            'Madrid': {'Sevilla': 0, 'Salamanca': 1, 'Leon': 2, 'Bilbao': 3, 'Barcelona': 4, 'Valencia': 5, 'Murcia': 6}
        }

        # Asignamos las distancias a cada trayecto.
        self.distances = {
        'A Coruna': {'Leon': 220, 'Salamanca': 452, 'Bilbao': 550},
        'Leon': {'A Coruna': 220, 'Salamanca': 205, 'Bilbao': 230, 'Madrid': 335},
        'Salamanca': {'A Coruna': 452, 'Leon': 205, 'Madrid': 212, 'Sevilla': 400},
        'Sevilla': {'Salamanca': 400, 'Madrid': 532, 'Murcia': 585},
        'Murcia': {'Sevilla': 585, 'Madrid': 400, 'Valencia': 240},
        'Valencia': {'Murcia': 240, 'Madrid': 352, 'Barcelona': 349},
        'Barcelona': {'Valencia': 349, 'Madrid': 621, 'Bilbao': 610},
        'Bilbao': {'Barcelona': 610, 'Madrid': 400, 'Leon': 230},
        'Madrid': {'Sevilla': 532, 'Salamanca': 212, 'Leon': 335, 'Bilbao': 400, 'Barcelona': 621, 'Valencia': 352, 'Murcia': 400}
        }

        # Listado de ciudades objetivo. Si no se especifica como parámetro tomamos todas.
        if len(self.cities_left) == 0:
            self.cities_left = list(self.all_cities.keys())

        # Inicializar penalización por permanecer en la misma ciudad y reward por exploración.
        self.penalty = False
        self.extra_reward = {}

        # Llevar recuento de ciudades anteriores, ciudades actuales e histórico de cada agente.
        self.previous_cities = {}
        self.agent_positions = {}
        self.agent_visited_cities = {f'truck_{i+1}': [] for i in range(self.num_agents)}

        # Definir espacios de observaciones y de acciones de cada agente.
        for i in range(num_agents):
            agent_name = f'truck_{i+1}'
            self.register_agent(
                agent_id=agent_name,
                observation_space=MultiDiscrete([9] * self.num_agents + [2] * 9),
                action_space=Discrete(7)
            )

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Resetear todo lo necesario.
        self.penalty = False
        self.cities_left = list(self.all_cities.keys())
        self.agent_visited_cities = {agent_id: [] for agent_id in self.agents}
        self.current_step = 1

        # Asignar ciudades random para agentes al inicio.
        random_cities = random.sample(self.all_cities.keys(), self.num_agents)

        for i, agent_id in enumerate(self.agents):
            self.agent_positions[agent_id] = random_cities[i]
            self.previous_cities[agent_id] = random_cities[i]
            self.agent_visited_cities[agent_id].append(random_cities[i])
            self.cities_left.remove(random_cities[i])

        # Actualizar las flags de las ciudades en función de las ya visitadas.
        self.city_flags = [0 if city in self.cities_left else 1 for city in self.all_cities]

        # Obtener posiciones de otros a ojos de cada agente.
        other_cities = [
            [self.all_cities[self.agent_positions[other_agent]] for other_agent in self.agents if other_agent != agent_id]
            for agent_id in self.agents
        ]
        self.other_agents_cities = {agent_id: other_cities[i] for i, agent_id in enumerate(self.agents)}

        # Generar observaciones.
        obs = {
            agent_id: [self.all_cities[self.agent_positions[agent_id]]] + self.other_agents_cities[agent_id] + self.city_flags
            for agent_id in self.agents
        }

        self.previous_observation = obs

        return obs, {"initial_positions": self.agent_positions}

    def step_agent(self, agent_id, action):
        # Para cada agente ejecutar acción elegida.
        current_city = self.agent_positions[agent_id]

        # Para cada caso obtener movimientos válidos.
        possible_actions = list(self.cities[current_city].keys())

        # De primeras no hay penalización por permanecer en la misma ciudad.
        self.penalty = False

        # Determinar próxima ciudad en base a la acción.
        next_city = self.get_city_by_action(agent_id, action)
        if next_city in possible_actions:
            # Actualizar la posición y guardar la anterior.
            self.previous_cities[agent_id] = current_city
            self.agent_positions[agent_id] = next_city

            # Actualizar lo que ven los demás agentes.
            other_cities = [
                [self.all_cities[self.agent_positions[other_agent]] for other_agent in self.agents if other_agent != agent_id]
                for agent_id in self.agents
            ]
            
            self.other_agents_cities = {agent_id: other_cities[i] for i, agent_id in enumerate(self.agents)}
        else:
            # Si la acción no es válida se aplicará una penalización.
            self.penalty = True

        # Comprobamos si las ciudades han sido visitadas previamente. Si no es así se dará recompensa extra.
        # También se tachan las ciudades de la lista si corresponde.
        if self.agent_positions[agent_id] in self.cities_left:
            self.extra_reward[agent_id] = True
            self.cities_left.remove(self.agent_positions[agent_id])
            self.city_flags = [0 if city in self.cities_left else 1 for city in self.all_cities]
        else:
            self.extra_reward[agent_id] = False

        # Añadir ciudad al histórico del agente.
        self.agent_visited_cities[agent_id].append(self.agent_positions[agent_id])

    def get_city_by_action(self, agent_id, action):
        # Obtener ciudades en base a acciones.
        current_city = self.agent_positions[agent_id]
        current_city_actions = self.cities[current_city]

        for neighbor, neighbor_action in current_city_actions.items():
            if action == neighbor_action:
                return neighbor

    def get_observation(self, agent_id):
        # Generamos la observación nueva de cada agente.
        current_city = self.agent_positions[agent_id]
        other_cities = self.other_agents_cities[agent_id]

        obs = [self.all_cities[current_city]] + other_cities + self.city_flags

        return obs

    def get_env_state_results(self):
        # Se dan las recompensas a cada agente.
        rewards = {}
        terminated = False
        truncated = False
        infos = {}
            
        # Vemos si los agentes van a la misma ciudad.
        city_count = Counter(self.agent_positions.values())
        repeated_agent = {agent: city_count[city] > 1 for agent, city in self.agent_positions.items()}

        for agent_id in self.agents:
            current_city = self.agent_positions[agent_id]
            # Se penaliza por cada step.
            reward = -1.0

            # Si no se escoge una acción válida se penaliza, si se explora hay reward y si no también se penaliza.
            if self.penalty:
                reward -= 10.0

            if self.previous_cities[agent_id] != current_city:
                # Se tienen en cuenta los km recorridos.
                reward -= self.distances[self.previous_cities[agent_id]][current_city] / 100.0

            if self.extra_reward[agent_id]:
                reward += 20.0
            else:
                reward -= 10.0

            # Si se han visitado todas las ciudades de la lista, se recompensa a los agentes y se finaliza el episodio.
            if len(self.cities_left) == 0:
                reward += 100.0
                terminated = True

            # Si ambos agentes van al mismo lugar se penaliza también.
            if repeated_agent[agent_id]:
                reward -= 10.0

            rewards[agent_id] = reward
            infos[agent_id] = {"current_city": current_city}

        # Comprobación de si se ha llegado al número máximo de steps.
        truncated = self.current_step >= self.max_steps

        # Si se ha indicado True se muestra el render del entorno.
        if (terminated or truncated) and self.render:
            self.draw_trajectories(img_path='Mapa.jpg')

        self.current_step += 1

        return rewards, terminated, truncated, infos
    
    def sync_wait_for_actions_completion(self):
        pass

    def close(self):
        pass

    def draw_trajectories(self, img_path: str):
        # Coordenadas y ciudades.
        city_coordinates = {
            'Madrid': (256, 221), 'Barcelona': (495, 168), 'Valencia': (391, 272),
            'Sevilla': (162, 388), 'Bilbao': (299, 63), 'A Coruna': (69, 66),
            'Leon': (162, 96), 'Salamanca': (183, 184), 'Murcia': (356, 360)
        }

        # Cargar la imagen base.
        img = cv2.imread(img_path)

        # Dimensiones de la imagen.
        height, width, _ = img.shape

        # Creamos un panel a la derecha.
        panel_width = 300
        base_with_panel = cv2.copyMakeBorder(img, 0, 0, 0, panel_width, cv2.BORDER_CONSTANT, value=(50, 50, 50))

        # Coordenadas para texto.
        text_x = width + 10
        text_y = 30
        line_spacing = 30

        # Lista de colores para dibujar.
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]

        # Crear una copia base para preservar marcas gráficas.
        img_with_panel_static = base_with_panel.copy()

        previous_positions = {agent_id: None for agent_id in self.agents}

        # Iterar sobre los pasos
        for step in range(len(self.agent_visited_cities['truck_1'])):

            panel_texts = [
                'Step:',
                step,
                '================'
            ]

            for i, agent_idx in enumerate(self.agents):
                current_city = self.agent_visited_cities[agent_idx][step]

                if step == 0:
                    previous_positions[agent_idx] = current_city

                    # Posiciones iniciales.
                    panel_texts.append(f"Agente {i+1}:")
                    panel_texts.append(f"Posicion inicial: {current_city}")

                    # Dibujar círculos iniciales.
                    cv2.circle(img_with_panel_static, center=city_coordinates[current_city], radius=5, color=colors[i], thickness=2)
                else:
                    previous_city = previous_positions[agent_idx]

                    # Actualizar texto.
                    panel_texts.append(f"Agente {i+1}:")
                    panel_texts.append(f"{previous_city} -> {current_city}")

                    # Dibujar líneas y círculos.
                    cv2.line(img_with_panel_static, city_coordinates[previous_city], city_coordinates[current_city], color=colors[i], thickness=1)
                    cv2.circle(img_with_panel_static, center=city_coordinates[current_city], radius=5, color=colors[i], thickness=2)

                    # Actualizar posición previa.
                    previous_positions[agent_idx] = current_city

            # Crear una copia dinámica para actualizar los textos.
            img_with_panel_dynamic = img_with_panel_static.copy()

            # Añadir los textos en la copia dinámica.
            for idx, text in enumerate(panel_texts):
                cv2.putText(img_with_panel_dynamic, str(text), (text_x, text_y + idx * line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Mostrar la imagen actualizada.
            start_time = time.time()
            while True:
                cv2.imshow('Mapa con Panel', img_with_panel_dynamic)

                # Verificar si han pasado 2 segundos.
                if time.time() - start_time >= 1:
                    break

                # Esperar un corto tiempo para permitir la actualización de la ventana.
                cv2.waitKey(1)

            cv2.imwrite('./ruta_final.png', img_with_panel_dynamic)

        cv2.destroyAllWindows()
