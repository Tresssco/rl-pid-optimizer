"""
Entrenamiento de agente RL (PPO) para optimizar parámetros PID
Utiliza datos sintéticos generados por synthetic_data_generator.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from synthetic_data_generator import RealisticPlantSimulator


class PIDOptimizationEnv(gym.Env):
    """
    Entorno Gymnasium para optimización de PID usando RL
    
    - Action space: [Kp, Ki, Kd]
    - Observation space: [error, d_error, integral_error, y]
    - Reward: basado en ISE, overshoot, esfuerzo de control
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, process_type: str = 'temperature', 
                 setpoint: float = 50,
                 episode_length: int = 500):
        """
        Args:
            process_type: tipo de proceso a simular
            setpoint: valor objetivo
            episode_length: pasos por episodio
        """
        super().__init__()
        
        self.process_type = process_type
        self.setpoint = setpoint
        self.episode_length = episode_length
        self.dt = 0.1
        
        # Espacio de acciones: [Kp, Ki, Kd]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 5.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Espacio de observaciones: [error, d_error, integral_error, y]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Inicializar simulador
        self.simulator = RealisticPlantSimulator(process_type)
        
        # Estado interno
        self.reset()
    
    def reset(self, seed=None):
        """Reinicia el entorno"""
        super().reset(seed=seed)
        
        if self.process_type == 'temperature':
            self.y = 20.0
        elif self.process_type == 'motor_speed':
            self.y = 0.0
        elif self.process_type == 'tank_level':
            self.y = 30.0
        else:
            self.y = 0.0
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.step_count = 0
        
        # Perturbación
        self.current_setpoint = self.setpoint
        
        obs = np.array([0.0, 0.0, 0.0, self.y], dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        """
        Ejecuta un paso de simulación
        
        Args:
            action: [Kp, Ki, Kd]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        kp, ki, kd = action
        
        # Perturbación: cambiar setpoint a mitad
        if self.step_count == self.episode_length // 2:
            self.current_setpoint = self.setpoint + 15
        
        # Cálculo de error
        error = self.current_setpoint - self.y
        
        # Integral (anti-windup)
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -100, 100)
        
        # Derivada
        if self.step_count > 0:
            d_error = (error - self.prev_error) / self.dt
        else:
            d_error = 0.0
        
        # Control PID
        u = kp * error + ki * self.integral_error + kd * d_error
        u = np.clip(u, -100, 100)
        
        # Simular planta
        dy_dt = self.simulator.plant_model(self.y, u, self.step_count * self.dt)
        self.y = self.y + dy_dt * self.dt
        
        # Agregar ruido (realista)
        y_meas = self.y + np.random.normal(0, 0.3)
        
        # Cálculo de reward
        reward = self._compute_reward(error, u)
        
        # Actualizar variables
        self.prev_error = error
        self.step_count += 1
        
        # Terminación
        done = self.step_count >= self.episode_length
        
        # Nueva observación
        obs = np.array([error, d_error, self.integral_error, self.y], 
                      dtype=np.float32)
        
        # Info para debugging
        info = {
            'error': float(error),
            'u': float(u),
            'y': float(self.y),
            'ise_component': float(error**2),
        }
        
        return obs, reward, done, False, info
    
    def _compute_reward(self, error: float, u: float) -> float:
        """
        Calcula reward
        
        Penaliza:
        - Error (ISE)
        - Overshoot
        - Esfuerzo de control
        """
        # Penalidad por error
        error_penalty = 0.5 * (error**2)
        
        # Penalidad por overshoot (si y > setpoint + margen)
        overshoot_penalty = 0.3 * max(0, self.y - self.current_setpoint - 2)**2
        
        # Penalidad por esfuerzo (reduce consumo de energía)
        energy_penalty = 0.2 * (u**2)
        
        reward = -(error_penalty + overshoot_penalty + energy_penalty)
        
        return reward
    
    def render(self):
        """Renderizar (opcional)"""
        pass


class TrainingCallback(BaseCallback):
    """Callback para monitorear entrenamiento"""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # No hacer nada por ahora
        return True


def train_optimizer(process_type: str = 'temperature',
                   total_timesteps: int = 100000,
                   learning_rate: float = 3e-4):
    """
    Entrena agente RL para optimizar PID
    
    Args:
        process_type: tipo de proceso
        total_timesteps: número total de pasos de entrenamiento
        learning_rate: tasa de aprendizaje
    
    Returns:
        modelo entrenado
    """
    
    print("=" * 80)
    print(f"ENTRENANDO AGENTE RL PARA {process_type.upper()}")
    print("=" * 80)
    
    # Crear entorno
    env = PIDOptimizationEnv(
        process_type=process_type,
        setpoint=50 if process_type == 'temperature' else 1500,
        episode_length=500
    )
    
    # Crear modelo
    print(f"\nCreando modelo PPO...")
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42
    )
    
    # Entrenar
    print(f"\n▶️  Iniciando entrenamiento ({total_timesteps:,} pasos)...")
    print(f"   Esto puede tomar 5-15 minutos...\n")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Guardar
    model_name = f'model_pid_optimizer_{process_type}'
    model.save(model_name)
    print(f"\nModelo entrenado y guardado: {model_name}.zip")
    
    return model, env


def evaluate_trained_model(model, env, num_episodes: int = 5):
    """
    Evalúa el modelo entrenado
    
    Args:
        model: modelo PPO entrenado
        env: entorno
        num_episodes: número de episodios para evaluar
    
    Returns:
        histórico de evaluaciones
    """
    
    print("\n" + "=" * 80)
    print("EVALUANDO MODELO ENTRENADO")
    print("=" * 80)
    
    all_results = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_data = {
            'time': [],
            'y': [],
            'setpoint': [],
            'error': [],
            'u': [],
        }
        
        for step in range(env.episode_length):
            # Predicción del modelo
            action, _states = model.predict(obs, deterministic=True)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            episode_data['time'].append(step * 0.1)
            episode_data['y'].append(info['y'])
            episode_data['setpoint'].append(env.current_setpoint)
            episode_data['error'].append(info['error'])
            episode_data['u'].append(info['u'])
            
            if done:
                break
        
        all_results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'actions': action.copy(),
            'data': pd.DataFrame(episode_data)
        })
        
        print(f"  Episodio {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Kp={action[0]:.3f}, Ki={action[1]:.3f}, Kd={action[2]:.3f}")
    
    # Promediar parámetros aprendidos
    avg_action = np.mean([r['actions'] for r in all_results], axis=0)
    
    print(f"\n📊 PARÁMETROS PROMEDIO APRENDIDOS:")
    print(f"  Kp = {avg_action[0]:.3f}")
    print(f"  Ki = {avg_action[1]:.3f}")
    print(f"  Kd = {avg_action[2]:.3f}")
    
    return all_results, avg_action


def compare_with_baseline(process_type: str, 
                          optimal_params: np.ndarray,
                          eval_results: list):
    """
    Compara parámetros RL con baseline manual
    """
    
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON TUNING MANUAL")
    print("=" * 80)
    
    # Parámetros baseline (tuning manual típico)
    baseline_params = {
        'temperature': {'Kp': 5.0, 'Ki': 0.2, 'Kd': 1.0},
        'motor_speed': {'Kp': 0.8, 'Ki': 0.05, 'Kd': 0.2},
        'tank_level': {'Kp': 10.0, 'Ki': 0.1, 'Kd': 2.0},
    }
    
    baseline = baseline_params.get(process_type, {})
    
    # Simular con baseline
    simulator = RealisticPlantSimulator(process_type)
    data_baseline = simulator.simulate_episode(
        kp=baseline.get('Kp', 1),
        ki=baseline.get('Ki', 0.1),
        kd=baseline.get('Kd', 0.1),
        setpoint=50,
        duration=50
    )
    metrics_baseline = simulator.calculate_metrics(data_baseline)
    
    # Simular con parámetros RL
    data_rl = simulator.simulate_episode(
        kp=optimal_params[0],
        ki=optimal_params[1],
        kd=optimal_params[2],
        setpoint=50,
        duration=50
    )
    metrics_rl = simulator.calculate_metrics(data_rl)
    
    print(f"\n{'Métrica':<20} {'Baseline':<15} {'RL':<15} {'Mejora':<15}")
    print("-" * 65)
    
    for metric in ['ISE', 'Overshoot', 'Settling_Time', 'Energy']:
        val_baseline = metrics_baseline[metric]
        val_rl = metrics_rl[metric]
        
        if val_baseline != 0:
            improvement = ((val_baseline - val_rl) / val_baseline) * 100
        else:
            improvement = 0
        
        print(f"{metric:<20} {val_baseline:<15.2f} {val_rl:<15.2f} {improvement:>+13.1f}%")


def plot_training_results(eval_results: list, optimal_params: np.ndarray):
    """Visualiza resultados de entrenamiento"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resultados del Entrenamiento RL para PID', 
                 fontsize=16, fontweight='bold')
    
    # Usar primer episodio para visualización
    data = eval_results[0]['data']
    
    # Respuesta temporal
    ax = axes[0, 0]
    ax.plot(data['time'], data['setpoint'], 'k--', linewidth=2, label='Setpoint')
    ax.plot(data['time'], data['y'], 'b-', linewidth=1.5, label='Respuesta')
    ax.fill_between(data['time'], data['y'], data['setpoint'], alpha=0.2)
    ax.set_title('Respuesta Temporal del Sistema')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error
    ax = axes[0, 1]
    ax.plot(data['time'], data['error'], 'r-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(data['time'], data['error'], 0, alpha=0.2, color='red')
    ax.set_title('Error del Control')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error')
    ax.grid(True, alpha=0.3)
    
    # Esfuerzo de control
    ax = axes[1, 0]
    ax.plot(data['time'], data['u'], 'g-', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(data['time'], data['u'], 0, alpha=0.2, color='green')
    ax.set_title('Esfuerzo de Control (u)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Salida')
    ax.grid(True, alpha=0.3)
    
    # Parámetros aprendidos
    ax = axes[1, 1]
    params_names = ['Kp', 'Ki', 'Kd']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(params_names, optimal_params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_title('Parámetros PID Aprendidos')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráficas guardadas: training_results.png")
    plt.show()


if __name__ == '__main__':
    # Entrenar para temperatura
    model, env = train_optimizer(
        process_type='temperature',
        total_timesteps=100000,
        learning_rate=3e-4
    )
    
    # Evaluar
    eval_results, optimal_params = evaluate_trained_model(model, env, num_episodes=5)
    
    # Comparar con baseline
    compare_with_baseline('temperature', optimal_params, eval_results)
    
    # Visualizar
    plot_training_results(eval_results, optimal_params)
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)