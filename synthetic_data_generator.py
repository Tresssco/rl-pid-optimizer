"""
Generador de datos sintéticos realistas para plantas de control
Simula dinámicas típicas con ruido, retrasos y perturbaciones
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class RealisticPlantSimulator:
    """
    Simula dinámicas realistas de diferentes procesos.
    Incluye ruido de sensor, retrasos, no-linealidades.
    """
    
    def __init__(self, process_type: str = 'temperature', seed: int = 42):
        """
        Args:
            process_type: 'temperature', 'motor_speed', 'tank_level', 'pressure'
            seed: para reproducibilidad
        """
        self.process_type = process_type
        self.seed = seed
        np.random.seed(seed)
        self.dt = 0.1  # Tiempo de muestreo: 100ms (típico PLC)
        
    def plant_model(self, y: float, u: float, t: float) -> float:
        """Modelo dinámico del proceso. dy/dt = f(y, u)"""
        
        if self.process_type == 'temperature':
            return self._temperature_model(y, u)
        elif self.process_type == 'motor_speed':
            return self._motor_speed_model(y, u)
        elif self.process_type == 'tank_level':
            return self._tank_level_model(y, u)
        elif self.process_type == 'pressure':
            return self._pressure_model(y, u)
        else:
            raise ValueError(f"Unknown process type: {self.process_type}")
    
    def _temperature_model(self, y: float, u: float) -> float:
        """
        Modelo de reactor/horno térmico
        dy/dt = -a*(y-T_amb) + b*u + ruido_ambiental
        """
        T_ambient = 20
        a = 0.15  # Velocidad de enfriamiento (constante térmica)
        b = 0.8   # Ganancia del actuador
        
        # Dead-zone en actuador (retraso inicialización)
        if abs(u) < 1.5:
            u_effective = 0
        else:
            u_effective = u - np.sign(u) * 1.5  # Hysteresis
        
        # Dinámica principal
        dy_dt = -a * (y - T_ambient) + b * u_effective
        
        return dy_dt
    
    def _motor_speed_model(self, y: float, u: float) -> float:
        """
        Modelo de motor DC con carga variable
        dy/dt = -c*y + d*u + perturbación
        """
        c = 0.2   # Fricción/amortiguamiento
        d = 1.5   # Ganancia del motor
        
        # Saturación de voltaje
        u_sat = np.clip(u, -100, 100)
        
        dy_dt = -c * y + d * u_sat
        
        return dy_dt
    
    def _tank_level_model(self, y: float, u: float) -> float:
        """
        Modelo de tanque cilíndrico con bomba
        dy/dt = u - c*sqrt(y)  (Torricelli, no-lineal)
        """
        c = 0.3  # Coeficiente de descarga
        
        # u es caudal de entrada
        inflow = u
        outflow = c * np.sqrt(max(0, y))  # No puede haber flujo negativo
        
        dy_dt = inflow - outflow
        
        return dy_dt
    
    def _pressure_model(self, y: float, u: float) -> float:
        """
        Modelo de compresor con válvula reguladora
        dy/dt = a*u - b*y  (respuesta exponencial)
        """
        a = 2.0
        b = 0.3
        
        dy_dt = a * u - b * y
        
        return dy_dt
    
    def add_sensor_noise(self, signal: np.ndarray, 
                        noise_type: str = 'realistic') -> np.ndarray:
        """
        Añade ruido realista de sensor
        
        Args:
            signal: señal limpia
            noise_type: 'gaussian' o 'realistic' (con drift)
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.5, len(signal))
        else:  # realistic
            # Ruido Gaussiano
            gaussian_noise = np.random.normal(0, 0.3, len(signal))
            
            # Drift lento (cambios de cero del sensor)
            drift = 0.02 * np.sin(np.linspace(0, 4*np.pi, len(signal)))
            
            # Outliers ocasionales
            outlier_mask = np.random.random(len(signal)) < 0.01
            outliers = outlier_mask * np.random.normal(0, 2, len(signal))
            
            noise = gaussian_noise + drift + outliers
        
        return signal + noise
    
    def add_sensor_delay(self, signal: np.ndarray, 
                        delay_steps: int = 2) -> np.ndarray:
        """Simula retraso de sensor (comunicación lenta)"""
        if delay_steps <= 0:
            return signal
        
        delayed_signal = np.zeros_like(signal)
        delayed_signal[:delay_steps] = signal[0]  # Valor inicial
        delayed_signal[delay_steps:] = signal[:-delay_steps]
        
        return delayed_signal
    
    def simulate_episode(self, kp: float, ki: float, kd: float,
                        setpoint: float = 50,
                        duration: float = 300,
                        perturbation_time: float = 150,
                        add_noise: bool = True) -> pd.DataFrame:
        """
        Simula un episodio completo de control PID
        
        Args:
            kp, ki, kd: parámetros del controlador PID
            setpoint: valor objetivo
            duration: duración en segundos (simulados)
            perturbation_time: momento en que cambia el setpoint
            add_noise: si añadir ruido realista
        
        Returns:
            DataFrame con histórico completo
        """
        steps = int(duration / self.dt)
        
        # Inicializar según tipo de proceso
        if self.process_type == 'temperature':
            y_initial = 20.0
        elif self.process_type == 'motor_speed':
            y_initial = 0.0
        elif self.process_type == 'tank_level':
            y_initial = 30.0
        elif self.process_type == 'pressure':
            y_initial = 0.0
        else:
            y_initial = 0.0
        
        # Arrays para almacenar datos
        data = {
            'time': np.zeros(steps),
            'setpoint': np.zeros(steps),
            'y_true': np.zeros(steps),
            'y_measured': np.zeros(steps),
            'error': np.zeros(steps),
            'u': np.zeros(steps),
            'integral_error': np.zeros(steps),
            'derivative_error': np.zeros(steps),
        }
        
        # Estado inicial
        y = y_initial
        integral_error = 0.0
        prev_error = 0.0
        
        # Simular
        for step in range(steps):
            # Actualizar setpoint (perturbación a mitad)
            if step >= int(perturbation_time / self.dt):
                sp = setpoint + 15  # Cambio de setpoint
            else:
                sp = setpoint
            
            # Cálculo de error
            error = sp - y
            
            # Integración
            integral_error += error * self.dt
            integral_error = np.clip(integral_error, -100, 100)  # Anti-windup
            
            # Derivada
            if step > 0:
                d_error = (error - prev_error) / self.dt
            else:
                d_error = 0.0
            
            # Control PID
            u = kp * error + ki * integral_error + kd * d_error
            u = np.clip(u, -100, 100)  # Saturación del actuador
            
            # Integración numérica
            dy_dt = self.plant_model(y, u, step * self.dt)
            y = y + dy_dt * self.dt
            
            # Ruido de sensor
            if add_noise:
                y_meas = self.add_sensor_noise(np.array([y]))[0]
            else:
                y_meas = y
            
            # Almacenar datos
            data['time'][step] = step * self.dt
            data['setpoint'][step] = sp
            data['y_true'][step] = y
            data['y_measured'][step] = y_meas
            data['error'][step] = error
            data['u'][step] = u
            data['integral_error'][step] = integral_error
            data['derivative_error'][step] = d_error
            
            prev_error = error
        
        return pd.DataFrame(data)
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas de control"""
        error = df['y_true'].values - df['setpoint'].values
        
        # ISE: Integral Squared Error
        ise = np.sum(error**2) * self.dt
        
        # Overshoot
        max_overshoot = max(0, np.max(df['y_true']) - np.max(df['setpoint']))
        overshoot_percent = (max_overshoot / np.max(df['setpoint'])) * 100 if np.max(df['setpoint']) > 0 else 0
        
        # Settling time (cuando error < 5%)
        settling_threshold = 0.05 * np.max(df['setpoint'])
        settling_indices = np.where(np.abs(error) <= settling_threshold)[0]
        if len(settling_indices) > 0:
            settling_time = settling_indices[0] * self.dt
        else:
            settling_time = np.nan
        
        # Steady-state error
        ss_error = np.mean(error[-50:])  # Últimos 5 segundos
        
        # Energy (esfuerzo de control)
        energy = np.sum(np.abs(df['u'].values)) * self.dt
        
        return {
            'ISE': ise,
            'Overshoot': overshoot_percent,
            'Settling_Time': settling_time,
            'SS_Error': ss_error,
            'Energy': energy,
            'Max_Output': np.max(np.abs(df['u'].values))
        }


def generate_all_scenarios():
    """Genera datos para todos los escenarios de uso"""
    
    scenarios = [
        {
            'name': 'Temperature Control',
            'process_type': 'temperature',
            'setpoint': 50,
            'kp_bad': 5.0,
            'ki_bad': 0.2,
            'kd_bad': 1.0,
            'kp_good': 2.34,
            'ki_good': 0.87,
            'kd_good': 0.45,
        },
        {
            'name': 'Motor Speed',
            'process_type': 'motor_speed',
            'setpoint': 1500,
            'kp_bad': 0.8,
            'ki_bad': 0.05,
            'kd_bad': 0.2,
            'kp_good': 0.45,
            'ki_good': 0.12,
            'kd_good': 0.18,
        },
        {
            'name': 'Tank Level',
            'process_type': 'tank_level',
            'setpoint': 50,
            'kp_bad': 10,
            'ki_bad': 0.1,
            'kd_bad': 2,
            'kp_good': 4.2,
            'ki_good': 0.35,
            'kd_good': 1.5,
        },
    ]
    
    print("=" * 80)
    print("GENERANDO DATOS SINTÉTICOS REALISTAS")
    print("=" * 80)
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Procesando: {scenario['name']}")
        print("-" * 80)
        
        simulator = RealisticPlantSimulator(scenario['process_type'])
        
        # Simular con PID malo (actual)
        print(f"  Simulando PID actual (Kp={scenario['kp_bad']}, Ki={scenario['ki_bad']}, Kd={scenario['kd_bad']})...")
        data_bad = simulator.simulate_episode(
            kp=scenario['kp_bad'],
            ki=scenario['ki_bad'],
            kd=scenario['kd_bad'],
            setpoint=scenario['setpoint'],
            duration=300
        )
        metrics_bad = simulator.calculate_metrics(data_bad)
        
        # Simular con PID óptimo
        print(f"  Simulando PID optimizado (Kp={scenario['kp_good']}, Ki={scenario['ki_good']}, Kd={scenario['kd_good']})...")
        data_good = simulator.simulate_episode(
            kp=scenario['kp_good'],
            ki=scenario['ki_good'],
            kd=scenario['kd_good'],
            setpoint=scenario['setpoint'],
            duration=300
        )
        metrics_good = simulator.calculate_metrics(data_good)
        
        # Guardar CSVs
        process_name = scenario['process_type']
        data_bad.to_csv(f'data_{process_name}_pid_actual.csv', index=False)
        data_good.to_csv(f'data_{process_name}_pid_optimized.csv', index=False)
        
        print(f"  ✓ Guardado: data_{process_name}_pid_actual.csv")
        print(f"  ✓ Guardado: data_{process_name}_pid_optimized.csv")
        
        # Mostrar comparación
        print(f"\n  COMPARACIÓN DE RESULTADOS:")
        print(f"  {'Métrica':<20} {'PID Actual':<15} {'PID Optimizado':<15} {'Mejora':<15}")
        print(f"  {'-'*65}")
        
        for metric in ['ISE', 'Overshoot', 'Settling_Time', 'Energy']:
            val_bad = metrics_bad[metric]
            val_good = metrics_good[metric]
            
            if val_bad == 0:
                improvement = 0
            else:
                improvement = ((val_bad - val_good) / val_bad) * 100
            
            print(f"  {metric:<20} {val_bad:<15.2f} {val_good:<15.2f} {improvement:>+13.1f}%")
        
        results[process_name] = {
            'data_bad': data_bad,
            'data_good': data_good,
            'metrics_bad': metrics_bad,
            'metrics_good': metrics_good,
            'scenario': scenario
        }
    
    print("\n" + "=" * 80)
    print("GENERACIÓN COMPLETADA")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    # Ejecutar generación
    results = generate_all_scenarios()
    
    # Crear visualización
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Comparación: PID Actual vs Optimizado por RL', fontsize=16, fontweight='bold')
    
    for idx, (process_name, result) in enumerate(results.items()):
        ax1 = axes[idx, 0]
        ax2 = axes[idx, 1]
        
        data_bad = result['data_bad']
        data_good = result['data_good']
        
        # Gráfico de respuesta temporal
        ax1.plot(data_bad['time'], data_bad['setpoint'], 'k--', linewidth=2, label='Setpoint')
        ax1.plot(data_bad['time'], data_bad['y_measured'], 'r-', linewidth=1.5, label='PID Actual', alpha=0.7)
        ax1.plot(data_good['time'], data_good['y_measured'], 'g-', linewidth=1.5, label='PID Optimizado', alpha=0.7)
        ax1.set_title(f'{process_name} - Respuesta Temporal')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de error
        error_bad = data_bad['error'].values
        error_good = data_good['error'].values
        
        ax2.plot(data_bad['time'], error_bad, 'r-', linewidth=1.5, label='PID Actual', alpha=0.7)
        ax2.plot(data_good['time'], error_good, 'g-', linewidth=1.5, label='PID Optimizado', alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title(f'{process_name} - Error')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_synthetic_data.png', dpi=150, bbox_inches='tight')
    print(f"\n Visualización guardada: comparison_synthetic_data.png")
    plt.show()
