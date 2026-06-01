"""
Integración con PLCSIM de Siemens TIA Portal
Simula la conexión OPC-UA con un PLC virtual
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Optional
import json


class PLCSIMBridge:
    """
    Simula la conexión OPC-UA con PLCSIM
    En producción, usarías: from opcua import Client
    
    Aquí simulamos el comportamiento típico de un PLC Siemens
    """
    
    def __init__(self, ip: str = '127.0.0.1', port: int = 4840,
                 process_type: str = 'temperature'):
        """
        Args:
            ip: dirección IP del PLC
            port: puerto OPC-UA
            process_type: tipo de proceso simulado
        """
        self.ip = ip
        self.port = port
        self.process_type = process_type
        self.is_connected = False
        
        # Variables del PLC (simuladas)
        self.plc_variables = {
            'Setpoint': 50.0,
            'ProcessValue': 20.0,
            'Kp': 5.0,
            'Ki': 0.2,
            'Kd': 1.0,
            'Output': 0.0,
            'Error': 0.0,
            'Integral': 0.0,
            'SystemTime': 0,
            'IsRunning': True,
            'AlarmFlag': False,
            'AlarmMessage': '',
        }
        
        # Histórico
        self.data_log = []
        self.log_enabled = False
        
    def connect(self) -> bool:
        """Conectar con PLCSIM"""
        try:
            print(f"🔌 Conectando con PLCSIM en {self.ip}:{self.port}...")
            # Simulamos la conexión
            time.sleep(0.5)
            self.is_connected = True
            print(f"Conexión establecida")
            return True
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Desconectar de PLCSIM"""
        try:
            self.is_connected = False
            print(f"Desconectado de PLCSIM")
            return True
        except Exception as e:
            print(f"❌ Error al desconectar: {e}")
            return False
    
    def read_variable(self, var_name: str) -> Optional[float]:
        """
        Lee una variable del PLC
        
        Args:
            var_name: nombre de la variable
        
        Returns:
            valor de la variable
        """
        if not self.is_connected:
            print("❌ No está conectado con PLC")
            return None
        
        if var_name not in self.plc_variables:
            print(f"❌ Variable '{var_name}' no existe en PLC")
            return None
        
        return self.plc_variables[var_name]
    
    def write_variable(self, var_name: str, value: float) -> bool:
        """
        Escribe una variable en el PLC
        
        Args:
            var_name: nombre de la variable
            value: nuevo valor
        
        Returns:
            True si fue exitoso
        """
        if not self.is_connected:
            print("❌ No está conectado con PLC")
            return False
        
        if var_name not in self.plc_variables:
            print(f"❌ Variable '{var_name}' no existe en PLC")
            return False
        
        # Simulamos validaciones del PLC
        if var_name == 'Kp':
            value = np.clip(value, 0.1, 20.0)  # Rango válido
        elif var_name == 'Ki':
            value = np.clip(value, 0.0, 5.0)
        elif var_name == 'Kd':
            value = np.clip(value, 0.0, 2.0)
        
        old_value = self.plc_variables[var_name]
        self.plc_variables[var_name] = value
        
        # Logging
        if self.log_enabled:
            self.data_log.append({
                'timestamp': time.time(),
                'variable': var_name,
                'old_value': old_value,
                'new_value': value
            })
        
        return True
    
    def update_pid_parameters(self, kp: float, ki: float, kd: float) -> bool:
        """
        Actualiza los tres parámetros PID simultaneamente
        
        Args:
            kp, ki, kd: nuevos parámetros
        
        Returns:
            True si fue exitoso
        """
        if not self.is_connected:
            return False
        
        print(f"\nActualizando parámetros PID en PLCSIM...")
        print(f"   Kp: {self.plc_variables['Kp']:.3f} → {kp:.3f}")
        print(f"   Ki: {self.plc_variables['Ki']:.3f} → {ki:.3f}")
        print(f"   Kd: {self.plc_variables['Kd']:.3f} → {kd:.3f}")
        
        success = True
        success &= self.write_variable('Kp', kp)
        success &= self.write_variable('Ki', ki)
        success &= self.write_variable('Kd', kd)
        
        if success:
            print(f"Parámetros actualizados exitosamente")
        
        return success
    
    def simulate_plc_cycle(self, dt: float = 0.1) -> bool:
        """
        Simula un ciclo PLC completo (lectura sensores, cálculo PID, escritura salida)
        
        Args:
            dt: tiempo de muestreo
        
        Returns:
            True si fue exitoso
        """
        if not self.is_connected:
            return False
        
        # Simular dinámica del proceso
        setpoint = self.plc_variables['Setpoint']
        y = self.plc_variables['ProcessValue']
        kp = self.plc_variables['Kp']
        ki = self.plc_variables['Ki']
        kd = self.plc_variables['Kd']
        integral = self.plc_variables['Integral']
        
        # Cálculo PID
        error = setpoint - y
        integral += error * dt
        integral = np.clip(integral, -100, 100)  # Anti-windup
        
        # Salida PID
        u = kp * error + ki * integral + kd * 0  # Sin derivada en este ciclo
        u = np.clip(u, -100, 100)
        
        # Simular dinámica simple del proceso
        # dy/dt = -0.1*(y-20) + 0.5*u
        dy_dt = -0.1 * (y - 20) + 0.5 * u
        y_new = y + dy_dt * dt
        
        # Agregar ruido
        y_new += np.random.normal(0, 0.2)
        
        # Actualizar variables del PLC
        self.plc_variables['ProcessValue'] = y_new
        self.plc_variables['Error'] = error
        self.plc_variables['Integral'] = integral
        self.plc_variables['Output'] = u
        self.plc_variables['SystemTime'] += dt
        
        # Detectar alarmas
        if abs(error) > 20:
            self.plc_variables['AlarmFlag'] = True
            self.plc_variables['AlarmMessage'] = f'Error alto: {error:.2f}'
        else:
            self.plc_variables['AlarmFlag'] = False
            self.plc_variables['AlarmMessage'] = ''
        
        # Logging
        if self.log_enabled:
            self.data_log.append({
                'timestamp': self.plc_variables['SystemTime'],
                'setpoint': setpoint,
                'process_value': y_new,
                'error': error,
                'output': u,
                'integral': integral,
                'kp': kp,
                'ki': ki,
                'kd': kd,
            })
        
        return True
    
    def get_plc_state(self) -> Dict:
        """Obtiene estado completo del PLC"""
        if not self.is_connected:
            return {}
        
        return self.plc_variables.copy()
    
    def enable_logging(self):
        """Habilita logging de datos"""
        self.log_enabled = True
        self.data_log = []
        print("📊 Logging habilitado")
    
    def disable_logging(self):
        """Deshabilita logging"""
        self.log_enabled = False
        print("📊 Logging deshabilitado")
    
    def get_logged_data(self) -> pd.DataFrame:
        """Obtiene datos registrados como DataFrame"""
        if not self.data_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.data_log)
    
    def save_logged_data(self, filename: str = 'plcsim_data.csv'):
        """Guarda datos registrados a CSV"""
        df = self.get_logged_data()
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"✓ Datos guardados en: {filename}")
            return True
        else:
            print("No hay datos para guardar")
            return False
    
    def generate_deployment_code(self, kp: float, ki: float, kd: float) -> str:
        """
        Genera código SCL para Siemens TIA Portal
        
        Args:
            kp, ki, kd: parámetros PID
        
        Returns:
            código SCL
        """
        
        scl_code = f"""
// ===================================================
// Bloque PID Optimizado por Reinforcement Learning
// Generado automáticamente
// ===================================================

FUNCTION_BLOCK FB_PID_Optimized
VAR_INPUT
    SetPoint: REAL;
    ProcessValue: REAL;
END_VAR

VAR_OUTPUT
    Output: REAL;
END_VAR

VAR
    Error: REAL;
    ErrorOld: REAL;
    ErrorSum: REAL;
    
    // Parámetros Optimizados por RL
    Kp: REAL := {kp:.6f};  // Ganancia Proporcional
    Ki: REAL := {ki:.6f};  // Ganancia Integral
    Kd: REAL := {kd:.6f};  // Ganancia Derivativa
    
    // Anti-windup
    MaxIntegral: REAL := 100.0;
    MinIntegral: REAL := -100.0;
    
    // Saturación
    MaxOutput: REAL := 100.0;
    MinOutput: REAL := -100.0;
    
    // Tiempo de muestreo
    dt: REAL := 0.1;  // 100ms
END_VAR

// Cálculo PID
Error := SetPoint - ProcessValue;
ErrorSum := ErrorSum + Error * dt;

// Anti-windup (integral clamping)
IF ErrorSum > MaxIntegral THEN
    ErrorSum := MaxIntegral;
END_IF;

IF ErrorSum < MinIntegral THEN
    ErrorSum := MinIntegral;
END_IF;

// Componentes PID
Output := Kp * Error + Ki * ErrorSum + Kd * (Error - ErrorOld) / dt;

// Saturación del actuador
IF Output > MaxOutput THEN
    Output := MaxOutput;
END_IF;

IF Output < MinOutput THEN
    Output := MinOutput;
END_IF;

// Guardar error anterior
ErrorOld := Error;

END_FUNCTION_BLOCK
"""
        
        return scl_code
    
    def export_xml_parameters(self, kp: float, ki: float, kd: float,
                             filename: str = 'optimized_parameters.xml') -> bool:
        """
        Exporta parámetros en formato XML para TIA Portal
        
        Args:
            kp, ki, kd: parámetros
            filename: nombre del archivo
        
        Returns:
            True si fue exitoso
        """
        
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<PIDParameters>
    <Header>
        <GeneratedBy>RL-PID Optimizer</GeneratedBy>
        <Timestamp>{pd.Timestamp.now()}</Timestamp>
        <Method>Reinforcement Learning (PPO)</Method>
    </Header>
    
    <OptimizedParameters>
        <Kp value="{kp:.6f}" min="0.1" max="20.0">
            <Description>Ganancia Proporcional</Description>
        </Kp>
        <Ki value="{ki:.6f}" min="0.0" max="5.0">
            <Description>Ganancia Integral</Description>
        </Ki>
        <Kd value="{kd:.6f}" min="0.0" max="2.0">
            <Description>Ganancia Derivativa</Description>
        </Kd>
    </OptimizedParameters>
    
    <ImplementationGuide>
        <Step1>Copiar valores en su bloque FC PID</Step1>
        <Step2>Cargar el programa en el PLC</Step2>
        <Step3>Probar en control cerrado</Step3>
        <Step4>Validar respuesta del sistema</Step4>
    </ImplementationGuide>
</PIDParameters>
"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            print(f"✓ Parámetros exportados: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error al exportar: {e}")
            return False


def test_plcsim_bridge():
    """Prueba la integración con PLCSIM"""
    
    print("=" * 80)
    print("PRUEBA DE INTEGRACIÓN PLCSIM")
    print("=" * 80)
    
    # Conectar
    bridge = PLCSIMBridge()
    bridge.connect()
    
    # Habilitar logging
    bridge.enable_logging()
    
    print("\n📋 Estado inicial del PLC:")
    state = bridge.get_plc_state()
    for key, value in state.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Actualizar parámetros
    print("\nActualizando parámetros PID...")
    bridge.update_pid_parameters(kp=2.34, ki=0.87, kd=0.45)
    
    # Simular 50 ciclos PLC
    print("\n▶️  Simulando 50 ciclos PLC (5 segundos)...")
    for i in range(50):
        bridge.simulate_plc_cycle()
        if i % 10 == 0:
            state = bridge.get_plc_state()
            print(f"  Ciclo {i:2d}: PV={state['ProcessValue']:.2f}°C, "
                  f"Error={state['Error']:.2f}, Output={state['Output']:.2f}")
    
    # Obtener y guardar datos
    print("\n💾 Guardando datos registrados...")
    bridge.save_logged_data('plcsim_simulation.csv')
    
    # Generar código SCL
    print("\n📄 Generando código SCL para TIA Portal...")
    scl_code = bridge.generate_deployment_code(2.34, 0.87, 0.45)
    
    with open('PID_Optimized.scl', 'w') as f:
        f.write(scl_code)
    print(f"✓ Código SCL guardado: PID_Optimized.scl")
    
    # Exportar parámetros XML
    print("\n📦 Exportando parámetros XML...")
    bridge.export_xml_parameters(2.34, 0.87, 0.45)
    
    # Desconectar
    bridge.disconnect()
    
    print("\n" + "=" * 80)
    print("PRUEBA COMPLETADA")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  - plcsim_simulation.csv")
    print("  - PID_Optimized.scl")
    print("  - optimized_parameters.xml")


if __name__ == '__main__':
    test_plcsim_bridge()
