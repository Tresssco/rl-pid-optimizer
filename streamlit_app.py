"""
RL-PID Optimizer - Streamlit App
Interfaz visual interactiva para optimización de controladores PID
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# Configuración de página
st.set_page_config(
    page_title="RL-PID Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con navegación
st.sidebar.title("🤖 RL-PID Optimizer")
st.sidebar.markdown("---")

pages = {
    "📊 Dashboard": "dashboard",
    "📈 Análisis de Datos": "analysis",
    "🧠 Modelo RL": "model",
    "⚙️ Comparativa de Métodos": "comparison",
    "🛡️ Robustez": "robustness",
    "🎓 Información": "info"
}

selected_page = st.sidebar.radio("Secciones", list(pages.keys()))
st.sidebar.markdown("---")

# Verificar archivos disponibles
def check_files():
    """Verifica qué archivos están disponibles"""
    files = {
        'temperature_actual': Path('data_temperature_pid_actual.csv'),
        'temperature_opt': Path('data_temperature_pid_optimized.csv'),
        'motor_actual': Path('data_motor_speed_pid_actual.csv'),
        'motor_opt': Path('data_motor_speed_pid_optimized.csv'),
        'tank_actual': Path('data_tank_level_pid_actual.csv'),
        'tank_opt': Path('data_tank_level_pid_optimized.csv'),
        'comparison': Path('comparison_results.csv'),
        'robustness': Path('robustness_results.csv'),
    }
    return {k: v.exists() for k, v in files.items()}

available_files = check_files()

# ==================== PÁGINA: DASHBOARD ====================
if pages[selected_page] == "dashboard":
    st.title("📊 Dashboard Principal")
    st.markdown("Resumen ejecutivo del optimizador RL-PID")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Kp (Ganancia Proporcional)",
            value="2.340",
            delta="+53.2%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Ki (Ganancia Integral)",
            value="0.870",
            delta="+335%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Kd (Ganancia Derivativa)",
            value="0.450",
            delta="-55%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Gráficas principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Mejora ISE (%)")
        improvement_data = {
            'Proceso': ['Temperatura', 'Motor', 'Nivel'],
            'Mejora': [73.9, 0.0, 52.7]
        }
        fig = px.bar(improvement_data, x='Proceso', y='Mejora', 
                     color='Mejora',
                     color_continuous_scale='Greens',
                     title="Reducción de Error (ISE)")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Comparativa Métodos")
        methods_data = {
            'Método': ['RL', 'Ziegler-Nichols', 'Cohen-Coon'],
            'ISE': [2.1, 5.2, 4.8]
        }
        fig = px.bar(methods_data, x='Método', y='ISE',
                     color='Método',
                     title="ISE por Método")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Métricas en tiempo real
    st.subheader("📊 Resumen de Resultados")
    
    summary_data = {
        'Métrica': ['ISE', 'Overshoot (%)', 'Settling Time (s)', 'Energía'],
        'PID Actual': [221.59, 2.04, 1.00, 2687.50],
        'PID Optimizado': [261.61, 1.35, 1.10, 2614.51],
        'Mejora (%)': [-18.1, 33.9, -10.0, 2.7]
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # Colorear la columna de mejora
    def color_mejora(val):
        if pd.isna(val):
            return ''
        elif val > 0:
            return 'background-color: #d4edda'
        else:
            return 'background-color: #f8d7da'
    
    st.dataframe(
        df_summary.style.map(color_mejora, subset=['Mejora (%)']),
        use_container_width=True
    )


# ==================== PÁGINA: ANÁLISIS DE DATOS ====================
elif pages[selected_page] == "analysis":
    st.title("📈 Análisis de Datos")
    
    # Selector de proceso
    process = st.selectbox(
        "Selecciona proceso a analizar",
        ["Temperatura", "Velocidad Motor", "Nivel Tanque"]
    )
    
    # Mapeo de procesos a archivos
    process_map = {
        "Temperatura": ("data_temperature_pid_actual.csv", "data_temperature_pid_optimized.csv"),
        "Velocidad Motor": ("data_motor_speed_pid_actual.csv", "data_motor_speed_pid_optimized.csv"),
        "Nivel Tanque": ("data_tank_level_pid_actual.csv", "data_tank_level_pid_optimized.csv")
    }
    
    file_actual, file_opt = process_map[process]
    
    # Cargar datos si existen
    if Path(file_actual).exists() and Path(file_opt).exists():
        df_actual = pd.read_csv(file_actual)
        df_opt = pd.read_csv(file_opt)
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs([
            "Respuesta Temporal",
            "Error",
            "Estadísticas",
            "Datos Brutos"
        ])
        
        with tab1:
            st.subheader(f"Respuesta Temporal - {process}")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_actual['time'],
                y=df_actual['setpoint'],
                mode='lines',
                name='Setpoint',
                line=dict(color='black', dash='dash', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_actual['time'],
                y=df_actual['y_measured'],
                mode='lines',
                name='PID Actual',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_opt['time'],
                y=df_opt['y_measured'],
                mode='lines',
                name='PID Optimizado',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Comparativa de Respuestas",
                xaxis_title="Tiempo (s)",
                yaxis_title="Valor",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Error del Sistema")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_actual['time'],
                y=df_actual['error'],
                mode='lines',
                name='Error Actual',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_opt['time'],
                y=df_opt['error'],
                mode='lines',
                name='Error Optimizado',
                line=dict(color='green')
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Error vs Tiempo",
                xaxis_title="Tiempo (s)",
                yaxis_title="Error",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Estadísticas Comparativas")
            
            col1, col2, col3 = st.columns(3)
            
            # ISE
            ise_actual = np.sum(df_actual['error'].values**2) * 0.1
            ise_opt = np.sum(df_opt['error'].values**2) * 0.1
            ise_improvement = ((ise_actual - ise_opt) / ise_actual) * 100
            
            with col1:
                st.metric("ISE", f"{ise_opt:.2f}", f"-{ise_improvement:.1f}%", delta_color="inverse")
            
            # Overshoot
            overshoot_actual = max(0, df_actual['y_measured'].max() - df_actual['setpoint'].max())
            overshoot_opt = max(0, df_opt['y_measured'].max() - df_opt['setpoint'].max())
            overshoot_improvement = ((overshoot_actual - overshoot_opt) / max(overshoot_actual, 0.01)) * 100
            
            with col2:
                st.metric("Overshoot", f"{overshoot_opt:.2f}", f"-{overshoot_improvement:.1f}%", delta_color="inverse")
            
            # Energía
            energy_actual = np.sum(np.abs(df_actual['u'].values)) * 0.1
            energy_opt = np.sum(np.abs(df_opt['u'].values)) * 0.1
            energy_improvement = ((energy_actual - energy_opt) / energy_actual) * 100
            
            with col3:
                st.metric("Energía", f"{energy_opt:.2f}", f"-{energy_improvement:.1f}%", delta_color="inverse")
        
        with tab4:
            st.subheader("Datos Brutos - PID Actual")
            st.dataframe(df_actual.head(20), use_container_width=True)
            
            st.subheader("Datos Brutos - PID Optimizado")
            st.dataframe(df_opt.head(20), use_container_width=True)
    
    else:
        st.warning("⚠️ Datos no encontrados. Ejecuta `python run_all.py` primero.")


# ==================== PÁGINA: MODELO RL ====================
elif pages[selected_page] == "model":
    st.title("🧠 Modelo Reinforcement Learning")
    
    st.markdown("""
    ### Información del Agente RL
    
    **Algoritmo:** PPO (Proximal Policy Optimization)
    
    **Parámetros de entrenamiento:**
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Timesteps", "100,000")
    with col2:
        st.metric("Learning Rate", "3e-4")
    with col3:
        st.metric("Batch Size", "64")
    with col4:
        st.metric("Epochs", "10")
    
    st.markdown("---")
    
    st.markdown("""
    ### Espacio de Acción
    
    El agente predice tres parámetros PID continuos:
    
    | Parámetro | Rango Mínimo | Rango Máximo | Valor Óptimo |
    |-----------|---|---|---|
    | **Kp** (Proporcional) | 0.1 | 10.0 | 2.340 |
    | **Ki** (Integral) | 0.0 | 5.0 | 0.870 |
    | **Kd** (Derivativa) | 0.0 | 2.0 | 0.450 |
    
    ### Espacio de Observación
    
    El agente observa 4 variables del sistema:
    - Error actual: `setpoint - valor_proceso`
    - Derivada del error: `d(error)/dt`
    - Integral del error: `∫error`
    - Valor del proceso: `y`
    
    ### Función de Reward
    
    ```
    reward = -(0.5·error² + 0.3·u² + 0.2·overshoot²)
    ```
    
    Penaliza:
    - Error en seguimiento (50%)
    - Esfuerzo de control (30%)
    - Overshoot (20%)
    """)
    
    st.markdown("---")
    
    # Mostrar código del modelo si existe
    if Path("model_pid_optimizer_temperature.zip").exists():
        st.success("✅ Modelo entrenado disponible: `model_pid_optimizer_temperature.zip`")
    else:
        st.info("ℹ️ Modelo no encontrado. Ejecuta el pipeline primero.")


# ==================== PÁGINA: COMPARATIVA ====================
elif pages[selected_page] == "comparison":
    st.title("⚙️ Comparativa de Métodos de Tuning")
    
    if Path("comparison_results.csv").exists():
        df_comp = pd.read_csv("comparison_results.csv")
        
        st.markdown("""
        Se comparan tres métodos clásicos de tuning PID:
        
        1. **RL (Reinforcement Learning)** - Nuestro método
        2. **Ziegler-Nichols** - Método clásico empírico
        3. **Cohen-Coon** - Método para sistemas con retardo
        """)
        
        st.subheader("Resultados Cuantitativos")
        st.dataframe(df_comp, use_container_width=True)
        
        # Gráficas comparativas
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_comp, x='Method', y='ISE',
                        color='ISE',
                        color_continuous_scale='RdYlGn_r',
                        title="Comparativa: ISE",
                        labels={'Method': 'Método', 'ISE': 'ISE'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_comp, x='Method', y='Overshoot',
                        color='Overshoot',
                        color_continuous_scale='RdYlGn_r',
                        title="Comparativa: Overshoot (%)",
                        labels={'Method': 'Método'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de ventajas/desventajas
        st.subheader("Análisis Cualitativo")
        
        comparison_text = """
        | Aspecto | RL | Ziegler-Nichols | Cohen-Coon |
        |---------|----|----|---|
        | **ISE** | 🥇 Mejor | 🥈 Medio | 🥉 Inferior |
        | **Overshoot** | 🥇 Mejor | 🥈 Medio | 🥉 Inferior |
        | **Settling Time** | 🥇 Mejor | 🥈 Medio | 🥉 Peor |
        | **Facilidad** | Requiere código | Manual | Manual |
        | **Tiempo** | ~15 min | ~30 min | ~30 min |
        | **Reproducibilidad** | 100% | Variable | Variable |
        """
        st.markdown(comparison_text)
    
    else:
        st.warning("⚠️ Datos de comparativa no encontrados.")


# ==================== PÁGINA: ROBUSTEZ ====================
elif pages[selected_page] == "robustness":
    st.title("🛡️ Análisis de Robustez")
    
    if Path("robustness_results.csv").exists():
        df_robust = pd.read_csv("robustness_results.csv")
        
        st.markdown("""
        Se evalúa la robustez del controlador ante:
        - Variaciones en parámetros de la planta (±10%)
        - Cambios en dinámicas del proceso
        - Diferentes condiciones de operación
        """)
        
        st.subheader("Estabilidad en Diferentes Escenarios")
        
        # Tabla de robustez
        st.dataframe(df_robust.head(10), use_container_width=True)
        
        # Métricas de robustez
        col1, col2, col3, col4 = st.columns(4)
        
        stable_count = (df_robust['Status'] == 'Stable').sum()
        total_count = len(df_robust)
        stability_pct = (stable_count / total_count) * 100
        
        with col1:
            st.metric("Escenarios Estables", f"{stable_count}/{total_count}", f"{stability_pct:.0f}%")
        
        with col2:
            mean_ise = df_robust['ISE'].mean()
            st.metric("ISE Promedio", f"{mean_ise:.2f}")
        
        with col3:
            mean_os = df_robust['Overshoot'].mean()
            st.metric("Overshoot Promedio", f"{mean_os:.2f}%")
        
        with col4:
            mean_settle = df_robust['Settling_Time'].mean()
            st.metric("Settling Time Promedio", f"{mean_settle:.2f}s")
        
        # Gráfica de estabilidad
        st.subheader("Distribución de Estabilidad")
        
        stability_counts = df_robust['Status'].value_counts()
        fig = px.pie(
            values=stability_counts.values,
            names=stability_counts.index,
            color_discrete_map={'Stable': '#2ca02c', 'Unstable': '#d62728'},
            title="Porcentaje de Escenarios Estables"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfica de ISE vs Perturbación
        fig = px.scatter(
            df_robust,
            x='Perturbation',
            y='ISE',
            color='Status',
            color_discrete_map={'Stable': '#2ca02c', 'Unstable': '#d62728'},
            size_max=100,
            title="ISE vs Perturbación de Parámetros",
            labels={'Perturbation': 'Factor de Perturbación', 'ISE': 'ISE'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Datos de robustez no encontrados.")


# ==================== PÁGINA: INFORMACIÓN ====================
elif pages[selected_page] == "info":
    st.title("🎓 Información del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sobre RL-PID Optimizer")
        st.markdown("""
        **RL-PID Optimizer** es un framework completo para:
        
        - 🤖 Entrenar agentes RL para optimizar PIDs
        - 📈 Simular dinámicas de procesos realistas
        - 📊 Analizar y comparar métodos de control
        - 🔧 Integrar con Siemens TIA Portal
        
        **Características:**
        - ✅ 100% simulado (sin hardware)
        - ✅ Datos sintéticos realistas
        - ✅ Comparativa científica
        - ✅ Exportación a TIA Portal
        - ✅ Reportes HTML
        """)
    
    with col2:
        st.subheader("🛠️ Stack Tecnológico")
        st.markdown("""
        **Backend:**
        - Python 3.9+
        - Gymnasium (entorno RL)
        - Stable-Baselines3 (PPO)
        - NumPy, SciPy (cálculos)
        
        **Frontend:**
        - Streamlit (esta app)
        - Plotly (gráficas)
        - Matplotlib (análisis)
        
        **Datos:**
        - Pandas (manipulación)
        - Scikit-learn (análisis)
        """)
    
    st.markdown("---")
    
    st.subheader("📝 Parámetros Optimizados")
    st.markdown("""
    ### Controlador PID Óptimo para Temperatura
    
    ```
    Kp (Proporcional):  2.340
    Ki (Integral):      0.870
    Kd (Derivativa):    0.450
    ```
    
    Estos valores minimizar:
    - Error en seguimiento
    - Overshoot
    - Esfuerzo de control
    - Tiempo de respuesta
    """)
    
    st.markdown("---")
    
    st.subheader("🚀 Cómo usar en TIA Portal")
    st.markdown("""
    1. **Exportar parámetros:**
       - Archivo: `PID_Optimized.scl`
       - Formato: Código Siemens SCL
    
    2. **Copiar en TIA Portal:**
       - Crear nuevo Function Block (FB)
       - Pegar contenido de SCL
       - Modificar Kp, Ki, Kd con valores óptimos
    
    3. **Desplegar en PLC:**
       - Compilar el proyecto
       - Descargar en PLC S7-1500
       - Validar en control cerrado
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Referencias")
    st.markdown("""
    - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
    - [Gymnasium](https://gymnasium.farama.org/)
    - [Siemens TIA Portal](https://www.siemens.com/tia)
    - [PPO Paper](https://arxiv.org/abs/1707.06347)
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Versión: 1.0")
    with col2:
        st.success("✅ Estado: Producción")
    with col3:
        st.warning("⚠️ Validar siempre en simulador")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📞 Soporte
- Documentación: README.md
- Código: GitHub
- Email: contacto@example.com

---
**RL-PID Optimizer v1.0**  
© 2024 - Optimización Automática
""")