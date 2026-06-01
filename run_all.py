"""
Script maestro: ejecuta todo el pipeline de RL-PID Optimizer
1. Generar datos sintéticos
2. Entrenar agente RL
3. Análisis comparativo
4. Integración PLCSIM
"""

import sys
import time
from synthetic_data_generator import generate_all_scenarios
from train_optimizer import train_optimizer, evaluate_trained_model, plot_training_results, compare_with_baseline
from analysis_and_reports import run_complete_analysis
from plcsim_integration import test_plcsim_bridge


def print_banner(text: str):
    """Imprime un banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def run_pipeline():
    """Ejecuta el pipeline completo"""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  RL-PID OPTIMIZER - PIPELINE COMPLETO".center(78) + "║")
    print("║" + "  Optimización automática de controladores PID".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    inicio = time.time()
    
    # PASO 1: Generar datos sintéticos
    print_banner("PASO 1: GENERAR DATOS SINTÉTICOS REALISTAS")
    try:
        print("Generando históricos de control para múltiples escenarios...\n")
        results_synthetic = generate_all_scenarios()
        print("\nGeneración de datos completada")
    except Exception as e:
        print(f"\nError en generación de datos: {e}")
        sys.exit(1)
    
    tiempo_1 = time.time() - inicio
    
    # PASO 2: Entrenar agente RL
    print_banner("PASO 2: ENTRENAR AGENTE RL (PPO)")
    try:
        print("Entrenando agente con 100,000 pasos de simulación...\n")
        model, env = train_optimizer(
            process_type='temperature',
            total_timesteps=100000,
            learning_rate=3e-4
        )
        
        print("\nEntrenamiento completado")
    except Exception as e:
        print(f"\nError en entrenamiento: {e}")
        sys.exit(1)
    
    tiempo_2 = time.time() - inicio - tiempo_1
    
    # PASO 3: Evaluar modelo
    print_banner("PASO 3: EVALUAR MODELO ENTRENADO")
    try:
        print("Evaluando política aprendida en 5 episodios...\n")
        eval_results, optimal_params = evaluate_trained_model(model, env, num_episodes=5)
        
        print("\nEvaluación completada")
    except Exception as e:
        print(f"\nError en evaluación: {e}")
        sys.exit(1)
    
    # PASO 4: Comparar con baseline
    print_banner("PASO 4: COMPARAR CON TUNING MANUAL")
    try:
        compare_with_baseline('temperature', optimal_params, eval_results)
        print("\nComparación completada")
    except Exception as e:
        print(f"\nError en comparación: {e}")
    
    # PASO 5: Visualizar resultados
    print_banner("PASO 5: VISUALIZAR RESULTADOS")
    try:
        print("Generando gráficas de entrenamiento...\n")
        plot_training_results(eval_results, optimal_params)
        print("\nVisualización completada")
    except Exception as e:
        print(f"\nError en visualización: {e}")
    
    tiempo_3 = time.time() - inicio - tiempo_1 - tiempo_2
    
    # PASO 6: Análisis comparativo
    print_banner("PASO 6: ANÁLISIS COMPARATIVO AVANZADO")
    try:
        print("Comparando con métodos clásicos y analizando robustez...\n")
        run_complete_analysis()
        print("\nAnálisis completado")
    except Exception as e:
        print(f"\nError en análisis: {e}")
    
    tiempo_4 = time.time() - inicio - tiempo_1 - tiempo_2 - tiempo_3
    
    # PASO 7: Integración PLCSIM
    print_banner("PASO 7: INTEGRACIÓN CON PLCSIM")
    try:
        print("Probando integración con PLCSIM de Siemens...\n")
        test_plcsim_bridge()
        print("\nIntegración completada")
    except Exception as e:
        print(f"\nError en integración: {e}")
    
    tiempo_5 = time.time() - inicio - tiempo_1 - tiempo_2 - tiempo_3 - tiempo_4
    
    # RESUMEN FINAL
    print_banner("PIPELINE COMPLETADO EXITOSAMENTE")
    
    print("\n📊 RESUMEN DE TIEMPOS:")
    print(f"  1. Generación de datos:      {tiempo_1:>6.1f}s")
    print(f"  2. Entrenamiento RL:          {tiempo_2:>6.1f}s")
    print(f"  3. Evaluación:                {tiempo_3:>6.1f}s")
    print(f"  4. Análisis comparativo:      {tiempo_4:>6.1f}s")
    print(f"  5. Integración PLCSIM:        {tiempo_5:>6.1f}s")
    print(f"  ─────────────────────────────────────")
    print(f"  TIEMPO TOTAL:                {time.time() - inicio:>6.1f}s")
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("  Datos:")
    print("    ├── data_temperature_pid_actual.csv")
    print("    ├── data_temperature_pid_optimized.csv")
    print("    ├── data_motor_speed_pid_actual.csv")
    print("    └── data_motor_speed_pid_optimized.csv")
    print("  Modelos:")
    print("    └── model_pid_optimizer_temperature.zip")
    print("  Visualizaciones:")
    print("    ├── comparison_synthetic_data.png")
    print("    ├── training_results.png")
    print("    ├── analysis_comparison.png")
    print("    └── analysis_sensitivity.png")
    print("  Reportes:")
    print("    ├── report.html  ← ABRE ESTO EN NAVEGADOR")
    print("    ├── comparison_results.csv")
    print("    └── robustness_results.csv")
    print("  Integración Siemens:")
    print("    ├── plcsim_simulation.csv")
    print("    ├── PID_Optimized.scl  ← COPIA A TIA PORTAL")
    print("    └── optimized_parameters.xml  ← IMPORTA EN TIA PORTAL")
    
    print("\nPRÓXIMOS PASOS:")
    print("  1. Abre report.html en tu navegador para ver análisis visual")
    print("  2. Copia PID_Optimized.scl a tu proyecto TIA Portal")
    print("  3. Importa optimized_parameters.xml con los valores:")
    print(f"     • Kp = {optimal_params[0]:.6f}")
    print(f"     • Ki = {optimal_params[1]:.6f}")
    print(f"     • Kd = {optimal_params[2]:.6f}")
    print("  4. Carga el programa en tu PLC")
    print("  5. Prueba en control cerrado y valida resultados")
    
    print("\nINFORMACIÓN ADICIONAL:")
    print("  • Documento técnico: README.md")
    print("  • Código fuente: *.py")
    print("  • Datos de validación: CSV")
    
    print("\n" + "=" * 80)
    print("¡Pipeline completado! 🎉")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
