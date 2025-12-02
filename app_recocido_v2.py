"""
Aplicación Streamlit para Simulación de Recocido en Campana
Versión 2.0 - Con modelo calibrado (405 corridas reales)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from steel_profiles import (SteelProfile, SteelProfileLibrary, Coil, 
                            FurnaceStack, create_quick_coil)
from bell_annealing_v2 import (BellAnnealingSimulatorV2, FurnaceConfig, 
                                AnnealingCycle, CalibratedModel)

# Configuración de página
st.set_page_config(
    page_title="Simulador de Recocido v2.0",
    page_icon="🔥",
    layout="wide"
)

# Inicializar perfiles
SteelProfileLibrary.initialize_defaults()

# Estado de sesión
if 'stack' not in st.session_state:
    st.session_state.stack = FurnaceStack()
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# =============================================================================
# SIDEBAR - Configuración del ciclo
# =============================================================================

st.sidebar.title("⚙️ Configuración del Ciclo")

st.sidebar.subheader("Perfil de Temperatura")

col1, col2 = st.sidebar.columns(2)
with col1:
    T_initial = st.number_input("T inicial (°C)", 20, 100, 50)
with col2:
    heating_time = st.number_input("Calentamiento (h)", 1.0, 20.0, 20.0, 0.5)

T_plateau = st.sidebar.slider("Temperatura Plateau (°C)", 600, 750, 703)
threshold = st.sidebar.slider("Umbral ΔT (°C)", 1.0, 10.0, 3.0, 0.5)

col1, col2 = st.sidebar.columns(2)
with col1:
    cooling_time = st.number_input("Enfriamiento (h)", 1.0, 20.0, 8.0, 0.5)
with col2:
    T_final = st.number_input("T final (°C)", 50, 200, 100)

st.sidebar.subheader("Parámetros del Horno")
psi = st.sidebar.slider("Factor ψ (convección)", 1.0, 5.0, 3.0, 0.1)
gas_flow = st.sidebar.slider("Flujo H₂ (m³/h)", 100, 250, 150)

# Mini gráfica del perfil
fig_mini, ax_mini = plt.subplots(figsize=(4, 2))
t_heat = [0, heating_time]
T_heat = [T_initial, T_plateau]
t_plat = [heating_time, heating_time + 5]  # Estimado
T_plat = [T_plateau, T_plateau]
t_cool = [heating_time + 5, heating_time + 5 + cooling_time]
T_cool = [T_plateau, T_final]

ax_mini.plot(t_heat, T_heat, 'r-', linewidth=2)
ax_mini.plot(t_plat, T_plat, 'orange', linewidth=2, linestyle='--')
ax_mini.plot(t_cool, T_cool, 'b-', linewidth=2)
ax_mini.set_xlabel("Tiempo (h)", fontsize=8)
ax_mini.set_ylabel("T (°C)", fontsize=8)
ax_mini.tick_params(labelsize=7)
ax_mini.grid(True, alpha=0.3)
st.sidebar.pyplot(fig_mini)
plt.close(fig_mini)

# =============================================================================
# CONTENIDO PRINCIPAL
# =============================================================================

st.title("🔥 Simulador de Recocido en Campana v2.0")
st.caption("Modelo calibrado con 405 corridas reales | Interacción térmica entre bobinas")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Perfiles de Acero", "🔩 Configurar Bobinas", 
                                   "▶️ Simular", "❓ Ayuda"])

# =============================================================================
# TAB 1: Perfiles de Acero
# =============================================================================

with tab1:
    st.header("Perfiles de Acero Disponibles")
    
    profiles = SteelProfileLibrary.list_profiles()
    
    cols = st.columns(3)
    for i, name in enumerate(profiles):
        with cols[i % 3]:
            profile = SteelProfileLibrary.get_profile(name)
            with st.expander(f"**{name}**"):
                st.write(f"**Descripción:** {profile.description}")
                st.write(f"- Densidad: {profile.density} kg/m³")
                st.write(f"- Emisividad: {profile.emissivity}")
                st.write(f"- Dureza: {profile.hardness/1e6:.1f} MPa")
    
    st.divider()
    
    # Crear perfil personalizado
    st.subheader("➕ Crear Perfil Personalizado")
    
    with st.form("nuevo_perfil"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Nombre del perfil")
            new_desc = st.text_input("Descripción")
            new_density = st.number_input("Densidad (kg/m³)", 7000, 8500, 7850)
        with col2:
            new_emissivity = st.number_input("Emisividad", 0.05, 0.5, 0.15)
            new_conductivity = st.number_input("Conductividad base (W/m·K)", 30.0, 60.0, 50.0)
        
        if st.form_submit_button("Crear Perfil"):
            if new_name:
                new_profile = SteelProfile(
                    name=new_name,
                    description=new_desc,
                    density=new_density,
                    emissivity=new_emissivity,
                    thermal_conductivity_coeffs=(new_conductivity, -0.01, 0.0)
                )
                SteelProfileLibrary.add_profile(new_profile)
                st.success(f"Perfil '{new_name}' creado exitosamente")
                st.rerun()

# =============================================================================
# TAB 2: Configurar Bobinas
# =============================================================================

with tab2:
    st.header("Configuración del Stack de Bobinas")
    
    col_form, col_stack = st.columns([1, 1])
    
    with col_form:
        st.subheader("➕ Agregar Bobina")
        
        with st.form("nueva_bobina"):
            coil_id = st.text_input("ID de la bobina", f"BOB-{st.session_state.stack.num_coils + 1:03d}")
            profile_name = st.selectbox("Perfil de acero", SteelProfileLibrary.list_profiles())
            
            col1, col2 = st.columns(2)
            with col1:
                od = st.number_input("Diámetro exterior (mm)", 1200, 2000, 1750, 10)
                id_val = st.number_input("Diámetro interior (mm)", 400, 800, 600, 10)
            with col2:
                width = st.number_input("Ancho (mm)", 600, 1500, 1100, 10)
                thickness = st.number_input("Espesor lámina (mm)", 0.3, 10.0, 1.0, 0.1)
            
            if st.form_submit_button("Agregar al Stack", use_container_width=True):
                new_coil = create_quick_coil(
                    coil_id, profile_name, od, id_val, width, thickness
                )
                st.session_state.stack.add_coil(new_coil)
                st.success(f"Bobina {coil_id} agregada")
                st.rerun()
    
    with col_stack:
        st.subheader("📦 Stack Actual")
        
        if st.session_state.stack.num_coils == 0:
            st.info("No hay bobinas en el stack. Agrega bobinas usando el formulario.")
        else:
            # Mostrar bobinas (de arriba a abajo visualmente)
            for i in range(st.session_state.stack.num_coils - 1, -1, -1):
                coil = st.session_state.stack.coils[i]
                pos = i + 1
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{pos}# - {coil.coil_id}**")
                        st.caption(f"OD:{coil.outer_diameter*1000:.0f}mm | "
                                  f"W:{coil.width*1000:.0f}mm | "
                                  f"t:{coil.thickness*1000:.2f}mm | "
                                  f"{coil.mass:.0f}kg")
                    with col2:
                        if st.button("⬇️", key=f"down_{i}", disabled=(i == 0)):
                            # Mover hacia abajo
                            st.session_state.stack.coils[i], st.session_state.stack.coils[i-1] = \
                                st.session_state.stack.coils[i-1], st.session_state.stack.coils[i]
                            st.rerun()
                    with col3:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.stack.coils.pop(i)
                            st.rerun()
                
                if i > 0:
                    st.divider()
            
            # Resumen
            st.metric("Total bobinas", st.session_state.stack.num_coils)
            total_mass = sum(c.mass for c in st.session_state.stack.coils)
            st.metric("Masa total", f"{total_mass:,.0f} kg")
            
            # Predicción rápida con modelo empírico
            if st.session_state.stack.num_coils >= 3:
                avg_width = np.mean([c.width * 1000 for c in st.session_state.stack.coils])
                avg_thick = np.mean([c.thickness * 1000 for c in st.session_state.stack.coils])
                n_bob = st.session_state.stack.num_coils
                
                t_pred = CalibratedModel.predict_plateau_time(avg_width, avg_thick, T_plateau, n_bob)
                
                st.info(f"⏱️ **Predicción rápida:** ~{t_pred:.1f}h de plateau")

# =============================================================================
# TAB 3: Simular
# =============================================================================

with tab3:
    st.header("Simulación")
    
    if st.session_state.stack.num_coils < 2:
        st.warning("Agrega al menos 2 bobinas al stack para simular")
    else:
        # Resumen de configuración
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📦 Stack")
            st.write(f"**{st.session_state.stack.num_coils} bobinas**")
            total_mass = sum(c.mass for c in st.session_state.stack.coils)
            st.write(f"Masa total: {total_mass:,.0f} kg")
        
        with col2:
            st.subheader("🌡️ Ciclo")
            st.write(f"T plateau: **{T_plateau}°C**")
            st.write(f"Calentamiento: {heating_time}h")
            st.write(f"Enfriamiento: {cooling_time}h")
        
        with col3:
            st.subheader("⚙️ Horno")
            st.write(f"Factor ψ: {psi}")
            st.write(f"Flujo H₂: {gas_flow} m³/h")
            st.write(f"Umbral ΔT: {threshold}°C")
        
        st.divider()
        
        # Predicción empírica
        avg_width = np.mean([c.width * 1000 for c in st.session_state.stack.coils])
        avg_thick = np.mean([c.thickness * 1000 for c in st.session_state.stack.coils])
        n_bob = st.session_state.stack.num_coils
        
        t_empirico = CalibratedModel.predict_plateau_time(avg_width, avg_thick, T_plateau, n_bob)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Predicción Empírica", f"{t_empirico:.1f} h", 
                     help="Basado en regresión con 405 corridas reales")
        
        # Botón de simulación
        if st.button("🚀 INICIAR SIMULACIÓN FÍSICA", type="primary", use_container_width=True):
            
            with st.spinner("Simulando... (esto puede tomar 1-2 minutos)"):
                # Configurar
                config = FurnaceConfig(
                    total_gas_flow=float(gas_flow),
                    convection_enhancement=float(psi),
                    inter_coil_conductance=50.0,
                    position_factor=0.15
                )
                
                cycle = AnnealingCycle(
                    T_plateau=float(T_plateau),
                    threshold=float(threshold),
                    T_initial=float(T_initial),
                    T_final=float(T_final),
                    heating_time=float(heating_time),
                    cooling_time=float(cooling_time)
                )
                
                # Simular
                simulator = BellAnnealingSimulatorV2(
                    st.session_state.stack, config, cycle
                )
                simulator.nr = 12
                simulator.nz = 8
                simulator.dt = 20.0
                
                results = simulator.simulate(max_time_h=40.0, save_interval=120)
                
                st.session_state.simulation_results = {
                    'results': results,
                    'cycle': cycle,
                    'annealing_time': cycle.annealing_time,
                    'plateau_duration': cycle.annealing_time - heating_time if cycle.annealing_time else None
                }
            
            st.success("Simulación completada")
            st.rerun()
        
        # Mostrar resultados
        if st.session_state.simulation_results:
            res = st.session_state.simulation_results
            results = res['results']
            
            st.divider()
            st.subheader("📈 Resultados")
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if res['annealing_time']:
                    st.metric("🔥 Tiempo de Recocido", f"{res['annealing_time']:.2f} h",
                             help="Tiempo cuando termina el plateau")
                else:
                    st.metric("🔥 Tiempo de Recocido", "No convergió")
            
            with col2:
                if res['plateau_duration']:
                    st.metric("⏸️ Duración Plateau", f"{res['plateau_duration']:.2f} h")
                    delta = res['plateau_duration'] - t_empirico
                    st.caption(f"vs empírico: {delta:+.2f}h")
            
            with col3:
                total_time = heating_time + (res['plateau_duration'] or 0) + cooling_time
                st.metric("🕐 Tiempo Total Estimado", f"{total_time:.1f} h")
            
            # Gráfica de temperaturas
            st.subheader("Evolución de Temperaturas")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            times = results['time']
            
            # Gas
            ax.plot(times, results['T_gas'], 'k--', linewidth=2, label='T_gas', alpha=0.7)
            
            # Bobinas
            colors = plt.cm.tab10(np.linspace(0, 1, st.session_state.stack.num_coils))
            for i in range(st.session_state.stack.num_coils):
                coil_id = st.session_state.stack.coils[i].coil_id
                ax.plot(times, results['T_hot'][i], color=colors[i], linewidth=1.5,
                       label=f'{coil_id} (hot)', linestyle='-')
                ax.plot(times, results['T_cold'][i], color=colors[i], linewidth=1.5,
                       label=f'{coil_id} (cold)', linestyle=':')
            
            # Línea de tiempo de recocido
            if res['annealing_time']:
                ax.axvline(res['annealing_time'], color='green', linewidth=2, 
                          linestyle='--', label='Fin plateau')
            
            ax.set_xlabel('Tiempo (h)', fontsize=12)
            ax.set_ylabel('Temperatura (°C)', fontsize=12)
            ax.set_title('Evolución de Temperaturas en el Stack', fontsize=14)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Tabla resumen por bobina
            st.subheader("Resumen por Bobina")
            
            data = []
            for i in range(st.session_state.stack.num_coils):
                coil = st.session_state.stack.coils[i]
                T_cold_max = max(results['T_cold'][i])
                T_hot_max = max(results['T_hot'][i])
                data.append({
                    'Posición': f"{i+1}#",
                    'ID': coil.coil_id,
                    'Ancho (mm)': f"{coil.width*1000:.0f}",
                    'Espesor (mm)': f"{coil.thickness*1000:.2f}",
                    'Masa (kg)': f"{coil.mass:.0f}",
                    'T_cold_max (°C)': f"{T_cold_max:.1f}",
                    'T_hot_max (°C)': f"{T_hot_max:.1f}",
                    'ΔT_max (°C)': f"{T_hot_max - T_cold_max:.1f}"
                })
            
            st.dataframe(data, use_container_width=True)

# =============================================================================
# TAB 4: Ayuda
# =============================================================================

with tab4:
    st.header("Guía de Uso")
    
    st.markdown("""
    ### 🔥 Simulador de Recocido en Campana v2.0
    
    Esta aplicación simula el proceso de recocido de bobinas de acero en hornos de campana
    con atmósfera de hidrógeno.
    
    #### Mejoras de la versión 2.0:
    - ✅ **Interacción térmica entre bobinas** (ya no son independientes)
    - ✅ **Efecto de posición** en el stack (bobinas del medio tardan más)
    - ✅ **Modelo empírico calibrado** con 405 corridas reales
    - ✅ **Predicción rápida** sin necesidad de simular
    
    ---
    
    ### Flujo de trabajo:
    
    1. **Configura el ciclo** en la barra lateral (temperatura, tiempos)
    2. **Agrega bobinas** en la pestaña "Configurar Bobinas"
    3. **Ejecuta la simulación** en la pestaña "Simular"
    4. **Analiza los resultados** (gráficas, tiempos, temperaturas)
    
    ---
    
    ### Conceptos clave:
    
    - **Hot spot**: Punto más caliente de la bobina (superficie exterior)
    - **Cold spot**: Punto más frío (centro radial y axial)
    - **Plateau**: Fase donde se mantiene la temperatura constante hasta homogeneizar
    - **Tiempo de recocido**: Momento en que el cold spot alcanza T_plateau - ΔT
    
    ---
    
    ### Precisión del modelo:
    
    | Modelo | Error típico | Uso recomendado |
    |--------|--------------|-----------------|
    | Empírico | ±0.7h | Predicción rápida |
    | Físico | ±1.0h | Comparación de configuraciones |
    
    ---
    
    ### Perfiles de acero predefinidos:
    
    - **SPCC**: Laminado en frío comercial (JIS G3141)
    - **DC01**: Para embutición (EN 10130)
    - **DC04**: Embutición profunda
    - **Q235**: Estructural (GB/T 700)
    - **AISI_1008**: Bajo carbono
    - **IF_Steel**: Libre de intersticiales
    """)

# Footer
st.sidebar.divider()
st.sidebar.caption("v2.0 | Calibrado con 405 corridas")
st.sidebar.caption("© 2025 - Modelo de Recocido")
