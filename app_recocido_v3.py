"""
Aplicación Streamlit para Simulación de Recocido en Campana
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from steel_profiles import (SteelProfile, SteelProfileLibrary, Coil, 
                            FurnaceStack, create_quick_coil)
from bell_annealing_v2 import (BellAnnealingSimulatorV2, FurnaceConfig, 
                                AnnealingCycle)

# Configuración de página
st.set_page_config(
    page_title="Simulador de Recocido",
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
    heating_time = st.number_input("Calentamiento (h)", 1.0, 30.0, 20.0, 0.5)

T_plateau = st.sidebar.slider("Temperatura Plateau (°C)", 600, 750, 680)
threshold = st.sidebar.slider("Umbral ΔT (°C)", 1.0, 10.0, 3.0, 0.5)

col1, col2 = st.sidebar.columns(2)
with col1:
    cooling_time = st.number_input("Enfriamiento (h)", 1.0, 20.0, 8.0, 0.5)
with col2:
    T_final = st.number_input("T final (°C)", 50, 200, 100)

st.sidebar.subheader("Parámetros del Horno")
psi = st.sidebar.slider("Factor ψ (convección)", 1.0, 10.0, 3.6, 0.1)
gas_flow = st.sidebar.slider("Flujo H₂ (m³/h)", 100, 250, 162)

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

st.title("🔥 Simulador de Recocido en Campana")

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
                st.write(f"- Conductividad: {profile.thermal_conductivity_coeffs[0]:.1f} W/m·K")
                st.write(f"- Calor específico: {profile.specific_heat_coeffs[0]:.0f} J/kg·K")
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
            new_cp = st.number_input("Calor específico (J/kg·K)", 400, 700, 500)
        
        if st.form_submit_button("Crear Perfil"):
            if new_name:
                new_profile = SteelProfile(
                    name=new_name,
                    description=new_desc,
                    density=new_density,
                    emissivity=new_emissivity,
                    thermal_conductivity_coeffs=(new_conductivity, -0.01, 0.0),
                    specific_heat_coeffs=(float(new_cp), 0.0, 0.0)  # Valor constante
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
                thickness = st.number_input("Espesor lámina (mm)", 0.30, 4.00, 1.30, 0.05)
            
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
        
        # Botón de simulación
        if st.button("🚀 INICIAR SIMULACIÓN", type="primary", use_container_width=True):
            
            with st.spinner("Simulando... (esto puede tomar 1-2 minutos)"):
                # Configurar
                config = FurnaceConfig(
                    total_gas_flow=float(gas_flow),
                    convection_enhancement=float(psi),
                    inter_coil_conductance=100.0,
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
            
            # =================================================================
            # MAPAS DE CALOR DE SECCIÓN TRANSVERSAL
            # =================================================================
            if 'T_fields' in results and len(results['T_fields']) > 0:
                st.subheader("🔥 Mapas de Calor - Sección Transversal")
                st.caption("Distribución de temperatura en la sección radial-axial de la bobina")
                
                # Selector de bobina para visualizar
                bobina_idx = st.selectbox(
                    "Seleccionar bobina para visualizar",
                    range(st.session_state.stack.num_coils),
                    format_func=lambda x: f"Bobina {x+1}# - {st.session_state.stack.coils[x].coil_id}"
                )
                
                # Obtener dimensiones de la bobina
                coil = st.session_state.stack.coils[bobina_idx]
                r_in = coil.inner_diameter * 1000 / 2  # mm
                r_out = coil.outer_diameter * 1000 / 2  # mm
                width = coil.width * 1000  # mm
                
                # Crear figura con 5 subplots
                n_snapshots = len(results['snapshot_times'])
                fig_hm, axes_hm = plt.subplots(1, n_snapshots, figsize=(3.5*n_snapshots, 4))
                
                if n_snapshots == 1:
                    axes_hm = [axes_hm]
                
                # Etiquetas para cada snapshot
                labels = ['Inicio', 'Mitad\ncalent.', 'Fin\ncalent.', 'Mitad\nplateau', 'Fin\nplateau']
                
                # Lista para guardar los colorbars individuales
                images = []
                
                for i, (t, ax_hm) in enumerate(zip(results['snapshot_times'], axes_hm)):
                    T_field = results['T_fields'][i][bobina_idx]
                    
                    # Crear meshgrid para la visualización
                    nr, nz = T_field.shape
                    r = np.linspace(r_in, r_out, nr)
                    z = np.linspace(0, width, nz)
                    R, Z = np.meshgrid(r, z, indexing='ij')
                    
                    # Escala individual para cada snapshot
                    T_min_local = T_field.min()
                    T_max_local = T_field.max()
                    
                    # Si ΔT es muy pequeño, expandir el rango para mejor visualización
                    if T_max_local - T_min_local < 5:
                        T_mid = (T_max_local + T_min_local) / 2
                        T_min_local = T_mid - 5
                        T_max_local = T_mid + 5
                    
                    # Graficar heatmap con escala individual
                    im = ax_hm.pcolormesh(Z, R, T_field, shading='auto', 
                                          cmap='hot', vmin=T_min_local, vmax=T_max_local)
                    images.append(im)
                    
                    # Colorbar individual para cada subplot
                    cbar = fig_hm.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02)
                    cbar.ax.tick_params(labelsize=7)
                    cbar.set_label('°C', fontsize=8)
                    
                    # Etiquetas
                    label = labels[i] if i < len(labels) else f"t={t:.1f}h"
                    ax_hm.set_title(f"{label}\nt = {t:.1f} h", fontsize=10)
                    ax_hm.set_xlabel("Axial z (mm)", fontsize=9)
                    if i == 0:
                        ax_hm.set_ylabel("Radio r (mm)", fontsize=9)
                    ax_hm.tick_params(labelsize=8)
                    
                    # Posiciones FIJAS de hot spot y cold spot (no buscar mínimo/máximo)
                    # Cold spot: centro geométrico de la sección
                    cold_r_idx = nr // 2
                    cold_z_idx = nz // 2
                    cold_r = r[cold_r_idx]
                    cold_z = z[cold_z_idx]
                    T_cold = T_field[cold_r_idx, cold_z_idx]
                    
                    # Hot spot: superficie exterior, centro axial
                    hot_r_idx = nr - 1  # Superficie exterior
                    hot_z_idx = nz // 2  # Centro axial
                    hot_r = r[hot_r_idx]
                    hot_z = z[hot_z_idx]
                    T_hot = T_field[hot_r_idx, hot_z_idx]
                    
                    # Marcar HOT SPOT (superficie exterior)
                    ax_hm.plot(hot_z, hot_r, 'o', markersize=12, markerfacecolor='none', 
                              markeredgecolor='lime', markeredgewidth=2.5)
                    ax_hm.annotate(f'HOT\n{T_hot:.0f}°C', xy=(hot_z, hot_r), 
                                  xytext=(hot_z + width*0.15, hot_r - (r_out-r_in)*0.1),
                                  fontsize=7, fontweight='bold', color='white',
                                  ha='left', va='top',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='darkred', alpha=0.9),
                                  arrowprops=dict(arrowstyle='->', color='lime', lw=1.5))
                    
                    # Marcar COLD SPOT (centro geométrico)
                    ax_hm.plot(cold_z, cold_r, 's', markersize=10, markerfacecolor='none',
                              markeredgecolor='cyan', markeredgewidth=2.5)
                    ax_hm.annotate(f'COLD\n{T_cold:.0f}°C', xy=(cold_z, cold_r),
                                  xytext=(cold_z - width*0.15, cold_r + (r_out-r_in)*0.1),
                                  fontsize=7, fontweight='bold', color='white',
                                  ha='right', va='bottom',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='darkblue', alpha=0.9),
                                  arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))
                    
                    # Mostrar ΔT en esquina
                    dT = T_hot - T_cold
                    ax_hm.text(0.05, 0.95, f"ΔT={dT:.0f}°C", 
                              transform=ax_hm.transAxes, fontsize=8, va='top',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
                
                fig_hm.suptitle(f"Evolución del Campo de Temperatura - {coil.coil_id}", 
                               fontsize=12, fontweight='bold', y=1.02)
                
                plt.tight_layout()
                st.pyplot(fig_hm)
                plt.close(fig_hm)
                
                # Explicación
                with st.expander("ℹ️ Interpretación de los mapas de calor"):
                    st.markdown("""
                    **Ejes:**
                    - **Eje horizontal (z)**: Dirección axial (ancho de la bobina)
                    - **Eje vertical (r)**: Dirección radial (desde el centro hacia afuera)
                    
                    **Marcadores:**
                    - 🔺 **Triángulo rojo (Hot spot)**: Punto más caliente (superficie exterior)
                    - ⭐ **Estrella azul (Cold spot)**: Punto más frío (centro de la bobina)
                    
                    **ΔT**: Diferencia entre temperatura máxima y mínima en cada instante.
                    
                    **Comportamiento típico:**
                    1. **Inicio**: Temperatura uniforme
                    2. **Calentamiento**: Gradiente radial significativo (exterior caliente, interior frío)
                    3. **Plateau**: Gradiente se reduce progresivamente
                    4. **Fin plateau**: Temperatura casi uniforme (ΔT ≈ umbral)
                    """)
            
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
    ### 🔥 Simulador de Recocido en Campana
    
    Esta aplicación simula el proceso de recocido de bobinas de acero en hornos de campana
    con atmósfera de hidrógeno.
    
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
st.sidebar.caption("© 2025 - Modelo de Recocido")
