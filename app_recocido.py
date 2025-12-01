"""
App de Simulación de Horno de Recocido
======================================

Ejecutar con: streamlit run app_recocido.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from steel_profiles import (
    SteelProfile, SteelProfileLibrary, Coil, FurnaceStack, create_quick_coil
)
from bell_annealing_modular import (
    BellAnnealingSimulator, FurnaceConfig, AnnealingCycle
)

# Configuración de página
st.set_page_config(
    page_title="Simulador de Recocido",
    page_icon="🔥",
    layout="wide"
)

# Inicializar perfiles
SteelProfileLibrary.initialize_defaults()

# Título
st.title("🔥 Simulador de Horno de Recocido Tipo Campana")
st.markdown("---")

# =============================================================================
# SIDEBAR - Configuración del ciclo y horno
# =============================================================================

st.sidebar.header("⚙️ Configuración del Ciclo")

T_plateau = st.sidebar.slider(
    "Temperatura de Plateau (°C)", 
    min_value=600, max_value=800, value=700, step=10
)

threshold = st.sidebar.slider(
    "Umbral ΔT para terminar plateau (°C)", 
    min_value=1.0, max_value=10.0, value=3.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.header("🌀 Configuración del Horno")

gas_flow = st.sidebar.slider(
    "Flujo de gas H₂ (m³/h)", 
    min_value=100, max_value=250, value=150, step=10
)

psi = st.sidebar.slider(
    "Factor ψ (convección)", 
    min_value=1.0, max_value=3.0, value=2.0, step=0.1
)

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📋 Perfiles de Acero", "🎯 Configurar Bobinas", "📊 Simular", "❓ Ayuda"])

# =============================================================================
# TAB 1: Perfiles de Acero
# =============================================================================

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Perfiles Disponibles")
        
        perfiles = SteelProfileLibrary.list_profiles()
        
        for nombre in perfiles:
            p = SteelProfileLibrary.get_profile(nombre)
            with st.expander(f"**{nombre}** - {p.description[:40]}..."):
                st.write(f"**Densidad:** {p.density} kg/m³")
                st.write(f"**Emisividad:** {p.emissivity}")
                st.write(f"**Conductividad @ 20°C:** {p.get_thermal_conductivity(293.15):.1f} W/(m·K)")
                st.write(f"**Dureza:** {p.hardness/1e6:.0f} MPa")
    
    with col2:
        st.subheader("➕ Crear Nuevo Perfil")
        
        with st.form("nuevo_perfil"):
            nombre_nuevo = st.text_input("Nombre del perfil", value="MI_ACERO")
            descripcion = st.text_input("Descripción", value="Acero personalizado")
            
            col_a, col_b = st.columns(2)
            with col_a:
                densidad = st.number_input("Densidad (kg/m³)", value=7850.0, step=10.0)
                emisividad = st.number_input("Emisividad", value=0.15, step=0.01, min_value=0.05, max_value=0.95)
            with col_b:
                conductividad = st.number_input("Conductividad @ 20°C (W/m·K)", value=50.0, step=1.0)
                dureza = st.number_input("Dureza (MPa)", value=1100.0, step=50.0)
            
            submitted = st.form_submit_button("Crear Perfil")
            
            if submitted:
                nuevo = SteelProfile(
                    name=nombre_nuevo,
                    description=descripcion,
                    density=densidad,
                    emissivity=emisividad,
                    thermal_conductivity_coeffs=(conductividad, -0.015, 0.0),
                    hardness=dureza * 1e6
                )
                SteelProfileLibrary.add_profile(nuevo)
                st.success(f"✅ Perfil '{nombre_nuevo}' creado exitosamente!")
                st.rerun()

# =============================================================================
# TAB 2: Configurar Bobinas
# =============================================================================

with tab2:
    st.subheader("Configurar Stack de Bobinas")
    
    # Inicializar estado
    if 'bobinas' not in st.session_state:
        st.session_state.bobinas = []
    
    # Formulario para agregar bobina
    st.markdown("### ➕ Agregar Bobina")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        coil_id = st.text_input("ID de Bobina", value=f"BOB-{len(st.session_state.bobinas)+1:03d}")
        perfil_sel = st.selectbox("Perfil de Acero", SteelProfileLibrary.list_profiles())
    
    with col2:
        od = st.number_input("Diámetro Exterior (mm)", value=1850, step=10, min_value=800, max_value=2500)
        id_val = st.number_input("Diámetro Interior (mm)", value=600, step=10, min_value=400, max_value=800)
    
    with col3:
        ancho = st.number_input("Ancho (mm)", value=1250, step=10, min_value=800, max_value=2000)
        espesor = st.number_input("Espesor de lámina (mm)", value=1.50, step=0.05, min_value=0.3, max_value=5.0)
    
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("➕ Agregar al Stack", type="primary"):
            nueva_bobina = {
                'coil_id': coil_id,
                'profile': perfil_sel,
                'od': od,
                'id': id_val,
                'width': ancho,
                'thickness': espesor
            }
            st.session_state.bobinas.append(nueva_bobina)
            st.success(f"Bobina {coil_id} agregada!")
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Limpiar Todo"):
            st.session_state.bobinas = []
            st.rerun()
    
    st.markdown("---")
    
    # Mostrar bobinas configuradas
    st.markdown("### 📦 Stack Actual (de abajo hacia arriba)")
    
    if len(st.session_state.bobinas) == 0:
        st.info("No hay bobinas configuradas. Agrega al menos una bobina para simular.")
    else:
        for i, bob in enumerate(st.session_state.bobinas):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Calcular masa aproximada
                r_out = bob['od'] / 2000
                r_in = bob['id'] / 2000
                vol = np.pi * (r_out**2 - r_in**2) * (bob['width']/1000)
                masa = vol * 7850
                
                st.markdown(f"""
                **{i+1}#** - `{bob['coil_id']}` ({bob['profile']})  
                OD: {bob['od']}mm | Ancho: {bob['width']}mm | Espesor: {bob['thickness']}mm | ~{masa:.0f} kg
                """)
            
            with col2:
                if i > 0:  # No se puede mover el primero hacia abajo
                    if st.button("⬇️", key=f"down_{i}"):
                        st.session_state.bobinas[i], st.session_state.bobinas[i-1] = \
                            st.session_state.bobinas[i-1], st.session_state.bobinas[i]
                        st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.bobinas.pop(i)
                    st.rerun()

# =============================================================================
# TAB 3: Simular
# =============================================================================

with tab3:
    st.subheader("🚀 Ejecutar Simulación")
    
    if len(st.session_state.bobinas) == 0:
        st.warning("⚠️ Primero configura las bobinas en la pestaña 'Configurar Bobinas'")
    else:
        # Resumen de configuración
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuración del Stack:**")
            st.write(f"- Bobinas: {len(st.session_state.bobinas)}")
            masa_total = sum(
                np.pi * ((b['od']/2000)**2 - (b['id']/2000)**2) * (b['width']/1000) * 7850
                for b in st.session_state.bobinas
            )
            st.write(f"- Masa total: ~{masa_total:.0f} kg")
        
        with col2:
            st.markdown("**Configuración del Ciclo:**")
            st.write(f"- Plateau: {T_plateau}°C")
            st.write(f"- Umbral: {threshold}°C")
            st.write(f"- Flujo gas: {gas_flow} m³/h")
        
        st.markdown("---")
        
        if st.button("▶️ INICIAR SIMULACIÓN", type="primary", use_container_width=True):
            
            with st.spinner("Simulando... esto puede tomar unos segundos"):
                
                # Crear stack
                stack = FurnaceStack()
                for bob in st.session_state.bobinas:
                    coil = create_quick_coil(
                        coil_id=bob['coil_id'],
                        profile=bob['profile'],
                        outer_diameter_mm=bob['od'],
                        inner_diameter_mm=bob['id'],
                        width_mm=bob['width'],
                        thickness_mm=bob['thickness']
                    )
                    stack.add_coil(coil)
                
                # Configuración
                config = FurnaceConfig(
                    total_gas_flow=gas_flow,
                    convection_enhancement=psi
                )
                
                cycle = AnnealingCycle(T_plateau=T_plateau, threshold=threshold)
                
                # Simular
                simulator = BellAnnealingSimulator(stack, config, cycle)
                results = simulator.simulate(max_time_h=50.0)
            
            st.success("✅ Simulación completada!")
            
            # Mostrar resultados
            st.markdown("### 📊 Resultados")
            
            # Métricas principales
            cols = st.columns(len(st.session_state.bobinas))
            for i, col in enumerate(cols):
                with col:
                    T_cold_max = max(results['coils'][i]['T_cold'])
                    st.metric(
                        label=f"{i+1}# {results['coils'][i]['coil_id']}",
                        value=f"{T_cold_max:.1f}°C",
                        delta=f"{T_cold_max - T_plateau:.1f}°C vs objetivo"
                    )
            
            st.write(f"**Tiempo total del ciclo:** {results['time'][-1]:.1f} horas")
            
            # Gráfica
            st.markdown("### 📈 Curvas de Temperatura")
            
            fig, axes = plt.subplots(1, len(st.session_state.bobinas), 
                                    figsize=(5*len(st.session_state.bobinas), 4))
            
            if len(st.session_state.bobinas) == 1:
                axes = [axes]
            
            time = np.array(results['time'])
            T_gas = np.array(results['T_gas'])
            
            for idx, ax in enumerate(axes):
                T_cold = np.array(results['coils'][idx]['T_cold'])
                T_hot = np.array(results['coils'][idx]['T_hot'])
                
                ax.plot(time, T_gas, 'k--', lw=1.5, label='Gas', alpha=0.7)
                ax.plot(time, T_hot, 'r-', lw=2, label='Hot spot')
                ax.plot(time, T_cold, 'b-', lw=2, label='Cold spot')
                
                # Zona donde cold > hot
                diff = T_cold - T_hot
                ax.fill_between(time, T_cold, T_hot, where=(diff > 0),
                               color='yellow', alpha=0.3)
                
                ax.set_title(f"{idx+1}# {results['coils'][idx]['coil_id']}\nT_cold máx: {max(T_cold):.1f}°C")
                ax.set_xlabel('Tiempo [h]')
                ax.set_ylabel('Temperatura [°C]')
                ax.legend(loc='lower right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 800])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla de resultados
            st.markdown("### 📋 Resumen por Bobina")
            
            data = []
            for i in range(len(st.session_state.bobinas)):
                T_cold = results['coils'][i]['T_cold']
                T_hot = results['coils'][i]['T_hot']
                data.append({
                    'Posición': f"{i+1}#",
                    'ID': results['coils'][i]['coil_id'],
                    'Perfil': results['coils'][i]['profile'],
                    'T_cold máx (°C)': f"{max(T_cold):.1f}",
                    'T_hot máx (°C)': f"{max(T_hot):.1f}",
                    'ΔT vs objetivo': f"{max(T_cold) - T_plateau:.1f}"
                })
            
            st.table(data)

# =============================================================================
# TAB 4: Ayuda
# =============================================================================

with tab4:
    st.subheader("❓ Guía de Uso")
    
    st.markdown("""
    ### ¿Qué es este simulador?
    
    Este simulador calcula la transferencia de calor en un **horno de recocido tipo campana** 
    para bobinas de acero laminado en frío. Está basado en el modelo de Yang et al. (2025).
    
    ### Pasos para usar:
    
    1. **Perfiles de Acero** (opcional)
       - Revisa los perfiles predefinidos o crea uno nuevo con propiedades personalizadas
    
    2. **Configurar Bobinas**
       - Agrega las bobinas que irán en el horno
       - Define las dimensiones de cada una (diámetros, ancho, espesor)
       - El orden de abajo hacia arriba es: 1#, 2#, 3#, 4#
    
    3. **Configurar Ciclo** (barra lateral)
       - **Temperatura de Plateau**: Temperatura objetivo del recocido (típico: 680-720°C)
       - **Umbral ΔT**: El plateau termina cuando el cold spot está a menos de este valor del gas
    
    4. **Simular**
       - Ejecuta la simulación y observa los resultados
       - El **cold spot** (centro de la bobina) debe alcanzar cerca de la temperatura objetivo
    
    ### Conceptos clave:
    
    - **Hot spot**: Esquina exterior de la bobina (se calienta primero)
    - **Cold spot**: Centro de la bobina (se calienta más lento)
    - **Plateau dinámico**: El tiempo de remojo se ajusta automáticamente hasta que el cold spot alcanza la temperatura
    
    ### Perfiles predefinidos:
    
    | Perfil | Descripción |
    |--------|-------------|
    | SPCC | Acero laminado en frío comercial (JIS) |
    | DC01 | Acero para embutición (EN 10130) |
    | DC04 | Acero para embutición profunda |
    | Q235 | Acero estructural (China) |
    | AISI_1008 | Acero bajo carbono |
    | IF_Steel | Acero libre de intersticiales |
    """)
