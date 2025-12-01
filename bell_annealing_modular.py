"""
Modelo de Horno de Recocido Tipo Campana - Versión Modular
==========================================================

Integra:
- Perfiles de acero personalizables
- Configuración flexible de bobinas
- Ciclo de recocido dinámico

Basado en: Yang et al., Scientific Reports (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import warnings

# Importar módulo de perfiles
from steel_profiles import (
    SteelProfile, SteelProfileLibrary, Coil, FurnaceStack, 
    create_quick_coil
)


# =============================================================================
# PROPIEDADES DEL GAS (HIDRÓGENO)
# =============================================================================

class HydrogenProperties:
    """Propiedades termofísicas del hidrógeno como función de la temperatura"""
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """Conductividad térmica del H2 [W/(m·K)]"""
        T_celsius = T - 273.15
        return 0.1672 + 4.86e-4 * T_celsius + 1.08e-7 * T_celsius**2
    
    @staticmethod
    def dynamic_viscosity(T: float) -> float:
        """Viscosidad dinámica del H2 [Pa·s]"""
        T_ref = 293.15
        mu_ref = 8.76e-6
        S = 72.0
        return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)
    
    @staticmethod
    def density(T: float, P: float = 101325.0) -> float:
        """Densidad del H2 [kg/m³]"""
        R_H2 = 4124.0
        return P / (R_H2 * T)
    
    @staticmethod
    def specific_heat() -> float:
        """Calor específico del H2 [J/(kg·K)]"""
        return 14300.0


# =============================================================================
# CONFIGURACIÓN DEL HORNO
# =============================================================================

@dataclass
class FurnaceConfig:
    """Configuración del horno de recocido"""
    
    # Dimensiones
    inner_cover_diameter: float = 2.5  # m
    
    # Flujo de gas
    total_gas_flow: float = 150.0  # m³/h
    
    # Coeficientes de modelo (CALIBRADOS)
    flow_attenuation: float = 0.909      # 1/ζ
    convection_enhancement: float = 2.0  # ψ
    radiation_enhancement: float = 1.2   # ξ
    emulsion_correction: float = 0.85    # η
    
    # Geometría del canal
    flow_channel_diameter: float = 0.02  # m
    flow_channel_length: float = 1.2     # m


# =============================================================================
# DISTRIBUCIÓN DE FLUJO
# =============================================================================

class FlowDistribution:
    """Calcula la distribución de flujo de gas protector"""
    
    def __init__(self, num_coils: int, zeta: float = 1.1):
        self.num_coils = num_coils
        self.zeta = zeta
        self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """Calcula coeficientes de distribución - Ecuación (5)"""
        fd = [0.0] * self.num_coils
        
        fd[-1] = 1.0 / (2 + self.num_coils)
        
        for i in range(self.num_coils - 2, -1, -1):
            fd[i] = self.zeta * fd[i + 1]
        
        self.fd = fd
    
    def get_flow_fraction(self, layer_index: int) -> Tuple[float, float]:
        """
        Obtiene fracciones de flujo para una capa.
        
        Returns:
            (inner_outer_frac, top_frac)
        """
        inner_outer_frac = 1.0
        for i in range(layer_index + 1):
            inner_outer_frac *= (1.0 - self.fd[i])
        
        if layer_index < self.num_coils:
            top_frac = self.fd[layer_index]
        else:
            top_frac = self.fd[-1]
        
        return inner_outer_frac, top_frac


# =============================================================================
# RESISTENCIA TÉRMICA
# =============================================================================

class ThermalResistanceCalculator:
    """Calcula resistencias térmicas radiales para una bobina"""
    
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
    
    def __init__(self, coil: Coil, compressive_stress: float = 8e6):
        self.coil = coil
        self.profile = coil.profile
        self.compressive_stress = compressive_stress
        self._calculate_contact_params()
    
    def _calculate_contact_params(self):
        """Calcula parámetros de contacto - Ecuaciones (14) y (15)"""
        P_MPa = self.compressive_stress / 1e6
        H_MPa = self.profile.hardness / 1e6
        
        self.phi = P_MPa / (H_MPa + P_MPa)
        self.gap = 42.7e-6 * np.exp(-5e-2 * P_MPa)
        self.tan_theta = 0.15
    
    def R_steel_half(self, T: float) -> float:
        """Resistencia de medio espesor de lámina - Ecuación (9)"""
        lambda_s = self.profile.get_thermal_conductivity(T)
        return self.coil.thickness / (2.0 * lambda_s)
    
    def R_oxide(self) -> float:
        """Resistencia de capa de óxido - Ecuación (10)"""
        return self.profile.oxide_thickness / self.profile.oxide_thermal_conductivity
    
    def R_radiation(self, T_mean: float) -> float:
        """Resistencia por radiación - Ecuación (11)"""
        eps = self.profile.emissivity
        T_mean = max(T_mean, 300.0)
        numerator = 2.0 - eps
        denominator = 4.0 * (1.0 - self.phi) * eps * self.STEFAN_BOLTZMANN * T_mean**3
        return numerator / max(denominator, 1e-10)
    
    def R_gas(self, T: float) -> float:
        """Resistencia del gas en gap - Ecuación (12)"""
        lambda_g = HydrogenProperties.thermal_conductivity(T)
        return self.gap / ((1.0 - self.phi) * lambda_g)
    
    def R_contact(self, T: float) -> float:
        """Resistencia por contacto - Ecuación (13)"""
        lambda_s = self.profile.get_thermal_conductivity(T)
        sigma_p = self.profile.surface_roughness
        denominator = 1.13 * lambda_s * self.tan_theta * (self.phi**0.94)
        return sigma_p / max(denominator, 1e-10)
    
    def radial_conductivity(self, T_mean: float, T_gas: float, 
                           emulsion_correction: float = 0.85) -> float:
        """
        Conductividad radial equivalente - Ecuación (17)
        """
        R_s2 = self.R_steel_half(T_mean)
        R_r = self.R_radiation(T_mean)
        R_g = self.R_gas(T_mean)
        R_d = self.R_contact(T_mean)
        
        R_parallel = 1.0 / (1.0/R_r + 1.0/R_g + 1.0/R_d)
        
        if T_gas >= 800:
            R_o = self.R_oxide()
            R_total = 2.0 * R_s2 + R_o + R_parallel
            thickness_total = self.coil.thickness + self.gap + self.profile.oxide_thickness
            lambda_r = thickness_total / R_total
        else:
            R_total = 2.0 * R_s2 + R_parallel
            thickness_total = self.coil.thickness + self.gap
            lambda_r = emulsion_correction * thickness_total / R_total
        
        return np.clip(lambda_r, 1.0, 25.0)


# =============================================================================
# COEFICIENTE DE CONVECCIÓN
# =============================================================================

class ConvectionCalculator:
    """Calcula coeficientes de convección"""
    
    def __init__(self, config: FurnaceConfig, flow_dist: FlowDistribution):
        self.config = config
        self.flow_dist = flow_dist
    
    def calculate(self, T_gas: float, layer_index: int) -> float:
        """
        Calcula coeficiente de convección - Ecuación (7)
        """
        inner_outer_frac, top_frac = self.flow_dist.get_flow_fraction(layer_index)
        
        # Propiedades del gas
        lambda_g = HydrogenProperties.thermal_conductivity(T_gas)
        mu_g = HydrogenProperties.dynamic_viscosity(T_gas)
        rho_g = HydrogenProperties.density(T_gas)
        
        # Velocidad
        flow_rate = self.config.total_gas_flow * inner_outer_frac / 3600.0
        channel_area = 0.15
        u_g = max(flow_rate * (T_gas / 293.15) / channel_area, 1.0)
        
        # Reynolds y Prandtl
        D = self.config.flow_channel_diameter
        L = self.config.flow_channel_length
        Re = rho_g * u_g * D / mu_g
        Pr = mu_g * HydrogenProperties.specific_heat() / lambda_g
        
        # Nusselt
        psi = self.config.convection_enhancement
        if Re > 2300:
            Nu = 0.023 * psi * (Re**0.8) * (Pr**0.4) * (1 + (D/L)**0.7)
        else:
            Nu = 3.66 + 0.0668 * (D/L) * Re * Pr / (1 + 0.04 * ((D/L) * Re * Pr)**(2/3))
            Nu *= psi
        
        h = 2 * Nu * lambda_g / D
        h = np.clip(h, 10.0, 400.0)
        
        # Factor de radiación
        if T_gas >= 800:
            h *= self.config.radiation_enhancement
        
        return h


# =============================================================================
# CICLO DE RECOCIDO
# =============================================================================

class AnnealingCycle:
    """Ciclo de recocido con plateau dinámico"""
    
    def __init__(self, T_plateau: float = 700.0, threshold: float = 3.0):
        """
        Args:
            T_plateau: Temperatura de plateau [°C]
            threshold: Umbral para terminar plateau [°C]
        """
        self.T_plateau = T_plateau + 273.15  # K
        self.threshold = threshold
        
        # Perfil de calentamiento [h, °C]
        self._heating_times = [0, 0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 4,
                               5, 6, 7, 8, 9, 10, 11, 12, 13]
        self._heating_temps = [50, 150, 250, 320, 380, 400, 350, 300, 350, 410, 450,
                               475, 510, 550, 590, 610, 650, 660, 680, T_plateau]
        self._heating_temps = [t + 273.15 for t in self._heating_temps]
        
        # Perfil de enfriamiento [h desde inicio enfriamiento, °C]
        self._cooling_times = [0, 0.2, 1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10]
        self._cooling_temps = [T_plateau, 660, 640, 610, 570, 520, 470, 380, 420, 380, 330, 280, 100]
        self._cooling_temps = [t + 273.15 for t in self._cooling_temps]
        
        # Estado
        self.phase = 'heating'
        self.plateau_start = None
        self.cooling_start = None
    
    def get_temperature(self, time_h: float) -> float:
        """Obtiene temperatura del gas según fase actual"""
        if self.phase == 'heating':
            if time_h <= self._heating_times[-1]:
                return np.interp(time_h, self._heating_times, self._heating_temps)
            return self.T_plateau
        
        elif self.phase == 'plateau':
            return self.T_plateau
        
        elif self.phase == 'cooling':
            t_rel = time_h - self.cooling_start
            return np.interp(t_rel, self._cooling_times, self._cooling_temps)
        
        return self.T_plateau
    
    def start_plateau(self, time_h: float):
        self.phase = 'plateau'
        self.plateau_start = time_h
    
    def start_cooling(self, time_h: float):
        self.phase = 'cooling'
        self.cooling_start = time_h
    
    def should_end_plateau(self, T_cold: float) -> bool:
        """Verifica si cold spot alcanzó el umbral"""
        if self.phase != 'plateau':
            return False
        delta = self.T_plateau - T_cold
        return delta <= self.threshold
    
    def is_finished(self, time_h: float) -> bool:
        """Verifica si el ciclo terminó"""
        if self.phase != 'cooling':
            return False
        t_rel = time_h - self.cooling_start
        return t_rel >= self._cooling_times[-1]


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class BellAnnealingSimulator:
    """
    Simulador del horno de recocido tipo campana.
    
    Usa perfiles de acero personalizables y configuración flexible de bobinas.
    """
    
    def __init__(self, stack: FurnaceStack, config: FurnaceConfig, cycle: AnnealingCycle):
        """
        Args:
            stack: Configuración del apilado de bobinas
            config: Configuración del horno
            cycle: Ciclo de recocido
        """
        self.stack = stack
        self.config = config
        self.cycle = cycle
        
        # Validar
        stack.validate()
        
        # Inicializar componentes
        self.flow_dist = FlowDistribution(stack.num_coils, 1.0 / config.flow_attenuation)
        self.convection = ConvectionCalculator(config, self.flow_dist)
        
        # Resistencias térmicas por bobina
        self.thermal_res = [
            ThermalResistanceCalculator(coil, stack.compressive_stress)
            for coil in stack.coils
        ]
        
        # Parámetros numéricos
        self.nr = 25  # Nodos radiales
        self.nz = 15  # Nodos axiales
        self.dt = 5.0  # Paso temporal [s]
    
    def _init_mesh(self, coil_idx: int) -> Tuple[np.ndarray, float, float]:
        """Inicializa malla para una bobina"""
        coil = self.stack.coils[coil_idx]
        
        r_inner = coil.inner_diameter / 2
        r_outer = coil.outer_diameter / 2
        
        dr = (r_outer - r_inner) / (self.nr - 1)
        dz = coil.width / (self.nz - 1)
        
        T = np.ones((self.nr, self.nz)) * 323.15  # 50°C inicial
        
        return T, dr, dz
    
    def _get_properties(self, coil_idx: int, T_mean: float, T_gas: float):
        """Obtiene propiedades térmicas"""
        coil = self.stack.coils[coil_idx]
        profile = coil.profile
        
        lambda_z = profile.get_thermal_conductivity(T_mean)
        lambda_r = self.thermal_res[coil_idx].radial_conductivity(
            T_mean, T_gas, self.config.emulsion_correction
        )
        rho = profile.density
        c = profile.get_specific_heat(T_mean)
        
        return lambda_r, lambda_z, rho, c
    
    def simulate(self, max_time_h: float = 50.0, save_interval: int = 60) -> dict:
        """
        Ejecuta la simulación.
        
        Args:
            max_time_h: Tiempo máximo [h]
            save_interval: Intervalo de guardado [pasos]
        
        Returns:
            Diccionario con resultados
        """
        n_steps = int(max_time_h * 3600 / self.dt)
        
        # Inicializar
        results = {
            'time': [],
            'T_gas': [],
            'coils': [{
                'T_cold': [], 'T_hot': [], 'coil_id': self.stack.coils[i].coil_id,
                'profile': self.stack.coils[i].profile_name
            } for i in range(self.stack.num_coils)]
        }
        
        T_coils = []
        mesh_params = []
        for i in range(self.stack.num_coils):
            T, dr, dz = self._init_mesh(i)
            T_coils.append(T)
            mesh_params.append((dr, dz))
        
        print(f"Simulando {self.stack.num_coils} bobinas...")
        print(f"Plateau @ {self.cycle.T_plateau - 273.15:.0f}°C, umbral: {self.cycle.threshold}°C")
        
        # Bucle temporal
        for step in range(n_steps):
            time_h = step * self.dt / 3600.0
            
            # Cold spot de referencia (bobina con peor transferencia)
            T_cold_ref = T_coils[0][self.nr // 2, self.nz // 2]
            
            # Transiciones de fase
            if self.cycle.phase == 'heating' and time_h >= self.cycle._heating_times[-1]:
                self.cycle.start_plateau(time_h)
                print(f"  PLATEAU @ {time_h:.1f}h")
            
            elif self.cycle.phase == 'plateau' and self.cycle.should_end_plateau(T_cold_ref):
                self.cycle.start_cooling(time_h)
                duration = time_h - self.cycle.plateau_start
                print(f"  ENFRIAMIENTO @ {time_h:.1f}h (plateau: {duration:.1f}h)")
            
            elif self.cycle.is_finished(time_h):
                print(f"  FIN @ {time_h:.1f}h")
                break
            
            T_gas = self.cycle.get_temperature(time_h)
            
            # Actualizar cada bobina
            for idx in range(self.stack.num_coils):
                T = T_coils[idx]
                dr, dz = mesh_params[idx]
                T_mean = np.mean(T)
                
                lambda_r, lambda_z, rho, c = self._get_properties(idx, T_mean, T_gas)
                h = self.convection.calculate(T_gas, idx)
                
                # Números de Fourier y Biot
                Fo_r = lambda_r * self.dt / (rho * c * dr**2)
                Fo_z = lambda_z * self.dt / (rho * c * dz**2)
                Bi_r = h * self.dt / (rho * c * dr)
                Bi_z = h * self.dt / (rho * c * dz)
                
                T_new = np.copy(T)
                
                # Interior
                for m in range(1, self.nr - 1):
                    for n in range(1, self.nz - 1):
                        T_new[m, n] = ((1 - 2*Fo_r - 2*Fo_z) * T[m, n] +
                                      Fo_r * (T[m+1, n] + T[m-1, n]) +
                                      Fo_z * (T[m, n+1] + T[m, n-1]))
                
                # Bordes
                for m in range(1, self.nr - 1):
                    # Superior
                    T_new[m, -1] = ((1 - 2*Fo_r - Fo_z - Bi_z) * T[m, -1] +
                                   Fo_r * (T[m+1, -1] + T[m-1, -1]) +
                                   Fo_z * T[m, -2] + Bi_z * T_gas)
                    # Inferior
                    T_new[m, 0] = ((1 - 2*Fo_r - Fo_z - Bi_z) * T[m, 0] +
                                  Fo_r * (T[m+1, 0] + T[m-1, 0]) +
                                  Fo_z * T[m, 1] + Bi_z * T_gas)
                
                for n in range(1, self.nz - 1):
                    # Exterior
                    T_new[-1, n] = ((1 - Fo_r - 2*Fo_z - Bi_r) * T[-1, n] +
                                   Fo_r * T[-2, n] +
                                   Fo_z * (T[-1, n+1] + T[-1, n-1]) + Bi_r * T_gas)
                    # Interior
                    T_new[0, n] = ((1 - Fo_r - 2*Fo_z - Bi_r) * T[0, n] +
                                  Fo_r * T[1, n] +
                                  Fo_z * (T[0, n+1] + T[0, n-1]) + Bi_r * T_gas)
                
                # Esquinas
                T_new[-1, -1] = ((1 - Fo_r - Fo_z - Bi_r - Bi_z) * T[-1, -1] +
                                Fo_r * T[-2, -1] + Fo_z * T[-1, -2] + (Bi_r + Bi_z) * T_gas)
                T_new[0, -1] = ((1 - Fo_r - Fo_z - Bi_r - Bi_z) * T[0, -1] +
                               Fo_r * T[1, -1] + Fo_z * T[0, -2] + (Bi_r + Bi_z) * T_gas)
                T_new[-1, 0] = ((1 - Fo_r - Fo_z - Bi_r - Bi_z) * T[-1, 0] +
                               Fo_r * T[-2, 0] + Fo_z * T[-1, 1] + (Bi_r + Bi_z) * T_gas)
                T_new[0, 0] = ((1 - Fo_r - Fo_z - Bi_r - Bi_z) * T[0, 0] +
                              Fo_r * T[1, 0] + Fo_z * T[0, 1] + (Bi_r + Bi_z) * T_gas)
                
                T_coils[idx] = T_new
            
            # Guardar resultados
            if step % save_interval == 0:
                results['time'].append(time_h)
                results['T_gas'].append(T_gas - 273.15)
                
                for idx in range(self.stack.num_coils):
                    T = T_coils[idx]
                    results['coils'][idx]['T_cold'].append(T[self.nr//2, self.nz//2] - 273.15)
                    results['coils'][idx]['T_hot'].append(T[-1, -1] - 273.15)
        
        self.results = results
        return results
    
    def plot_results(self):
        """Grafica resultados"""
        results = self.results
        time = np.array(results['time'])
        T_gas = np.array(results['T_gas'])
        
        n_coils = self.stack.num_coils
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx in range(min(4, n_coils)):
            ax = axes[idx // 2, idx % 2]
            
            T_cold = np.array(results['coils'][idx]['T_cold'])
            T_hot = np.array(results['coils'][idx]['T_hot'])
            
            ax.plot(time, T_gas, 'k--', lw=2, label='Gas', alpha=0.7)
            ax.plot(time, T_hot, 'r-', lw=2, label='Hot spot')
            ax.plot(time, T_cold, 'b-', lw=2, label='Cold spot')
            
            # Zona donde cold > hot
            diff = T_cold - T_hot
            ax.fill_between(time, T_cold, T_hot, where=(diff > 0),
                           color='yellow', alpha=0.3, label='Cold > Hot')
            
            coil = self.stack.coils[idx]
            ax.set_title(f"{idx+1}# {coil.coil_id} ({coil.profile_name}) - Cold máx: {np.max(T_cold):.1f}°C")
            ax.set_xlabel('Tiempo [h]')
            ax.set_ylabel('Temperatura [°C]')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 800])
        
        # Ocultar subplots vacíos
        for idx in range(n_coils, 4):
            axes[idx // 2, idx % 2].axis('off')
        
        plt.suptitle(f'Simulación de Recocido - {n_coils} bobinas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        results = self.results
        
        print("\n" + "=" * 60)
        print("RESUMEN DE RESULTADOS")
        print("=" * 60)
        
        for idx in range(self.stack.num_coils):
            coil = self.stack.coils[idx]
            T_cold = np.array(results['coils'][idx]['T_cold'])
            T_hot = np.array(results['coils'][idx]['T_hot'])
            
            print(f"\n{idx+1}# {coil.coil_id} ({coil.profile_name}):")
            print(f"   T_cold máx: {np.max(T_cold):.1f}°C")
            print(f"   T_hot máx:  {np.max(T_hot):.1f}°C")
            print(f"   Masa: {coil.mass:.0f} kg")
        
        print(f"\nTiempo total: {results['time'][-1]:.1f} h")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Inicializar perfiles
    SteelProfileLibrary.initialize_defaults()
    
    print("=" * 60)
    print("DEMOSTRACIÓN - MODELO MODULAR DE RECOCIDO")
    print("=" * 60)
    
    # Crear bobinas con DIFERENTES perfiles
    coil1 = create_quick_coil("BOB-001", "SPCC", 1869, width_mm=1272, thickness_mm=1.50)
    coil2 = create_quick_coil("BOB-002", "SPCC", 1867, width_mm=1250, thickness_mm=1.50)
    coil3 = create_quick_coil("BOB-003", "DC01", 1837, width_mm=1272, thickness_mm=1.48)
    coil4 = create_quick_coil("BOB-004", "DC04", 1761, width_mm=1272, thickness_mm=1.50)
    
    # Configurar stack
    stack = FurnaceStack()
    stack.add_coil(coil1)  # 1# (fondo)
    stack.add_coil(coil2)  # 2#
    stack.add_coil(coil3)  # 3# 
    stack.add_coil(coil4)  # 4# (arriba)
    
    print(stack.summary())
    
    # Configuración del horno
    config = FurnaceConfig(
        total_gas_flow=150.0,
        convection_enhancement=2.0,
        radiation_enhancement=1.2
    )
    
    # Ciclo dinámico
    cycle = AnnealingCycle(T_plateau=700.0, threshold=3.0)
    
    # Simular
    print("\n" + "-" * 60)
    simulator = BellAnnealingSimulator(stack, config, cycle)
    results = simulator.simulate(max_time_h=50.0)
    print("-" * 60)
    
    # Resultados
    simulator.print_summary()
    simulator.plot_results()
