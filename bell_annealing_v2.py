"""
Modelo de Recocido en Campana v3.0 (CORREGIDO)
==============================================
Correcciones basadas en Yang et al., Scientific Reports (2025):

1. Conductividad radial calculada según Ecuaciones 9-17 del paper
2. Gap entre capas calculado con Ec. 15: b = 42.7e-6 * exp(-0.05*P)
3. Ratio de contacto φ según Ec. 14: φ = P/(H+P)
4. Eliminado el factor de "enhancement" incorrecto

Autor: Corrección basada en análisis del paper
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from steel_profiles import SteelProfile, SteelProfileLibrary, Coil, FurnaceStack


# =============================================================================
# PROPIEDADES DEL GAS (Hidrógeno) - Sin cambios
# =============================================================================

class HydrogenProperties:
    """Propiedades termofísicas del H2 vs temperatura"""
    
    @staticmethod
    def thermal_conductivity(T):
        """λ en W/(m·K), T en K"""
        return 0.0786 + 4.29e-4 * T - 4.0e-8 * T**2
    
    @staticmethod
    def density(T, P=101325):
        """ρ en kg/m³, T en K, P en Pa"""
        M = 2.016e-3  # kg/mol
        R = 8.314
        return P * M / (R * T)
    
    @staticmethod
    def specific_heat(T):
        """cp en J/(kg·K), T en K"""
        return 14300 + 0.5 * (T - 300)
    
    @staticmethod
    def viscosity(T):
        """μ en Pa·s, T en K"""
        return 4.0e-6 + 2.5e-8 * T
    
    @staticmethod
    def prandtl(T):
        """Número de Prandtl"""
        cp = HydrogenProperties.specific_heat(T)
        mu = HydrogenProperties.viscosity(T)
        k = HydrogenProperties.thermal_conductivity(T)
        return cp * mu / k


# =============================================================================
# CONFIGURACIÓN DEL HORNO
# =============================================================================

@dataclass
class FurnaceConfig:
    """Parámetros del horno de recocido"""
    total_gas_flow: float = 150.0  # m³/h de H2
    convection_enhancement: float = 1.5  # Factor ψ (reducido - más realista)
    radiation_enhancement: float = 1.0  # Factor ξ
    inter_coil_conductance: float = 30.0  # W/(m²·K) - conductancia entre bobinas
    position_factor: float = 0.15  # Penalización para bobinas del medio
    compressive_stress: float = 8e6  # Pa - Presión entre capas


# =============================================================================
# CICLO DE RECOCIDO - Sin cambios significativos
# =============================================================================

class AnnealingCycle:
    """Ciclo de recocido con plateau dinámico"""
    
    def __init__(self, T_plateau: float = 700.0, threshold: float = 3.0,
                 T_initial: float = 50.0, T_final: float = 100.0,
                 heating_time: float = 13.0, cooling_time: float = 10.0):
        self.T_plateau = T_plateau + 273.15  # K
        self.T_plateau_C = T_plateau
        self.threshold = threshold
        self.T_initial = T_initial
        self.T_final = T_final
        self.heating_time = heating_time
        self.cooling_time = cooling_time
        
        self._heating_times = [0, heating_time]
        self._heating_temps = [T_initial + 273.15, T_plateau + 273.15]
        
        self._cooling_times = [0, cooling_time]
        self._cooling_temps = [T_plateau + 273.15, T_final + 273.15]
        
        self.phase = 'heating'
        self.plateau_start = None
        self.cooling_start = None
        self.annealing_time = None
    
    def get_temperature(self, time_h: float) -> float:
        if self.phase == 'heating':
            return np.interp(time_h, self._heating_times, self._heating_temps)
        elif self.phase == 'plateau':
            return self.T_plateau
        else:
            t_rel = time_h - self.cooling_start
            return np.interp(t_rel, self._cooling_times, self._cooling_temps)
    
    def start_plateau(self, time_h: float):
        self.phase = 'plateau'
        self.plateau_start = time_h
    
    def should_end_plateau(self, T_cold: float) -> bool:
        delta = self.T_plateau - T_cold
        return delta <= self.threshold
    
    def start_cooling(self, time_h: float):
        self.phase = 'cooling'
        self.cooling_start = time_h
        self.annealing_time = time_h
    
    def is_finished(self, time_h: float) -> bool:
        if self.phase != 'cooling' or self.cooling_start is None:
            return False
        return (time_h - self.cooling_start) >= self.cooling_time


# =============================================================================
# CALCULADOR DE PROPIEDADES TÉRMICAS - CORREGIDO
# =============================================================================

class ThermalCalculatorCorrected:
    """
    Calcula propiedades térmicas efectivas según Yang et al. (2025)
    
    CORRECCIONES PRINCIPALES:
    1. λ_r calculado con fórmulas del paper (Ec. 9-17)
    2. Gap calculado con Ec. 15
    3. Sin factor de enhancement artificial
    """
    
    STEFAN_BOLTZMANN = 5.67e-8
    
    def __init__(self, coil: Coil, compressive_stress: float = 8e6):
        self.coil = coil
        self.profile = SteelProfileLibrary.get_profile(coil.profile_name)
        self.P = compressive_stress  # Presión entre capas [Pa]
        self._calc_geometry()
        self._calc_contact_params()
    
    def _calc_geometry(self):
        """Calcula parámetros geométricos"""
        self.r_out = self.coil.outer_diameter / 2
        self.r_in = self.coil.inner_diameter / 2
        self.thickness = self.coil.thickness  # Espesor de lámina
        self.width = self.coil.width
        
        radial_thickness = self.r_out - self.r_in
        self.n_layers = radial_thickness / self.thickness
    
    def _calc_contact_params(self):
        """
        Calcula parámetros de contacto según el paper.
        Ecuaciones 14 y 15.
        """
        H = self.profile.hardness  # Dureza [Pa]
        
        # Ecuación 14: ratio de contacto
        self.phi = self.P / (H + self.P)
        
        # Ecuación 15: gap entre capas [m]
        # b = 42.7 × 10^-6 exp(-5 × 10^-2 P) donde P en MPa
        P_MPa = self.P / 1e6
        self.b = 42.7e-6 * np.exp(-5e-2 * P_MPa)
        
        # Espesor de óxido (del paper)
        self.b_O = self.profile.oxide_thickness  # ~10 μm
    
    def steel_conductivity(self, T: float) -> float:
        """Conductividad del acero [W/(m·K)]"""
        a, b, c = self.profile.thermal_conductivity_coeffs
        T_C = T - 273.15
        return max(a + b * T_C + c * T_C**2, 10.0)
    
    def steel_specific_heat(self, T: float) -> float:
        """Calor específico del acero [J/(kg·K)]"""
        a, b, c = self.profile.specific_heat_coeffs
        T_C = T - 273.15
        return max(a + b * T_C + c * T_C**2, 400.0)
    
    def radial_conductivity(self, T_mean: float, T_gas: float) -> float:
        """
        Conductividad efectiva radial según Ecuaciones 9-17 del paper.
        
        La resistencia térmica radial incluye:
        - R_S/2: Media capa de acero (x2)
        - R_O: Capa de óxido (para T > 800K)
        - R_R: Radiación entre capas
        - R_G: Conducción del gas en el gap
        - R_D: Contacto mecánico entre capas
        """
        lambda_steel = self.steel_conductivity(T_mean)
        lambda_gas = HydrogenProperties.thermal_conductivity(T_mean)
        lambda_O = self.profile.oxide_thermal_conductivity
        
        S = self.thickness
        eps = self.profile.emissivity
        tan_theta = 0.1  # Pendiente promedio de superficie (típico)
        sigma_p = self.profile.surface_roughness
        
        # Ecuación 9: R_S/2 - Resistencia de media capa de acero
        R_S2 = S / (2 * lambda_steel)
        
        # Ecuación 10: R_O - Resistencia de óxido
        R_O = self.b_O / lambda_O
        
        # Ecuación 11: R_R - Resistencia por radiación
        h_rad = 4 * (1 - self.phi) * eps * self.STEFAN_BOLTZMANN * T_mean**3 / (2 - eps)
        R_R = 1 / h_rad if h_rad > 1e-10 else 1e6
        
        # Ecuación 12: R_G - Resistencia del gas
        R_G = self.b / ((1 - self.phi) * lambda_gas) if self.phi < 1 else 1e6
        
        # Ecuación 13: R_D - Resistencia de contacto
        R_D = sigma_p / (1.13 * lambda_steel * tan_theta) * self.phi**(-0.94)
        
        # Resistencias en paralelo para el gap
        R_parallel = 1 / (1/R_R + 1/R_G + 1/R_D)
        
        # Ecuación 16: Resistencia total
        if T_gas >= 800 + 273.15:  # > 800°C
            R_total = 2 * R_S2 + R_O + R_parallel
            numerator = S + self.b + self.b_O
        else:
            # A baja temperatura, considerar emulsión (simplificado)
            R_E = 5e-5  # Resistencia de emulsión estimada
            R_total = 2 * R_S2 + R_E + R_parallel
            numerator = S + self.b
        
        # Ecuación 17: Conductividad radial efectiva
        lambda_r = numerator / R_total
        
        return lambda_r
    
    def axial_conductivity(self, T: float) -> float:
        """
        Conductividad efectiva axial (dirección del ancho).
        En esta dirección, la conducción es principalmente a través del acero.
        """
        return self.steel_conductivity(T) * 0.95  # Pequeña reducción por gaps de aire


# =============================================================================
# SIMULADOR CORREGIDO
# =============================================================================

class BellAnnealingSimulatorV3:
    """
    Simulador de recocido con modelo térmico corregido.
    """
    
    def __init__(self, stack: FurnaceStack, config: FurnaceConfig, cycle: AnnealingCycle):
        self.stack = stack
        self.config = config
        self.cycle = cycle
        
        # Crear calculadores térmicos para cada bobina
        self.calculators = [
            ThermalCalculatorCorrected(coil, config.compressive_stress) 
            for coil in stack.coils
        ]
        
        # Parámetros de discretización
        self.nr = 20  # Nodos radiales
        self.nz = 15  # Nodos axiales (ancho)
        self.dt = 10.0  # Paso de tiempo [s]
        
        self._calc_position_factors()
    
    def _calc_position_factors(self):
        """Calcula factores de penalización por posición"""
        n = self.stack.num_coils
        self.position_factors = []
        
        for i in range(n):
            pos = i / (n - 1) if n > 1 else 0.5
            distance_from_edge = 1 - abs(2 * pos - 1)
            factor = 1 - self.config.position_factor * distance_from_edge
            self.position_factors.append(factor)
    
    def _init_temperatures(self) -> List[np.ndarray]:
        """Inicializa campos de temperatura"""
        T_init = self.cycle.T_initial + 273.15
        return [np.ones((self.nr, self.nz)) * T_init for _ in range(self.stack.num_coils)]
    
    def _calc_convection_coeff(self, T_gas: float, coil_idx: int) -> float:
        """
        Calcula coeficiente de convección.
        Valores típicos industriales: h = 50-150 W/(m²·K)
        """
        h_base = 60.0  # W/(m²·K) - más conservador
        T_factor = (T_gas / 973.15) ** 0.4
        h = h_base * T_factor * self.config.convection_enhancement
        h *= self.position_factors[coil_idx]
        return max(h, 15.0)
    
    def _step_coil(self, T: np.ndarray, T_gas: float, calc: ThermalCalculatorCorrected, 
                   h_conv: float, T_above: Optional[np.ndarray], 
                   T_below: Optional[np.ndarray]) -> np.ndarray:
        """Avanza un paso de tiempo para una bobina."""
        T_new = T.copy()
        
        coil = calc.coil
        profile = calc.profile
        
        rho = profile.density
        T_mean = np.mean(T)
        cp = calc.steel_specific_heat(T_mean)
        lambda_r = calc.radial_conductivity(T_mean, T_gas)
        lambda_z = calc.axial_conductivity(T_mean)
        
        alpha_r = lambda_r / (rho * cp)
        alpha_z = lambda_z / (rho * cp)
        
        dr = (coil.outer_diameter - coil.inner_diameter) / 2 / self.nr
        dz = coil.width / self.nz
        
        Fr = alpha_r * self.dt / dr**2
        Fz = alpha_z * self.dt / dz**2
        
        # Verificar estabilidad
        if Fr > 0.25 or Fz > 0.25:
            substeps = int(max(Fr, Fz) / 0.2) + 1
            dt_sub = self.dt / substeps
            Fr = alpha_r * dt_sub / dr**2
            Fz = alpha_z * dt_sub / dz**2
        else:
            substeps = 1
        
        for _ in range(substeps):
            T_old = T_new.copy()
            
            # Interior
            for i in range(1, self.nr - 1):
                r = coil.inner_diameter/2 + (i + 0.5) * dr
                for j in range(1, self.nz - 1):
                    d2T_dr2 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j])
                    dT_dr = (T_old[i+1, j] - T_old[i-1, j]) / 2
                    term_r = Fr * (d2T_dr2 + dT_dr * dr / r)
                    
                    d2T_dz2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1])
                    term_z = Fz * d2T_dz2
                    
                    T_new[i, j] = T_old[i, j] + term_r + term_z
            
            # Condiciones de frontera
            
            # Superficie exterior: convección con gas
            Bi_out = h_conv * dr / lambda_r
            for j in range(self.nz):
                T_new[-1, j] = (T_old[-2, j] + Bi_out * T_gas) / (1 + Bi_out)
            
            # Superficie interior: convección reducida
            h_in = h_conv * 0.4  # Menor convección en canal central
            Bi_in = h_in * dr / lambda_r
            for j in range(self.nz):
                T_new[0, j] = (T_old[1, j] + Bi_in * T_gas) / (1 + Bi_in)
            
            # Superficie superior
            h_top = self.config.inter_coil_conductance if T_above is not None else h_conv * 0.6
            T_boundary_top = T_above[:, 0] if T_above is not None else np.full(self.nr, T_gas)
            Bi_top = h_top * dz / lambda_z
            for i in range(self.nr):
                T_new[i, -1] = (T_old[i, -2] + Bi_top * T_boundary_top[i]) / (1 + Bi_top)
            
            # Superficie inferior
            h_bot = self.config.inter_coil_conductance if T_below is not None else h_conv * 0.6
            T_boundary_bot = T_below[:, -1] if T_below is not None else np.full(self.nr, T_gas)
            Bi_bot = h_bot * dz / lambda_z
            for i in range(self.nr):
                T_new[i, 0] = (T_old[i, 1] + Bi_bot * T_boundary_bot[i]) / (1 + Bi_bot)
        
        return T_new
    
    def simulate(self, max_time_h: float = 50.0, save_interval: int = 60,
                 verbose: bool = True) -> Dict:
        """Ejecuta la simulación completa."""
        T_coils = self._init_temperatures()
        
        results = {
            'time': [],
            'T_gas': [],
            'T_hot': [[] for _ in range(self.stack.num_coils)],
            'T_cold': [[] for _ in range(self.stack.num_coils)],
            'T_mean': [[] for _ in range(self.stack.num_coils)],
            'delta_T': [[] for _ in range(self.stack.num_coils)],
            'phase': []
        }
        
        time_s = 0
        max_time_s = max_time_h * 3600
        last_save = -save_interval
        
        if verbose:
            print(f"Simulando {self.stack.num_coils} bobinas (modelo v3 corregido)...")
        
        while time_s < max_time_s:
            time_h = time_s / 3600
            
            # Cold spot más frío de todas las bobinas
            T_cold_min = min(T_coils[i][self.nr//2, self.nz//2] 
                            for i in range(self.stack.num_coils))
            
            # Transiciones de fase
            if self.cycle.phase == 'heating' and time_h >= self.cycle.heating_time:
                self.cycle.start_plateau(time_h)
                if verbose:
                    print(f"  PLATEAU @ {time_h:.1f}h")
            
            elif self.cycle.phase == 'plateau' and self.cycle.should_end_plateau(T_cold_min):
                self.cycle.start_cooling(time_h)
                duration = time_h - self.cycle.plateau_start
                if verbose:
                    print(f"  ENFRIAMIENTO @ {time_h:.1f}h (plateau: {duration:.1f}h)")
            
            elif self.cycle.is_finished(time_h):
                if verbose:
                    print(f"  FIN @ {time_h:.1f}h")
                break
            
            T_gas = self.cycle.get_temperature(time_h)
            
            h_convs = [self._calc_convection_coeff(T_gas, i) 
                      for i in range(self.stack.num_coils)]
            
            T_coils_new = []
            for i in range(self.stack.num_coils):
                T_below = T_coils[i-1] if i > 0 else None
                T_above = T_coils[i+1] if i < self.stack.num_coils - 1 else None
                
                T_new = self._step_coil(
                    T_coils[i], T_gas, self.calculators[i], h_convs[i],
                    T_above, T_below
                )
                T_coils_new.append(T_new)
            
            T_coils = T_coils_new
            
            if time_s - last_save >= save_interval:
                results['time'].append(time_h)
                results['T_gas'].append(T_gas - 273.15)
                results['phase'].append(self.cycle.phase)
                
                for i in range(self.stack.num_coils):
                    T_hot = T_coils[i][-1, self.nz//2] - 273.15
                    T_cold = T_coils[i][self.nr//2, self.nz//2] - 273.15
                    T_mean = np.mean(T_coils[i]) - 273.15
                    
                    results['T_hot'][i].append(T_hot)
                    results['T_cold'][i].append(T_cold)
                    results['T_mean'][i].append(T_mean)
                    results['delta_T'][i].append(T_hot - T_cold)
                
                last_save = time_s
            
            time_s += self.dt
        
        # Convertir a arrays
        results['time'] = np.array(results['time'])
        results['T_gas'] = np.array(results['T_gas'])
        for i in range(self.stack.num_coils):
            results['T_hot'][i] = np.array(results['T_hot'][i])
            results['T_cold'][i] = np.array(results['T_cold'][i])
            results['T_mean'][i] = np.array(results['T_mean'][i])
            results['delta_T'][i] = np.array(results['delta_T'][i])
        
        return results


# =============================================================================
# FUNCIÓN DE COMPARACIÓN
# =============================================================================

def compare_models():
    """Compara el modelo original v2 con el corregido v3"""
    from bell_annealing_v2 import BellAnnealingSimulatorV2, ThermalCalculator
    from bell_annealing_v2 import FurnaceConfig as FurnaceConfigV2
    from bell_annealing_v2 import AnnealingCycle as AnnealingCycleV2
    
    SteelProfileLibrary.initialize_defaults()
    
    # Crear stack
    from steel_profiles import create_quick_coil
    stack = FurnaceStack()
    stack.add_coil(create_quick_coil('B1', 'SPCC', 1869, 600, 1272, 1.50))
    stack.add_coil(create_quick_coil('B2', 'SPCC', 1867, 600, 1250, 1.50))
    stack.add_coil(create_quick_coil('B3', 'SPCC', 1837, 600, 1272, 1.48))
    stack.add_coil(create_quick_coil('B4', 'SPCC', 1761, 600, 1272, 1.50))
    
    print("="*70)
    print("COMPARACIÓN: MODELO ORIGINAL (v2) vs CORREGIDO (v3)")
    print("="*70)
    
    # Comparar λ_r
    coil = stack.coils[0]
    calc_v2 = ThermalCalculator(coil)
    calc_v3 = ThermalCalculatorCorrected(coil)
    
    T_test = 700 + 273.15
    T_gas = 750 + 273.15
    
    lambda_r_v2 = calc_v2.radial_conductivity(T_test, T_gas)
    lambda_r_v3 = calc_v3.radial_conductivity(T_test, T_gas)
    
    print(f"\nConductividad radial a 700°C:")
    print(f"  v2 (original):  λ_r = {lambda_r_v2:.2f} W/(m·K)")
    print(f"  v3 (corregido): λ_r = {lambda_r_v3:.2f} W/(m·K)")
    print(f"  Ratio v2/v3:    {lambda_r_v2/lambda_r_v3:.1f}x")
    
    return lambda_r_v2, lambda_r_v3


if __name__ == "__main__":
    compare_models()
