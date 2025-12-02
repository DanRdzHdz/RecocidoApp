"""
Módulo de Perfiles de Acero y Configuración de Bobinas
======================================================

Este módulo permite:
1. Definir perfiles de acero con propiedades físicas
2. Crear bobinas con geometría específica usando un perfil
3. Configurar el orden de bobinas en el horno

Autor: Basado en Yang et al., Scientific Reports (2025)
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable
from pathlib import Path


# =============================================================================
# PERFILES DE ACERO - Solo propiedades físicas del material
# =============================================================================

@dataclass
class SteelProfile:
    """
    Perfil de acero - Contiene SOLO propiedades físicas del material.
    
    Las dimensiones NO van aquí, ya que el mismo acero puede tener
    diferentes geometrías (espesores, anchos, etc.)
    
    Attributes:
        name: Nombre identificador del perfil (ej: "SPCC", "DC01", "Q235")
        description: Descripción opcional del acero
        
        # Propiedades térmicas
        density: Densidad [kg/m³]
        thermal_conductivity: Conductividad térmica [W/(m·K)] o función de T
        specific_heat: Calor específico [J/(kg·K)] o función de T
        
        # Propiedades de superficie
        emissivity: Emisividad [-]
        surface_roughness: Rugosidad superficial [m]
        oxide_thickness: Espesor típico de capa de óxido [m]
        
        # Propiedades mecánicas (para cálculo de contacto entre capas)
        hardness: Dureza [Pa]
        
        # Propiedades de óxido
        oxide_thermal_conductivity: Conductividad del óxido [W/(m·K)]
    """
    name: str
    description: str = ""
    
    # Propiedades térmicas base
    density: float = 7850.0  # kg/m³
    
    # Conductividad térmica - valor base a 20°C
    thermal_conductivity_base: float = 50.0  # W/(m·K)
    # Coeficientes para variación con temperatura: λ = a + b*T + c*T²
    thermal_conductivity_coeffs: tuple = (50.0, -0.01, 0.0)
    
    # Calor específico - valor base
    specific_heat_base: float = 480.0  # J/(kg·K)
    # Coeficientes para variación con temperatura: c = a + b*T + c*T²
    specific_heat_coeffs: tuple = (420.0, 0.3, -1e-4)
    
    # Propiedades de superficie
    emissivity: float = 0.15
    surface_roughness: float = 3.2e-6  # m (3.2 μm típico)
    oxide_thickness: float = 10e-6  # m (10 μm típico)
    
    # Propiedades mecánicas
    hardness: float = 1133.86e6  # Pa (dureza Vickers ~115 HV para acero bajo carbono)
    
    # Propiedades del óxido
    oxide_thermal_conductivity: float = 0.5  # W/(m·K)
    
    def get_thermal_conductivity(self, T: float) -> float:
        """
        Obtiene conductividad térmica a temperatura T [K]
        
        Args:
            T: Temperatura en Kelvin
        
        Returns:
            Conductividad térmica [W/(m·K)]
        """
        a, b, c = self.thermal_conductivity_coeffs
        T_celsius = T - 273.15
        lambda_val = a + b * T_celsius + c * T_celsius**2
        return max(lambda_val, 10.0)  # Mínimo físicamente razonable
    
    def get_specific_heat(self, T: float) -> float:
        """
        Obtiene calor específico a temperatura T [K]
        
        Args:
            T: Temperatura en Kelvin
        
        Returns:
            Calor específico [J/(kg·K)]
        """
        a, b, c = self.specific_heat_coeffs
        T_celsius = T - 273.15
        cp = a + b * T_celsius + c * T_celsius**2
        return max(cp, 400.0)  # Mínimo físicamente razonable
    
    def to_dict(self) -> dict:
        """Convierte el perfil a diccionario para serialización"""
        return {
            'name': self.name,
            'description': self.description,
            'density': self.density,
            'thermal_conductivity_base': self.thermal_conductivity_base,
            'thermal_conductivity_coeffs': list(self.thermal_conductivity_coeffs),
            'specific_heat_base': self.specific_heat_base,
            'specific_heat_coeffs': list(self.specific_heat_coeffs),
            'emissivity': self.emissivity,
            'surface_roughness': self.surface_roughness,
            'oxide_thickness': self.oxide_thickness,
            'hardness': self.hardness,
            'oxide_thermal_conductivity': self.oxide_thermal_conductivity
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SteelProfile':
        """Crea un perfil desde un diccionario"""
        data = data.copy()
        if 'thermal_conductivity_coeffs' in data:
            data['thermal_conductivity_coeffs'] = tuple(data['thermal_conductivity_coeffs'])
        if 'specific_heat_coeffs' in data:
            data['specific_heat_coeffs'] = tuple(data['specific_heat_coeffs'])
        return cls(**data)
    
    def __str__(self):
        return f"SteelProfile('{self.name}', ρ={self.density} kg/m³, ε={self.emissivity})"


# =============================================================================
# PERFILES PREDEFINIDOS - Aceros comunes
# =============================================================================

class SteelProfileLibrary:
    """
    Biblioteca de perfiles de acero predefinidos.
    Los usuarios pueden agregar sus propios perfiles.
    """
    
    _profiles: Dict[str, SteelProfile] = {}
    
    @classmethod
    def initialize_defaults(cls):
        """Inicializa perfiles predefinidos comunes"""
        
        # SPCC - Acero laminado en frío comercial (JIS)
        cls.add_profile(SteelProfile(
            name="SPCC",
            description="Acero laminado en frío comercial (JIS G3141)",
            density=7850.0,
            thermal_conductivity_coeffs=(51.9, -0.017, 0.0),
            specific_heat_coeffs=(420.0, 0.36, -1.2e-4),
            emissivity=0.15,
            surface_roughness=3.2e-6,
            hardness=1133.86e6
        ))
        
        # DC01 - Acero laminado en frío (EN 10130)
        cls.add_profile(SteelProfile(
            name="DC01",
            description="Acero laminado en frío para embutición (EN 10130)",
            density=7850.0,
            thermal_conductivity_coeffs=(52.0, -0.018, 0.0),
            specific_heat_coeffs=(425.0, 0.35, -1.1e-4),
            emissivity=0.12,
            surface_roughness=2.8e-6,
            hardness=1100.0e6
        ))
        
        # DC04 - Acero para embutición profunda
        cls.add_profile(SteelProfile(
            name="DC04",
            description="Acero para embutición profunda (EN 10130)",
            density=7850.0,
            thermal_conductivity_coeffs=(50.5, -0.016, 0.0),
            specific_heat_coeffs=(430.0, 0.34, -1.0e-4),
            emissivity=0.10,
            surface_roughness=2.5e-6,
            hardness=1050.0e6
        ))
        
        # Q235 - Acero estructural chino
        cls.add_profile(SteelProfile(
            name="Q235",
            description="Acero estructural al carbono (GB/T 700)",
            density=7850.0,
            thermal_conductivity_coeffs=(48.0, -0.015, 0.0),
            specific_heat_coeffs=(450.0, 0.32, -1.0e-4),
            emissivity=0.20,
            surface_roughness=4.0e-6,
            hardness=1200.0e6
        ))
        
        # AISI 1008 - Acero bajo carbono
        cls.add_profile(SteelProfile(
            name="AISI_1008",
            description="Acero bajo carbono (AISI 1008)",
            density=7870.0,
            thermal_conductivity_coeffs=(51.0, -0.016, 0.0),
            specific_heat_coeffs=(420.0, 0.35, -1.1e-4),
            emissivity=0.14,
            surface_roughness=3.0e-6,
            hardness=1100.0e6
        ))
        
        # IF Steel - Acero libre de intersticiales
        cls.add_profile(SteelProfile(
            name="IF_Steel",
            description="Acero libre de intersticiales (Interstitial-Free)",
            density=7850.0,
            thermal_conductivity_coeffs=(55.0, -0.020, 0.0),
            specific_heat_coeffs=(415.0, 0.38, -1.3e-4),
            emissivity=0.08,
            surface_roughness=2.0e-6,
            hardness=950.0e6
        ))
    
    @classmethod
    def add_profile(cls, profile: SteelProfile):
        """Agrega un perfil a la biblioteca"""
        cls._profiles[profile.name] = profile
    
    @classmethod
    def get_profile(cls, name: str) -> SteelProfile:
        """Obtiene un perfil por nombre"""
        if not cls._profiles:
            cls.initialize_defaults()
        
        if name not in cls._profiles:
            available = list(cls._profiles.keys())
            raise ValueError(f"Perfil '{name}' no encontrado. Disponibles: {available}")
        
        return cls._profiles[name]
    
    @classmethod
    def list_profiles(cls) -> List[str]:
        """Lista todos los perfiles disponibles"""
        if not cls._profiles:
            cls.initialize_defaults()
        return list(cls._profiles.keys())
    
    @classmethod
    def save_to_file(cls, filepath: str):
        """Guarda todos los perfiles en un archivo JSON"""
        data = {name: profile.to_dict() for name, profile in cls._profiles.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Carga perfiles desde un archivo JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for name, profile_data in data.items():
            cls.add_profile(SteelProfile.from_dict(profile_data))


# =============================================================================
# BOBINA - Geometría + Referencia a Perfil
# =============================================================================

@dataclass
class Coil:
    """
    Bobina de acero - Geometría específica + referencia a un perfil de acero.
    
    Attributes:
        coil_id: Identificador único de la bobina
        profile_name: Nombre del perfil de acero a usar
        
        # Geometría
        outer_diameter: Diámetro exterior [m]
        inner_diameter: Diámetro interior [m]
        width: Ancho de la bobina [m]
        thickness: Espesor de la lámina [m]
        
        # Masa (opcional, se calcula si no se proporciona)
        mass: Masa de la bobina [kg]
    """
    coil_id: str
    profile_name: str
    
    # Geometría
    outer_diameter: float  # m
    inner_diameter: float  # m
    width: float  # m
    thickness: float  # m
    
    # Masa opcional
    mass: Optional[float] = None
    
    # Referencia al perfil (se asigna al validar)
    _profile: Optional[SteelProfile] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Valida y carga el perfil"""
        self.validate()
    
    def validate(self):
        """Valida la bobina y carga el perfil"""
        # Validar geometría
        if self.outer_diameter <= self.inner_diameter:
            raise ValueError(f"Diámetro exterior ({self.outer_diameter}) debe ser mayor que interior ({self.inner_diameter})")
        
        if self.thickness <= 0:
            raise ValueError(f"Espesor debe ser positivo: {self.thickness}")
        
        if self.width <= 0:
            raise ValueError(f"Ancho debe ser positivo: {self.width}")
        
        # Cargar perfil
        self._profile = SteelProfileLibrary.get_profile(self.profile_name)
        
        # Calcular masa si no se proporcionó
        if self.mass is None:
            self.mass = self.calculate_mass()
    
    @property
    def profile(self) -> SteelProfile:
        """Obtiene el perfil de acero"""
        if self._profile is None:
            self._profile = SteelProfileLibrary.get_profile(self.profile_name)
        return self._profile
    
    def calculate_mass(self) -> float:
        """
        Calcula la masa de la bobina basándose en geometría y densidad.
        
        Returns:
            Masa [kg]
        """
        r_outer = self.outer_diameter / 2
        r_inner = self.inner_diameter / 2
        
        # Volumen del anillo cilíndrico
        volume = np.pi * (r_outer**2 - r_inner**2) * self.width
        
        return volume * self.profile.density
    
    def calculate_length(self) -> float:
        """
        Calcula la longitud total de la lámina enrollada.
        
        Returns:
            Longitud [m]
        """
        r_outer = self.outer_diameter / 2
        r_inner = self.inner_diameter / 2
        
        # Área de sección transversal
        area = np.pi * (r_outer**2 - r_inner**2)
        
        # Longitud = área / espesor
        return area / self.thickness
    
    @property
    def radial_thickness(self) -> float:
        """Espesor radial de la bobina [m]"""
        return (self.outer_diameter - self.inner_diameter) / 2
    
    def to_dict(self) -> dict:
        """Convierte la bobina a diccionario"""
        return {
            'coil_id': self.coil_id,
            'profile_name': self.profile_name,
            'outer_diameter': self.outer_diameter,
            'inner_diameter': self.inner_diameter,
            'width': self.width,
            'thickness': self.thickness,
            'mass': self.mass
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Coil':
        """Crea una bobina desde un diccionario"""
        return cls(**data)
    
    def __str__(self):
        return (f"Coil('{self.coil_id}', profile='{self.profile_name}', "
                f"OD={self.outer_diameter*1000:.0f}mm, "
                f"width={self.width*1000:.0f}mm, "
                f"mass={self.mass:.0f}kg)")


# =============================================================================
# CONFIGURACIÓN DEL HORNO
# =============================================================================

@dataclass
class FurnaceStack:
    """
    Configuración del apilado de bobinas en el horno.
    
    Define qué bobinas van en cada posición (1# abajo, 4# arriba).
    """
    coils: List[Coil] = field(default_factory=list)
    
    # Esfuerzo compresivo entre capas [Pa]
    compressive_stress: float = 8e6  # 8 MPa típico
    
    def add_coil(self, coil: Coil, position: Optional[int] = None):
        """
        Agrega una bobina al stack.
        
        Args:
            coil: Bobina a agregar
            position: Posición (0=fondo). Si None, agrega al final.
        """
        if position is None:
            self.coils.append(coil)
        else:
            self.coils.insert(position, coil)
    
    def remove_coil(self, position: int) -> Coil:
        """Remueve y retorna la bobina en la posición dada"""
        return self.coils.pop(position)
    
    def get_coil(self, position: int) -> Coil:
        """Obtiene la bobina en la posición dada"""
        return self.coils[position]
    
    @property
    def num_coils(self) -> int:
        """Número de bobinas en el stack"""
        return len(self.coils)
    
    @property
    def total_mass(self) -> float:
        """Masa total del stack [kg]"""
        return sum(c.mass for c in self.coils)
    
    @property
    def total_height(self) -> float:
        """Altura total del stack [m]"""
        return sum(c.width for c in self.coils)
    
    def validate(self):
        """Valida la configuración del stack"""
        if len(self.coils) == 0:
            raise ValueError("El stack debe tener al menos una bobina")
        
        if len(self.coils) > 5:
            raise ValueError("Máximo 5 bobinas por stack en hornos típicos")
        
        # Validar que los diámetros sean compatibles
        od_max = max(c.outer_diameter for c in self.coils)
        od_min = min(c.outer_diameter for c in self.coils)
        if (od_max - od_min) > 0.3:  # 300mm de diferencia máxima
            print(f"Advertencia: Gran diferencia en diámetros ({od_min*1000:.0f}-{od_max*1000:.0f}mm)")
    
    def summary(self) -> str:
        """Genera un resumen del stack"""
        lines = ["=" * 60]
        lines.append("CONFIGURACIÓN DEL STACK DE BOBINAS")
        lines.append("=" * 60)
        
        for i, coil in enumerate(self.coils):
            pos = f"{i+1}#"
            lines.append(f"\nPosición {pos} (desde abajo):")
            lines.append(f"  ID: {coil.coil_id}")
            lines.append(f"  Perfil: {coil.profile_name}")
            lines.append(f"  Dimensiones: OD={coil.outer_diameter*1000:.0f}mm, "
                        f"ID={coil.inner_diameter*1000:.0f}mm, "
                        f"W={coil.width*1000:.0f}mm")
            lines.append(f"  Espesor lámina: {coil.thickness*1000:.2f}mm")
            lines.append(f"  Masa: {coil.mass:.0f} kg")
            lines.append(f"  Longitud: {coil.calculate_length():.0f} m")
        
        lines.append("\n" + "-" * 60)
        lines.append(f"Total bobinas: {self.num_coils}")
        lines.append(f"Masa total: {self.total_mass:.0f} kg")
        lines.append(f"Altura total: {self.total_height*1000:.0f} mm")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convierte el stack a diccionario"""
        return {
            'coils': [c.to_dict() for c in self.coils],
            'compressive_stress': self.compressive_stress
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FurnaceStack':
        """Crea un stack desde un diccionario"""
        stack = cls(compressive_stress=data.get('compressive_stress', 8e6))
        for coil_data in data['coils']:
            stack.add_coil(Coil.from_dict(coil_data))
        return stack
    
    def save_to_file(self, filepath: str):
        """Guarda la configuración en un archivo JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FurnaceStack':
        """Carga la configuración desde un archivo JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# FUNCIONES DE AYUDA
# =============================================================================

def create_quick_coil(
    coil_id: str,
    profile: str,
    outer_diameter_mm: float,
    inner_diameter_mm: float = 600,
    width_mm: float = 1200,
    thickness_mm: float = 1.5,
    mass_kg: Optional[float] = None
) -> Coil:
    """
    Función de conveniencia para crear bobinas rápidamente.
    
    Args:
        coil_id: Identificador de la bobina
        profile: Nombre del perfil de acero
        outer_diameter_mm: Diámetro exterior [mm]
        inner_diameter_mm: Diámetro interior [mm] (default 600)
        width_mm: Ancho [mm] (default 1200)
        thickness_mm: Espesor de lámina [mm] (default 1.5)
        mass_kg: Masa [kg] (opcional, se calcula si no se da)
    
    Returns:
        Coil configurada
    """
    return Coil(
        coil_id=coil_id,
        profile_name=profile,
        outer_diameter=outer_diameter_mm / 1000,
        inner_diameter=inner_diameter_mm / 1000,
        width=width_mm / 1000,
        thickness=thickness_mm / 1000,
        mass=mass_kg
    )


# =============================================================================
# DEMO Y PRUEBAS
# =============================================================================

if __name__ == "__main__":
    # Inicializar biblioteca
    SteelProfileLibrary.initialize_defaults()
    
    print("PERFILES DE ACERO DISPONIBLES:")
    print("-" * 40)
    for name in SteelProfileLibrary.list_profiles():
        profile = SteelProfileLibrary.get_profile(name)
        print(f"  {name}: {profile.description}")
    
    print("\n" + "=" * 60)
    print("EJEMPLO: Crear configuración de bobinas")
    print("=" * 60)
    
    # Crear bobinas con diferentes perfiles
    coil1 = create_quick_coil("L179046800T", "SPCC", 1869, width_mm=1272, thickness_mm=1.50)
    coil2 = create_quick_coil("L179038800T", "SPCC", 1867, width_mm=1250, thickness_mm=1.50)
    coil3 = create_quick_coil("L179046600T", "DC01", 1837, width_mm=1272, thickness_mm=1.48)
    coil4 = create_quick_coil("L179046700T", "DC04", 1761, width_mm=1272, thickness_mm=1.50)
    
    # Crear stack
    stack = FurnaceStack()
    stack.add_coil(coil1)  # 1# (fondo)
    stack.add_coil(coil2)  # 2#
    stack.add_coil(coil3)  # 3#
    stack.add_coil(coil4)  # 4# (arriba)
    
    # Mostrar resumen
    print(stack.summary())
    
    # Guardar configuración
    stack.save_to_file("/tmp/stack_config.json")
    print("\nConfiguración guardada en: /tmp/stack_config.json")
    
    # Mostrar propiedades térmicas a diferentes temperaturas
    print("\n" + "=" * 60)
    print("PROPIEDADES TÉRMICAS vs TEMPERATURA")
    print("=" * 60)
    
    profile = SteelProfileLibrary.get_profile("SPCC")
    print(f"\nPerfil: {profile.name}")
    print(f"{'T [°C]':>10} {'λ [W/(m·K)]':>15} {'cp [J/(kg·K)]':>15}")
    print("-" * 42)
    for T_celsius in [20, 100, 200, 400, 600, 700]:
        T_kelvin = T_celsius + 273.15
        lambda_val = profile.get_thermal_conductivity(T_kelvin)
        cp_val = profile.get_specific_heat(T_kelvin)
        print(f"{T_celsius:>10} {lambda_val:>15.1f} {cp_val:>15.1f}")
