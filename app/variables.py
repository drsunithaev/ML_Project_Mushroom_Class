from pydantic import BaseModel

class MushroomVariables(BaseModel):
    cap_shape: int
    cap_surface: int
    cap_color: int
    bruises : int
    odor: int
    gill_attachment: int
    gill_spacing: int
    gill_size: int
    gill_color: int
    stalk_shape: int
    stalk_root: int
    stalk_surface_above_ring: int
    stalk_surface_below_ring: int
    stalk_color_above_ring: int
    stalk_color_below_ring: int
    veil_type: int
    veil_color: int
    ring_number: int
    ring_type: int
    spore_print_color: int
    population: int
    habitat: int