from .core import Hexagon

def hex_add(a:Hexagon,b:Hexagon) -> Hexagon:
    return Hexagon(a.coordinate + b.coordinate)