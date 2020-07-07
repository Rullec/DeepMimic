from shapevar.shapegen.mcmc_shape_gen import MCMCShapeGen
from shapevar.shapegen.uniform_shape_gen import UniformShapeGen
from shapevar.shapegen.grid_search_gen import GridSearchGen
from shapevar.shapegen.fixed_shape_gen import FixedShapeGen


def build_shape_generator(shape_type):
    if shape_type == MCMCShapeGen.NAME:
        return MCMCShapeGen
    elif shape_type == UniformShapeGen.NAME:
        return UniformShapeGen
    elif shape_type == GridSearchGen.NAME:
        return GridSearchGen
    elif shape_type == FixedShapeGen.NAME:
        return FixedShapeGen
    else:
        print("Please Set Correct Shape Generator Type")
        exit(-1)
