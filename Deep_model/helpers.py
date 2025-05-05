
import numpy as np

import matplotlib.pyplot as plt

import gempy as gp
import gempy_viewer as gpv


def create_initial_gempy_model(refinement, filename='prior_model.png', save=True):
    """ Create an initial gempy model objet

    Args:
        refinement (int): Refinement of grid
        save (bool, optional): Whether you want to save the image

    """
    geo_model_test = gp.create_geomodel(
    project_name='Gempy_abc_Test',  
    extent=[0, 1, -0.1, 0.1, 0, 1], 
    resolution=[100,10,100],             
    refinement=refinement,
    structural_frame= gp.data.StructuralFrame.initialize_default_structure()
    )
   
    brk1 = 0.25
    brk2 = 0.5
    brk3 = 0.75
    
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[0.0, 0.25, 0.5, 0.75, 1.0],
        y=[0.0, 0.0, 0.0, 0.0, 0.0],
        z=[brk1, brk1 , brk1, brk1, brk1],
        elements_names=['surface1','surface1', 'surface1', 'surface1','surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[0.5],
        y=[0.0],
        z=[0.0],
        elements_names=['surface1'],
        pole_vector=[[0, 0, 0.5]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)

    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=[0.0, 0.25, 0.5, 0.75, 1.0],
            y=[0.0, 0.0, 0.0, 0.0, 0.0],
            z=np.array([brk2, brk2, brk2, brk2, brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    geo_model_test.structural_frame.structural_groups[0].append_element(element2)
    
    element3 = gp.data.StructuralElement(
        name='surface3',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=[0.0, 0.25, 0.5, 0.75, 1.0],
            y=[0.0, 0.0, 0.0, 0.0, 0.0],
            z=np.array([brk3, brk3, brk3, brk3, brk3]),
            names='surface3'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    geo_model_test.structural_frame.structural_groups[0].append_element(element3)
    
    gp.add_orientations(
        geo_model=geo_model_test,
        x=[0.5],
        y=[0.0],
        z=[1.0],
        elements_names=['surface3'],
        pole_vector=[[0, 0, 0.5]]
    )
    
    num_elements = len(geo_model_test.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test.structural_frame.structural_groups[0].elements[i], geo_model_test.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test.structural_frame.structural_groups[0].elements[i + 1], geo_model_test.structural_frame.structural_groups[0].elements[i]
    # geo_model_test.structural_frame.structural_groups[0].elements[0], geo_model_test.structural_frame.structural_groups[0].elements[1] = \
    # geo_model_test.structural_frame.structural_groups[0].elements[1], geo_model_test.structural_frame.structural_groups[0].elements[0]

    gp.compute_model(geo_model_test)
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    # if False:
    #     plt.show()
    # if save:
    #     plt.savefig(filename)
    
    return geo_model_test