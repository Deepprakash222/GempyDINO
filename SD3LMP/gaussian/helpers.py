
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
   
    brk1 = 0.3
    brk2 = 0.7
    grad = 1.0
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[0.1, 0.5, 0.9],
        y=[0.0, 0.0, 0.0],
        z=[brk1 , brk1 , brk1],
        elements_names=['surface1', 'surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-0.1, 1.1],
        y=[0.0, 0.0],
        z=[brk1, brk1],
        elements_names=['surface1', 'surface1'],
        pole_vector=[[0, 0, grad], [0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)


    
    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=[ 0.1, 0.5, 0.9],
            y=[ 0.0, 0.0, 0.0],
            z=np.array([brk2, brk2, brk2]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    geo_model_test.structural_frame.structural_groups[0].append_element(element2)
    
    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-0.1, 1.1],
        y=[0.0, 0.0],
        z=[brk2, brk2],
        elements_names=['surface2', 'surface2'],
        pole_vector=[[0, 0, grad], [0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    num_elements = len(geo_model_test.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test.structural_frame.structural_groups[0].elements[i], geo_model_test.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test.structural_frame.structural_groups[0].elements[i + 1], geo_model_test.structural_frame.structural_groups[0].elements[i]
    
    gp.compute_model(geo_model_test)
    
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if False:
        plt.show()
    if save:
        plt.savefig(filename)
    
    return geo_model_test

def create_final_gempy_model(posterior_data, refinement=7,  filename='posterior_model.png', save=True):
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
    
    
    
    brk1 = 0.3
    brk2 = 0.7
    grad = 1.0
    
    gp.add_surface_points(
        geo_model=geo_model_test,
        x=[0.1, 0.5, 0.9],
        y=[0.0, 0.0, 0.0],
        z=[posterior_data[3,2] ,posterior_data[4,2]  , posterior_data[5,2]],
        elements_names=['surface1', 'surface1', 'surface1']
    )

    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-0.1, 1.1],
        y=[0.0, 0.0],
        z=[brk1, brk1],
        elements_names=['surface1', 'surface1'],
        pole_vector=[[0, 0, grad], [0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)


    
    element2 = gp.data.StructuralElement(
        name='surface2',
        color=next(geo_model_test.structural_frame.color_generator),
        surface_points=gp.data.SurfacePointsTable.from_arrays(
            x=[ 0.1, 0.5, 0.9],
            y=[ 0.0, 0.0, 0.0],
            z=np.array([posterior_data[0,2] ,posterior_data[1,2]  , posterior_data[2,2]]),
            names='surface2'
        ),
        orientations=gp.data.OrientationsTable.initialize_empty()
    )
    
    
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    geo_model_test.structural_frame.structural_groups[0].append_element(element2)
    
    gp.add_orientations(
        geo_model=geo_model_test,
        x=[-0.1, 1.1],
        y=[0.0, 0.0],
        z=[brk2, brk2],
        elements_names=['surface2', 'surface2'],
        pole_vector=[[0, 0, grad], [0, 0, grad]]
    )
    geo_model_test.update_transform(gp.data.GlobalAnisotropy.NONE)
    
    num_elements = len(geo_model_test.structural_frame.structural_groups[0].elements) - 1  # Number of elements - 1 for zero-based index
    for swap_length in range(num_elements, 0, -1):  
        for i in range(swap_length):
            # Perform the swap for each pair (i, i+1)
            geo_model_test.structural_frame.structural_groups[0].elements[i], geo_model_test.structural_frame.structural_groups[0].elements[i + 1] = \
            geo_model_test.structural_frame.structural_groups[0].elements[i + 1], geo_model_test.structural_frame.structural_groups[0].elements[i]
    
    gp.compute_model(geo_model_test)
    
    picture_test = gpv.plot_2d(geo_model_test, cell_number=5, legend='force')
    if False:
        plt.show()
    if save:
        plt.savefig(filename)
    
    return geo_model_test