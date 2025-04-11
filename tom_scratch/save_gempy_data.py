import os,sys
from datetime import datetime
import numpy as np

import gempy as gp
import gempy_engine
import gempy_viewer as gpv


from helpers import *
from generate_samples import *

def main():
    
    nodes = 256
    directory_path = "../Results/Nodes_"+str(nodes)
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    mesh_coordinates =np.load(directory_path +"/mesh_data.npy")
    
    filename = directory_path + "/data.json"
    data = generate_input_output_gempy_data(mesh_coordinates=mesh_coordinates, number_samples=10000,filename=filename)
    c,m_data, dmdc_data = np.array(data["input"]),np.array(data["Gempy_output"]), np.array(data["Jacobian_Gempy"])
    

if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()

    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")
