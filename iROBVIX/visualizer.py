import argparse
import os
import vis as v
import open3d as o3d
import pickle

meshes = []
meshes_paths = []
frame_id = -1
meshroot = ''
LEFT = 263
RIGHT = 262
SPACE = 32

def draw_frame(vis):
    global meshes, frame_id, meshes_paths
    print(f'Visualizing {os.path.basename(meshes_paths[frame_id])}')
    vis.clear_geometries()
    init_params = vis.get_view_control().convert_to_pinhole_camera_parameters()    
    vis.add_geometry(meshes[frame_id])
    vis.get_view_control().convert_from_pinhole_camera_parameters(init_params)
    vis.update_renderer()
    vis.poll_events()
    
def draw_next_frame(vis):
    global meshes, frame_id
    frame_id = (frame_id + 1)%len(meshes)
    draw_frame(vis)

def draw_prev_frame(vis):
    global meshes, frame_id
    frame_id = (frame_id + len(meshes) - 1)%len(meshes)
    draw_frame(vis)

def space_pressed(vis):
    print('Space pressed')
    global meshes, meshes_paths, meshroot
    all_meshes = sorted([os.path.join(meshroot, f) for f in os.listdir(meshroot) if f.endswith('.pkl')])
    
    # now only add the meshes that are not already in the list
    for mesh_path in all_meshes:
        if mesh_path not in meshes_paths:
            with open(mesh_path,'rb') as fp:
                print(f'Loading Mesh {os.path.basename(mesh_path)}. Please wait.')
                meshes.append(v.trimesh_to_open3d(pickle.load(fp)))
                meshes_paths.append(mesh_path)


def draw_frame_first_time():
    pass
    
def main():
    global meshes, meshes_paths, meshroot
    parser = argparse.ArgumentParser(description='Visualizer')
    parser.add_argument('--path',default='/raid/home/niranjan20090/IRob/vMAP/logs/iMAP/room0/scene_mesh',type=str)
    args = parser.parse_args()
    meshroot = args.path

    global meshes, meshes_paths
    meshes_paths = sorted([os.path.join(args.path,f) for f in os.listdir(args.path) if f.endswith('.pkl')])
    
    for mesh_path in meshes_paths:
        with open(mesh_path,'rb') as fp:
            meshes.append(v.trimesh_to_open3d(pickle.load(fp)))
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800,height=600)
    
    draw_frame_first_time()

    vis.register_key_callback(RIGHT, draw_next_frame)
    vis.register_key_callback(LEFT, draw_prev_frame)
    vis.register_key_callback(SPACE, space_pressed)

    vis.run()
    

if __name__ == "__main__":
    main()
