import open3d as o3d
import numpy as np

LEFT = 263
RIGHT = 262
SPACE = 32

frame_id = 0
params = o3d.camera.PinholeCameraParameters()
params.intrinsic.set_intrinsics(600,600,350,350,299.5,299.5)
print(params.intrinsic.intrinsic_matrix)
params.extrinsic = np.eye(4)

def get_pose(vis):
    print(np.asarray(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic))

def draw_frame(vis):
    global traj,frame_id,params
    print('Drawing',frame_id)
    ext = np.linalg.inv(traj[frame_id])
    params.extrinsic = ext
    vis.get_view_control().convert_from_pinhole_camera_parameters(params)
    print(get_pose(vis))

def draw_next_frame(vis):
    global traj,frame_id
    frame_id = (frame_id + 1)%traj.shape[0]
    draw_frame(vis)


def draw_prev_frame(vis):
    global traj,frame_id
    frame_id = (frame_id + traj.shape[0] - 1)%traj.shape[0]
    draw_frame(vis)


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=600,height=600)
t = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic

traj = np.loadtxt('traj_w_c.txt').reshape(-1,4,4)
mesh = o3d.io.read_triangle_mesh('mesh.ply')
vis.add_geometry(mesh)

draw_frame(vis)

# vis.register_key_callback(RIGHT, draw_next_frame)
# vis.register_key_callback(LEFT, draw_prev_frame)
# vis.register_key_callback(SPACE, get_pose)
vis.run()
