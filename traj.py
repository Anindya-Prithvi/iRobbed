import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import signal
import threading
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
import socket
import trimesh
import cv2
DEBUG = False

class LiveImapApp:
    LOAD_MESH = 1
    MENU_QUIT = 2
    CAPTURE = 3
    STOP_CAPTURE = 4

    def __init__(self,camera_config):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if not DEBUG:
            self.sock.connect(("192.168.3.95", 34512)) # listening server
        self.traj = np.loadtxt('traj_w_c.txt').reshape(-1,4,4)
        self.uploader = ThreadPoolExecutor(max_workers=1) # forced to remain sequential
        self.frame = 0
        self.capturing = False
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "Live IMAP Demo", camera_config["w"], (camera_config["h"] + 28))
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                                   [10, 10, 10])
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_config["w"],camera_config["h"],camera_config["fx"],
                                                      camera_config["fy"],camera_config["cx"],camera_config["cy"])
        
        self.scene.setup_camera(self.intrinsic, np.eye(4), bbox)
        self.scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self.window.add_child(self.scene)

        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Load Mesh", LiveImapApp.LOAD_MESH)
            file_menu.add_separator()
            file_menu.add_item("Quit", LiveImapApp.MENU_QUIT)

            capture_menu = gui.Menu()
            capture_menu.add_item("Start", LiveImapApp.CAPTURE)
            capture_menu.add_item("Stop", LiveImapApp.STOP_CAPTURE)

            self.menu = gui.Menu()
            self.menu.add_menu("File", file_menu)
            self.menu.add_menu("Capture", capture_menu)
            gui.Application.instance.menubar = self.menu
            self.menu.set_enabled(LiveImapApp.CAPTURE,False)
            self.menu.set_enabled(LiveImapApp.STOP_CAPTURE,False)

        self.window.set_on_menu_item_activated(LiveImapApp.CAPTURE,self._on_start_capture)
        self.window.set_on_menu_item_activated(LiveImapApp.STOP_CAPTURE,self._on_stop_capture)
        self.window.set_on_menu_item_activated(LiveImapApp.LOAD_MESH,self._on_menu_load_mesh)
        self.window.set_on_menu_item_activated(LiveImapApp.MENU_QUIT,self._on_menu_quit)
        self.window.set_on_close(self._on_menu_quit)
    
    def _on_start_capture(self):
        self.menu.set_enabled(LiveImapApp.CAPTURE,False)
        self.menu.set_enabled(LiveImapApp.STOP_CAPTURE,True)
        print("[debug] Start Capturing")
        self.capturing = True
        def thread_capture():
            while self.capturing:
                gui.Application.instance.post_to_main_thread(
                    self.window, self.capture_rgbdT)
                
                #Sending to Server
                time.sleep(2.5)
                self.frame += 1
            print("[debug] Stop Capturing")
        threading.Thread(target=thread_capture).start()
    
    def _on_stop_capture(self):
        self.menu.set_enabled(LiveImapApp.CAPTURE,True)
        self.menu.set_enabled(LiveImapApp.STOP_CAPTURE,False)
        self.capturing = False

    def _on_menu_load_mesh(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Load .ply file", self.window.theme)
        dlg.add_filter(".ply","Triangle mesh files (.ply)")
        
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)
    
    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load_mesh(filename)

    def _on_menu_quit(self):
        print('[debug] Quitting...')
        gui.Application.instance.quit()
        current_pid = os.getpid()
        os.kill(current_pid, signal.SIGKILL)

    def load_mesh(self,path):
        self.scene.scene.clear_geometry()
        self.mesh = o3d.io.read_triangle_model(path)
        self.scene.scene.add_model("__model__", self.mesh)
        bbox = self.scene.scene.bounding_box
        self.scene.setup_camera(self.intrinsic, np.eye(4), bbox)
        self.camera = self.scene.scene.view.get_camera()
        self.menu.set_enabled(LiveImapApp.CAPTURE,True)

    def thread_post(self, data, frame, type):
        print('[socket] Posting...',frame,data.shape)
        #Post data to server
        # data is always np array
        # frame is always int
        # determine size of data
        
        # data serialized
        data = data.tobytes()

        # data size
        size = len(data)

        # create a header of 64 bytes and set it to the size of the data and frame number
        header = f"{size:<64}{frame:<63}{type:<1}".encode('utf-8')

        # create a socket
        self.sock.sendall(header)
        self.sock.sendall(data)
    

    def opencv_to_opengl_camera(self,transform=None):
        if transform is None:
            transform = np.eye(4)
        return transform @ trimesh.transformations.rotation_matrix(
            np.deg2rad(180), [1, 0, 0]
        )

    def opengl_to_opencv_camera(self,transform=None):
        if transform is None:
            transform = np.eye(4)
        return transform @ trimesh.transformations.rotation_matrix(
            np.deg2rad(-180), [1, 0, 0]
        )

    def get_pose(self):
        pose = np.asarray(self.camera.get_model_matrix())
        # T = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        # return np.linalg.inv(pose)
        return (pose @ self.opengl_to_opencv_camera()).astype(np.float32)

    def set_camera(self,M):
        #center is lookat_
        #eye is eye_
        #up is up
        up = -M[1, 0:3][:, np.newaxis]
        eye = (np.linalg.inv(M[:3,:3]) @ -M[:3,3])[:, np.newaxis]
        front = -M[2, 0:3][:, np.newaxis]
        center = eye - front
        return center, eye, up
    

    def capture_rgbdT(self):
        # pose = np.linalg.inv(np.asarray(self.camera.get_model_matrix()))
        print(self.frame)
        c,e,u = self.set_camera(np.linalg.inv(self.traj[self.frame]))
        self.camera.look_at(c,e,u)
        pose = self.get_pose()
        if not DEBUG:
            t = self.uploader.submit(self.thread_post, pose, self.frame,2)
            if t.result is not None:
                print(t.result())
        
        self.scene.scene.scene.render_to_image(self.rgb_callback)
        self.scene.scene.scene.render_to_depth_image(self.depth_callback)

    def depth_callback(self,depth_image):
        depth = (np.asarray(depth_image)).astype(np.float32)
        print(np.min(depth),np.max(depth),np.mean(depth),np.std(depth))
        z_near = self.camera.get_near()
        z_far = self.camera.get_far()
        print(z_near,z_far)
        depth_1 = depth != 1
        depth[depth_1] = 2.0*z_near*z_far/(z_far + z_near - (2.0 * (depth[depth_1] - 1.0))*(z_far - z_near))
        depth = np.clip(np.round(5*depth),0.0,65535.0)
        if DEBUG:
            print(np.min(depth),np.max(depth),np.mean(depth),np.std(depth))
            print(depth)
            cv2.imwrite('pred_3.png',depth.astype(np.uint16))
            # plt.imshow(depth,cmap='gray')
            # plt.show()
        print(self.frame,'done depth',depth.shape)
        if not DEBUG:
            self.uploader.submit(self.thread_post, depth, self.frame, 1)


    def rgb_callback(self,rgb_image):
        rgb = np.asarray(rgb_image)
        print(self.frame,'done rgb',rgb.shape)
        if not DEBUG:
            self.uploader.submit(self.thread_post, rgb, self.frame, 0)

def main():

    camera_config = {
        "w": 1200,
        "h": 680,
        "fx": 600.0,
        "fy": 600.0,
        "cx": 599.5,
        "cy": 339.5,
        "mw": 0,
        "mh": 0
    }
    gui.Application.instance.initialize()
    LiveImapApp(camera_config)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()