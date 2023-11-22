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


class LiveImapApp:
    LOAD_MESH = 1
    MENU_QUIT = 2
    CAPTURE = 3
    STOP_CAPTURE = 4

    def __init__(self,camera_config):
        self.uploader = ThreadPoolExecutor(max_workers=1) # forced to remain sequential
        self.frame = 0
        self.capturing = False
        self._id = 0
        self.window = gui.Application.instance.create_window(
            "Live IMAP Demo", camera_config["w"], (camera_config["h"] + 24))
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
                self.frame += 1
                time.sleep(1)
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

    def thread_post(data, frame):
        #Post data to server
        import socket
        # data is always np array
        # frame is always int
        # determine size of data
        
        # data serialized
        data = data.tobytes()

        # data size
        size = len(data)

        # create a header of 64 bytes and set it to the size of the data and frame number
        header = f"{size:<64}{frame:<64}".encode('utf-8')

        # create a socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("192.168.3.95", 34512)) # listening server
        s.send(header)
        s.send(data)
        s.close()

    def capture_rgbdT(self):
        pose = self.camera.get_model_matrix()
        self.uploader.submit(self.thread_post,np.asarray(pose), frame=self.frame)
        
        self.scene.scene.scene.render_to_image(self.rgb_callback)
        self.scene.scene.scene.render_to_depth_image(self.depth_callback)
        #RGB DEPTH AND POSE
        #Send To Server
        # print(self.rgb)
        # print(self.depth)
        # print(self.pose.shape)
        # self.rgb_lock.release()
        # self.depth_lock.release()

    def depth_callback(self,depth_image):
        depth = np.asarray(depth_image)
        print(self.frame,'done depth',self.depth.shape)
        self.uploader.submit(self.thread_post,depth, frame=self.frame)

    def rgb_callback(self,rgb_image):
        rgb = np.asarray(rgb_image)
        print(self.frame,'done rgb',self.rgb.shape)
        self.uploader.submit(self.thread_post,rgb, frame=self.frame)

def main():
    camera_config = {
        "w": 600,
        "h": 600,
        "fx": 350.0,
        "fy": 350.0,
        "cx": 299.5,
        "cy": 299.5,
        "mw": 0,
        "mh": 0
    }
    gui.Application.instance.initialize()
    LiveImapApp(camera_config)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()