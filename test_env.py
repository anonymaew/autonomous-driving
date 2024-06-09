import glob
import math
import os
import sys
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from lib.resnet_model_levi_1 import Network
from lib.dataset import unnormalize, resnet_transforms
from lib.model import ResNet, ViTNet

# from carla.models import cnn_version1

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False


class CarEnv:
    STEER_AMT = 1.0
    SHOW_CAM = SHOW_PREVIEW
    SECONDS_PER_EPISODE = 60
    IMG_WIDTH = 200
    IMG_HEIGHT = 88
    front_camera = None

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        # self.world = self.client.load_world('Town10')
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_random_device_seed(0)
        self.blueprint_library = self.world.get_blueprint_library()
        self.car = self.blueprint_library.filter("model3")[0]
        self.models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']

        self.reset()

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # spawn vehicle agent
        self.transform = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(self.car, self.transform)
        self.actor_list.append(self.vehicle)
        self.world.debug.draw_string(self.transform.location, "Start Point", life_time=5)

        # spawn camera
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.IMG_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.IMG_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")
        cam_transform = carla.Transform(carla.Location(x=2.5, z=1))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_cam, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)
        self.rgb_sensor.listen(lambda data: self.process_img(data))
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(4)

        # spawn collision sensor
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, cam_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))
        time.sleep(2)

        # spawn traffic
        spawn_points = self.world.get_map().get_spawn_points()[1:]
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), life_time=2)

        blueprints = []
        for vehicle in self.world.get_blueprint_library().filter('*vehicle*'):
            if any(model in vehicle.id for model in self.models):
                blueprints.append(vehicle)

        # Set a max number of vehicles and prepare a list for those we spawn
        max_vehicles = 50
        max_vehicles = min([max_vehicles, len(spawn_points)])

        # Take a random sample of the spawn points and spawn some vehicles
        '''vehicles = []
        for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
            temp = self.world.try_spawn_actor(random.choice(blueprints), spawn_point)
            if temp is not None:
                temp.set_autopilot(True)
                self.actor_list.append(temp)
                vehicles.append(temp)
        '''
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.front_camera

    def cleanup(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        done = False
        # model
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        khm = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:
            self.episode_end = time.time()
            done = True
        elif khm < 50:
            done = False

        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            self.episode_end = time.time()
            done = True

        return self.front_camera, done, None

    def step_2(self, throttle, steer_amt, brake):
        done = False
        # model
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer_amt * self.STEER_AMT, brake=brake))

        v = self.vehicle.get_velocity()
        khm = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:
            self.episode_end = time.time()
            done = True
        #elif khm < 50:
        #    done = False

        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            self.episode_end = time.time()
            done = True

        return self.front_camera, done, None

    def step_3(self, target_speed, steer_amt):
        done = False

        v = self.vehicle.get_velocity()
        khm = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        throttle = 0.0
        brake = 0.0

        if target_speed > khm:
            throttle = 1
            brake = 0.0
        elif target_speed < khm:
            throttle = 0.0
            brake = 0.5
        # print(khm, throttle, brake)

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_amt * self.STEER_AMT))
        if len(self.collision_hist) != 0:
            self.episode_end = time.time()
            done = True
        #elif khm < 50:
        #    done = False

        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            self.episode_end = time.time()
            done = True

        return self.front_camera, done, None


if __name__ == "__main__":
    FPS = 60
    random.seed(10)
    np.random.seed(10)

    env = CarEnv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTNet()
    model.load('./best_vit.pth', device)
    model.to(device)
    model.eval()

    throttle = 0.0  # Set a constant throttle value
    brake = 0.0  # Set the brake value
    next_image, done, _ = env.step(1)
    while True:
        time.sleep(1 / FPS)
        with torch.no_grad():
            img = resnet_transforms()(next_image).to(device).unsqueeze(0)
            pred = model.model_input(img)
            pred = unnormalize(pred.cpu()[0])
        speed = round(float(pred[0]), 2)
        steer_amt = float(pred[1])

        print(f'speed: {speed} | steer: {steer_amt}')
        next_image, done, _ = env.step_3(speed, steer_amt)
        if done:
            break

    env.cleanup()
    print(f'Car drove for {env.episode_end - env.episode_start} sec')
