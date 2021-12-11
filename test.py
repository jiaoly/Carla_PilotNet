import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.image import resize
import model as md

# Remove warnings
#  tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load pretrained model
model = md.getPilotNetModel()
model.load_weights('model/model-weights.h5')

# Constants
actor_list = []
W, H = 320, 180
SHOW_CAM = True
front_camera = None

# TODO: predict steering angle using trained model
def predict_steering(img_rgb, pretrained = False):
    input_img = img_rgb
    if pretrained:
        input_img = resize(input_img, (66, 200))
    model = load_model('model/baseline3.h5') # the model input shape is (180,320,3)
    input_img = expand_dims(input_img, 0)  # Create batch axis
    steering_pred= model.predict(input_img)[0][0]
    return steering_pred

# TODO: call_back function for RGB image
def action(image):
    print("Image shape (bgr): ", image.shape)
    angle = predict_steering(image)

    # angle /= 70
    if angle >= 1.0:
        angle = 1.0
    elif angle <= -1.0:
        angle = -1.0
    else:
        angle = angle
    print("Steering angle: ", angle)
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=float(angle)))

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((H, W, 4)) # RGBA
    i3 = i2[:, :, :3]  # RGBA -> RGB
    cv2.imshow('image', i3)
    cv2.waitKey(1)
    #  img_norm = i3/255.0
    #  action(img_norm)
    #  path = '_out1/'  + str(image.frame) + '.jpg'
    #  cv2.imwrite(path, i3)

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # seconds

    world = client.get_world()
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = 0.05
    # settings.no_rendering_mode = False
    # world.apply_settings(settings)
    world.unload_map_layer(carla.MapLayer.All)

    blueprint_library = world.get_blueprint_library()

    # bp = random.choice(blueprint_library.filter('vehicle'))
    bp = blueprint_library.filter('model3')[0]
    transform = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, transform)
    # vehicle.set_autopilot(True)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)

    # Find the blueprint of the sensor.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', str(W))
    camera_bp.set_attribute('image_size_y', str(H))
    camera_bp.set_attribute('fov', '40')

    # Set the time in seconds between sensor captures
    camera_bp.set_attribute('sensor_tick', '1.0')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    actor_list.append(camera)
    print('created %s' % camera.type_id)

    cc = carla.ColorConverter.Raw
    camera.listen(lambda image: process_img(image))
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))

    time.sleep(100)
finally:

    print('destroying actors')
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('done.')
