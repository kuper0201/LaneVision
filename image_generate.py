import carla
import random
import os
import numpy as np
import cv2
import time

def save_segmentation_image(image, image_id, output_dir):
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_data = np.reshape(image_data, (image.height, image.width, 4))
    segmentation_mask = image_data[:, :, 2]

    # Generate blank black image
    result_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    
    result_image[0, :, :] = 1 # Background
    result_image[segmentation_mask == 1] = [0, 0, 1] # Road
    result_image[segmentation_mask == 24] = [0, 1, 0] # Lane

    # Save the image
    filename = os.path.join(output_dir, f"segmentation_{image_id:05d}.npy")
    np.save(filename, result_image)
    
    #filename_png = os.path.join(output_dir, f"segmentation_{image_id:05d}.png")
    #cv2.imwrite(filename_png, result_image)
    
def save_rgb_image(image, image_id, output_dir):
    # Convert Carla image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
    array = array[:, :, :3]  # Drop the alpha channel

    # Convert to BGR for OpenCV
    array = array[:, :, ::-1]

    # Save the image
    filename = os.path.join(output_dir, f"rgb_{image_id:05d}.png")
    cv2.imwrite(filename, array)

def main():
    # Connect to Carla server
    client = carla.Client('192.168.0.10', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town03')
    
    # Set up the output directory
    seg_output_dir = 'segmentation_images'
    rgb_output_dir = 'rgb_images'
    os.makedirs(seg_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    # Load the default blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn a vehicle at a random location
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Attach a segmentation camera to the vehicle
    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '160')
    camera_bp.set_attribute('image_size_y', '80')
    camera_bp.set_attribute('fov', '45')
    
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '160')
    rgb_bp.set_attribute('image_size_y', '80')
    rgb_bp.set_attribute('fov', '45')

    # Camera position and rotation
    camera_transform = carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    rgb_cam = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)

    # Enable autopilot
    vehicle.set_autopilot(True)
    # Ignore traffic lights
    traffic_manager = client.get_trafficmanager()
    traffic_manager.ignore_lights_percentage(vehicle, 100.0)

    # Store the images
    seg_id, rgb_id = 0, 0
    image_count = seg_id + 5000

    # Set up the callback to save images
    def save_segmentation(image):
        nonlocal seg_id
        if seg_id < image_count:
            save_segmentation_image(image, seg_id, seg_output_dir)
            seg_id += 1

    def save_rgb(image):
        nonlocal rgb_id
        if rgb_id < image_count:
            save_rgb_image(image, rgb_id, rgb_output_dir)
            rgb_id += 1

    camera.listen(lambda image: save_segmentation(image))
    rgb_cam.listen(lambda image: save_rgb(image))

    try:
        while seg_id < image_count:
            world.tick()

    finally:
        print(f"Saved {seg_id} images to '{seg_output_dir}' and '{rgb_output_dir}'.")

        # Clean up
        camera.stop()
        rgb_cam.stop()
        vehicle.destroy()

if __name__ == '__main__':
    main()
