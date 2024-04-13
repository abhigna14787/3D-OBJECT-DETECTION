# pip install mediapipe
# pip install open3d
# pip install opencv-contrib-python

# cup model
# shoe model
# camera model
# chair model

# Import the required modules

import cv2
import mediapipe as mp
import open3d as o3d # Import the open3d library

# Load the image or video
image = cv2.imread('download1.jpg')

# Initialize the objectron solution
mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.99, model_name='shoe')

# Initialize the drawing utils
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)


# video = cv2.VideoCapture('shoe.mp4')

# Load the Lidar data
lidar_data = 'lidar.pcd' # Change to your Lidar data file path
point_cloud = o3d.io.read_point_cloud(lidar_data) # Read the point cloud from the file
point_cloud.paint_uniform_color([0.5, 0.5, 0.5]) # Assign a gray color to the point cloud


# Convert the image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flip the image horizontally for a selfie-view display
image = cv2.flip(image, 1)

# Perform object detection
results = objectron.process(image)

# Create a list of geometries to display
geometries = [point_cloud] # Add the point cloud to the list

# Draw the 3D bounding boxes and graphical representation on the image
if results.detected_objects:
    for detected_object in results.detected_objects:
        mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS, drawing_spec, drawing_spec)
        mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
        # Create a 3D bounding box from the detected object's parameters
        bbox = o3d.geometry.OrientedBoundingBox(center=detected_object.translation, R=detected_object.rotation, extent=detected_object.scale)
        bbox.color = (0, 1, 0) # Assign a green color to the 3D bounding box
        geometries.append(bbox) # Add the 3D bounding box to the list

# Convert the image back to BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display the image or video
cv2.imshow('MediaPipe Objectron', image)
# if cv2.waitKey(5) & 0xFF == 27:
#     breakzzzzz

# Display the point cloud and the 3D bounding boxes
o3d.visualization.draw_geometries(geometries)

# Release the resources
# video.release()
cv2.destroyAllWindows()
