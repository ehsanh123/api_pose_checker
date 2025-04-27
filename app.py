from flask import Flask, request, send_file
from io import BytesIO
import mimetypes
import cv2
import numpy as np
import mediapipe as mp
import math

################

def calculate_angle(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    # Calculate vectors
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes (lengths) of the vectors
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # Calculate cosine of the angle
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Handle cases where points are the same
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # Ensure cos_theta is within the valid range [-1, 1] due to potential floating-point errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # Calculate the angle in radians and convert to degrees
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

# Function to extract the coordinates of a landmark
def get_landmark_coordinates(landmarks, landmark_id):
    try:
        x = landmarks.landmark[landmark_id].x
        y = landmarks.landmark[landmark_id].y
        x=x/1
        y=y/1
        return np.array([x, y])
    except:
        return np.array([0, 0])  # Return [0, 0] if landmark is not detected

################
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

###########################
def calculate_angles_landmarks(landmarks):
        # Left side angles
        left_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        #
        right_shoulder = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        #
        left_hip = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        #
        right_hip = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        right_knee = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        #
        left_ankle = get_landmark_coordinates(landmarks,mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coordinates(landmarks,mp_pose.PoseLandmark.RIGHT_ANKLE)
        #

        if np.any(left_shoulder) and np.any(left_elbow) and np.any(left_wrist):
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        else: left_elbow_angle = 0

        if np.any(left_shoulder) and np.any(left_hip) and np.any(left_knee):
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        else:  left_hip_angle = 0

        if np.any(right_shoulder) and np.any(right_elbow) and np.any(right_wrist):
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        else:   right_elbow_angle = 0

        if np.any(right_shoulder) and np.any(right_hip) and np.any(right_knee):
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        else:  right_hip_angle = 0

        if np.any(right_knee) and np.any(right_ankle) and np.any(right_hip):
            right_knee_angle = calculate_angle(right_knee, right_ankle, right_hip)
        else:   right_knee_angle = 0

        if np.any(left_knee) and np.any(left_ankle) and np.any(left_hip):
            left_knee_angle = calculate_angle(left_knee, left_ankle, left_hip)
        else:  left_knee_angle = 0


        # Additional angles (e.g., shoulders, knees, etc.)
        head0 = 0
        num = 0
        head = get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.NOSE)
        if not np.array_equal(head, head0): num += 1
        head0 = head

        head += get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_EYE)
        if (head[0]==head0[0]) & (head[1]==head0[1]): num += 1
        head0 = head

        head += get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_EYE)
        if (head[0]==head0[0]) & (head[1]==head0[1]): num += 1
        head0 = head

        head += get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.RIGHT_EAR)
        if (head[0]==head0[0]) & (head[1]==head0[1]): num += 1
        head0 = head

        head += get_landmark_coordinates(landmarks, mp_pose.PoseLandmark.LEFT_EAR)
        if (head[0]==head0[0]) & (head[1]==head0[1]): num += 1
        head0 = head

        head /= num

        if np.any(head) and np.any(left_shoulder) and np.any(left_elbow):
            left_arm_angle = calculate_angle(head, left_shoulder, left_elbow)
        else:
            left_arm_angle = 0

        if np.any(head) and np.any(right_shoulder) and np.any(right_elbow):
            right_arm_angle = calculate_angle(head, right_shoulder, right_elbow)
        else:
            right_arm_angle = 0

        if np.any(head) and np.any(left_shoulder) and np.any(left_hip):
            left_back_angle = calculate_angle(head, left_shoulder, left_hip)
        else:
            left_back_angle = 0

        if np.any(head) and np.any(right_shoulder) and np.any(right_hip):
            right_back_angle = calculate_angle(head, right_shoulder, right_hip)
        else:
            right_back_angle = 0
        ###############
        angles = {
                "left_elbow_angle": left_elbow_angle,
                "left_hip_angle": left_hip_angle,
                "right_elbow_angle": right_elbow_angle,
                "right_hip_angle": right_hip_angle,
                "left_arm_angle": left_arm_angle,
                "right_arm_angle": right_arm_angle,
                "left_back_angle": left_back_angle,
                "right_back_angle": right_back_angle,
                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle
        }

        locations={
            "left_shoulder":left_shoulder,
            "left_elbow":left_elbow,
            "left_wrist":left_wrist,
            "right_shoulder":right_shoulder,
            "right_elbow":right_elbow,
            "right_wrist":right_wrist,

            "head":head,

            "left_hip":left_hip,
            "left_knee":left_knee,
            "left_ankle":left_ankle,

            "right_hip":right_hip,
            "right_knee":right_knee,
            "right_ankle":right_ankle
        }
        return angles,locations

# from PIL import Image
# Function to calculate the angles for both left and right sides and overlay on image
def calculate_and_display_angles(image_path, return_angles=False):
    
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image_path.seek(0)  # rewind to the beginning
    file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        ang , locs = calculate_angles_landmarks(results.pose_landmarks)
        
        #if only angles needed
        if return_angles:
          return ang

        # Draw the pose landmarks on the image
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        
        # Display the angle values at corresponding locations
        def overlay_angle(angle, location, color=(0, 255, 0)):
            # Coordinates are normalized, scale them to the image size
            x = int(location[0] * image.shape[1])
            y = int(location[1] * image.shape[0])
            
            cv2.putText(image, f'{angle:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Overlay angles on the image
        if np.any(ang['left_elbow_angle']!=0):
            overlay_angle(ang['left_elbow_angle'], locs['left_elbow'], (0, 255, 0))  # Left elbow angle
        if np.any(ang['right_elbow_angle']!=0):
            overlay_angle(ang['right_elbow_angle'], locs['right_elbow'], (0, 255, 0))  # Right elbow angle
        if np.any(ang['left_hip_angle']!=0):
            overlay_angle(ang['left_hip_angle'], locs['left_hip'], (0, 255, 0))  # Left hip angle
        if np.any(ang['right_hip_angle']!=0):
            overlay_angle(ang['right_hip_angle'], locs['right_hip'], (0, 255, 0))  # Right hip angle
        if np.any(ang['left_arm_angle']!=0):
            overlay_angle(ang['left_arm_angle'], (locs['left_shoulder']
                                                   + locs['left_elbow'])/2, (0, 255, 0))  # Left arm angle
        if np.any(ang['right_arm_angle']!=0):
            overlay_angle(ang['right_arm_angle'], (locs['right_shoulder']
                                                   + locs['right_elbow'])/2, (0, 255, 0))
        if np.any(ang['left_back_angle']!=0):
            overlay_angle(ang['left_back_angle'], locs['left_shoulder'], (0, 255, 0))  # Left back angle
        if np.any(ang['right_back_angle']!=0):
            overlay_angle(ang['right_back_angle'], locs['right_shoulder'], (0, 255, 0))  # Right back angle

        if np.any(ang['left_knee_angle']!=0):
            overlay_angle(ang['left_knee_angle'], locs['left_knee'], (0, 255, 0))  # Left knee angle

        if np.any(ang['right_knee_angle']!=0):
            overlay_angle(ang['right_knee_angle'], locs['right_knee'], (0, 255, 0))  # Right knee angle

        # Return the image with the overlayed angles
        return image
    else:
        return None

#########################################
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app = Flask(__name__)

#############################################
@app.route('/')
def hello():
    return "This is my python code Adew"
    
from flask import Response
########################################
@app.route('/1', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return {"message": "No file part"}, 400
    file = request.files['image']
    if file.filename == '':
        return {"message": "No selected file"}, 400
    
    # send file with angles on image or just angles
    if file and allowed_file(file.filename):
        try:
            # file.seek(0)
            # mime_type, _ = mimetypes.guess_type(file.filename)
            
            f1 = calculate_and_display_angles(file)

            success, encoded_image = cv2.imencode('.jpg', f1)
            if not success:
                return "Image encoding failed", 500

            return Response(encoded_image.tobytes(), mimetype='image/jpeg')

            # return send_file(
            #     BytesIO(file.read()), mimetype=mime_type,
            #     as_attachment=False, download_name=file.filename
            # )
        
        except Exception as e:
            return {"message": f"Error processing file: {str(e)}"}, 500
    return {"message": "File not allowed"}, 400

#####################################
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)