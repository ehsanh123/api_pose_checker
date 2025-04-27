from fastapi import FastAPI, Request,UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
#
import os
import json
#
import mediapipe as mp
import cv2
import numpy as np
import math

#
import base64
########################### === Setup server === ###########################

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
################angle detection and pose detection##################

ref_1_angle = None
ref_2_angle = None
#####################
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

    angle_degrees = angle_degrees-angle_degrees%1
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

def calculate_angles_landmarks(landmarks, mp_pose):
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
                "L_Elbow": left_elbow_angle,
                "L_Hip": left_hip_angle,
                "R_Elbow": right_elbow_angle,
                "R_Hip": right_hip_angle,
                "L_Arm": left_arm_angle,
                "R_Arm": right_arm_angle,
                "L_Back": left_back_angle,
                "R_Back": right_back_angle,
                "L_Knee": left_knee_angle,
                "R_Knee": right_knee_angle
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

# def get_locs_selected_angles(selected_angles, mp_pose):
#     angle_to_points = {
#     "L_Elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
#     "L_Hip": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],

#     "R_Elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
#     "R_Hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],

#     "L_Knee": [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HIP],
#     "R_Knee": [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HIP],

#     "L_Arm": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE],
#     "R_Arm": [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE],

#     "L_Back": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.NOSE],
#     "R_Back": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.NOSE]
#     }
#     needed_points = set()
    
#     for angle in selected_angles:
#         points = angle_to_points.get(angle, [])
#         needed_points.update(points)  # avoid duplicates
    
#     return list(needed_points)

def putItems_on_image(image1,angs,locs,selected_angles,mp_pose,results):
    image = image1
       
    decoded_angles = decode_selected_angles(selected_angles)
        
    # locations = get_locs_selected_angles(decoded_angles, mp_pose)
    ang = angs.copy()
    for key in angs.keys():
        ang[key] = angs[key] if key in decoded_angles else 0

        ###########
    angle_to_points = {
            "L_Elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            "L_Hip": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],

            "R_Elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
            "R_Hip": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],

            "L_Knee": [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP],
            "R_Knee": [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP],

            "L_Arm": [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE],
            "R_Arm": [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE],

            "L_Back": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE],
            "R_Back": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE]
    }
        
    for angle_name, value in ang.items():
        if value != 0:
            if angle_name in angle_to_points:
                points = angle_to_points[angle_name]
            
                outer1 = points[0]
                middle = points[1]
                outer2 = points[2]
            
                # Get pixel locations
                h, w, _ = image.shape
                landmark_outer1 = results.pose_landmarks.landmark[outer1]
                landmark_middle = results.pose_landmarks.landmark[middle]
                landmark_outer2 = results.pose_landmarks.landmark[outer2]
            
                cx1, cy1 = int(landmark_outer1.x * w), int(landmark_outer1.y * h)
                cxm, cym = int(landmark_middle.x * w), int(landmark_middle.y * h)
                cx2, cy2 = int(landmark_outer2.x * w), int(landmark_outer2.y * h)
            
                # Draw circles
                cv2.circle(image, (cx1, cy1), 3, (0, 0, 255), cv2.FILLED)  # outer1
                cv2.circle(image, (cxm, cym), 3, (255, 0, 0), cv2.FILLED)  # middle
                cv2.circle(image, (cx2, cy2), 3, (0, 0, 255), cv2.FILLED)  # outer2
            
                # Draw lines
                cv2.line(image, (cx1, cy1), (cxm, cym), (0, 255, 0), thickness=1)
                cv2.line(image, (cx2, cy2), (cxm, cym), (0, 255, 0), thickness=1)

    return image

# Function to calculate the angles for both left and right sides and overlay on image
def calculate_and_display_angles(image1,ang_select, return_angles=False):
    # pose = mp_pose.Pose(
        # min_detection_confidence=0.5, min_tracking_confidence=0.5,
        # static_image_mode=False, model_complexity=0,
        # enable_segmentation=False)

    # if return_angles == False:
        
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)
        
    image = image1#.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        ang , locs = calculate_angles_landmarks(results.pose_landmarks , mp_pose)
        if return_angles:
            return ang
        
        image = putItems_on_image(image , ang , locs , ang_select , mp_pose , results)

        return image ,ang
    else:
        return image , None

##################pose chek
def decode_selected_angles(binary_value):
    angle_mapping = {
        "L_Elbow": 1,    # 2^0
        "L_Hip": 2,      # 2^1
        "R_Elbow": 4,    # 2^2
        "R_Hip": 8,      # 2^3
        "L_Arm": 16,     # 2^4
        "R_Arm": 32,     # 2^5
        "L_Back": 64,    # 2^6
        "R_Back": 128,   # 2^7
        "L_Knee": 256,   # 2^8
        "R_Knee": 512    # 2^9
    }

    selected_angles = []

    for angle, value in angle_mapping.items():
        if binary_value & value:  # Bitwise AND: checks if this bit is on
            selected_angles.append(angle)

    return selected_angles

curent_angle = None
def compare_angles(angles3, angles_s):
    global ref_1_angle, ref_2_angle, curent_angle

    new_key = "Compared to"
    new_value = "ref image 1"

    if curent_angle is None or curent_angle == ref_1_angle:
            curent_angle = ref_1_angle
    if curent_angle is None or curent_angle == ref_2_angle:
            new_value = "ref image 2"
            curent_angle = ref_2_angle
    if curent_angle is None:
            new_value = "None"
        
    angles2 = curent_angle.copy()
    angles1 = angles3.copy()
    selected_angles = decode_selected_angles(angles_s)

    if angles1 is None or angles2 is None:
        return angles1[selected_angles]
    
    

    angle_diff = {}
    b= 0
    for key in selected_angles:
        if key in angles1 and key in angles2:
            b+=1
            angle_diff[key] = abs(angles1[key] - angles2[key])
        else:
            angle_diff[key] = None  # Missing in one of the sets

    a = sum(angle_diff.values())
    if a< 10*b:
        if curent_angle == ref_1_angle:
            new_value = "Changed"
            curent_angle = ref_2_angle
        else:
            new_value = "Changed"
            curent_angle = ref_1_angle

    items = list(angle_diff.items())
    items.insert(0, (new_key, new_value))
    angles = dict(items)

    return angles

@app.post("/upload-photo")
async def Pose_detector(request: Request):
    data = await request.json()
    image_base64 = data["image"].split(",")[1]  # strip the header

    ref_name = data["name"]
    ang_select = data["angle"]

    # Decode base64 to image bytes
    img_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # return JSONResponse(content={"processed_image": None , "angles": {'sda':ang_select}})

    image2, ang = None , None#calculate_and_display_angles(image, return_angles=False)

    #caculate and copmare angles
    if ref_name == "1" or ref_name == "2":
        image2, ang = calculate_and_display_angles(image,ang_select, return_angles=False)    
    global ref_1_angle, ref_2_angle,curent_angle

    if ref_name == "1":
        ref_1_angle = ang
    elif ref_name == "2":
        ref_2_angle = ang

    elif ref_name == "3":
        ang = calculate_and_display_angles(image,ang_select , return_angles=True)
        ang = compare_angles(ang , ang_select)
        
        return JSONResponse(content={"processed_image": None , "angles": ang})
    else:
        image2, ang = calculate_and_display_angles(image,ang_select, return_angles=False) 
        ang = compare_angles(ang , ang_select)
            

    
    
    _, buffer = cv2.imencode(".png", image2)
    processed_base64 = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/png;base64,{processed_base64}"

    return JSONResponse(content={"processed_image": data_url, "angles": ang})


###########basic API for testing ###########`
# `
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
