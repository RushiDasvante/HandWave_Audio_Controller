# Importing libraries

import math
import threading
import time
import warnings
from ctypes import POINTER, cast

import cv2
import mediapipe as mp
import numpy as np
from comtypes import CLSCTX_ALL
from playsound import playsound
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pygame

# Welcome message and program start information
print("                                     Welcome!...Program is starting in few seconds.......")

# Instructions for controlling the prāāogram
print('''\nHow to Control Program.......\n
      1.For adjusting Volume : Open Thumb and Index Finger\n
      2.For Volume set : Open any finger while adjusting volume\n
      3.For Exiting Program : Open All the fingers
      ''')

# Message indicating the demo song is playing
print("\n                                   ...........🎵 Playing  Demo  song 🎵............")

# Flag to indicate the program is running
a = True
# Initialize pygame mixer for controlling the song
pygame.mixer.init()

# Function to play the demo song
pygame.mixer.music.load(r"D:\Desktop\my work\Project_audio_controller\demo_song.mp3")

pygame.mixer.music.play(-1)  # Play the song in a loop (-1 means loop indefinitely)


# Ignore potential warning messages
warnings.filterwarnings("ignore")

# Mediapipe setup for hand detection
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Pycaw setup for audio volume control
dev = AudioUtilities.GetSpeakers()
iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
vol_ctl = cast(iface, POINTER(IAudioEndpointVolume))

# Get the range of possible volume levels
vol_range = vol_ctl.GetVolumeRange()
v_min, v_max = vol_range[0], vol_range[1]

# Camera setup for capturing video
w_cam, h_cam = 800, 600
cam = cv2.VideoCapture(0)
cam.set(3, w_cam)
cam.set(4, h_cam)

# Function to determine which fingers are open based on landmark positions
def c_f(lm_list):
    # Determine if the hand is left or right
    is_right = lm_list[4][1] < lm_list[3][1]
    # Thumb logic: for right hand, thumb is to the left of index finger MCP, vice versa for left hand
    thumb_open = (is_right and lm_list[4][1] < lm_list[3][1]) or (not is_right and lm_list[4][1] > lm_list[3][1])
    # Other fingers: compare y-coordinate of tip with pip
    fingers = [8, 12, 16, 20]
    open_fingers = [thumb_open]
    for i in range(1, 5):
        open_fingers.append(lm_list[fingers[i - 1]][2] < lm_list[fingers[i - 1] - 2][2])
    return open_fingers

# Initialize Mediapipe Hands object for hand detection
with mp_hands.Hands(
        model_complexity=1,  # Set the model complexity for hand detection
        min_detection_confidence=0.5, # Minimum confidence for hand detection
        min_tracking_confidence=0.5 # Minimum confidence for tracking existing hands
        ) as hands:
    # Start the main loop for video processing
    while cam.isOpened():
        # Read a frame from the camera
        success, img = cam.read()
        # Exit the loop if the camera is not working
        if not success:
            break
        # Convert the image from BGR to RGB for Mediapipe processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image with Mediapipe Hands
        results = hands.process(img)
        # Convert the image back to BGR for display
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Initialize an empty list to store hand landmarks
        lm_list = []
        # If any hands are detected
        if results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_lms in results.multi_hand_landmarks:
                # Draw the hand landmarks and connections on the image
                mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                # Loop through each landmark in the hand
                for i, lm in enumerate(hand_lms.landmark):
                    # Get the image dimensions
                    h, w, c = img.shape
                    # Calculate the x and y coordinates of the landmark
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Add the landmark information to the lm_list
                    lm_list.append([i, cx, cy])
            # If lm_list is not empty (hand landmarks are detected)
            if lm_list:
                # Calculate which fingers are open
                fingers_open = c_f(lm_list)
                # Check if all fingers are open (exit condition)
                if all(fingers_open):
                    print("All fingers are open. Exiting program.")
                    pygame.mixer.music.stop()  # Stop the song 
                    time.sleep(5)
                    break

                # Check if only thumb and index finger are open (volume control)
                if fingers_open[0] and fingers_open[1] and not any(fingers_open[2:]):
                    # Get the coordinates of the thumb and index finger tips
                    x1, y1 = lm_list[4][1], lm_list[4][2]
                    x2, y2 = lm_list[8][1], lm_list[8][2]
                    # Draw circles at the finger tips
                    cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
                    # Draw a line connecting the finger tips
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Calculate the distance between the finger tips
                    length = math.hypot(x2 - x1, y2 - y1)
                    # Change line color to red if distance is less than 50 pixels
                    if length < 50:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # Map the distance to a volume level
                    vol = np.interp(length, [50, 220], [v_min, v_max])
                    # Set the master volume level
                    vol_ctl.SetMasterVolumeLevel(vol, None)
                    # Calculate the position of the volume bar
                    vol_bar = np.interp(length, [50, 220], [400, 150])
                    # Calculate the volume percentage
                    vol_per = np.interp(length, [50, 220], [0, 100])
                    # Draw the volume bar and display the volume percentage
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        # Display the processed image with hand landmarks and volume visualization
        cv2.imshow('Hand Detector', img)
        warnings.filterwarnings("ignore")

        # Check if the 'q' key is pressed to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            

            break

# Release the camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
exit
