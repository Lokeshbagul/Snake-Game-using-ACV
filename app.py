import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Game parameters
width, height = 640, 480
speed = 15

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class SnakeGame(VideoTransformerBase):
    def __init__(self):
        self.reset_game()
    
    def reset_game(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.snake_direction = 'RIGHT'
        self.change_to = self.snake_direction
        self.food_pos = [random.randrange(1, (width // 10)) * 10, random.randrange(1, (height // 10)) * 10]
        self.food_spawn = True
        self.score = 0
        self.game_over = False

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Draw hand landmarks and control snake movement
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                index_finger_tip = landmarks[8]
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)

                # Determine movement direction based on index finger tip position
                if abs(self.snake_pos[0] - x) > abs(self.snake_pos[1] - y):
                    if x > self.snake_pos[0]:
                        self.change_to = 'RIGHT'
                    elif x < self.snake_pos[0]:
                        self.change_to = 'LEFT'
                else:
                    if y > self.snake_pos[1]:
                        self.change_to = 'DOWN'
                    elif y < self.snake_pos[1]:
                        self.change_to = 'UP'

        # Update the snake's direction and position
        if self.change_to == 'RIGHT' and not self.snake_direction == 'LEFT':
            self.snake_direction = 'RIGHT'
        if self.change_to == 'LEFT' and not self.snake_direction == 'RIGHT':
            self.snake_direction = 'LEFT'
        if self.change_to == 'UP' and not self.snake_direction == 'DOWN':
            self.snake_direction = 'UP'
        if self.change_to == 'DOWN' and not self.snake_direction == 'UP':
            self.snake_direction = 'DOWN'

        # Move the snake
        if self.snake_direction == 'RIGHT':
            self.snake_pos[0] += 10
        if self.snake_direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.snake_direction == 'UP':
            self.snake_pos[1] -= 10
        if self.snake_direction == 'DOWN':
            self.snake_pos[1] += 10

        # Snake body growing mechanism
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos == self.food_pos:
            self.food_spawn = False
            self.score += 1
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (width // 10)) * 10, random.randrange(1, (height // 10)) * 10]
        self.food_spawn = True

        # Game Over conditions
        if self.snake_pos[0] < 0 or self.snake_pos[0] > width - 10 or self.snake_pos[1] < 0 or self.snake_pos[1] > height - 10:
            self.game_over = True
        for block in self.snake_body[1:]:
            if self.snake_pos == block:
                self.game_over = True

        # Draw snake and food
        for pos in self.snake_body:
            cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + 10, pos[1] + 10), (0, 255, 0), -1)
        cv2.rectangle(frame, (self.food_pos[0], self.food_pos[1]), (self.food_pos[0] + 10, self.food_pos[1] + 10), (0, 0, 255), -1)

        # Display score
        cv2.putText(frame, f'Score: {self.score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check if game over
        if self.game_over:
            cv2.putText(frame, 'Game Over', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.reset_game()

        return frame

st.title("Hand-Tracked Snake Game")
st.write("Control the snake using your hand! Open your camera, and move your index finger to control the snake's direction.")

# Start the game
webrtc_streamer(key="snake-game", video_transformer_factory=SnakeGame)
