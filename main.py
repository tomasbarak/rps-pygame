import pygame, sys
from pygame.locals import *
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier
from model import PointHistoryClassifier

from cameraView import get_frame

pygame.init()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


# Set up the window in full screen mode
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

# Set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)

# Set up the fonts
font = pygame.font.SysFont(None, 48)

com_images = [
    pygame.image.load("assets/images/rock.png"),
    pygame.image.load("assets/images/paper.png"),
    pygame.image.load("assets/images/scissors.png")
]

def main_menu():
    play_button = font.render("Jugar", True, WHITE)
    play_button_rect = play_button.get_rect()
    play_button_rect.center = screen.get_rect().center
    play_button_rect.y += 0

    exit_button = font.render("Salir", True, WHITE)
    exit_button_rect = exit_button.get_rect()
    exit_button_rect.center = screen.get_rect().center
    exit_button_rect.y += 100

    menu = True

    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if play_button_rect.collidepoint(mouse_pos):
                    game()
                if exit_button_rect.collidepoint(mouse_pos):
                    menu = False
                    pygame.quit()
                    sys.exit()
            #MOuse hover
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()
                if play_button_rect.collidepoint(mouse_pos):
                    play_button = font.render("Jugar", True, LIGHT_GRAY)
                else:
                    play_button = font.render("Jugar", True, WHITE)
                if exit_button_rect.collidepoint(mouse_pos):
                    exit_button = font.render("Salir", True, LIGHT_GRAY)
                else:
                    exit_button = font.render("Salir", True, WHITE)


        screen.fill(BLACK)
        screen.blit(play_button, play_button_rect)
        screen.blit(exit_button, exit_button_rect)

        pygame.display.update()

    pygame.quit()
    sys.exit()
def game():
    #Clear screen
    screen.fill(BLACK)
    pygame.display.update()
    clock = pygame.time.Clock()

    com_possible_moves = [0, 1, 2]
    #Choose random move for computer
    com_move = np.random.choice(com_possible_moves)

    com_cam = pygame.Rect(0, 0, screen.get_width() / 2, screen.get_height())
    user_cam = pygame.Rect(screen.get_width() / 2, 0, screen.get_width() / 2, screen.get_height())

    com_cam_backround = pygame.Surface((com_cam.width, com_cam.height))
    com_cam_backround.fill(WHITE)

    user_cam_backround = pygame.Surface((user_cam.width, user_cam.height))
    user_cam_backround.fill(BLACK)

    playing = True
    waiting_player_move = True

    def check_winner(player_move, com_move):
        if player_move == com_move:
            return "Tie"
        elif player_move == 0 and com_move == 2:
            return "Player"
        elif player_move == 1 and com_move == 0:
            return "Player"
        elif player_move == 2 and com_move == 1:
            return "Player"
        else:
            return "Com"

    def draw_text(x, y, string, col, size, window):
        _font = pygame.font.SysFont(None, 100)
        text = _font.render(string, True, col)
        textbox = text.get_rect()
        textbox.center = (x, y)
        window.blit(text, textbox)

    def display_com_hand(number, window):
        img = com_images[number]
        scaled_img = pygame.transform.scale(img, (com_cam.width/4, com_cam.height/4))
        com_image_rect = scaled_img.get_rect()
        com_image_rect.center = com_cam.center
        window.blit(scaled_img, com_image_rect)
    
    def fade_in(image, window):
        alpha = 0
        while alpha < 255:
            image.set_alpha(alpha)
            window.blit(image, (0, 0))
            pygame.display.update()
            pygame.time.delay(10)
            alpha += 1
    def fade_in_text(text, window):
        alpha = 0
        while alpha < 255:
            text.set_alpha(alpha)
            window.blit(text, (0, 0))
            pygame.display.update()
            pygame.time.delay(10)
            alpha += 1

    last_player_move = -1
    last_player_frame = None
    while playing:
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    playing = False
        screen.fill(BLACK)
        screen.blit(com_cam_backround, com_cam)
        screen.blit(user_cam_backround, user_cam)

        frame_info = get_frame()


        frame = frame_info[0]
        hand_sign = frame_info[1]
        hand_sign_label = hand_sign[1]
        pygame_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pygame_frame = np.rot90(pygame_frame)
        pygame_frame = pygame.surfarray.make_surface(pygame_frame)
        #Scale maintaining aspect ratio
        #Calc aspect ratio
        aspect_ratio = pygame_frame.get_width() / pygame_frame.get_height()
        #Scale to width
        pygame_frame = pygame.transform.scale(pygame_frame, (int(user_cam.width), int(user_cam.width / aspect_ratio)))
        pygame_frame = pygame.transform.flip(pygame_frame, True, False)
        screen.blit(pygame_frame, user_cam)


        if frame_info[1][0] == -1 and waiting_player_move:
            print("No hand detected")
            continue

        if waiting_player_move:
            print("Waiting for player move")
            last_player_move = frame_info[1][0]
            last_player_frame = pygame_frame
            waiting_player_move = False
            continue

        if not waiting_player_move:
            #Show com hand
            print("Showing com hand")
            print(last_player_move, com_move)
            screen.blit(last_player_frame, user_cam)
            display_com_hand(com_move, screen)
            winner = check_winner(last_player_move, com_move)
            if winner == "Player":
                draw_text(screen.get_width() / 2, screen.get_height() / 2, "Player wins!", (0, 255, 0), 100, screen)
            elif winner == "Com":
                draw_text(screen.get_width() / 2, screen.get_height() / 2, "Computer wins!", (255, 0, 0), 100, screen)
            else:
                draw_text(screen.get_width() / 2, screen.get_height() / 2, "Empate!", (255, 255, 0), 100, screen)

            #Add replay button at right bottom corner
            replay_button = font.render("Replay", True, WHITE)
            replay_button_rect = replay_button.get_rect()
            replay_button_rect.bottomright = screen.get_rect().bottomright
            replay_button_rect.x -= 10
            replay_button_rect.y -= 10
            screen.blit(replay_button, replay_button_rect)

            #add function to replay button
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if replay_button_rect.collidepoint(mouse_pos):
                        game()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        playing = False
                    elif event.key == pygame.K_r:
                        game()
    pygame.display.update()
    
main_menu()