import pygame
import glob
import time
import os
import cv2
import subprocess
import datetime
import contextlib
import sys

@contextlib.contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fileno = sys.stderr.fileno()

    saved_stderr = os.dup(stderr_fileno)
    os.dup2(devnull, stderr_fileno)
    os.close(devnull)

    try:
        yield
    finally:
        os.dup2(saved_stderr, stderr_fileno)
        os.close(saved_stderr)

def get_video_filename():
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d%H%M')
    filename = f"{timestamp}_output_video.mp4"
    return os.path.join('pred', filename)
def is_package_installed(package_name):
    result = subprocess.run(
        ["dpkg", "-s", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def install_if_missing(package_list):
    for package in package_list:
        if not is_package_installed(package):
            subprocess.run(["sudo", "apt", "install", "-y", package], check=True)

packages = ["fonts-nanum","ubuntu-restricted-extras"]

install_if_missing(packages)

BUTTON_COLOR_IDLE = (50, 150, 50)
BUTTON_COLOR_HOVER = (70, 200, 70)
BUTTON_COLOR_ACTIVE = (200, 50, 50)
BUTTON_TEXT_COLOR = (255, 255, 255)
FPS = 5
VIDEO_PATH = get_video_filename()


pygame.init()
font = pygame.font.Font("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 24)

def load_latest_image():
    img_files = glob.glob('pred/img/image_*.png')
    img_files = [f for f in img_files if os.path.isfile(f)]
    try:
        img_files.sort(key=os.path.getmtime, reverse=True)
    except FileNotFoundError:
        return None

    for file in img_files:
        try:
            with suppress_stderr():
                img = cv2.imread(file)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
    return None


def draw_button(surface, rect, text, is_hovered, is_active):
    color = BUTTON_COLOR_HOVER if is_hovered else BUTTON_COLOR_IDLE
    if is_active:
        color = BUTTON_COLOR_ACTIVE
    pygame.draw.rect(surface, color, rect, border_radius=8)

    label = font.render(text, True, BUTTON_TEXT_COLOR)
    label_rect = label.get_rect(center=rect.center)
    surface.blit(label, label_rect)

recording = False
record_start_time = None
video_writer = None

img = load_latest_image()
if img is not None:
    h, w, _ = img.shape
else:
    w, h = 1280, 720
screen = pygame.display.set_mode((w, h + 60))
pygame.display.set_caption("Live Image Viewer")


DEFAULT_BUTTON_WIDTH = 120
DEFAULT_BUTTON_HEIGHT = 40
HOVER_SCALE = 1.1

button_rect = pygame.Rect(20, h + 10, DEFAULT_BUTTON_WIDTH, DEFAULT_BUTTON_HEIGHT)
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if video_writer:
                video_writer.release()
            pygame.quit()
            exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                recording = not recording
                if recording:
                    record_start_time = time.time()
                    video_writer = None
                else:
                    if video_writer:
                        video_writer.release()
                        video_writer = None

    img = load_latest_image()
    if img is not None:
        h, w, _ = img.shape
        if screen.get_width() != w or screen.get_height() != h + 60:
            screen = pygame.display.set_mode((w, h + 60))
        img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(img_surface, (0, 0))
        screen.fill((0, 0, 0), rect=pygame.Rect(0, h, w, 60))

        if recording:
            if video_writer is None:
                VIDEO_PATH = get_video_filename()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (w, h))
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

    button_width = DEFAULT_BUTTON_WIDTH
    button_height = DEFAULT_BUTTON_HEIGHT
    button_x = 20
    button_y = h + 10

    mouse_pos = pygame.mouse.get_pos()
    is_hovered = pygame.Rect(button_x, button_y, button_width, button_height).collidepoint(mouse_pos)

    if is_hovered:
        button_width = int(DEFAULT_BUTTON_WIDTH * HOVER_SCALE)
        button_height = int(DEFAULT_BUTTON_HEIGHT * HOVER_SCALE)


    button_rect = pygame.Rect(
        button_x - (button_width - DEFAULT_BUTTON_WIDTH) // 2,
        button_y - (button_height - DEFAULT_BUTTON_HEIGHT) // 2,
        button_width,
        button_height
    )

    draw_button(screen, button_rect, "Stop Rec" if recording else "Start Rec", is_hovered, recording)

    if recording and record_start_time:
        elapsed = int(time.time() - record_start_time)
        time_text = font.render(f"Recording: {elapsed}s", True, (255, 255, 255))
        screen.blit(time_text, (button_rect.right + 20, h + 20))
    pygame.display.flip()
    clock.tick(FPS)
