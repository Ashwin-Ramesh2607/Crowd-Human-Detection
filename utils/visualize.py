import cv2
from PIL import Image, ImageFont, ImageDraw


def draw_status(original_image, bird_view_image, person_data, person_status):

    # Make into PIL Image
    bird_view_image = Image.fromarray(bird_view_image)

    # Get a drawing context
    draw = ImageDraw.Draw(im_p)
    font = ImageFont.truetype("Arial Unicode.ttf", 32)
    tick = str(emoji.emojize(':heavy_check_mark:'))

    for idx, coords in person_data:

        if person_status[idx] == 'unsafe':
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        draw.text((int(coords['feet_point'][0]), int(coords['feet_point'][1])), tick, (255,255,255), font=font)

        # Visualization on Bird View Image
        # cv2.circle(
        #     bird_view_image,
        #     (int(coords['feet_point'][0]), int(coords['feet_point'][1])),
        #     5, color, -1)

        # Visualization on Video Feed
        cv2.rectangle(
            original_image, 
            (int(coords['bbox'][0]), int(coords['bbox'][1])),
            (int(coords['bbox'][2]), int(coords['bbox'][3])),
            color, 2)

    return original_image, bird_view_image


def draw_connections(bird_view_image, person_data, track_connections):

    for id_pair, status in track_connections.items():
        point1 = tuple(map(int, person_data[id_pair[0]]['feet_point']))
        point2 = tuple(map(int, person_data[id_pair[1]]['feet_point']))

        if status == 'unsafe':
            color = (0, 0, 255)  # Red
        elif status == 'family':
            color = (255, 0, 0)  # Blue
        elif status == 'safe':
            continue  # No connecting lines if they are at safe distance
        
        # Visualization on Bird View Image
        cv2.line(bird_view_image, point1, point2, color, thickness=2)

    return bird_view_image


def show_violations(original_image, bird_view_image, person_data, person_status, track_connections):

    original_image, bird_view_image = draw_status(original_image, bird_view_image, person_data, person_status)
    bird_view_image = draw_connections(bird_view_image, dict(person_data), track_connections)

    return original_image, bird_view_image
