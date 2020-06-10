import cv2
import numpy as np


def distance_check(point1, point2, distance_threshold):

    # Calculate the Euclidean distance between the feet points of the 2 detection boxes
    euclid_distance = np.linalg.norm(point1 - point2)

    return 'higher' if euclid_distance > distance_threshold else 'lower'


def social_distance_check(person_data):

    person_status = {}
    person_connections = {}

    for i in range(len(person_data) - 1):

        person_1_id = person_data[i][0]
        if person_1_id not in person_status:
            person_status[person_1_id] = 'safe'

        for j in range(i + 1, len(person_data)):
            person_2_id = person_data[j][0]

            # If that person has never been encountered, initialize them as safe
            if person_2_id not in person_status:
                person_status[person_2_id] = 'safe'
            person_connections[(person_1_id, person_2_id)] = 'safe'

            # Elaborate Checks if the 2 persons are within the social distance threshold
            if distance_check(person_data[i][1]['feet_point'], person_data[j][1]['feet_point'], 40) == 'lower':
                person_status[person_1_id] = 'unsafe'
                person_status[person_2_id] = 'unsafe'
                person_connections[(person_1_id, person_2_id)] = 'unsafe'

    return person_status, person_connections
