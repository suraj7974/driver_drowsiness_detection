import os
import cv2
import numpy as np


# def display_images(image_paths):
#     for image_path in image_paths:
#         img = cv2.imread(image_path)
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()


def face_for_yawn(direc="./Dataset", face_cas_path="./archive/haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(
                path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y),
                                    (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


def get_data(dir_path="./Dataset", face_cas="./archive/haarcascade_frontalface_default.xml", eye_cas="./archive/haarcascade.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num += 2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(
                    os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data


def append_data():
    yaw_no = face_for_yawn()
    data = get_data()

    images = []
    labels = []

    # Count number of images in each class
    num_yawn_images = 0
    num_no_yawn_images = 0
    num_closed_images = 0
    num_open_images = 0

    for image, label in yaw_no:
        images.append(image)
        labels.append(label)
        if label == 0:
            num_yawn_images += 1
        elif label == 1:
            num_no_yawn_images += 1

    for image, label in data:
        images.append(image)
        labels.append(label)
        if label == 2:
            num_closed_images += 1
        elif label == 3:
            num_open_images += 1

    print("Number of yawn images:", num_yawn_images)
    print("Number of no yawn images:", num_no_yawn_images)
    print("Number of closed eyes images:", num_closed_images)
    print("Number of open eyes images:", num_open_images)
    print("Total number of images in final dataset:", len(images))

    return np.array(images), np.array(labels)


new_data = append_data()
