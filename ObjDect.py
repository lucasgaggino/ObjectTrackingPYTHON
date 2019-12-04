# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

#para Comunicacion Serial
import serial
import io
import base64

#para manejo de IMGS
import cv2
import math

# Check available GPU devices.
# print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

def display_image(image):
    if (camIndex == 1):
        plt.imsave('./im1.jpg', image)
        cv2.namedWindow("CAM1", cv2.WND_PROP_AUTOSIZE)
        cv2.setWindowProperty("CAM1", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        im = cv2.imread('./im1.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('CAM1', im)
    elif (camIndex == 2):
        plt.imsave('./im2.jpg', image)
        cv2.namedWindow("CAM2", cv2.WND_PROP_AUTOSIZE)
        cv2.setWindowProperty("CAM2", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        im = cv2.imread('./im2.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('CAM2', im)

   #plt.imshow(image)
   #plt.show()






def load_and_resize_image(imageName, new_width=256, new_height=256,
                          display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_center_of_target(image, x, y):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    draw = ImageDraw.Draw(image)
    #print('Width, Height', ( im_width, im_height))
    #print('Center X, Y', (x*im_width, y*im_height))
    draw.rectangle([x*im_width-5, y*im_height-5, x*im_width+5, y*im_height+5], None, 'red')



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    # print(im_width,im_height)
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.01):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        #print("Font not found, using default font.")
        font = ImageFont.load_default()
    b = [None]
    for x in range(100):
        if class_names[x].decode("ascii") == "Person"\
                :
            b.append(x)
        if class_names[x].decode("ascii") == "Man":
            b.append(x)
        if class_names[x].decode("ascii") == "Human face":
            b.append(x)
        if class_names[x].decode("ascii") == "Woman":
            b.append(x)

    if len(b) > 1:
        most_probale_target_index = b[1]
        for i in b[1:]:
            if scores[i] > scores[most_probale_target_index]:
                most_probale_target_index = i
        i = most_probale_target_index
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_center_of_target(image_pil, (xmax+xmin)/2, (ymax+ymin)/2)
            np.copyto(image, np.array(image_pil))
            calculate_angles((xmax+xmin)/2,(ymax+ymin)/2,640, 480)

            # print(class_names[i].decode("ascii"), [ymin * 640, ymax * 640, xmin * 480, xmax * 480]);

    return image


def calculate_angles(xcentre, ycentre,im_width, im_height):
    #valido para 640x480
    #im_width = 640
    #im_height = 480
    xcentre=xcentre*im_width
    ycentre=ycentre*im_height
    D_FOV = math.radians(75)
    diag = 2 * math.tan(D_FOV / 2)
    hor = diag * 4 / 5
    ver = diag * 3 / 5
    H_FOV = 2 * math.atan(hor / 2)
    V_FOV = 2 * math.atan(ver / 2)
    alfa = math.atan((xcentre - (im_width / 2)) * math.tan(H_FOV / 2) / (im_width / 2))
    alfa = math.degrees(alfa)
    beta = -math.atan((ycentre - (im_height / 2)) * math.tan(V_FOV / 2) / (im_height / 2))
    beta = math.degrees(beta)
    if(camIndex==1):
        global alfa1
        alfa1 = alfa
        global beta1
        beta1=beta
    elif(camIndex==2):
        global alfa2
        alfa2 = alfa
        global beta2
        beta2 = beta
    #print("alfa:", alfa, " -  beta:", beta)

# image_url = "https://britainatwar.keypublishing.com/wp-content/uploads/sites/9/2018/08/Hood_Vancouver_Harbour.jpg"
# downloaded_image_path = download_and_resize_image(image_url, 674, 437, True)
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(module_handle).signatures['default']


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    #print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])


    display_image(image_with_boxes)

def float_2_fixed_len_string(number, len, presicion):
    instruction = '{:+' + str(len) + '.' + str(presicion) + 'f}'
    string = instruction.format(number)
    return string


# print(downloaded_image_path)

def main_Loop():
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()
    cv2.imwrite('./image1.png', frame)
    cv2.imwrite('./image2.png', frame2)
    global camIndex
    camIndex = 1
    run_detector(detector, ".\image1.png")
    camIndex= 2
    run_detector(detector, ".\image2.png")
    print("alfa1:", alfa1, " -  beta1:", beta1)
    print("alfa2:", alfa2, " -  beta1:", beta2)
    send_target_data()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap1.release()
        cv2.destroyAllWindows()

def send_target_data():
    ser.write('xxPleaseSYNCwithMExx'.encode())
    ser.write(float_2_fixed_len_string(alfa1, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(beta1, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(alfa2, 10, 6).encode())
    ser.write('@'.encode())
    ser.write(float_2_fixed_len_string(beta2, 10, 6).encode())
    ser.write('@'.encode())
    ser.write('xxPleaseSYNCwithMExx'.encode())
    time.sleep(0.1)
    print(str('xxPleaseSYNCwithMExx') + float_2_fixed_len_string(alfa1, 10, 6) + '@' + float_2_fixed_len_string(
        beta1, 10, 6) + '@' + float_2_fixed_len_string(alfa2, 10, 6) + '@' + float_2_fixed_len_string(beta2, 10,
                                                                                                      6) + '@')


# --------------------------------MAIN -----------------------##
# Capture frame-by-frame from CAM 1
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

alfa1=0.0
beta1=0.0
alfa2=0.0
beta2=0.0
camIndex=0


ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM4'
ser.timeout=1

try:
    ser.open()
except serial.SerialException:
    print("Unable to open Port")
    time.sleep(60)

while(True):
    main_Loop()


