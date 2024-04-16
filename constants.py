IMAGE_MAX_SIZE = (512, 512)

DETECT_PEOPLE_MODEL = "lib/saved_models/faster_rcnn-inception_resnet_v2"
DETECT_PEOPLE_THRESHOLD = 0.3
DETECT_PEOPLE_CLASS_LABELS = [69]  # 69 is the label for person

DETECT_FACES_MODEL = "lib/saved_models/faster_rcnn-inception_resnet_v2"
DETECT_FACES_THRESHOLD = 0.3
DETECT_FACES_CLASS_LABELS = [502]  # 502 is the label for human faces


URL_FOR_SCRAP = "https://he.wikipedia.org/wiki/%D7%A2%D7%9E%D7%95%D7%93_%D7%A8%D7%90%D7%A9%D7%99"