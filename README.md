# Image-Generation-Facial-Recognition
Implementations:

1- Implementing a facial recognition system.

2- Training a feature extractor on the LFW dataset.

3- During training, every N epochs, an assessment is run of the quality of the extractor and shown in the logs the best found threshold and its F1 measure.

4- Implementing the face alignment function (face cut + rotate + resize).

5- Testing the pipeline (detection + alignment + feature extraction) on real images.

6- Recognition results on video are displayed (reading each frame using cv2.VideoCapture and drawing these frames into the output video via cv2.VideoWriter. Drawing face boxes on the frames, the name of the face and the confidence of the face match).

Steps:
1- Loading Yolov8n-face:
~~~
detector_model = YOLO('yolov8n-face.pt')
res = detector_model.predict(source=test_img_path,
                              show=False,
                              save=False,
                              conf=0.4,
                              save_txt=False,
                              save_crop=False,
                              verbose=False,
                              device=device,
                              half=fp16,
                              )[0]
~~~
2- Implementing the function for drawing detection results:
~~~

def draw_detection_results(image, face_info):
    conf = face_info['conf']
    box_points = face_info['box_points']
    key_points = face_info['key_points']

    x0, y0, x1, y1 = tuple(box_points)
    color = (0, 0, 255)
    text = f'{conf:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, 0.5, 2)
    text_x = int((x1 + x0 - text_size[0]) // 2)
    text_y = int(y0 - text_size[1])
    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
    cv2.putText(image, text, (text_x, text_y), font, 0.5, color, 2)

    key_points_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (128, 0, 128)]
    radius = 3
    for point, cur_color in zip(key_points, key_points_colors):
        cv2.circle(image, (int(point[0]), int(point[1])), radius, cur_color, -1)
~~~

3- An Example of results:
![image](https://github.com/ghfranj/Image-Generation-Facecial-Recognition/assets/98123238/fae04fb6-a1f6-43ab-ade6-f69d6855eb88)

4- Features extraction model:
~~~

class EfficientNetB0(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.05, inplace=True),
            nn.Linear(in_features=1280,
                      out_features=embedding_size)
        )

    def forward(self, x):
        return self.model(x)
~~~
 5- ArcFace Loss:
 ~~~

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input).to(device), F.normalize(self.weight).to(device))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output
~~~
6- Results of analyzing thresholds for `cosine` metric...:
![image](https://github.com/ghfranj/Image-Generation-Facecial-Recognition/assets/98123238/3914e8cd-79e5-4ced-bee3-b3ac8c0c6fcf)

7- Results of analyzing thresholds for `euclidian` metric:
![image](https://github.com/ghfranj/Image-Generation-Facecial-Recognition/assets/98123238/92f056d7-12ee-491f-9a13-ced8a3541008)

8- Results of testing on a video:
![image](https://github.com/ghfranj/Image-Generation-Facecial-Recognition/assets/98123238/9d67436b-73e1-44fb-b63f-48da17d762f1)
