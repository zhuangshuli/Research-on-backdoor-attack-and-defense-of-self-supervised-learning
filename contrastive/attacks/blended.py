from PIL import Image
import os

class Blended(object):
    def __init__(self, img_size, num, mode=0, target=0, args=None):
        super(Blended, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, 'triggers', 'blended.jpg')
        self.trigger = Image.open(image_path).resize((self.img_size, self.img_size), Image.BILINEAR)
        self.args = args

    def __call__(self, img, target, backdoor, idx):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if (self.mode == 0 and idx > self.num) or self.mode == 2:
            target, backdoor = self.target, 1
            img = Image.blend(img, self.trigger, self.args.blended_ratio)
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode
