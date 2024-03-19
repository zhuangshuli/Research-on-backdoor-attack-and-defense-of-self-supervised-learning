from PIL import Image
import os

class Patched(object):
    def __init__(self, img_size, num, mode=0, target=0, args=None):
        super(Patched, self).__init__()
        self.img_size = img_size
        self.num = num
        self.mode = mode
        self.target = target
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, 'triggers', f'patched{img_size}_{args.patched_size}.png')
        self.trigger = Image.open(image_path).resize((self.img_size, self.img_size), Image.NEAREST)
        self.mask = Image.open(image_path).resize((self.img_size, self.img_size), Image.NEAREST).convert('L')

    def __call__(self, img, target, backdoor, idx):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if (self.mode == 0 and idx > self.num) or self.mode == 2:
            target, backdoor = self.target, 1
            img = Image.composite(self.trigger, img, self.mask)
        return img, target, backdoor

    def set_mode(self, mode):
        self.mode = mode
