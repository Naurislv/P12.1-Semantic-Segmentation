"""Image augmentation and generator."""

# Standard imports
from random import randint, randrange, uniform
import logging

# Dependecy imports
import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

__all__ = [
    "RANDOM_NOISE",
    "RANDOM_ROTATIONS",
    "RANDOM_SHIFTS",
    "RANDOM_BRIGHTNESS",
    "RANDOM_ZOOMS",
    "RANDOM_BLUR",
]


class Augmentation(object):
    """Augmentation and generator class."""

    def __init__(self, degree=10, output_size=(28, 28), min_bright=-33, max_bright=33, amount=5):
        """Initialize class variables."""
        self.degree = degree
        self.output_h, self.output_w = output_size
        self.min_bright = min_bright
        self.max_bright = max_bright
        self.amount = amount
        self.zoom_in = 0.5
        self.zoom_out = 0.5
        self.blur_range = None  # Must be tuple min, max --> e.g. 0, 16

        # Generate distribution for blur
        self.distro = None

    def random_rotations(self, image, degree, label=None):
        """Rotate image randomly."""

        (h_img, w_img, ch_img) = image.shape[:3]
        center = (w_img / 2, h_img / 2)

        # rotation = np.random.uniform(-1, 1, 1)[0] * degree
        rotation = uniform(-degree, degree)
        rot_mtrx = cv2.getRotationMatrix2D(center, rotation, 1.0)

        image = cv2.warpAffine(image, rot_mtrx, (w_img, h_img)).reshape(h_img, w_img, ch_img)
        if label is not None:
            label = cv2.warpAffine(label, rot_mtrx, (w_img, h_img))

            return image, label, rotation

        return image, rotation

    def random_shifts(self, image, label, h_shift, v_shift):
        """Add random horizontal/vertical shifts to image dataset to imitate
        steering away from sides."""

        rows = image.shape[0]
        cols = image.shape[1]

        horizontal = uniform(- h_shift / 2, h_shift / 2)
        vertical = uniform(- v_shift / 2, v_shift / 2)

        mtx = np.float32([[1, 0, horizontal], [0, 1, vertical]])

        # change also corresponding lable -> steering angle
        image = cv2.warpAffine(image, mtx, (cols, rows))
        label = cv2.warpAffine(label, mtx, (cols, rows))

        return image, label, (h_shift, v_shift)

    def random_brightness(self, images, min_val=0, max_val=255, min_bright=-50, max_bright=40):
        """Add random brightness to give image dataset to imitate day/night."""
        # random_bright = np.random.uniform(min_bright, max_bright, 1)[0]
        random_bright = randrange(min_bright, max_bright)
        data_type = images.dtype
        if random_bright > 0:
            # add brightness
            images = np.where((max_val - images) < random_bright, max_val, images + random_bright)
        elif random_bright < 0:
            # remove brightness
            images = np.where((images + random_bright) <= min_val, min_val, images + random_bright)

        return images.astype(data_type), random_bright

    def random_noise(self, images, amount=20, noise_chance=0.5):
        """Add random noise to image dataset.

        noise_chance: probability that noise will be applied.
        """
        noise_applied = False
        if uniform(0, 1) > noise_chance:
            noise = np.zeros_like(images, images.dtype)  # needs preallocated input image
            noise = cv2.randn(noise, (0), (amount))

            images = np.where((255 - images) < noise, 255, images + noise)
            noise_applied = True

        return images, noise_applied

    def random_padding(self, image, output_size, override_random=None):
        """Add random horizontal/vertical shifts and increases size of image to output_size."""
        h_img, w_img, ch_img = image.shape
        h_output, w_output = output_size

        asser_msg = ("For Random padding input image Hight must be less or equal to "
                     "output_size hight")
        assert h_img <= h_output, asser_msg
        assert_msg = ("For Random padding input image Width must be less or equal to "
                      "output_size width")
        assert w_img <= w_output, assert_msg

        output_image = np.zeros((h_output, w_output, ch_img), dtype=np.float32)

        if override_random is None:
            pad_h_up = randint(0, h_output - h_img)
            pad_w_left = randint(0, w_output - w_img)
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left
        else:
            pad_h_up = override_random[0]
            pad_w_left = override_random[1]
            pad_h_down = h_output - h_img - pad_h_up
            pad_w_right = w_output - w_img - pad_w_left

        output_image = np.pad(image, ((pad_h_up, pad_h_down), (pad_w_left, pad_w_right), (0, 0)),
                              'constant', constant_values=0)

        return output_image, (pad_h_up, pad_w_left)

    def random_zooms(self, image, label, zoom_in=0.5, zoom_out=0.5):
        """Randomly zoom image."""
        output_size_h = image.shape[0]
        output_size_w = image.shape[1]

        rand_size = uniform(-1, 1)

        if rand_size < 0:
            max_zoom = output_size_h * zoom_out
            random_size_h = int(output_size_h + max_zoom * rand_size)
        else:
            max_zoom = output_size_h * zoom_in
            random_size_h = int(output_size_h + max_zoom * rand_size)

        random_size_w = output_size_w * random_size_h // output_size_h
        if random_size_w == output_size_w:
            return image, label, rand_size

        # Image zooming
        image = cv2.resize(image, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)

        if random_size_w < output_size_w:
            image, _ = self.random_padding(image, output_size=(output_size_h, output_size_w))
        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            image = image[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
        else:
            logging.info("Failed random_zooms ? %s", image.shape)

        # Label zooming
        label = cv2.resize(label, (random_size_w, random_size_h), interpolation=cv2.INTER_AREA)

        if random_size_w < output_size_w:
            label, _ = self.random_padding(label, output_size=(output_size_h, output_size_w))
        elif random_size_w > output_size_w:
            diff_w = random_size_w - output_size_w
            diff_h = random_size_h - output_size_h

            label = label[diff_h // 2: -diff_h // 2, diff_w // 2: -diff_w // 2, :]
        else:
            logging.info("Failed random_zooms ? %s", label.shape)

        return image, label, rand_size

    def random_blur(self, image, blur_range=None, many=False):
        """Randomly blur image or if many=True --> images."""

        _image = image.copy()
        im_shape = _image.shape

        if blur_range is None:
            if self.distro is None:
                self.distro = []
                for i, weight in zip([0, 2, 4, 6, 8, 10, 12, 14], [50, 5, 5, 5, 10, 10, 10, 5]):
                    self.distro += [i] * weight

                rand_blur = self.distro[randint(0, 99)]
            else:
                rand_blur = self.distro[randint(0, 99)]

        else:
            blur_min, blur_max = blur_range
            rand_blur = randint(blur_min, blur_max)

        if rand_blur == 0:
            return image, rand_blur

        if rand_blur % 2 == 0:
            rand_blur += 1

        if many:
            for i, img in enumerate(_image):
                _image[i] = cv2.GaussianBlur(img, (rand_blur, rand_blur), 0).reshape(im_shape[1],
                                                                                     im_shape[2],
                                                                                     im_shape[3])
            return _image, rand_blur

        return cv2.GaussianBlur(_image, (rand_blur, rand_blur), 0), rand_blur

    def random_flip(self, image, label, horizontal=False):
        """Apply random flip to single image and label."""

        flip = 1
        rand_float = uniform(0, 1)

        if horizontal:
            # 1 == vertical flip
            # 0 == horizontal flip
            flip = randint(0, 1)

        if rand_float > 0.5:
            image = cv2.flip(image, flip)
            label = cv2.flip(label, flip)

        return image, label


_INIT = Augmentation()
RANDOM_NOISE = _INIT.random_noise
RANDOM_ROTATIONS = _INIT.random_rotations
RANDOM_SHIFTS = _INIT.random_shifts
RANDOM_BRIGHTNESS = _INIT.random_brightness
RANDOM_ZOOMS = _INIT.random_zooms
RANDOM_BLUR = _INIT.random_blur
RANDOM_FLIP = _INIT.random_flip
