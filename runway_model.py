# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import shutil

import runway
from runway.data_types import image

from PIL import Image

from util.util import tensor2im
from options.test_options import TestOptions
from data.base_dataset import get_transform
from models import create_model


@runway.setup(options={'generator_checkpoint': runway.file(description="Checkpoint for the generator",
                                                           extension='.pth')})
def setup(opts):
    generator_checkpoint_path = opts['generator_checkpoint']
    try:
        os.makedirs('checkpoints/pretrained/')
    except OSError:
        pass
    shutil.copy(generator_checkpoint_path, 'checkpoints/pretrained/latest_net_G.pth')

    opt = TestOptions(args=['--dataroot', '',
                            '--name', 'pretrained',
                            '--model', 'pix2pix',
                            '--direction', 'BtoA']).parse()
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)
    model.setup(opt)
    return {'model': model, 'opt': opt}


@runway.command(name='generate',
                inputs={ 'image': image(description='Input image') },
                outputs={ 'image': image(description='Output image') })
def generate(model, args):
    opt = model['opt']
    model = model['model']

    orig_image = args['image'].convert('RGB')
    orig_size = orig_image.size
    input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
    transform = get_transform(opt, grayscale=(input_nc == 1))
    A = transform(orig_image)
    input_obj = {'A': A.unsqueeze(0), 'A_paths': '', 'B': A.unsqueeze(0), 'B_paths': ''}

    model.set_input(input_obj)
    model.test()
    visuals = model.get_current_visuals()
    im = tensor2im(visuals['fake_B'])

    return {
        'image': Image.fromarray(im).resize(orig_size)
    }


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888)
