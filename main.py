import moviepy.editor as mp
from IPython.display import display, HTML
from IPython.display import display
from PIL import Image
import numpy as np
from PIL.Image import NONE
from networks import model_cache, load_G, load_D
from networks import std_gen, std_gen_interpolate
from networks import trunc_gen, trunc_gen_interpolate
from networks import std_enc, std_enc_with_D
load_G('model/anime-biggan-256px-run39-607250.generator')


def concat_imgs_bsz8(imgs):
    np_imgs = [np.asarray(img) for img in imgs]
    img1 = np.concatenate(np_imgs[:4], 1)
    img2 = np.concatenate(np_imgs[4:], 1)
    img = Image.fromarray(np.concatenate([img1, img2], 0))
    return img.resize([img.size[0]//2, img.size[1]//2])


# imgs=[]
# Timage = std_gen(1, seed=233)
# for i in range(8):
#     Timage = std_gen(1, seed=None)
#     imgs += Timage

# for i, img in enumerate(imgs):
#     img.save(f'data/std_seed233_{str(i).zfill(3)}.png')
# concat_imgs_bsz8(imgs).save("test.png")


def display_mp4(path):
    print(f'Read from {path}')
    from base64 import b64encode
    mp4 = open(path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML("""
    <video controls loop autoplay>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url))
    print('Display finished.')


# std_gen_interpolate(8, seed=2048, out_path='data/std_out', interpolate_mode=1)
# clip = mp.VideoFileClip("data\std_out.gif")
# clip.write_videofile("data\std_out.mp4")

std_gen_interpolate(8, seed=2333, out_path='data/std_inter0', levels="z11;z12")
std_gen_interpolate(8, seed=2333, out_path='data/std_inter1', levels="z21;z22")
std_gen_interpolate(8, seed=2333, out_path='data/std_inter2', levels="z31;z32")
std_gen_interpolate(8, seed=2333, out_path='data/std_inter3', levels="z41;z42")
std_gen_interpolate(8, seed=2333, out_path='data/std_inter4', levels="z51;z52")
std_gen_interpolate(8, seed=2333, out_path='data/std_inter5', levels="z61;z62")