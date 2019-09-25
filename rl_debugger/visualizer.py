import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import cv2
from .hooks import StepHook

class SaliencyHook(StepHook):
    def __init__(self, model, phi, videowriter=None):
        self.videowriter = videowriter
        self.phi = phi
        self.model = SaliencyWrapper(model, phi)

    def __call__(self, env, agent, result, step):
        obs, _, _, _ = result
        map_img, sight_img = env.render(return_rgb=True)
        p_map, v_map = get_saliency_map(self.model, obs)
        result_img = plot_saliency_on_img(p_map, sight_img, channel=2)
        result_img = plot_saliency_on_img(v_map, result_img, channel=0)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR)
        result_img = np.concatenate((map_img, result_img), axis=1)
        if self.videowriter is not None:
            self.videowriter.write(result_img)
        cv2.imshow('saliency', result_img)
        cv2.waitKey(10)
    
    def close_videowriter(self):
        self.videowriter.release()
 

class SaliencyWrapper:
    """ Wrapper for get_saliency_map 
        The wrapped model return policy digit and value estimation
    """

    def __init__(self, model, phi):
        self.model = model
        self.phi = phi

    def __call__(self, obs):
        pout, vout = self.model(self.phi(obs[None]))
        pout = np.squeeze(pout.logits.data)
        vout = np.squeeze(vout.data)
        return pout, vout


def get_mask(center, size, r):
    # get mask to add blur
    y, x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size)
    mask[keep] = 1  # select a circle of pixels
    # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    mask = gaussian_filter(mask, sigma=r)
    return mask/mask.max()


def get_saliency_map(model, obs, stride=5, radius=5):
    """get_saliency_map 

    Arguments:
        model {SaliencyWrapper} -- [RL model that returns policy digits and value estimation]
        obs {ndarray} -- [input to the model]

    Keyword Arguments:
        stride {int} -- [Stride to apply blurring] (default: {5})
        radius {int} -- [Radius of gaussian blur] (default: {5})

    Returns:
        [ndarray, ndarray] -- [obs size saliency map]
    """

    def occlude(I, mask):
        # return blurred image
        img = I*(1-mask) + gaussian_filter(I, sigma=3)*mask
        return img

    pout, vout = model(obs)
    obs_size = obs.shape[-2:]

    # saliency map S(t,i,j)
    p_map = np.zeros((int(obs_size[0]/stride)+1, int(obs_size[1]/stride)+1)).astype("float32")
    v_map = np.zeros((int(obs_size[0]/stride)+1, int(obs_size[1]/stride)+1)).astype("float32")

    for i in range(0, obs_size[0], stride):
        for j in range(0, obs_size[1], stride):
            mask = get_mask(center=[i, j], size=obs_size, r=radius)
            occluded_obs = occlude(obs, mask)
            occ_p, occ_v = model(occluded_obs)
            score_p = np.sum((pout - occ_p) ** 2) / 2.0
            score_v = np.sum((vout - occ_v) ** 2) / 2.0
            p_map[int(i/stride), int(j/stride)] = score_p
            v_map[int(i/stride), int(j/stride)] = score_v

    pmax = p_map.max()
    vmax = v_map.max()

    p_map = np.asarray(Image.fromarray(
        p_map).resize(obs_size, Image.BILINEAR))
    v_map = np.asarray(Image.fromarray(
        v_map).resize(obs_size, Image.BILINEAR))
    return p_map, v_map


def plot_saliency_on_img(saliency, img, intensity=1e4, alpha=0.4, channel=2, sigma=0.0):
    """plot_saliency_on_img 

    Arguments:
        saliency {[ndarray]} -- [saliency map]
        img {[ndarray]} -- [base image]

    Keyword Arguments:
        intensity {int} -- [Higher value shows clearer saliency map] (default: {1e4})
        alpha {float} -- [transparency ratio of saliency map] (default: {0.4})
        channel {int} -- [channel to draw the saliency map] (default: {2})
        sigma {float} -- [sometimes saliency maps are a bit clearer if you blur them.] (default: {0.0})

    Returns:
        [ndarray] -- [image with saliency map]
    """

    smax, smin = saliency.max(), saliency.min()
    saliency = np.asarray(Image.fromarray(
        saliency).resize(img.shape[:2], Image.BILINEAR))
    saliency = saliency if sigma == 0 else gaussian_filter(saliency, sigma=sigma)
    saliency = intensity * (saliency - smin) / (smax - smin)  # normalize
    saliency = saliency.clip(0, 255).astype('uint8')

    s_img = np.zeros_like(img).astype('uint8')
    s_img[:, :, channel] += saliency
    s_img = Image.fromarray(s_img)
    s_img.putalpha(Image.fromarray((saliency * alpha).astype('uint8')))

    base_img = Image.fromarray(img)
    base_img.putalpha(255)
    base_img.paste(s_img, mask=s_img)

    return np.asarray(base_img)
