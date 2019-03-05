import holoviews as hv
from holoviews import opts
import numpy
import torch

def get_random_images(k:int, images: torch.Tensor):
    k = max(k,0)
    n = len(images)
    indexes = torch.randperm(n)
    rand_idx = indexes[:k]
    return images[rand_idx]

def tensors_to_numpy(*args):
    return (x.numpy() for x in args)

def tensor_dict_to_numpy(args:{})->{}:
    return {k:v.numpy() for k,v in args.items()}

def images_plot(k:int, 
                images : numpy.array, 
                targets: numpy.array, 
                predictions : numpy.array, 
                scores : numpy.array,
                plot_columns : int=5) -> hv.NdLayout:
    def image_text(i):
        data = {
            "i" : i,
            "t" : targets[i],
            "p" : predictions[i],
            "s" : scores[i],
        }
        text_list = [f"{k}:{v}" for k,v in data.items()]
        return ','.join(text_list)
    def image_plot(i):
        text = image_text(i)
        return hv.Image(images[i]).opts(title=text)
        
    n = len(images)
    random_full_idx = numpy.random.permutation(n).tolist()
    coords = {image_text(i):image_plot(i) for i in random_full_idx[:k]}
    plot = hv.NdLayout(coords,kdims=["Desc"])
    return plot