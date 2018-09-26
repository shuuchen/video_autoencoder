import torch
from torchvision import models

class Img2Vec():

    def __init__(self, model_path='./fine_tuning_dict.pt'):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.model = torch.load(model_path) # because the model was trained on a cuda machine
        else:
            self.model = torch.load(model_path, map_location='cpu')

        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model = self.model.to(self.device)
        self.model.eval()


    def get_vec(self, image):

        image = image.to(self.device)

        num_imgs = image.size(0)

        my_embedding = torch.zeros(num_imgs, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return my_embedding.view(num_imgs, -1)
