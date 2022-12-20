import torch
from PIL import Image
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.transform import resize
from torchvision import transforms as T
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import imageio

from model_template_class import ModelABC, settings


class SegmentationModel(ModelABC):

    def __init__(self, model_type: str):  # model_type='all'/'long'/'cross'
        super().__init__()
        self.model_type = model_type
        self._model2 = None
        self.device = torch.device('cpu')
        self.load(path=settings['segmentation'][self.model_type])

    def load(self, path: list) -> None:
        weight_c1 = path[0]
        weight_c2 = path[1]
        with torch.no_grad():
            self._model = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
            self._model.to(self.device)
            self._model.load_state_dict(torch.load(
                weight_c1, map_location=self.device))
            self._model.eval()
            self._model2 = smp.DeepLabV3Plus(
                encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
            self._model2.to(self.device)
            self._model2.load_state_dict(torch.load(
                weight_c2, map_location=self.device))
            self._model2.eval()

    @staticmethod
    def tif2png(path: str) -> list:
        images = []
        image = Image.open(path)
        i = 0
        while True:
            try:
                image.seek(i)
                image_array = np.array(image)
                images.append(image_array)
                i += 1
            except EOFError:
                break
        print(f'Tif processed. Number of images: {len(images)}')
        return images

    @staticmethod
    def preprocessing(img: object) -> object:
        img = Image.fromarray(img)
        img = img.convert(mode='L')
        Transform = T.Compose([T.ToTensor()])
        img_tensor = Transform(img)
        img_dtype = img_tensor.dtype
        img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()
        img_array = np.array(img, dtype=np.float32)
        or_shape = img_array.shape
        if or_shape == (735, 975):
            x_cut_min = 130
            x_cut_max = 655
            y_cut_min = 155
            y_cut_max = 700
        elif or_shape == (528, 687):
            x_cut_min = 15
            x_cut_max = 420
            y_cut_min = 40
            y_cut_max = 640
        else:
            value_x = np.mean(img, 1)
            value_y = np.mean(img, 0)
            x_hold_range = list(
                (len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
            y_hold_range = list(
                (len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
            value_thresold = 5
            x_cut = np.argwhere((value_x <= value_thresold) == True)
            x_cut_min = list(x_cut[x_cut <= x_hold_range[0]])
            if x_cut_min:
                x_cut_min = max(x_cut_min)
            else:
                x_cut_min = 0
            x_cut_max = list(x_cut[x_cut >= x_hold_range[1]])
            if x_cut_max:
                x_cut_max = min(x_cut_max)
            else:
                x_cut_max = or_shape[0]
            y_cut = np.argwhere((value_y <= value_thresold) == True)
            y_cut_min = list(y_cut[y_cut <= y_hold_range[0]])
            if y_cut_min:
                y_cut_min = max(y_cut_min)
            else:
                y_cut_min = 0
            y_cut_max = list(y_cut[y_cut >= y_hold_range[1]])
            if y_cut_max:
                y_cut_max = min(y_cut_max)
            else:
                y_cut_max = or_shape[1]
        cut_image = img_array_fromtensor[x_cut_min:x_cut_max,
                                         y_cut_min:y_cut_max]
        cut_image_orshape = cut_image.shape
        cut_image = resize(cut_image, (256, 256), order=3)
        cut_image_tensor = torch.tensor(data=cut_image, dtype=img_dtype)
        return [cut_image_tensor, cut_image_orshape, or_shape, [x_cut_min, x_cut_max, y_cut_min, y_cut_max]]

    @staticmethod
    def largest_connect_component(bw_img: object) -> object:
        if np.sum(bw_img) == 0:
            return bw_img
        labeled_img, num = sklabel(
            bw_img, connectivity=1, background=0, return_num=True)
        if num == 1:
            return bw_img
        max_label = 0
        max_num = 0
        for i in range(0, num):
            if np.sum(labeled_img == (i + 1)) > max_num:
                max_num = np.sum(labeled_img == (i + 1))
                max_label = i + 1
        mcr = (labeled_img == max_label)
        return mcr.astype(np.int)

    @staticmethod
    def preprocessing2(mask_c1_array_biggest: object) -> list:
        if np.sum(mask_c1_array_biggest) == 0:
            minr, minc, maxr, maxc = [0, 0, 256, 256]
        else:
            region = regionprops(mask_c1_array_biggest)[0]
            minr, minc, maxr, maxc = region.bbox
        dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
        max_length = max(maxr - minr, maxc - minc)
        max_lengthl = int((256 / 256) * 80)
        preprocess1 = int((256 / 256) * 19)
        pp22 = int((256 / 256) * 31)
        if max_length > max_lengthl:
            ex_pixel = preprocess1 + max_length // 2
        else:
            ex_pixel = pp22 + max_length // 2
        dim1_cut_min = dim1_center - ex_pixel
        dim1_cut_max = dim1_center + ex_pixel
        dim2_cut_min = dim2_center - ex_pixel
        dim2_cut_max = dim2_center + ex_pixel
        if dim1_cut_min < 0:
            dim1_cut_min = 0
        if dim2_cut_min < 0:
            dim2_cut_min = 0
        if dim1_cut_max > 256:
            dim1_cut_max = 256
        if dim2_cut_max > 256:
            dim2_cut_max = 256
        return [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]

    def predict(self, path: str) -> str:
        images = self.tif2png(path)
        masks_list = []

        print('Inference...')
        with torch.no_grad():
            for index, im in enumerate(images):
                print(index + 1, '/', len(images))
                with torch.no_grad():
                    img, cut_image_orshape, or_shape, location = self.preprocessing(
                        im)
                    img = torch.unsqueeze(img, 0)
                    img = torch.unsqueeze(img, 0)
                    img = img.to(self.device)
                    img_array = (torch.squeeze(img)).data.cpu().numpy()

                    with torch.no_grad():
                        mask_c1 = self._model(img)
                        mask_c1 = torch.sigmoid(mask_c1)
                        mask_c1_array = (torch.squeeze(
                            mask_c1)).data.cpu().numpy()
                        mask_c1_array = (mask_c1_array > 0.5)
                        mask_c1_array = mask_c1_array.astype(np.float32)
                        mask_c1_array_biggest = self.largest_connect_component(
                            mask_c1_array.astype(np.int))

                    with torch.no_grad():
                        dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = self.preprocessing2(
                            mask_c1_array_biggest)
                        img_array_roi = img_array[dim1_cut_min:dim1_cut_max,
                                                  dim2_cut_min:dim2_cut_max]
                        img_array_roi_shape = img_array_roi.shape
                        img_array_roi = resize(
                            img_array_roi, (512, 512), order=3)
                        img_array_roi_tensor = torch.tensor(
                            data=img_array_roi, dtype=img.dtype)
                        img_array_roi_tensor = torch.unsqueeze(
                            img_array_roi_tensor, 0)
                        img_array_roi_tensor = torch.unsqueeze(
                            img_array_roi_tensor, 0).to(self.device)

                        mask_c2 = self._model2(img_array_roi_tensor)
                        mask_c2 = torch.sigmoid(mask_c2)
                        mask_c2_array = (torch.squeeze(
                            mask_c2)).data.cpu().numpy()
                        cascade2_t = 0.5
                        mask_c2_array = (mask_c2_array > cascade2_t)
                        mask_c2_array = mask_c2_array.astype(np.float32)
                        mask_c2_array = resize(
                            mask_c2_array, img_array_roi_shape, order=0)
                        mask_c1_array_biggest[dim1_cut_min:dim1_cut_max,
                                              dim2_cut_min:dim2_cut_max] = mask_c2_array
                        mask_c1_array_biggest = mask_c1_array_biggest.astype(
                            np.float32)

                    mask_c1_array_biggest = mask_c1_array_biggest.astype(
                        np.float32)
                    final_mask = np.zeros(
                        shape=or_shape, dtype=mask_c1_array_biggest.dtype)
                    mask_c1_array_biggest = resize(
                        mask_c1_array_biggest, cut_image_orshape, order=1)
                    final_mask[location[0]:location[1], location[2]:location[3]] = mask_c1_array_biggest
                    final_mask = (final_mask > 0.5)
                    final_mask = final_mask.astype(np.float32)
                masks_list.append(final_mask)

        img_fpath = '/'.join(path.split('/')[:-1])
        result_path = img_fpath + '/' + \
            path.split('/')[-1].split('.')[0] + '_result.tif'
        imageio.mimwrite(result_path, np.array(masks_list))
        print('Result saved')
        return result_path


if __name__ == '__main__':
    tif_path = 'D:/Study/CV_Project/FOR Github/dlv3p_segmentation/example/177_TIRADS5_cross.tif'
    segmentation_model = SegmentationModel(model_type='cross')
    segmentation_model.predict(path=tif_path)
