from collections import defaultdict
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


class MM_CAD_Dataset(Dataset):
    def __init__(self, datalist=['ADNI-train']) -> None:
        super().__init__()
        df_l = []
        for d in datalist:
            f_n = f'./local_data/{d}.csv'
            with open(f_n) as f:
                for ln in f.readlines():
                    p, r = ln.strip('\n').split('\t')
                    df_l.append((p, r))
        self.df = df_l

    def _p_im(self, im, sz=224):
        x, y, z = im.shape
        im = im.unsqueeze(0).unsqueeze(0)
        m_s = max(x, y, z)
        n_s = (int(sz * x / m_s), int(sz * y / m_s), int(sz * z / m_s))
        im = F.interpolate(im, size=n_s, mode='trilinear', align_corners=True)
        nx, ny, nz = n_s
        n_im = torch.zeros((1, 1, sz, sz, sz))
        xm = int((sz - nx) / 2)
        ym = int((sz - ny) / 2)
        zm = int((sz - nz) / 2)
        n_im[:, :, xm:xm + nx, ym:ym + ny, zm:zm + nz] = im
        return n_im

    def _n_im(self, im):
        return (im - im.min()) / (im.max() - im.min())

    def __getitem__(self, idx):
        p, r = self.df[idx]
        im = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(float))
        im = self._p_im(self._n_im(im))
        return im, r

    def __len__(self):
        return len(self.df)


class MM_Collator:
    def __init__(self):
        pass

    def __call__(self, b):
        inp = defaultdict(list)
        for d in b:
            inp['images'].append(d[0])
            inp['reports'].append(d[1])
        inp['images'] = torch.cat(inp['images'], 0)
        return inp


class ZSEvalSet(Dataset):
    def __init__(self, datalist=['ADNI-train']) -> None:
        super().__init__()
        df_l = []
        for d in datalist:
            f_n = f'./local_data/{d}.csv'
            with open(f_n) as f:
                for ln in f.readlines():
                    p, r = ln.strip('\n').split('\t')
                    df_l.append((p, r))
        self.df = df_l

    def _p_im(self, im, sz=224):
        x, y, z = im.shape
        im = im.unsqueeze(0).unsqueeze(0)
        m_s = max(x, y, z)
        n_s = (int(sz * x / m_s), int(sz * y / m_s), int(sz * z / m_s))
        im = F.interpolate(im, size=n_s, mode='trilinear', align_corners=True)
        nx, ny, nz = n_s
        n_im = torch.zeros((1, 1, sz, sz, sz))
        xm = int((sz - nx) / 2)
        ym = int((sz - ny) / 2)
        zm = int((sz - nz) / 2)
        n_im[:, :, xm:xm + nx, ym:ym + ny, zm:zm + nz] = im
        return n_im

    def _n_im(self, im):
        return (im - im.min()) / (im.max() - im.min())

    def __getitem__(self, idx):
        p, r = self.df[idx]
        im = torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(float))
        im = self._p_im(self._n_im(im))
        return im, r

    def __len__(self):
        return len(self.df)


class ZSEvalCollator:
    def __init__(self):
        pass

    def __call__(self, b):
        inp = defaultdict(list)
        for d in b:
            inp['images'].append(d[0])
            inp['reports'].append(d[1])
        inp['images'] = torch.cat(inp['images'], 0)
        return inp