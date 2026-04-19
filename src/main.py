import pandas as pd
from torchvision.models import DenseNet

from models.vision_encoder import Xencoder
from data import MIMICReduced


if __name__ == '__main__':
    enc = Xencoder()

    train_ds = MIMICReduced(
        df=pd.read_csv('./ds_train.csv'),
        label_column='hospital_expire_flag',
        images_extension='jpg',
        images_base_dir='../mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.1.0/files',
        debug=True
    )

    image, x, y = train_ds[3]
    print(image.shape)
    print(x)
    print(y)
