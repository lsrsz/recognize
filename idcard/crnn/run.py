import os
from crnn.train import CBC


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def main():
    model_save_path = "model/"
    img_size = (256, 32)
    num_classes = 11
    max_label_length = 26
    aug_number = 50
    epochs = 100

    model = CBC.build(img_size, num_classes, max_label_length)
    CBC.train(model,
              src_dir="D:/idcard/dataset/card_nbr/imgs/",
              save_dir=model_save_path,
              img_size=img_size,
              batch_size=16,
              aug_nbr=aug_number,
              max_label_length=max_label_length,
              epochs=epochs)

if __name__ == '__main__':
    main()
