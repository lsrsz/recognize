from PIL import Image
import numpy as np

def main():
    path="D:/idcard/crnn/train.npz"
    data=np.load(path)
    print(data.files)
    #print(data['data'])
    #x = data["data"]
    #for i in range(15):
        #im = Image.fromarray(np.uint8(x[1]))
        #im.show()
if __name__ == '__main__':
    main()
