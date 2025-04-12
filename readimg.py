import cv2 # opencvをインポート
import numpy as np
import matplotlib.pyplot as plt
import os

def show_image(img, title='', fontsize=20):
    fig, ax = plt.subplots()
    plt.gray() # グレースケールでプロットする際に必要
    ax.imshow(img, vmin=0, vmax=255)
    ax.axis('off')
    ax.set_title(title, fontsize=fontsize)
    plt.show()

# 以下を実行するとファイルをアップロードするフォームが出てくるので
# 「ファイル選択（Choose Files）」から画像ファイルをアップロードする


img_num=20 #画像枚数
SIZE=256 #画像サイズ

# 画像フォルダのパス
img_dir = './image'  # 同じディレクトリ内の "image" フォルダ
img=np.zeros((img_num,SIZE,SIZE))

for i in range(img_num):
    file_path = os.path.join(img_dir, str(i) + ".bmp")
    img[i] = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

##画像を分割
## 1  2  3  4
## 5  6  7  8
## 9 10 11 12
##13 14 15 16 という順

SIZE=256
size=16 #分割後の画像サイズ
num=int((SIZE/size)**2) #分割数
print(num)
new_image=np.zeros((img_num,256,256))
split_image=np.zeros((img_num,num,size,size))
image=np.zeros((img_num,num,size,size))

for i in range(img_num):
  v_size = img[i].shape[0] // size * size
  h_size = img[i].shape[1] // size * size
  img[i] = img[i][:v_size, :h_size]

  v_split = img[i].shape[0] // size
  h_split = img[i].shape[1] // size
  out_img = []
  [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img[i], v_split)]

  for j in range(num):
    image[i][j]=out_img[j]

