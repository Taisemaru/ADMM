import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from readimg import SIZE,size,num,img_num,image,img
import cv2
from skimage.metrics import structural_similarity as ssim

#諸々設定======================================
N_pre=SIZE*SIZE #原画像のピクセル数
N=size*size   #分割後のピクセル数
M=N  #AはM×Nの行列(今回は単位行列)
LAM=0.03 #ラムダ
GAM=10 #ガンマ1定義
EP=50 #反復回数
GAM_num=1 #比べるγの数
amp=256
stan_devi=np.sqrt((amp-1)*2.56*(10**(-9))) #付加雑音の標準偏差
print(stan_devi*stan_devi)
max=255
#=============================================

#雑音
LOC=0 #正規分布の平均
SCALE=10 #正規分布の標準偏差

#観測行列を生成
A=np.eye(N)
A_T=A.transpose()

#全変動の行列Dを定義
D_v=np.eye(N)*(-1) #縦差分
for i in range(N-1):
  D_v[i+1][i]=1
D_v[0][N-1]=1
D_u=np.eye(N)*(-1) #横差分
j=0
for i in range(size,N):
  D_u[i][j]=1
  j=j+1
j=0
for i in range(N-size,N):
  D_u[j][i]=1
  j=j+1
D=np.concatenate((D_v,D_u))
D_T=D.transpose()

# ノイズの配列をあらかじめ用意
noise=np.zeros((num,M,1))
amp_noise=np.zeros((num,EP,2*M,1))
no_noise=np.zeros((num,EP,2*M,1))
for i in range(num):
  noise[i]=np.random.normal(LOC, SCALE, (M,1))
  for j in range(EP):
    amp_noise[i][j]=np.random.normal(LOC, stan_devi, (2*M,1))
print('標準偏差：',stan_devi)

#画像を1次元ベクトルに変換
def transvector(k,img_origin):
  index=0
  vector=np.zeros((N,1))
  for i in range(size):
    for j in range(size):
      vector[index][0]=img_origin[j][i]
      index+=1
  return vector

#1次元ベクトルを画像のサイズに変換
def transimg(k,vector_origin):
  index=0
  img_size=np.zeros((size,size))
  for i in range(size):
    for j in range(size):
      img_size[j][i]=int(vector_origin[index])
      index+=1
  return img_size




class ADMMDeepUnfolding(nn.Module):
    def __init__(self, N, D, EP, lam):
        super(ADMMDeepUnfolding, self).__init__()
        self.N = N
        self.D = torch.tensor(D, dtype=torch.float32)
        self.DT = self.D.T
        self.lam = lam
        self.EP = EP

        # γ̃（学習用パラメータ）→ γ = softplus(γ̃)
        self.gamma_raw = nn.Parameter(torch.tensor(2.0))  # 初期値 log(exp(10)-1)
        self.softplus = nn.Softplus()

    def prox(self, x, lam):
        x1 = x[:self.N]
        x2 = x[self.N:]
        norm = torch.sqrt(x1**2 + x2**2 + 1e-8)
        factor = torch.clamp(1 - lam / norm, min=0.0)
        return torch.cat([x1 * factor, x2 * factor], dim=0)

    def forward(self, y, x_0):
       gamma = self.softplus(self.gamma_raw)  # 正の γ を取得
       C = torch.eye(self.N, device=y.device) + (self.DT @ self.D) / gamma
       C_inv = torch.inverse(C)
       GAMLAM = gamma * self.lam
       
       x = y.clone()
       z = torch.zeros((2 * self.N, 1), device=y.device)
       v = torch.zeros((2 * self.N, 1), device=y.device)
       losses = []
       
       for _ in range(self.EP):

        noise_amp = torch.randn((2 * self.N, 1), device=x.device) * stan_devi

        x = C_inv @ (y + self.DT @ (z - v) / gamma)
        z_input = self.D @ x + v + noise_amp
        z = self.prox(z_input, GAMLAM)
        v = z_input - z

        mse = torch.mean((x - x_0) ** 2)
        losses.append(mse)
        return x, sum(losses) / self.EP
    
    def get_gamma(self):
        return self.softplus(self.gamma_raw).item()
    
# 1画像だけ使ってγ学習したい場合（複数画像でも可）
n = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADMMDeepUnfolding(N=N, D=D, EP=EP, lam=LAM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

epochs = 50
num_blocks = num  # 分割数

for epoch in range(epochs):
    total_loss = 0
    for h in range(num_blocks):  # 各画像ブロックに対して
        x0_np = transvector(h, image[0][h]) / max  # 正規化
        y_np = x0_np + noise[h]                    # ノイズ追加

        x_0 = torch.tensor(x0_np, dtype=torch.float32).to(device)
        y = torch.tensor(y_np, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        _, loss = model(y, x_0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / num_blocks:.6f}, γ = {model.get_gamma():.6f}")

GAM = model.get_gamma()
print('GAM=',GAM)

# 結果を入れるリストを用意
result_whole=np.zeros((img_num,EP+1))
PS=np.zeros((img_num,2,EP+1))
SS=np.zeros((img_num,2,EP+1))
#行列を定義
x_noise=np.zeros((img_num,num,size,size)) #ノイズ画像用
x_img=np.zeros((img_num,2,num,size,size)) #推定画像用
#x_img_ampnoise=np.zeros((GAM_num,num,size,size)) #推定画像用(増幅器ノイズあり)
#ADMMアルゴリズム(戻り値はMSE,PSNR,SSIMの反復ごとの値を格納した配列)
def ADMM_alg(noised,whi,number):
  gamma=GAM
  lammda=LAM
  C=np.eye(N)+D.T@D/gamma
  C_I=np.linalg.inv(C) #逆行列
  GAMLAM=gamma*lammda
  result=np.zeros((img_num,num,EP+1)) #分割した画像ごとのMSEを格納

  #l1.2normのprox
  def prox(x,lam):
    xg=np.zeros((N,1))
    temp=np.zeros((N,1))
    for i in range(N):
      xg[i]=np.sqrt(x[i]**2+x[i+N]**2)
    temp=np.maximum(1-lam/xg,0)
    pre = np.concatenate((temp,temp), axis = 0)
    return pre*x

  #　hごとに分割した各画像で処理を行う
  for h in range(num):

      #画像を1次元列ベクトルに変換
      x_0=np.zeros((N,1))
      x_0=transvector(h,image[number][h])
      D_vu=D@(x_0/max)
      #観測ベクトルを生成&値を0~1に正規化
      y=A@x_0+noise[h]
      y=y/max

      #ノイズありの画像を保存(後で画像を比べるため)
      x_noise[number][h]=transimg(h,y*max)

      #ADMMの初期値の設定
      x_t=y
      z_t=np.zeros((2*N,1))
      v_t=np.zeros((2*N,1))
      x_0_temp=np.zeros((size,size)) #SSIM計算用の整数を代入する配列
      x_t_temp=np.zeros((size,size)) #SSIM計算用の整数を代入する配列
      temp_alg=np.zeros((2*N,1))    #アルゴリズム中の計算省略用

      #MSE,PSNR,SSIMを最初に計算
      result[number][h][0]+=np.linalg.norm(x_t-x_0/max)**2/N
      x_0_temp=transimg(h,x_0)
      x_t_temp=transimg(h,x_t*max)
      PS[number][whi][0]+=cv2.PSNR(x_0_temp, x_t_temp)
      SS[number][whi][0]+=ssim(x_0_temp, x_t_temp, data_range=255)

      ######## ADMM ###################
      for i in range(EP):
        x_t=C_I@(y+D_T@(z_t-v_t)/gamma)
        temp_alg=D@x_t+v_t+noised[h][i]
        z_t=prox(temp_alg,GAMLAM)
        v_t=temp_alg-z_t

        #結果を追加
        result[number][h][i+1]+=np.linalg.norm(x_t-x_0/max)**2/N
        x_t_temp=transimg(h,x_t*max)
        PS[number][whi][i+1]+=cv2.PSNR(x_0_temp, x_t_temp)
        SS[number][whi][i+1]+=ssim(x_0_temp, x_t_temp, data_range=255)

      ##################################

      #1次元列ベクトルから画像用の行列に変換
      x_img[number][whi][h]=transimg(h,x_t*max)

      #γ毎の画像全体のMSEの推移を格納
      for i in range(EP+1):
        result_whole[number][i]+=result[number][h][i]/num #MSEの平均

  return result_whole[number],PS[number][whi],SS[number][whi]



result_no_noise=np.zeros((img_num,EP+1))
result_amp_noise=np.zeros((img_num,EP+1))
PSNR_no_noise=np.zeros((img_num,EP+1))
PSNR_amp_noise=np.zeros((img_num,EP+1))
SSIM_no_noise=np.zeros((img_num,EP+1))
SSIM_amp_noise=np.zeros((img_num,EP+1))
for image_number in range(img_num):
  result_no_noise[image_number],PSNR_no_noise[image_number],SSIM_no_noise[image_number]=ADMM_alg(no_noise,0,image_number)
  result_amp_noise[image_number],PSNR_amp_noise[image_number],SSIM_amp_noise[image_number]=ADMM_alg(amp_noise,1,image_number)

side=16
path='C:/Users/taise/iCloudDrive/卒論資料/python/local'
for n in range(img_num):
  noised_img=np.vstack([np.hstack(x_noise[n][i*side:(i+1)*side])for i in range(side)])
  estimate_img0=np.vstack([np.hstack(x_img[n][0][i*side:(i+1)*side])for i in range(side)])
  estimate_img1=np.vstack([np.hstack(x_img[n][1][i*side:(i+1)*side])for i in range(side)])

print(cv2.PSNR(img[0],  noised_img))
print(cv2.PSNR(img[0],  estimate_img1))