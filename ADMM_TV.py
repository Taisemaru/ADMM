
import numpy as np
from readimg import SIZE,size,num,img_num,img,image

#諸々設定======================================
N_pre=SIZE*SIZE #原画像のピクセル数
N=size*size   #分割後のピクセル数
M=N  #AはM×Nの行列(今回は単位行列)
LAM=0.03 #ラムダ
GAM=2 #ガンマ1定義
EP=50 #反復回数
GAM_num=5 #比べるγの数
stan_devi=np.sqrt(6.53*(10**(-4))) #付加雑音の標準偏差
max=255
#=============================================

#LAM_array=np.array([0.01, 0.03, 0.05, 0.07])
LAM_array=np.array([0.03])
GAM_array=np.array([0.1, 0.5, 1, 5, 10])

import cv2
from skimage.metrics import structural_similarity as ssim

#雑音
LOC=0 #正規分布の平均
SCALE=10 #正規分布の標準偏差

#観測行列を生成
A=np.eye(N)
#A=np.random.randn(M,N)
A_T=A.transpose()

print(N)

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

#行列を定義
x_noise=np.zeros((img_num,num,size,size)) #ノイズ画像用
x_img=np.zeros((img_num,2,GAM_num,num,size,size)) #推定画像用
#x_img_ampnoise=np.zeros((GAM_num,num,size,size)) #推定画像用(増幅器ノイズあり)

# 結果を入れるリストを用意
result_whole=np.zeros((img_num,GAM_num,EP+1))
PS=np.zeros((img_num,2,GAM_num,EP+1))
SS=np.zeros((img_num,2,GAM_num,EP+1))

#l1.2normのprox
def prox(x,lam):
  xg=np.zeros((N,1))
  temp=np.zeros((N,1))
  for i in range(N):
    xg[i]=np.sqrt(x[i]**2+x[i+N]**2)
  temp=np.maximum(1-lam/xg,0)
  pre = np.concatenate((temp,temp), axis = 0)
  return pre*x

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

#ADMMアルゴリズム(戻り値はMSE,PSNR,SSIMの反復ごとの値を格納した配列)
def ADMM_alg(g,noised,whi,number):
  gamma=GAM_array[g] #γの値
  lammda=LAM
  C=np.eye(N)+D.T@D/gamma
  C_I=np.linalg.inv(C) #逆行列
  GAMLAM=gamma*lammda
  result=np.zeros((img_num,num,EP+1)) #分割した画像ごとのMSEを格納

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
      #PS[画像番号][増幅器雑音ありか][γ][反復回数]
      PS[number][whi][g][0]+=cv2.PSNR(x_0_temp, x_t_temp)
      SS[number][whi][g][0]+=ssim(x_0_temp, x_t_temp, data_range=255)

      ######## ADMM ###################
      for i in range(EP):
        x_t=C_I@(y+D_T@(z_t-v_t)/gamma)
        temp_alg=D@x_t+v_t+noised[h][i]
        z_t=prox(temp_alg,GAMLAM)
        v_t=temp_alg-z_t

        #結果を追加
        result[number][h][i+1]+=np.linalg.norm(x_t-x_0/max)**2/N
        x_t_temp=transimg(h,x_t*max)
        PS[number][whi][g][i+1]+=cv2.PSNR(x_0_temp, x_t_temp)
        SS[number][whi][g][i+1]+=ssim(x_0_temp, x_t_temp, data_range=255)

      #################################

      #print("   PSNR=",cv2.PSNR(x_0, x_t*max))
      #print("   SSIM=",ssim(x_0_temp, x_t_temp, data_range=255))

      #1次元列ベクトルから画像用の行列に変換
      x_img[number][whi][g][h]=transimg(h,x_t*max)

      #γ毎の画像全体のMSEの推移を格納
      for i in range(EP+1):
        result_whole[number][g][i]+=result[number][h][i]/num #MSEの平均

  return result_whole[number][g],PS[number][whi][g],SS[number][whi][g]

#アルゴリズムを繰り返す
#===============================================================================
result_no_noise=np.zeros((img_num,GAM_num,EP+1))
result_amp_noise=np.zeros((img_num,GAM_num,EP+1))
PSNR_no_noise=np.zeros((img_num,GAM_num,EP+1))
PSNR_amp_noise=np.zeros((img_num,GAM_num,EP+1))
SSIM_no_noise=np.zeros((img_num,GAM_num,EP+1))
SSIM_amp_noise=np.zeros((img_num,GAM_num,EP+1))

for image_number in range(img_num):
  print(image_number+1,image_number+1,image_number+1,image_number+1,image_number+1,image_number+1,image_number+1,image_number+1,image_number+1,image_number+1)
  for l in range(GAM_num): #　gごとにγの値変わる
    result_no_noise[image_number][l],PSNR_no_noise[image_number][l],SSIM_no_noise[image_number][l]=ADMM_alg(l,no_noise,0,image_number)
    result_amp_noise[image_number][l],PSNR_amp_noise[image_number][l],SSIM_amp_noise[image_number][l]=ADMM_alg(l,amp_noise,1,image_number)
    #print(PSNR_no_noise[image_number][l][EP], PSNR_amp_noise[image_number][l][EP])

#===============================================================================

for image_number in range(img_num): #画像ごとに
  for g in range(GAM_num): #γごと
    for i in range(EP+1): #反復回数ごと
      PSNR_no_noise[image_number][g][i]/=num
      PSNR_amp_noise[image_number][g][i]/=num
      SSIM_no_noise[image_number][g][i]/=num
      SSIM_amp_noise[image_number][g][i]/=num

#すべての画像のPSNRとSSIMの平均をとる
MSE=np.zeros((GAM_num,EP+1))
PSNR=np.zeros((GAM_num,EP+1))
PSNR_noise=np.zeros((GAM_num,EP+1))
SSIM=np.zeros((GAM_num,EP+1))
SSIM_noise=np.zeros((GAM_num,EP+1))
for image_number in range(img_num):
  for g in range(GAM_num):
    for i in range(EP+1):
      PSNR[g][i]+=PSNR_no_noise[image_number][g][i]
      PSNR_noise[g][i]+=PSNR_amp_noise[image_number][g][i]
      SSIM[g][i]+=SSIM_no_noise[image_number][g][i]
      SSIM_noise[g][i]+=SSIM_amp_noise[image_number][g][i]

for g in range(GAM_num):
  for i in range(EP+1):
    PSNR[g][i]/=img_num
    PSNR_noise[g][i]/=img_num
    SSIM[g][i]/=img_num
    SSIM_noise[g][i]/=img_num


#グラフプロット用にテキストファイルへ書き込み
np.savetxt('γ5_PSNR_noiseless.txt', PSNR.reshape(-1, 1))
np.savetxt('γ5_SSIM_noiseless.txt', SSIM.reshape(-1, 1))
np.savetxt('γ5_PSNR_noisy.txt', PSNR_noise.reshape(-1, 1))
np.savetxt('γ5_SSIM_noisy.txt', SSIM_noise.reshape(-1, 1))


side=16
path='/Users/katotaisei/Library/CloudStorage/OneDrive-個人用/2025年研究/ADMM/ADMM_result/image'
for n in range(img_num):
  noised_img=np.vstack([np.hstack(x_noise[n][i*side:(i+1)*side])for i in range(side)])
  estimate_img0=np.vstack([np.hstack(x_img[n][0][0][i*side:(i+1)*side])for i in range(side)])
  estimate_img1=np.vstack([np.hstack(x_img[n][1][0][i*side:(i+1)*side])for i in range(side)])
  filename_noise=f'γ5_{n}(noise).png'
  filename_0=f'γ5_{n}(no_circuit_noise).png'
  filename_1=f'γ5_{n}(circuit_noise).png'
  cv2.imwrite(path+"/"+filename_noise, noised_img)
  cv2.imwrite(path+"/"+filename_0, estimate_img0)
  cv2.imwrite(path+"/"+filename_1, estimate_img1)