import matplotlib.pyplot as plt
import numpy as np
import cv2

color_array=['blue','red','violet','turquoise','green']
marker_array = ['o', 's', '^', 'D', 'x']  # 円, 四角, 上三角, ダイヤモンド, バツ
plt.rcParams["font.size"] = 14
plt.tight_layout()

gam_num=5
GAM_array=np.array([0.1, 0.5, 1, 5, 10])
EP=50

# テキストファイルを読み込み、shapeを(51, 5)に整える
PSNR = np.loadtxt('γ5_PSNR_noiseless.txt').reshape(gam_num,EP+1)
SSIM = np.loadtxt('γ5_SSIM_noiseless.txt').reshape(gam_num, EP+1)
PSNR_noise = np.loadtxt('γ5_PSNR_noisy.txt').reshape(gam_num,EP+1)
SSIM_noise = np.loadtxt('γ5_SSIM_noisy.txt').reshape(gam_num, EP+1)

#PSNRの推移を表示
x=np.linspace(0, EP+1, EP+1)

#fig=plt.figure(figsize=(8.6))
#ax=fig.add_subplot(1,1,1)

for i in range(gam_num):
  print(i)
  plt.plot(x, PSNR[i, :], color=color_array[i], label=f'noiseless(γ={GAM_array[i]})', linestyle='--', marker=marker_array[i], markevery=10)
  plt.plot(x, PSNR_noise[i, :], color=color_array[i], label=f'noisy(γ={GAM_array[i]})', marker=marker_array[i], markevery=10)
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('PSNR')
#plt.xticks([10,20,30,40,50],fontsize=9,color='black')
plt.legend(loc='lower left', fontsize=14, bbox_to_anchor=(1, 0))
#plt.ylim(21.5,35)
plt.savefig('PSNR_plot.png', dpi=300, bbox_inches='tight')
plt.show()

#SSIMの推移を表示
x=np.linspace(0, EP+1, EP+1)
for i in range(gam_num):
  plt.plot(x, SSIM[i, :], color=color_array[i], label=f'noiseless(γ={GAM_array[i]})',linestyle='--', marker=marker_array[i], markevery=10)
  plt.plot(x, SSIM_noise[i, :], color=color_array[i], label=f'noisy(γ={GAM_array[i]})', marker=marker_array[i], markevery=10)
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('SSIM')
plt.legend(loc='lower left', fontsize=14, bbox_to_anchor=(1, 0))
#plt.ylim(0.35,0.95)
plt.savefig('SSIM_plot.png', dpi=300, bbox_inches='tight')
plt.show()
