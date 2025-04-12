from ADMM_TV import plt,np,EP,PSNR,SSIM,PSNR_noise,SSIM_noise,LAM_array

color_array=['blue','red','violet','turquoise','green']
plt.rcParams["font.size"] = 11
plt.tight_layout()

index_array=[0,1,2,3]

n=0

#PSNRの推移を表示
x=np.linspace(0, EP+1, EP+1)

for i in index_array:
  plt.plot(x, PSNR[i], color=color_array[i], linestyle='--', label=f'no-noise(λ={LAM_array[i]})')#, marker="o")
  plt.plot(x, PSNR_noise[i], color=color_array[i], label=f'noise(λ={LAM_array[i]})')
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('PSNR')
#plt.xticks([10,20,30,40,50],fontsize=9,color='black')
plt.legend(loc='lower left', fontsize=11, bbox_to_anchor=(1, 0))
#plt.ylim(21.5,35)
plt.show()
for i in index_array:
  print(PSNR_noise[i][EP])


#SSIMの推移を表示
x=np.linspace(0, EP+1, EP+1)
for i in index_array:
  plt.plot(x, SSIM[i], color=color_array[i], linestyle='--', label=f'no-noise(λ={LAM_array[i]})')
  plt.plot(x, SSIM_noise[i], color=color_array[i], label=f'noise(λ={LAM_array[i]})')
#plt.title('ADMM_TV')
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Number of iterations')
plt.ylabel('SSIM')
plt.legend(loc='lower left', fontsize=12, bbox_to_anchor=(1, 0))
#plt.ylim(0.35,0.95)
plt.show()
for i in index_array:
  print(SSIM_noise[i][EP])