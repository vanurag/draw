# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy

import matplotlib
import sys
import numpy as np

interactive = False  # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


def xrecons_grid(X, BB, B, A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize = 1
	padval = .5
	ph = B + 2 * padsize
	pw = A + 2 * padsize
	batch_size = X.shape[0]
	N = int(np.sqrt(batch_size))
	X = X.reshape((N, N, B, A))
	BB = BB.reshape((N, N, 3))
	img = np.ones((N * ph, N * pw)) * padval
	for i in range(N):
		for j in range(N):
			startr = i * ph + padsize
			endr = startr + B
			startc = j * pw + padsize
			endc = startc + A
			img[startr:endr, startc:endc] = X[i, j, :, :]
			BB_startr = BB[i, j, 1].astype(int) - np.sqrt(BB[i, j, 2]).astype(int)
			BB_startr = max(0, min(BB_startr, B))
			BB_endr = BB[i, j, 1].astype(int) + np.sqrt(BB[i, j, 2]).astype(int)
			BB_endr = max(0, min(BB_endr, B))
			BB_startc = BB[i, j, 0].astype(int) - np.sqrt(BB[i, j, 2]).astype(int)
			BB_startc = max(0, min(BB_startc, A))
			BB_endc = BB[i, j, 0].astype(int) + np.sqrt(BB[i, j, 2]).astype(int)
			BB_endc = max(0, min(BB_endc, A))
			img[startr + BB_startr, startc + BB_startc:startc + BB_endc] = 0.5
			img[startr + BB_endr, startc + BB_startc:startc + BB_endc] = 0.5
			img[startr + BB_startr:startr + BB_endr, startc + BB_startc] = 0.5
			img[startr + BB_startr:startr + BB_endr, startc + BB_endc] = 0.5
	return img


if __name__ == '__main__':
	prefix = sys.argv[1]
	out_file = sys.argv[2]
	[In, C, BBs, Lxs, Lzs] = np.load(out_file)
	T, batch_size, img_size = C.shape
	X = 1.0 / (1.0 + np.exp(-C))  # x_recons=sigmoid(canvas)
	B = A = int(np.sqrt(img_size))
	input_img = xrecons_grid(In, BBs[0, :, :], B, A)
	if interactive:
		f, arr = plt.subplots(2, T)
	for t in range(T):
		img = xrecons_grid(X[t, :, :], BBs[t, :, :], B, A)
		if interactive:
			arr[0, t].matshow(input_img, cmap=plt.cm.gray)
			arr[1, t].matshow(img, cmap=plt.cm.gray)
			arr[0, t].set_xticks([])
			arr[1, t].set_xticks([])
			arr[0, t].set_yticks([])
			arr[1, t].set_yticks([])
		else:
			plt.matshow(img, cmap=plt.cm.gray)
			imgname = '%s_%d.png' % (prefix, t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
	if not interactive:
		plt.matshow(input_img, cmap=plt.cm.gray)
		imgname = '%s_ref.png' % (prefix)
		plt.savefig(imgname)
		print(imgname)
	f = plt.figure()
	plt.plot(Lxs, label='Reconstruction Loss Lx')
	plt.plot(Lzs, label='Latent Loss Lz')
	plt.xlabel('iterations')
	plt.legend()
	if interactive:
		plt.show()
	else:
		plt.savefig('%s_loss.png' % (prefix))
