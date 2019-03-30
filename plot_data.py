# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy

import matplotlib
import sys
import numpy as np

interactive = False  # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


def xrecons_grid(X, rBB, wBB, B, A, draw_with_white):
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
	if rBB is not None:
		rBB = rBB.reshape((N, N, 3))
	if wBB is not None:
		wBB = wBB.reshape((N, N, 3))
	img = np.ones((N * ph, N * pw)) * padval
	rgb_img = np.stack((img,) * 3, axis=-1)
	for i in range(N):
		for j in range(N):
			startr = i * ph + padsize
			endr = startr + B
			startc = j * pw + padsize
			endc = startc + A
			rgb_img[startr:endr, startc:endc, 0] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
			rgb_img[startr:endr, startc:endc, 1] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
			rgb_img[startr:endr, startc:endc, 2] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]

			# Reading bounding box
			if rBB is not None:
				BB_startr = rBB[i, j, 1].astype(int) - (rBB[i, j, 2] / 2).astype(int)
				BB_startr = max(0, min(BB_startr, B))
				BB_endr = rBB[i, j, 1].astype(int) + (rBB[i, j, 2] / 2).astype(int)
				BB_endr = max(0, min(BB_endr, B))
				BB_startc = rBB[i, j, 0].astype(int) - (rBB[i, j, 2] / 2).astype(int)
				BB_startc = max(0, min(BB_startc, A))
				BB_endc = rBB[i, j, 0].astype(int) + (rBB[i, j, 2] / 2).astype(int)
				BB_endc = max(0, min(BB_endc, A))
				rgb_img[startr + BB_startr, startc + BB_startc:startc + BB_endc, :] = [1, 0, 0]
				rgb_img[startr + BB_endr, startc + BB_startc:startc + BB_endc, :] = [1, 0, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_startc, :] = [1, 0, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_endc, :] = [1, 0, 0]
			
			# Writing bounding box
			if wBB is not None:
				BB_startr = wBB[i, j, 1].astype(int) - (wBB[i, j, 2] / 2).astype(int)
				BB_startr = max(0, min(BB_startr, B))
				BB_endr = wBB[i, j, 1].astype(int) + (wBB[i, j, 2] / 2).astype(int)
				BB_endr = max(0, min(BB_endr, B))
				BB_startc = wBB[i, j, 0].astype(int) - (wBB[i, j, 2] / 2).astype(int)
				BB_startc = max(0, min(BB_startc, A))
				BB_endc = wBB[i, j, 0].astype(int) + (wBB[i, j, 2] / 2).astype(int)
				BB_endc = max(0, min(BB_endc, A))
				rgb_img[startr + BB_startr, startc + BB_startc:startc + BB_endc, :] = [0, 1, 0]
				rgb_img[startr + BB_endr, startc + BB_startc:startc + BB_endc, :] = [0, 1, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_startc, :] = [0, 1, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_endc, :] = [0, 1, 0]
	return rgb_img


if __name__ == '__main__':
	prefix = sys.argv[1]
	out_file = sys.argv[2]
	draw_with_white = False
	[In, C, rBBs, wBBs, Lxs, Lzs] = np.load(out_file)
	T, batch_size, img_size = C.shape
# 	X = 1.0 / (1.0 + np.exp(-C))  # x_recons=sigmoid(canvas)
	X = (np.exp(2 * C) - 1) / (np.exp(2 * C) + 1)  # x_recons=tanh(canvas)
	B = A = int(np.sqrt(img_size))
	input_img = xrecons_grid(In, None, None, B, A, draw_with_white)
	if interactive:
		f, arr = plt.subplots(2, T)
	for t in range(T):
		img = xrecons_grid(X[t, :, :], rBBs[t, :, :], wBBs[t, :, :], B, A, draw_with_white)
		if interactive:
			arr[0, t].imshow(input_img, vmin=0, vmax=1)
			arr[1, t].imshow(img, vmin=0, vmax=1)
			arr[0, t].set_xticks([])
			arr[1, t].set_xticks([])
			arr[0, t].set_yticks([])
			arr[1, t].set_yticks([])
		else:
			plt.imshow(img, vmin=0, vmax=1)
			imgname = '%s_%d.png' % (prefix, t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
	if not interactive:
		plt.imshow(input_img, vmin=0, vmax=1)
		imgname = '%s_ref.png' % (prefix)
		plt.savefig(imgname)
		print(imgname)
		res_img = xrecons_grid(X[-1, :, :], None, None, B, A, draw_with_white)
		plt.imshow(res_img, vmin=0, vmax=1)
		imgname = '%s_result.png' % (prefix)
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
