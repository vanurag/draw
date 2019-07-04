# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy

import matplotlib
import sys
import numpy as np
from numpy import dtype

interactive = False  # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


def xrecons_grid(X, rBB, wBB, B, A, draw_with_white, T, n_samples, sample_rgb_img):
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
	draw_locations = np.empty(shape=(0, 2))
	for i in range(N):
		for j in range(N):
			sid = i + N * j
			startr = i * ph + padsize
			endr = startr + B
			startc = j * pw + padsize
			endc = startc + A
			rgb_img[startr:endr, startc:endc, 0] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
			rgb_img[startr:endr, startc:endc, 1] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
			rgb_img[startr:endr, startc:endc, 2] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
			
			if rBB is not None:
				if np.isnan(rBB[i, j, 0]):
					continue
				if sample_rgb_img is not None and sid < n_samples:
					sample_rgb_img[sid * B:(sid + 1) * B, T * A:(T + 1) * A, 0] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
					sample_rgb_img[sid * B:(sid + 1) * B, T * A:(T + 1) * A, 1] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]
					sample_rgb_img[sid * B:(sid + 1) * B, T * A:(T + 1) * A, 2] = X[i, j, :, :] if draw_with_white else 1.0 - X[i, j, :, :]

			# Reading bounding box
			if rBB is not None:
				if np.isnan(rBB[i, j, 0]):
					continue
				BB_startr = rBB[i, j, 1].astype(int) - (rBB[i, j, 2] / 2).astype(int)
				BB_startr = max(0, min(BB_startr, B - 1))
				BB_endr = rBB[i, j, 1].astype(int) + (rBB[i, j, 2] / 2).astype(int)
				BB_endr = max(0, min(BB_endr, B - 1))
				BB_startc = rBB[i, j, 0].astype(int) - (rBB[i, j, 2] / 2).astype(int)
				BB_startc = max(0, min(BB_startc, A - 1))
				BB_endc = rBB[i, j, 0].astype(int) + (rBB[i, j, 2] / 2).astype(int)
				BB_endc = max(0, min(BB_endc, A - 1))
				rgb_img[startr + BB_startr, startc + BB_startc:startc + BB_endc, :] = [1, 0, 0]
				rgb_img[startr + BB_endr, startc + BB_startc:startc + BB_endc, :] = [1, 0, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_startc, :] = [1, 0, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_endc, :] = [1, 0, 0]
				if sample_rgb_img is not None and sid < n_samples:
					sample_rgb_img[sid * B + BB_startr, T * A + BB_startc:T * A + BB_endc, :] = [1, 0, 0]
					sample_rgb_img[sid * B + BB_endr, T * A + BB_startc:T * A + BB_endc, :] = [1, 0, 0]
					sample_rgb_img[sid * B + BB_startr:BB_endr, T * A + BB_startc, :] = [1, 0, 0]
					sample_rgb_img[sid * B + BB_startr:i * B + BB_endr, T * A + BB_endc, :] = [1, 0, 0]
			
			# Writing bounding box
			if wBB is not None:
				if np.isnan(wBB[i, j, 0]):
					continue
				BB_startr = wBB[i, j, 1].astype(int) - (wBB[i, j, 2] / 2).astype(int)
				BB_startr = max(0, min(BB_startr, B - 1))
				BB_endr = wBB[i, j, 1].astype(int) + (wBB[i, j, 2] / 2).astype(int)
				BB_endr = max(0, min(BB_endr, B - 1))
				BB_centerr = (BB_startr + BB_endr) / 2
				BB_startc = wBB[i, j, 0].astype(int) - (wBB[i, j, 2] / 2).astype(int)
				BB_startc = max(0, min(BB_startc, A - 1))
				BB_endc = wBB[i, j, 0].astype(int) + (wBB[i, j, 2] / 2).astype(int)
				BB_endc = max(0, min(BB_endc, A - 1))
				BB_centerc = (BB_startc + BB_endc) / 2
				rgb_img[startr + BB_startr, startc + BB_startc:startc + BB_endc, :] = [0, 1, 0]
				rgb_img[startr + BB_endr, startc + BB_startc:startc + BB_endc, :] = [0, 1, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_startc, :] = [0, 1, 0]
				rgb_img[startr + BB_startr:startr + BB_endr, startc + BB_endc, :] = [0, 1, 0]
				if sample_rgb_img is not None and sid < n_samples:
					sample_rgb_img[sid * B + BB_startr, T * A + BB_startc:T * A + BB_endc, :] = [0, 1, 0]
					sample_rgb_img[sid * B + BB_endr, T * A + BB_startc:T * A + BB_endc, :] = [0, 1, 0]
					sample_rgb_img[sid * B + BB_startr:i * B + BB_endr, T * A + BB_startc, :] = [0, 1, 0]
					sample_rgb_img[sid * B + BB_startr:i * B + BB_endr, T * A + BB_endc, :] = [0, 1, 0]
					
# 				draw_locations = np.append(draw_locations, [[startr + BB_centerr, startc + BB_centerc]], axis=0)
# 				draw_locations = np.append(draw_locations, [[startc + BB_centerc, N * ph - (startr + BB_centerr)]], axis=0)
				draw_locations = np.append(draw_locations, [[startc + BB_centerc, startr + BB_centerr]], axis=0)
	return rgb_img, draw_locations


if __name__ == '__main__':
	prefix = sys.argv[1]
	out_file = sys.argv[2]
	[In, C, rBBs, wBBs, wTs, draw_with_white] = np.load(out_file)
	T, batch_size, img_size = C.shape
	n_samples = int(np.sqrt(batch_size))  # batch_size // 10
	B = A = int(np.sqrt(img_size))
	draw_trajectory = np.empty(shape=(0, 2))
	
# 	X = 1.0 / (1.0 + np.exp(-C))  # x_recons=sigmoid(canvas)
	X = (np.exp(2 * C) - 1) / (np.exp(2 * C) + 1)  # x_recons=tanh(canvas)
	C_final = C[wTs - 1, np.arange(batch_size), :]
	X_final = (np.exp(2 * C_final) - 1) / (np.exp(2 * C_final) + 1)  # x_recons=tanh(canvas)
	input_img, _ = xrecons_grid(In, None, None, B, A, draw_with_white, 0, 0, None)
	sample_img = np.ones((n_samples * B, T * A))
	sample_rgb_img = np.stack((sample_img,) * 3, axis=-1)
	if interactive:
		f, arr = plt.subplots(2, T)
	for t in range(T):
		X_plot = np.empty(shape=(0, img_size), dtype=np.float32)
		rBB_plot = np.empty(shape=(0, 3), dtype=np.float32)
		wBB_plot = np.empty(shape=(0, 3), dtype=np.float32)
		for b in range(batch_size):
			X_plot = np.vstack((X_plot, X[t, b, :] if t < wTs[b] else X_final[b, :]))
			rBB_plot = np.vstack((rBB_plot, rBBs[t, b, :] if t < wTs[b] else np.array([np.nan, np.nan, np.nan])))
			wBB_plot = np.vstack((wBB_plot, wBBs[t, b, :3] if t < wTs[b] else np.array([np.nan, np.nan, np.nan])))
		img, draw_locations = xrecons_grid(X_plot, rBB_plot, wBB_plot, B, A, draw_with_white, t, n_samples, sample_rgb_img)
		draw_trajectory = np.append(draw_trajectory, draw_locations, axis=0)
		if interactive:
			arr[0, t].imshow(input_img, vmin=0, vmax=1)
			arr[1, t].imshow(img, vmin=0, vmax=1)
			arr[1, t].scatter(draw_trajectory[:, 0], draw_trajectory[:, 1], alpha=0.5, color='blue', s=0.1)
			arr[0, t].set_xticks([])
			arr[1, t].set_xticks([])
			arr[0, t].set_yticks([])
			arr[1, t].set_yticks([])
		else:
			plt.clf()
			plt.imshow(img, vmin=0, vmax=1)
			plt.scatter(draw_trajectory[:, 0], draw_trajectory[:, 1], alpha=0.5, color='blue', s=0.1)
			plt.xlim(0, input_img.shape[1])
			plt.ylim(input_img.shape[0], 0)
			plt.xticks([])
			plt.yticks([])
			imgname = '%s_%d.png' % (prefix, t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
	if not interactive:
		plt.clf()
		plt.imshow(input_img, vmin=0, vmax=1)
		imgname = '%s_ref.png' % (prefix)
		plt.savefig(imgname)
		print(imgname)
		res_img, _ = xrecons_grid(X_final, None, None, B, A, draw_with_white, T - 1, 0, None)
		plt.clf()
		plt.imshow(res_img, vmin=0, vmax=1)
		imgname = '%s_result.png' % (prefix)
		plt.savefig(imgname)
		print(imgname)
		# Sample
# 		plt.clf()
# 		plt.imshow(sample_rgb_img, vmin=0, vmax=1)
# 		imgname = '%s_sample.png' % (prefix)
# 		plt.savefig(imgname)
# 		print(imgname)
		print(sample_rgb_img.shape)
		fig = plt.figure(figsize=(1, 1))
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(sample_rgb_img, vmin=0, vmax=1)
		imgname = '%s_sample.png' % (prefix)
		plt.savefig(imgname, dpi=T * A)  # bbox_inches='tight', pad_inches=0)
		print(imgname)
