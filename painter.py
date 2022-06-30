import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from sklearn.cluster import KMeans
from PIL import Image

class Painter():
	def __init__(self, image):
		self.image = image
		self.smol_img = np.array(Image.fromarray(image).resize((image.shape[0]//10,image.shape[1]//10),0))
		self.color_df = pd.DataFrame(data={color:self.image[:,:,ii].ravel() for ii,color in enumerate(['r','g','b']) })
		self.smol_color_df = pd.DataFrame(data={color:self.smol_img[:,:,ii].ravel() for ii,color in enumerate(['r','g','b']) })
		self.fitted_kmeans = {ii:None for ii in range(1,10)}
		self.painted_images = {ii:None for ii in range(1,10)}

	def _fit_model(self, n_clusters):
		kmeans = KMeans(n_clusters = n_clusters)
		kmeans.fit(self.smol_color_df)
		self.fitted_kmeans[n_clusters] = kmeans

	def paint_image(self, n_clusters):
		if self.fitted_kmeans[n_clusters] == None:
			self._fit_model(n_clusters)

		kmeans = self.fitted_kmeans[n_clusters]
		return np.array([kmeans.cluster_centers_[label] for label in kmeans.predict(self.color_df)], dtype=np.uint8).reshape(self.image.shape)


if __name__ == '__main__':
	p = Painter(imread("images/roses.jpeg"))
	for ii in range(1,10): p._paint_image(ii)