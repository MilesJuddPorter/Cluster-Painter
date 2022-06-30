import streamlit as st
import pandas as pd
import numpy as np
from painter import Painter

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
matplotlib.use('agg')
from PIL import Image

import cv2


### SET UP ###
def get_image(image, n_clusters):
	if image == "uploaded":
		painter_dict["uploaded"] = Painter(imread(image_file))
	p = painter_dict[image]
	return p.paint_image(n_clusters)

def create_selected_color_image(selected_colors):
	base_img = np.zeros((100,1000,3), dtype=np.uint8)
	for c_num, color in enumerate(selected_colors):
		c = np.array(color)
		for ii in range(100):
			for jj in range(100):
				base_img[ii, jj+(c_num*100), :] = c

	return base_img

def set_uploaded_pixel_plot(p):
	fig = plt.figure(figsize=(5,5))
	ax = plt.axes(projection='3d')
	xdata = p.smol_color_df['r'].values
	ydata = p.smol_color_df['g'].values
	zdata = p.smol_color_df['b'].values
	cdata = p.smol_color_df.to_numpy()/255
	ax.scatter3D(xdata,ydata,zdata,c=cdata, alpha=0.8)
	fig.canvas.draw()
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	len_data = len(data)
	data = data.reshape((500,500,3))
	return data

painter_dict = {name: Painter(imread(f"images/{name}.jpeg")) for name in ['roses', 'nebula', 'beach']}
uploaded_pixel_plot = None

rose_p = Painter(imread("images/roses.jpeg"))
nebula_p = Painter(imread("images/nebula.jpeg"))
beach_p = Painter(imread("images/beach.jpeg"))


### SIDE BAR ###
image = st.sidebar.radio("Image", ("roses", "nebula", "beach", "uploaded"))
st.sidebar.write("\n")
n_clusters = st.sidebar.slider("n_clusters", 1,9,3)
st.sidebar.write("\n")
image_file = st.sidebar.file_uploader("Upload Files")


### MAIN SCREEN ###

st.write("""
# KMeans Clustering For Image Colors

Select your image and use the n_cluster slider on the left
to how a computer can use KMeans unsupervised clustering algorithm to repaint an image!

	""")

col1, col2 = st.columns(2)

with col1:
    st.header("Base Image")
    if image == "uploaded":
    	st.image(imread(image_file), width=350)
  
    else:
    	st.image(imread(f"images/{image}.jpeg"), width=350)
    

with col2:
    st.header(f"Repainted: {n_clusters} Color(s)")
    st.image(get_image(image, n_clusters), width=350)

st.write("Colors in the Palette")
st.image(create_selected_color_image(painter_dict[image].fitted_kmeans[n_clusters].cluster_centers_))

st.write("Plotted Pixels")
if image != "uploaded":
	st.image(imread(f"plotted_pixels/{image}.jpg"))
else:
	if uploaded_pixel_plot == None:
		st.image(set_uploaded_pixel_plot(painter_dict["uploaded"]))

### MORE INFO ###
st.write("""
## More Info
Every image contains lots of pixels, each with a specific color. This color is represented
by RGB (red, green, blue) values. If you were to take all the RGB values from an image
you could create a dataset where each row is a pixel and the columns are the rgb values for that pixel.

You could imagine this being plotted in 3 dimensions where X axis is red, Y axis is blue,
and a Z axis going up would be green. Every pixel would exist somehwere in this plot.
(This plot can be seen above)

The goal of clustering is to summarize or group data into a certain number of clusters.
Thinking again about the plot, what clustering would be doing is finding clusters of
pixels that were all close together (or similar colors) and creating a point in the center of those.

What is nice about clustering is we can choose how many of these clusters we want to look for and
as we change this number we start to get closer to original representation of the data. Clustering
in a way is a compression or summary of our data.

Each cluster that we find in our plot would also represent a specific color. This allows us to go
repaint the image (by setting each pixel to it's closest cluster center - or color).

The more clusters you allow, the more colors you let the computer use in it's repainting.
	""")