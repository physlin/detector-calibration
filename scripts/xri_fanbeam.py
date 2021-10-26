import astra
import xri
import h5py
import numpy as np
from PIL import Image
from tomopy.misc.corr import circ_mask

indir = "/home/lindacac/yd88/Linda/Delta_X-ray_Lab/delta_2021-02-11_Croton_Rings/phantom_320ms_30kV_23W_sod50_sid150_1/"
volume = "corr_volume_sm50_fixed_3sig"
fb_slice = 516
skip_num = 5
source_origin = 0.50
origin_det = 1.00
det_spacing = 50.e-6
pixelsize = det_spacing / ((source_origin + origin_det) / source_origin)
pixel_offset = -2.8
sample_offset = pixel_offset * pixelsize
full_rotation = 360.
outname = indir+"RECON/recon_FBP_cor"+str("{:.1f}".format(pixel_offset))+"_rot"+str("{:.1f}".format(full_rotation))+"_"+volume+"_"+str(fb_slice).zfill(4)+".tif"

print("Loading sinogram...")
print("...slice",fb_slice)

with h5py.File(indir+volume+".h5", "r") as hf:
    sinogram = hf[volume][skip_num:,:,fb_slice]

sinogram[np.where(sinogram<=0.)] = 1e-08
sinogram[np.where(np.isnan(sinogram))] = 1e-08
sinogram = -np.log(sinogram)

outim = Image.fromarray(sinogram)
outim.save("sino.tif")

num_of_projections = sinogram.shape[0]
angles_deg = np.linspace(0, full_rotation, num=num_of_projections, endpoint=False)

print("Reconstructing slice using sample offset",sample_offset,"m "+str(repr("(")[1:-1])+str(pixel_offset),"pixels) and",full_rotation,"degrees of rotation over",sinogram.shape[0],"projections...")

#recon = xri.ct._astra.reconstruct_fanflat(sinogram=sinogram, source_origin=source_origin, origin_det=origin_det, angles_deg=angles_deg, det_spacing=det_spacing, sample_offset=sample_offset, astra_reconstruction_algorithm="FBP_CUDA")
recon = xri.ct._astra.FBP_fanflat(sinogram=sinogram, source_origin=source_origin, origin_det=origin_det, angles_deg=angles_deg, det_spacing=det_spacing, sample_offset=sample_offset)

recon = np.expand_dims(recon, 0)
recon = circ_mask(recon, axis=0, ratio=1, val=0.0)
recon = np.squeeze(recon, 0)
recon /= pixelsize

outim = Image.fromarray(recon)
outim.save(outname)

