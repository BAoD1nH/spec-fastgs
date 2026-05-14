# ============================================================
# COLMAP Loader (Final: FastGS base)
# ============================================================

import numpy as np
import collections
import struct


CameraModel = collections.namedtuple(
	"CameraModel", ["model_id", "model_name", "num_params"]
)

Camera = collections.namedtuple(
	"Camera", ["id", "model", "width", "height", "params"]
)

BaseImage = collections.namedtuple(
	"Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

Point3D = collections.namedtuple(
	"Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


# ------------------------------------------------------------
# CAMERA MODELS
# ------------------------------------------------------------

CAMERA_MODELS = {
	CameraModel(0, "SIMPLE_PINHOLE", 3),
	CameraModel(1, "PINHOLE", 4),
	CameraModel(2, "SIMPLE_RADIAL", 4),
	CameraModel(3, "RADIAL", 5),
	CameraModel(4, "OPENCV", 8),
	CameraModel(5, "OPENCV_FISHEYE", 8),
	CameraModel(6, "FULL_OPENCV", 12),
	CameraModel(7, "FOV", 5),
	CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
	CameraModel(9, "RADIAL_FISHEYE", 5),
	CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}

CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}


# ------------------------------------------------------------
# ROTATION
# ------------------------------------------------------------

def qvec2rotmat(q):
	return np.array([
    	[1 - 2*q[2]**2 - 2*q[3]**2,
     	2*q[1]*q[2] - 2*q[0]*q[3],
     	2*q[3]*q[1] + 2*q[0]*q[2]],

    	[2*q[1]*q[2] + 2*q[0]*q[3],
     	1 - 2*q[1]**2 - 2*q[3]**2,
     	2*q[2]*q[3] - 2*q[0]*q[1]],

    	[2*q[3]*q[1] - 2*q[0]*q[2],
     	2*q[2]*q[3] + 2*q[0]*q[1],
     	1 - 2*q[1]**2 - 2*q[2]**2]
	])


class Image(BaseImage):
	def qvec2rotmat(self):
    	return qvec2rotmat(self.qvec)


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def read_next_bytes(fid, num_bytes, fmt, endian="<"):
	return struct.unpack(endian + fmt, fid.read(num_bytes))


# ------------------------------------------------------------
# POINT CLOUD
# ------------------------------------------------------------

def read_points3D_text(path):
	num = sum(1 for line in open(path) if line.strip() and line[0] != "#")

	xyzs = np.empty((num, 3))
	rgbs = np.empty((num, 3))
	errors = np.empty((num, 1))

	with open(path, "r") as f:
    	i = 0
    	for line in f:
        	if line.strip() and line[0] != "#":
            	elems = line.split()
            	xyzs[i] = np.array(elems[1:4], float)
            	rgbs[i] = np.array(elems[4:7], int)
            	errors[i] = float(elems[7])
            	i += 1

	return xyzs, rgbs, errors


# ------------------------------------------------------------
# EXTRINSICS / INTRINSICS
# ------------------------------------------------------------

def read_intrinsics_text(path):
	cams = {}

	with open(path, "r") as f:
    	for line in f:
        	if line.strip() and line[0] != "#":
            	elems = line.split()

            	cid = int(elems[0])
            	model = elems[1]

            	assert model == "PINHOLE"

            	cams[cid] = Camera(
                	id=cid,
                	model=model,
                	width=int(elems[2]),
                	height=int(elems[3]),
                	params=np.array(elems[4:], float)
            	)

	return cams


def read_extrinsics_text(path):
	images = {}

	with open(path, "r") as f:
    	while True:
        	line = f.readline()
        	if not line:
            	break

        	if line.strip() and line[0] != "#":
            	elems = line.split()

            	iid = int(elems[0])
            	qvec = np.array(elems[1:5], float)
            	tvec = np.array(elems[5:8], float)

            	cid = int(elems[8])
            	name = elems[9]

            	elems = f.readline().split()
            	xys = np.column_stack([
                	elems[0::3],
                	elems[1::3]
            	]).astype(float)

            	point_ids = np.array(elems[2::3], int)

            	images[iid] = Image(
                	iid, qvec, tvec, cid, name, xys, point_ids
            	)

	return images
