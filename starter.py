#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)

files = glob('deploy/trainval/*/*_image.jpg')
# idx = np.random.randint(0, len(files))
# snapshot = files[idx]

pd_writer = pd.DataFrame(files)
pd_writer.to_csv('train.txt', index=False, header=False)

# data = []

for idx, snapshot in enumerate(files):
    img = plt.imread(snapshot)

    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])

    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)

    bbox = bbox.reshape([-1, 11])

    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]

    # clr = np.linalg.norm(xyz, axis=0)
    # fig1 = plt.figure(1, figsize=(16, 9))
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.imshow(img)
    # ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)
    # ax1.axis('scaled')
    # fig1.tight_layout()

    # fig2 = plt.figure(2, figsize=(8, 8))
    # ax2 = Axes3D(fig2)
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')

    # step = 5
    # ax2.scatter(
    #     xyz[0, ::step], xyz[1, ::step], xyz[2, ::step],
    #     c=clr[::step], marker='.', s=1
    # )

    # colors = ['C{:d}'.format(i) for i in range(10)]

    # print(bbox)
    # import cv2

    # implot = plt.imshow(img)
    coords = []

    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]

        # print(vert_2D)
        # print(vert_2D.shape)
        coords = np.array([axis for axis in vert_2D.T])

    max_x = 0
    max_y = 0
    min_x = float('inf')
    min_y = float('inf')

    for point in coords:
        # print(point)
        max_x = max(max_x, point[0])
        max_y = max(max_y, point[1])
        min_x = min(min_x, point[0])
        min_y = min(min_y, point[1])

    # print(min_x, min_y, max_x, max_y )

    min_x = max(0, min_x - 30)
    min_y = max(0, min_y - 30)
    max_x = min(max_x + 30, img.shape[1])
    max_y = min(max_y + 30, img.shape[0])
    # w = img.shape[0]
    # h = img.shape[1]
    # b = [min_x, max_x, min_y, max_y]
    # bb = convert((w,h), b)
    # label = int(bbox[0][-2])
    # txt_outpath = snapshot.replace('.jpg', '.txt')
    # print("Output:" + txt_outpath)
    # txt_outfile = open(txt_outpath, "w")
    # txt_outfile.write(str(label) + " " + " ".join([str(a) for a in bb]) + '\n')
    # txt_outfile.close()
    # plt.plot(min_x, min_y, 'o')
    # plt.plot(min_x, max_y, 'o')
    # plt.plot(max_x, min_y, 'o')
    # plt.plot(max_x, max_y, 'o')
    # print(img.shape)
    # print(min_x, min_y, max_x, max_y)

    crop_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]

    cv2.imwrite(snapshot.replace('.jpg', '_cropped.jpg'), crop_img)
    print('Completed image: {}'.format(idx))

    # pd_writer = pd.DataFrame(data, columns=['a','b','c','d','e'])
    # pd_writer.to_csv(snapshot.replace('.jpg', '') + '.txt', index=False, header=False)
    # print(snapshot)
    # break
    # clr = colors[np.mod(k, len(colors))]
    # for e in edges.T:
    #     ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
    #     ax2.plot(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e], color=clr)

    # c = classes[int(b[9])]
    # ignore_in_eval = bool(b[10])
    # if ignore_in_eval:
    #     ax2.text(t[0], t[1], t[2], c, color='r')
    # else:
    #     ax2.text(t[0], t[1], t[2], c)

# ax2.auto_scale_xyz([-40, 40], [-40, 40], [0, 80])
# ax2.view_init(elev=-30, azim=-90)

# for e in np.identity(3):
#     ax2.plot([0, e[0]], [0, e[1]], [0, e[2]], color=e)

# plt.show()
