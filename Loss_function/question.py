import numpy as np
import open3d as o3d
from optimizer import *
from support import *
depthpath = "./depth/" + str(50) + ".png"
img = cv2.imread(depthpath, 2) / 1000
points,pixel = depthImage2PointCloud(img)

posepath = "./pose/"+str(50)+".txt"
pose = readCameraMatrix(posepath)
pose_verse= np.linalg.inv(pose)
print(pose)
print(pose_verse)

pcdhelper = get3dPointCloud(50)
#print("before point",np.array(pcdhelper.points))
#pcdhelper.transform(pose)
#pcdhelper.transform(pose_verse)
print("later point",np.array(pcdhelper.points))

INDEX = 0
opt = Optimizer()
Intrinsic =np.array([[1169.621094,0.000000 ,646.295044],
                    [0.000000, 1167.105103, 489.927032],
                    [0.000000, 0.000000, 1.000000]])
kx, ky =640/1296,480/968
Intrinsic[0] = Intrinsic[0]*kx
Intrinsic[1] = Intrinsic[1]*ky

intrinsic = np.array([[577.590698, 0, 318.905426, 0.000000],
                     [0, 578.729797, 242.683609, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
Intrinsic = Intrinsic[:3,:3]
pose = np.eye(4)

pointlist1=np.array(pcdhelper.points)
matchlist1 = np.array(pixel)

def custom_draw_geometry_with_view_tracking(meshes):
    def track_view(vis):
        global matchlist1, pointlist1, INDEX, opt, pose, E0, Intrinsic
        if (INDEX % 1000 == 0):
            # reverse
            # import pdb
            # pdb.set_trace()
            #print(E0)
            print("--------------")
            #################
            verse = np.eye(4)
            verse[:3,:3] = np.linalg.inv(pose[:3,:3])
            #arrow2.transform(pose)  # 变回去
            #################
            E = []
            for i in range(len(matchlist1)):
                E.append(E0.copy())
            # print("***",matchlist1,pointlist1)
            print("interation", str(INDEX / 1000), "--loss:",
                  opt.loss(np.zeros((1, 3)), np.zeros((1, 3)), Intrinsic,E, matchlist1, pointlist1)[0][0])
            E = opt.image_R(E, matchlist1, pointlist1,Intrinsic)
            E = opt.image_t(E, matchlist1, pointlist1,Intrinsic)
            ###################
            pose = np.eye(4)
            pose[:3, :3] = E[0][:3,:3]

            ###################

            print(E[0])
            E0 = E[0].copy()
            vis.update_geometry()
        INDEX += 1

    o3d.draw_geometries_with_animation_callback(meshes, track_view)


custom_draw_geometry_with_view_tracking([])
