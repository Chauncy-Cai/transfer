import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from sklearn.neighbors import NearestNeighbors
import random
import gauss

def decompose(index,imageshape):
    l = imageshape[1]
    #print("should be 640",l)
    y = index%l
    x = index//l
    return [x,y]

def store_vec(filename,vec):
    f = open(filename)
    for v in vec:
        f.write(str(v)+"\t")
    return

def depthImage2PointCloud(img,intrinsic = None):
    #print(intrinsic)
    if intrinsic is None:
        intrinsic = [[577.590698, 0, 318.905426, 0.000000],
                     [0, 578.729797, 242.683609, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
    print(intrinsic)
    intrinsic = np.array(intrinsic)
    height,width = img.shape
    print(height,width)
    ys,xs=np.meshgrid(range(height),range(width),indexing='ij')
    vertex = np.zeros([height*width,3])
    vertex[:, 2] = img.flatten()
    vertex[:, 0] = ((xs - intrinsic[0, 2]) / intrinsic[0, 0]).flatten() * vertex[:, 2]
    vertex[:, 1] = ((ys - intrinsic[1, 2]) / intrinsic[1, 1]).flatten() * vertex[:, 2]
    return vertex

def blockStat(bs,match,indexlist1,indexlist2,image1,image2):
    print("match:", len(match),"\t bs:",bs*2+1)
    try:
        matchtemp = random.sample(match,500)
    except:
        matchtemp = match
        print("too less sample")
    print("extract 1000 points for calculation")

    possibilityMatrix = gauss.getGuassPossiblityTensor()

    # 正确匹配
    midloss_1 = []
    for i in range(len(matchtemp)):
        mapindex1 = indexlist1[matchtemp[i][0]]
        mapindex2 = indexlist2[matchtemp[i][1]]
        x1, y1 = decompose(mapindex1, image1.shape)
        x2, y2 = decompose(mapindex2, image2.shape)  #

        image1xmin = max(x1 - bs, 0)
        image1ymin = max(y1 - bs, 0)
        image2xmin = max(x2 - bs, 0)
        image2ymin = max(y2 - bs, 0)
        # print(x1,x2,y1,y2)
        # get the block
        temp1 = image1[image1xmin:(x1 + bs), image1ymin:(y1 + bs)]
        temp2 = image2[image2xmin:(x2 + bs), image2ymin:(y2 + bs)]
        temp1 = temp1.reshape(-1, 3)
        temp2 = temp2.reshape(-1, 3)
        # find the color center => Guass seperate
        ##initialize
        tensor1 = np.zeros((256, 256, 256))
        tensor2 = np.zeros((256, 256, 256))

        ##collection information from two small pixel-blocks
        for j in range(len(temp1)):
            try:
                tensor1 = gauss.blockReplace(
                    tensor1, temp1[j], possibilityMatrix)
            except:
                print(temp1[j])
        for j in range(len(temp2)):
            try:
                tensor2 = gauss.blockReplace(
                    tensor2, temp2[j], possibilityMatrix)
            except:
                print(temp2[j])
        tensor1 = tensor1 / (np.sum(tensor1) + 1e-7)
        tensor2 = tensor2 / (np.sum(tensor2) + 1e-7)
        loss = np.linalg.norm(tensor1.flatten() - tensor2.flatten())
        # print("loss",i," ",loss)
        midloss_1.append(loss)
    plt.hist(midloss_1, bins=100)
    plt.ylabel(str(bs*2+1)+'right match')
    plt.show()
    plt.savefig(str(bs*2+1)+'right match.jpg')
    store_vec(str(bs*2+1)+'_right_distribute.txt')
    # select
    # 错误匹配
    midloss_2 = []
    for i in range(len(matchtemp)):
        mapindex1 = indexlist1[matchtemp[i][0]]
        mapindex2 = indexlist2[matchtemp[(i + 3) % len(matchtemp)][1]]
        x1, y1 = decompose(mapindex1, image1.shape)
        x2, y2 = decompose(mapindex2, image2.shape)
        image1xmin = max(x1 - bs, 0)
        image1ymin = max(y1 - bs, 0)
        image2xmin = max(x2 - bs, 0)
        image2ymin = max(y2 - bs, 0)
        # print(x1,x2,y1,y2)
        # get the block
        temp1 = image1[image1xmin:(x1 + bs), image1ymin:(y1 + bs)]
        temp2 = image2[image2xmin:(x2 + bs), image2ymin:(y2 + bs)]
        # find the color center => Guass seperate
        ##initialize
        tensor1 = np.zeros((256, 256, 256))
        tensor2 = np.zeros((256, 256, 256))
        temp1 = temp1.reshape(-1, 3)
        temp2 = temp2.reshape(-1, 3)
        ##collection information from two small pixel-blocks
        for j in range(len(temp1)):
            try:
                tensor1 = gauss.blockReplace(
                    tensor1, temp1[j], possibilityMatrix)
            except:
                print(temp1[j])
        for j in range(len(temp2)):
            try:
                tensor2 = gauss.blockReplace(
                    tensor2, temp2[j], possibilityMatrix)
            except:
                print(temp2[j])
        tensor1 = tensor1 / (np.sum(tensor1) + 1e-7)
        tensor2 = tensor2 / (np.sum(tensor2) + 1e-7)
        loss = np.linalg.norm(tensor1.flatten() - tensor2.flatten())
        # print("loss",i," ",loss)
        midloss_2.append(loss)
    plt.hist(midloss_2, bins=100)
    plt.ylabel(str(bs*2+1)+'wrong match')
    plt.show()
    plt.savefig(str(bs*2+1)+'teeth wrong match.jpg')
    store_vec(str(bs * 2 + 1) + '_wrong_distribute.txt')

    # 计算区分度
    for i in range(99):
        gt = np.percentile(np.array(midloss_1), i + 1)  # right
        wm = np.percentile(np.array(midloss_2), 99 - i)  # wrong
        if (gt - wm > 0):
            print(i)
            # print("hopeing",abs(gt-wm))
            break

    return

def see2pcd(fid,depth,image,pcd,intrinsics):
    intrinsics = np.array(intrinsics)
    depthshape = depth.shape
    reshapesize = (depthshape[1],depthshape[0])
    color = cv2.resize(image,reshapesize)
    colorlist = np.array(color).reshape((-1,3))/256
    colorlist = colorlist[0::8]
    points = depthImage2PointCloud(depth,intrinsic=intrinsics)
    indexlist = range(len(points))
    points = points[0::8]
    indexlist = indexlist[0::8]
    #print(len(indexlist),len(points))
    assert(len(indexlist)==len(points))
    pcdnew = o3d.PointCloud()
    pcdnew.points = o3d.utility.Vector3dVector(points)
    pcdnew.colors = o3d.utility.Vector3dVector(colorlist)
    #defaultT = np.eye(4)
    #defaultT[:,3]=np.array([2,3,4,5 ])#seperate
    o3d.write_point_cloud(str(fid)+ "_new.ply",pcdnew,write_ascii=True)#还没转动
    o3d.write_point_cloud(str(fid)+ "_old.ply",pcd,write_ascii=True)
    return pcdnew,color,indexlist

def compare(parameter):
    picindex1 = 000
    picindex2 = 111
    print("welcome to compare")
    pc1,image1,indexlist1 = parameter['0']
    pc2,image2,indexlist2 = parameter['1']
    o3d.write_point_cloud("combine.ply",pc1+pc2,write_ascii=True)
    print("image shape",image1.shape)
    points1 = np.array(pc1.points)
    points2 = np.array(pc2.points)
    print("point count:",len(points1))
    print("point count:",len(points2))
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(points2)
    print("neighbor match finish")
    distance, indices = nbrs.kneighbors(points1)
    del nbrs
    #thersh = np.sqrt(distance)
    thersh = np.percentile(np.array(distance),10)
    #thersh = 2e-2
    #showing
    print("thersh is",thersh)
    plt.hist(distance, bins=100)
    plt.ylabel('Distance of neighbor points'+str(picindex1)+"+"+str(picindex2))
    plt.show()
    #match
    match = []
    print("count ",len(distance))
    for i in range(len(distance)):
        if points1[i][0]+points1[i][1]+points1[i][2] ==0:
            continue
        if distance[i]<thersh:
            match.append([i,indices[i][0]])
    #o3d.visualization.draw_geometries([pc1, pc2])
    ###watch mid point
    ################

    bs = 3
    blockStat(bs, match, indexlist1, indexlist2, image1, image2)
    bs = 7
    blockStat(bs, match, indexlist1, indexlist2, image1, image2)
    bs = 15
    blockStat(bs, match, indexlist1, indexlist2, image1, image2)
    bs = 31
    blockStat(bs, match, indexlist1, indexlist2, image1, image2)
    bs = 63
    blockStat(bs, match, indexlist1, indexlist2, image1, image2)
    return

frame = 0
parameter = {}
def AngularDistance(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R = R1.dot(R2.T)
    theta = np.arccos(np.clip((np.sum(np.diag(R)) - 1.0)/2.0, -1, 1))
    theta = theta / np.pi * 180.0
    terr = np.linalg.norm(T1[:3, 3] - T2[:3, 3], 2)
    if (theta > 3) or (terr >= 0.3):
        return True
    else:
        return False

period = 20
o3d.set_verbosity_level(o3d.VerbosityLevel.Debug)
if True:
    import glob
    pose_files = glob.glob('stream/*.pose')
    num_frames = len(pose_files)
    frame_trace = [i for i in range(num_frames) if i % period == 0]
    #frame_trace += [i for i in range(450, 480, 2)]
    #frame_trace += [445]
    #compareA = args.compareA #对比的两个点云
    #compareB = args.compareB
    frame_trace = sorted(frame_trace)

pcd_partial = o3d.PointCloud()
last_T = None

def custom_draw_geometry_with_view_tracking(mesh,generate,period):
    def track_view(vis):
        global frame, poses
        global num_frames, last_T
        global parameter
        ctr = vis.get_view_control()
        if frame == 0:
            params = ctr.convert_to_pinhole_camera_parameters()
            intrinsics = params.intrinsic
            extrinsics = params.extrinsic

            pose = np.array([[-0.8188861922, 0.3889273405, -0.4220911372, -14.6068376600],
                    [-0.1157361687, -0.8321937190, -0.5422718444, 23.0477832143],
                    [-0.5621659395, -0.3952077147, 0.7264849060, 4.1193224787],
                    [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])
            params = o3d.camera.PinholeCameraParameters()
            params.extrinsic = pose
            params.intrinsic = intrinsics
            ctr.convert_from_pinhole_camera_parameters(params)
        fid = frame % num_frames

        keyindex1 = 230
        keyindex2 = 400

        if True:
            intrinsics = o3d.read_pinhole_camera_intrinsic("stream/%d.intrinsics.json" % fid)
            T = np.loadtxt('stream/%d.pose' % fid)
            params = o3d.camera.PinholeCameraParameters()
            params.extrinsic = T
            params.intrinsic = intrinsics
            ctr.convert_from_pinhole_camera_parameters(params)#很重要，不能删除
            if (fid == keyindex1)|(fid==keyindex2):
                """ Generate Point Cloud """
                if (last_T is None) or ((fid in frame_trace) or (AngularDistance(T, last_T))):
                    print('%d/%d' % (fid, num_frames))
                    depth = vis.capture_depth_float_buffer(False)
                    depth = np.array(depth)
                    idx = np.where(depth > 30)
                    depth[idx] = 0
                    depth = o3d.Image(depth)
                    image = vis.capture_screen_float_buffer(False)
                    image = o3d.Image(np.array(np.array(image)*255).astype(np.uint8))
                    rgbd = o3d.create_rgbd_image_from_color_and_depth(
                                image, depth, convert_rgb_to_intensity = False)
                    rgbd.depth = o3d.Image(np.array(rgbd.depth)*1000)
                    pcd = o3d.create_point_cloud_from_rgbd_image(rgbd, intrinsics)
                    pcd.transform(np.linalg.inv(T))
                    o3d.write_point_cloud("stream/%d.ply" % fid, pcd, write_ascii=True)
                    cv2.imwrite("stream/%d.png" % fid, np.array(image))
                    depth = np.array(depth)
                    #depth = np.array(depth)*1000
                    #depth.astype(np.uint16)
                    #cv2.imwrite("stream/%d_depth.png" % fid, depth)##
                    last_T = T
                    if (fid==keyindex1):
                        intrinsic_matrix = intrinsics.intrinsic_matrix

                        pcd0,image,indexlist = see2pcd(fid,depth,np.array(image),pcd,intrinsics.intrinsic_matrix)
                        pcd0.transform(np.linalg.inv(T))
                        #print(T)
                        parameter['0']=[pcd0,image,indexlist]
                    if (fid==keyindex2):
                        intrinsic_matrix = intrinsics.intrinsic_matrix

                        pcd1,image,indexlist = see2pcd(fid,depth,np.array(image),pcd,intrinsics.intrinsic_matrix)
                        pcd1.transform(np.linalg.inv(T))
                        #print(T)
                        parameter['1']=[pcd1,image,indexlist]
                        vis.destroy_window()
                        #o3d.draw_geometries([parameter['0'][0],pcd1])
                        compare(parameter)
                        os.system("pause")
            if fid == num_frames - 1:
                exit()
        frame += 1

    o3d.draw_geometries_with_animation_callback([mesh, pcd_partial], track_view,width=1920, height=1080)


def main():
    mesh = o3d.read_triangle_mesh("box.ply")#########change here
    generate = True
    period = 20
    custom_draw_geometry_with_view_tracking(mesh,generate,period)






if __name__ == "__main__":
    main()
