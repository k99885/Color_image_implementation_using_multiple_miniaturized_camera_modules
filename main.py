import cv2
import numpy as np
# ############## 분리된 합성###################
#
# # # 이미지 불러오기
img_names = ['images4/RED_5.jpg', 'images4/GREEN_5.jpg','images4/BLUE_5.jpg']

# r_channel = cv2.imread(img_names[0] ,cv2.IMREAD_GRAYSCALE)
# g_channel = cv2.imread(img_names[1] ,cv2.IMREAD_GRAYSCALE)
# b_channel = cv2.imread(img_names[2], cv2.IMREAD_GRAYSCALE)

r_channel = cv2.imread(img_names[0])
g_channel = cv2.imread(img_names[1])
b_channel = cv2.imread(img_names[2])

# width1,height1=512,682
width1,height1=r_channel.shape[1],r_channel.shape[0]

print(r_channel.shape)

r_channel_resized = cv2.resize(r_channel, (width1, height1))
g_channel_resized = cv2.resize(g_channel, (width1, height1))
b_channel_resized = cv2.resize(b_channel, (width1, height1))

print(r_channel_resized.shape)

cv2.imshow("r_channel_resized",r_channel_resized)
cv2.imshow("g_channel_resized",g_channel_resized)
cv2.imshow("b_channel_resized",b_channel_resized)

# merged_image = cv2.merge((b_channel, g_channel, r_channel))
#
# # Save or display the merged image
# cv2.imwrite('merged_image.jpg', merged_image)
# print(r_channel[500][555])
# print(r_channel.shape)

_, _, r_channel_resized = cv2.split(r_channel_resized)
_, g_channel_resized, _ = cv2.split(g_channel_resized)
b_channel_resized, _, _ = cv2.split(b_channel_resized)
#
# print(r_channel.shape)
cv2.imshow("r_channel_resized_c1",r_channel_resized)
cv2.imshow("g_channel_resized_c1",g_channel_resized)
cv2.imshow("b_channel_resized_c1",b_channel_resized)

def align_images(im1, im2):
    # ORB 특징 검출 및 디스크립터 계산
    # orb = cv2.SIFT_create()#edgeThreshold=10)#, fastThreshold=20)
    orb = cv2.ORB_create()#edgeThreshold=10)#, fastThreshold=20)

    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)


    # 특징 매칭
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # 매칭 점수를 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    print('len(matches):',len(matches))
    # matches = matches[:300]

    matched_image = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 좋은 매칭의 위치 추출
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)


    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # 호모그래피 찾기
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 호모그래피를 사용하여 이미지 변환
    height, width = im1.shape
    im1_aligned = cv2.warpPerspective(im1, h, (width, height))

    return im1_aligned,matched_image





orb = cv2.ORB_create()
# orb = cv2.SIFT_create()


keypoints_r = orb.detect(r_channel_resized, None)
keypoints_g = orb.detect(g_channel_resized, None)
keypoints_b = orb.detect(b_channel_resized, None)
#
#
image_with_keypoints_r = cv2.drawKeypoints(r_channel_resized, keypoints_r, None, color=(0,255,0))
image_with_keypoints_g = cv2.drawKeypoints(g_channel_resized, keypoints_g, None, color=(0,255,0))
image_with_keypoints_b = cv2.drawKeypoints(b_channel_resized, keypoints_b, None, color=(0,255,0))

# 이미지 표시
cv2.imshow("image_with_keypoints_r", image_with_keypoints_r)
cv2.imshow("image_with_keypoints_g", image_with_keypoints_g)
cv2.imshow("image_with_keypoints_b", image_with_keypoints_b)


#######레드 기준###########
# g_channel_aligned = align_images(g_channel_resized, r_channel_resized)
# b_channel_aligned = align_images(b_channel_resized, r_channel_resized)
#
# merged_image = cv2.merge((b_channel_aligned, g_channel_aligned, r_channel_resized))
#######그린 기준###########

r_channel_aligned,matched_image_r_g  = align_images(r_channel_resized, g_channel_resized)
b_channel_aligned,matched_image_b_g  = align_images(b_channel_resized, g_channel_resized)

merged_image = cv2.merge((b_channel_aligned, g_channel_resized, r_channel_aligned))

cv2.imshow('r_channel_aligned', r_channel_aligned)
cv2.imshow('b_channel_aligned', b_channel_aligned)
# ######  블루기준 ############
# r_channel_aligned = align_images(r_channel_resized, b_channel_resized)
# g_channel_aligned = align_images(g_channel_resized, b_channel_resized)
#
# merged_image = cv2.merge((b_channel_resized, g_channel_aligned, r_channel_aligned))

#
# # 병합된 이미지 저장 또는 표시
cv2.imwrite('aligned_merged_image.jpg', merged_image)
cv2.imshow('aligned_merged_image.jpg', merged_image)

cv2.imshow('Matched Keypoints R-G', matched_image_r_g)
cv2.imshow('Matched Keypoints B-G', matched_image_b_g)

cv2.waitKey(0)
cv2.destroyAllWindows()
